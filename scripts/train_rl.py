"""Online GRPO training for multi-fault debugging.

Trains a small model to fix code bugs using reinforcement learning.
The reward signal comes from the functional judge (pytest T_fail tests).

Strategy:
  - Train on n=1 corruptions (single bug) — tractable for a small model
  - Reward: functional score (0.0 if bug remains, 1.0 if all T_fail tests pass)
  - GRPO with G=8 rollouts per prompt — variance reduction over REINFORCE
  - LoRA fine-tuning for memory efficiency
  - Eval on n=1,3,5 every --eval-interval steps

Dependencies (install if missing):
  pip install trl>=0.12 peft accelerate

Usage:
    # Baseline eval before training
    python scripts/run_eval_local.py \\
        --model Qwen/Qwen2.5-Coder-1.5B-Instruct \\
        --domain sklearn --catalog sgs \\
        --tp 1 --counts 1,3,5 --trials 3

    # RL training
    python scripts/train_rl.py \\
        --model Qwen/Qwen2.5-Coder-1.5B-Instruct \\
        --domain sklearn \\
        --steps 200 \\
        --G 8 \\
        --train-n 1 \\
        --out checkpoints/rl_1.5b_sklearn

    # Post-training eval
    python scripts/run_eval_local.py \\
        --model checkpoints/rl_1.5b_sklearn \\
        --domain sklearn --catalog sgs \\
        --tp 1 --counts 1,3,5 --trials 3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HOME", "/mnt/localssd/cdbench/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from scripts.benchmark_ablation import (
    load_snippet_files, apply_corruptions, extract_fixed_files,
    select_by_diversity,
)
from scripts.functional_judge import SANDBOX_ROOT, judge_fix
import scripts.functional_judge as _fj
from scripts.run_eval_local import (
    DOMAIN_CONFIG, load_tfail_cache, filter_healthy,
    load_sgs_catalog, build_domain_prompt, build_findreplace_prompt,
    extract_findreplace_fixes, _apply_findreplace_to_sandbox,
)


# ── Reward function ───────────────────────────────────────────────────────────

def score_completion(
    completion: str,
    corruptions: list,
    sandbox: Path,
    tfail_cache: dict,
    git_target: str,
    fmt: str = "full_file",
) -> float:
    """Apply model fix to sandbox, run T_fail tests, return functional score."""
    import subprocess as sp

    if fmt == "find_replace":
        fixes = extract_findreplace_fixes(completion)
        if not fixes:
            return 0.0
        full_fixed = _apply_findreplace_to_sandbox(fixes, corruptions, sandbox)
    else:
        fixed = extract_fixed_files(completion)
        if not fixed:
            return 0.0
        full_fixed = {}
        for rel_path, snippet_fix in fixed.items():
            target = sandbox / rel_path
            if not target.exists():
                continue
            full_text = target.read_text()
            for c in corruptions:
                if c.source_file != rel_path:
                    continue
                if c.find in snippet_fix and c.replace not in snippet_fix:
                    if c.replace in full_text:
                        full_text = full_text.replace(c.replace, c.find, 1)
            full_fixed[rel_path] = full_text

    if not full_fixed:
        return 0.0

    sp.run(["git", "checkout", "--", git_target], cwd=sandbox, check=True,
           capture_output=True)
    try:
        for c in corruptions:
            target = sandbox / c.source_file
            if target.exists():
                txt = target.read_text()
                if c.find in txt:
                    target.write_text(txt.replace(c.find, c.replace, 1))
        result = judge_fix(corruptions, full_fixed, sandbox, tfail_cache=tfail_cache)
        return float(result.score)
    except Exception as e:
        print(f"    [judge] ERROR: {e}")
        return 0.0
    finally:
        sp.run(["git", "checkout", "--", git_target], cwd=sandbox, check=True,
               capture_output=True)


def make_reward_fn(healthy: list, tfail_cache: dict, cfg: dict, catalog_by_id: dict,
                   fmt: str = "full_file"):
    """Return a reward function compatible with TRL GRPOTrainer."""
    sandbox = cfg["sandbox"]
    git_target = cfg["git_target"]

    def reward_fn(prompts, completions, corruption_ids_json, **kwargs):
        scores = []
        for completion, cids_json in zip(completions, corruption_ids_json):
            try:
                cids = json.loads(cids_json)
                corruptions = [catalog_by_id[cid] for cid in cids if cid in catalog_by_id]
                if not corruptions:
                    scores.append(0.0)
                    continue
                score = score_completion(
                    completion, corruptions, sandbox, tfail_cache, git_target, fmt=fmt
                )
            except Exception as e:
                print(f"    [reward] error: {e}")
                score = 0.0
            scores.append(score)
        return scores

    return reward_fn


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_training_dataset(
    healthy: list,
    cfg: dict,
    train_ns: list[tuple[int, float]],  # [(n_bugs, weight), ...] — curriculum
    diversity_modes: list[str],
    total_examples: int,
    seed: int = 42,
    fmt: str = "full_file",
) -> list[dict]:
    """Build GRPO training examples with a curriculum over bug counts.

    train_ns is a weighted distribution, e.g.:
      [(1, 0.6), (3, 0.3), (5, 0.1)]
    means 60% of examples have 1 bug, 30% have 3, 10% have 5.

    Starting with mostly n=1 ensures the model gets positive reward signal early.
    Including n=3 ensures the model sees multi-bug prompts — without this, n=3
    generalization at eval time is much weaker (distribution shift).
    """
    from scripts.benchmark_ablation import load_snippet_files, apply_corruptions, select_by_diversity
    rng = random.Random(seed)
    library = cfg["library"]
    system_prompt = cfg["system_prompt"]
    examples = []

    ns, weights = zip(*train_ns)
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    for _ in range(total_examples):
        n = rng.choices(ns, weights=weights, k=1)[0]
        div = rng.choice(diversity_modes)

        if n > len(healthy):
            n = 1  # fallback if not enough healthy corruptions

        pool = list(healthy)
        rng.shuffle(pool)
        selected = select_by_diversity(pool, n, div)
        if not selected:
            continue

        snippets = load_snippet_files(selected)
        corrupted = apply_corruptions(snippets, selected)
        descs = [c.description for c in selected]
        prompt_fn = build_findreplace_prompt if fmt == "find_replace" else build_domain_prompt
        user_prompt = prompt_fn(corrupted, n, descs, library)

        examples.append({
            "prompt": user_prompt,
            "system": system_prompt,
            "corruption_ids_json": json.dumps([c.corruption_id for c in selected]),
            "n": n,
            "diversity": div,
        })

    rng.shuffle(examples)
    return examples


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--domain", default="sklearn", choices=list(DOMAIN_CONFIG))
    ap.add_argument("--catalog", default="sgs", choices=["hand", "sgs", "all"])
    ap.add_argument("--train-ns", default="1:0.6,3:0.3,5:0.1",
                    help="curriculum: comma-separated n:weight pairs. "
                         "Default 60%% n=1 / 30%% n=3 / 10%% n=5. "
                         "Use '1:1.0' for n=1 only.")
    ap.add_argument("--diversity", default="scattered",
                    help="diversity modes for training examples (default: scattered only). "
                         "Use 'clustered,scattered' to include both.")
    ap.add_argument("--steps", type=int, default=200,
                    help="total GRPO training steps")
    ap.add_argument("--G", type=int, default=8,
                    help="number of rollouts per prompt (GRPO group size)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="max new tokens per rollout (8192 prevents clipped_ratio=1 on full-file outputs)")
    ap.add_argument("--out", default=None,
                    help="output checkpoint dir (default: checkpoints/rl_{model_slug}_{domain})")
    ap.add_argument("--eval-interval", type=int, default=20,
                    help="eval on held-out examples every N steps")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--total-examples", type=int, default=400,
                    help="total training examples in the dataset")
    ap.add_argument("--format", default="find_replace",
                    choices=["full_file", "find_replace"],
                    help="output format for training (default: find_replace — short outputs, no clipping)")
    args = ap.parse_args()

    cfg = DOMAIN_CONFIG[args.domain]
    _fj.TFAIL_CACHE = cfg["tfail_cache"]

    # Override source files
    from pathlib import Path as _P
    import scripts.benchmark_ablation as _ba
    _ba.SOURCE_FILES = cfg["source_files"](cfg["package_root"])

    # Load catalog
    import importlib
    mod_name, attr_name = cfg["catalog_module"]
    mod = importlib.import_module(mod_name)
    hand_catalog = list(getattr(mod, attr_name))

    if args.catalog == "hand":
        catalog = hand_catalog
    elif args.catalog == "sgs":
        catalog = load_sgs_catalog(args.domain)
    else:
        catalog = hand_catalog + load_sgs_catalog(args.domain)

    tfail = load_tfail_cache(catalog, cfg["tfail_cache"])
    healthy = filter_healthy(catalog, tfail)
    catalog_by_id = {c.corruption_id: c for c in catalog}

    # Parse curriculum
    train_ns = []
    for part in args.train_ns.split(","):
        n_str, w_str = part.strip().split(":")
        train_ns.append((int(n_str), float(w_str)))
    max_n = max(n for n, _ in train_ns)

    print(f"Domain: {args.domain}, Catalog: {args.catalog}")
    print(f"Total: {len(catalog)} corruptions, {len(healthy)} healthy")
    print(f"Curriculum: {train_ns}, G={args.G}, steps={args.steps}")

    if len(healthy) < max_n:
        raise SystemExit(f"Need at least {max_n} healthy corruptions for curriculum, have {len(healthy)}")

    # Hold out 20% for eval
    rng = random.Random(args.seed)
    shuffled = list(healthy)
    rng.shuffle(shuffled)
    n_eval = max(1, len(shuffled) // 5)
    eval_pool = shuffled[:n_eval]
    train_pool = shuffled[n_eval:]
    print(f"Train pool: {len(train_pool)}, Eval pool: {len(eval_pool)}")
    # Persist held-out IDs so run_eval_local.py can filter to clean held-out set
    _held_out_ids = [c.corruption_id for c in eval_pool]
    _train_ids = [c.corruption_id for c in train_pool]

    # Build dataset
    dataset_list = build_training_dataset(
        train_pool, cfg,
        train_ns=train_ns,
        diversity_modes=args.diversity.split(","),
        total_examples=args.total_examples,
        seed=args.seed,
        fmt=args.format,
    )
    print(f"Training examples: {len(dataset_list)}")

    ns_tag = "_".join(f"n{n}" for n, _ in train_ns)
    out_dir = Path(args.out) if args.out else (
        ROOT / "checkpoints" /
        f"rl_{args.model.split('/')[-1]}_{args.domain}_{args.catalog}_{ns_tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading {args.model}...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType
    import torch

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=os.environ["HF_HOME"], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=os.environ["HF_HOME"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Build reward function ─────────────────────────────────────────────────
    reward_fn = make_reward_fn(healthy, tfail, cfg, catalog_by_id, fmt=args.format)

    # ── TRL GRPOTrainer ───────────────────────────────────────────────────────
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    hf_dataset = Dataset.from_list(dataset_list)

    grpo_config = GRPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=max(1, args.steps * 1 // max(1, len(dataset_list))),
        max_steps=args.steps,
        per_device_train_batch_size=1,
        generation_batch_size=args.G,
        gradient_accumulation_steps=1,
        num_generations=args.G,
        max_completion_length=args.max_tokens,
        learning_rate=args.lr,
        logging_steps=1,
        save_steps=args.eval_interval,
        save_total_limit=3,
        bf16=True,
        remove_unused_columns=False,
        seed=args.seed,
        report_to="tensorboard",
        # GRPO-specific
        beta=0.04,           # KL penalty (low = more exploration)
        temperature=0.9,     # rollout temperature
        top_p=0.95,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=hf_dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\nStarting GRPO training for {args.steps} steps...")
    print(f"Checkpoints: {out_dir}")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/3600:.1f}h")

    # Save final merged model (LoRA merged into base)
    print("Saving merged model...")
    merged = model.merge_and_unload()
    merged.save_pretrained(str(out_dir / "merged"))
    tokenizer.save_pretrained(str(out_dir / "merged"))
    print(f"Merged model saved to {out_dir / 'merged'}")

    # ── Quick eval on held-out corruptions ────────────────────────────────────
    print(f"\nQuick eval on {len(eval_pool)} held-out corruptions (n=1)...")
    from vllm import LLM, SamplingParams as SP
    eval_llm = LLM(
        model=str(out_dir / "merged"),
        tensor_parallel_size=1,
        gpu_memory_utilization=0.35,
        max_model_len=8192,
        dtype="bfloat16",
        download_dir=os.environ["HF_HOME"],
    )
    eval_tokenizer = eval_llm.get_tokenizer()

    correct = 0
    for c in eval_pool:
        snippets = load_snippet_files([c])
        corrupted = apply_corruptions(snippets, [c])
        user_prompt = build_domain_prompt(corrupted, 1, [c.description], cfg["library"])
        msgs = [
            {"role": "system", "content": cfg["system_prompt"]},
            {"role": "user", "content": user_prompt},
        ]
        chat_prompt = eval_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        out = eval_llm.generate([chat_prompt], SP(temperature=0.0, max_tokens=4096))
        score = score_completion(
            out[0].outputs[0].text, [c], cfg["sandbox"], tfail, cfg["git_target"]
        )
        correct += score >= 1.0
        print(f"  {c.corruption_id}: {'FIXED' if score >= 1.0 else 'MISS'}")

    acc = correct / max(1, len(eval_pool))
    print(f"\nPost-RL eval: {correct}/{len(eval_pool)} = {acc:.0%} on n=1 held-out")

    # Save eval results + held-out/train split IDs for clean downstream eval
    (out_dir / "eval_results.json").write_text(json.dumps({
        "model": args.model,
        "post_rl_model": str(out_dir / "merged"),
        "domain": args.domain,
        "catalog": args.catalog,
        "format": args.format,
        "train_ns": args.train_ns,
        "steps": args.steps,
        "G": args.G,
        "n1_accuracy_held_out": acc,
        "n_eval_examples": len(eval_pool),
        "training_time_h": round(elapsed / 3600, 2),
        "held_out_ids": _held_out_ids,
        "train_ids": _train_ids,
    }, indent=2))
    print(f"Results saved to {out_dir}/eval_results.json")


if __name__ == "__main__":
    main()
