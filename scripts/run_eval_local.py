"""Local-vLLM ablation runner with dual scoring (text-match + functional).

This is the H100 replacement for benchmark_ablation.py's API-driven runner.
Differences:
  - Uses vLLM offline batched inference (no API)
  - Loads the model once, batches all prompts
  - Saves prompts, raw outputs, parsed fixes, AND both scores
  - Honors the T_fail cache produced by discover_tfail.py
  - Filters corruptions to those marked HEALTHY by the cache

Usage:
    python run_eval_local.py --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --domain sklearn --counts 1,3,5,7,10 --trials 3

    python run_eval_local.py --model Qwen/Qwen2.5-Coder-32B-Instruct \
        --tp 4 --domain fastapi --counts 1,3,5,7,10 --trials 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HOME", "/mnt/localssd/cdbench/hf_cache")

import scripts.benchmark_ablation as _ba
from scripts.functional_judge import SANDBOX_ROOT, judge_fix
import scripts.functional_judge as _fj

# ── Domain configuration ──────────────────────────────────────────────────────

_SANDBOX_BASE = Path(os.environ.get("CDBENCH_SANDBOX", "/mnt/localssd/cdbench/sandboxes"))
_CACHE_BASE = Path(os.environ.get("CDBENCH_TFAIL_CACHE", "/mnt/localssd/cdbench/tfail_cache"))

DOMAIN_CONFIG = {
    "sklearn": {
        "sandbox": _SANDBOX_BASE / "sklearn",
        "tfail_cache": _CACHE_BASE,
        "package_root": _SANDBOX_BASE / "sklearn" / "sklearn",
        "source_files": lambda pkg: {
            "sklearn/metrics/_classification.py": pkg / "metrics" / "_classification.py",
            "sklearn/metrics/_ranking.py":        pkg / "metrics" / "_ranking.py",
            "sklearn/preprocessing/_data.py":     pkg / "preprocessing" / "_data.py",
            "sklearn/linear_model/_ridge.py":      pkg / "linear_model" / "_ridge.py",
            "sklearn/ensemble/_forest.py":         pkg / "ensemble" / "_forest.py",
            "sklearn/model_selection/_split.py":   pkg / "model_selection" / "_split.py",
        },
        "git_target": "sklearn/",
        "library": "scikit-learn",
        "catalog_module": ("scripts.benchmark_ablation", "HAND_CORRUPTIONS"),
        "system_prompt": (
            "You are an expert Python debugging agent specializing in scikit-learn "
            "internals. Find and fix bugs. Return the complete fixed code for each "
            "affected file in separate ```python blocks with a # FILE: header."
        ),
    },
    "fastapi": {
        "sandbox": _SANDBOX_BASE / "fastapi",
        "tfail_cache": _CACHE_BASE.parent / "tfail_cache_fastapi",
        "package_root": _SANDBOX_BASE / "fastapi" / "fastapi",
        "source_files": lambda pkg: {
            "fastapi/dependencies/utils.py": pkg / "dependencies" / "utils.py",
            "fastapi/routing.py":            pkg / "routing.py",
            "fastapi/applications.py":       pkg / "applications.py",
            "fastapi/params.py":             pkg / "params.py",
        },
        "git_target": "fastapi/",
        "library": "FastAPI",
        "catalog_module": ("scripts.fastapi_corruptions", "FASTAPI_CORRUPTIONS"),
        "system_prompt": (
            "You are an expert Python debugging agent specializing in FastAPI internals. "
            "Find and fix bugs. Return the complete fixed code for each affected file "
            "in separate ```python blocks with a # FILE: header."
        ),
    },
}


def load_tfail_cache(catalog: list, cache_dir: Path) -> dict[str, list[str]]:
    """Map corruption_id -> T_fail test nodeids, read from per-bug cache files."""
    out: dict[str, list[str]] = {}
    if not cache_dir.exists():
        return out
    for c in catalog:
        for f in cache_dir.glob(f"{c.corruption_id}_*.json"):
            data = json.loads(f.read_text())
            out[c.corruption_id] = data.get("tfail", [])
            break
    return out


def filter_healthy(catalog: list, tfail: dict[str, list[str]]) -> list:
    """Drop bugs with no T_fail tests (silent or missing)."""
    return [c for c in catalog if tfail.get(c.corruption_id)]


def load_sgs_catalog(domain: str) -> list:
    """Load persisted SGS corruptions from results/sgs_corruptions_{domain}.json."""
    from scripts.generators.coding_diffusion import CorruptionSpec
    path = ROOT / "results" / f"sgs_corruptions_{domain}.json"
    if not path.exists():
        raise SystemExit(
            f"SGS catalog not found: {path}\n"
            f"Run: python scripts/generate_sgs_vllm.py --domain {domain}"
        )
    raw = json.loads(path.read_text())
    return [CorruptionSpec(**c) for c in raw]


def make_chat_prompt(tokenizer, user_text: str, system_text: str,
                     thinking: bool = False) -> str:
    msgs = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
    try:
        # Always pass enable_thinking explicitly so Qwen3 doesn't default to thinking
        return tokenizer.apply_chat_template(msgs, **kwargs, enable_thinking=thinking)
    except TypeError:
        pass  # tokenizer doesn't support enable_thinking
    return tokenizer.apply_chat_template(msgs, **kwargs)


def build_domain_prompt(corrupted: dict[str, str], n_bugs: int,
                        corruption_descs: list[str], library: str) -> str:
    files_section = ""
    for fname, code in corrupted.items():
        files_section += f"\n### {fname}\n```python\n{code}\n```\n"
    desc_list = "\n".join(f"  {i+1}. {d}" for i, d in enumerate(corruption_descs))
    return (
        f"You are an expert debugging agent. The following {library} code has "
        f"{n_bugs} bug(s) introduced by a recent refactor.\n\n"
        f"Your task: find and fix ALL {n_bugs} bug(s).\n\n"
        f"Rules:\n"
        f"- Each bug is a single-point change (operator swap, off-by-one, wrong axis, condition inversion)\n"
        f"- Fix ONLY the bugs — do not refactor or add new features\n"
        f"- Return the COMPLETE fixed code for EACH affected file in separate ```python blocks\n"
        f"- Start each code block with a comment like # FILE: {list(corrupted.keys())[0]}\n\n"
        f"Bug symptom hints:\n{desc_list}\n\n"
        f"Here is the corrupted code:\n{files_section}\n"
        f"Return the COMPLETE fixed code for each affected file."
    )


_FEWSHOT_EXAMPLE = """\
Example — 2 bugs across 2 files:

FILE: mylib/stats.py
FIND: return np.sum(x, axis=1)
FIX: return np.sum(x, axis=0)

FILE: mylib/split.py
FIND: if n_samples < min_size:
FIX: if n_samples <= min_size:

(One FILE/FIND/FIX block per bug. No code fences, no extra text.)
"""


def build_findreplace_prompt(corrupted: dict[str, str], n_bugs: int,
                             corruption_descs: list[str], library: str) -> str:
    files_section = ""
    for fname, code in corrupted.items():
        files_section += f"\n### {fname}\n```python\n{code}\n```\n"
    desc_list = "\n".join(f"  {i+1}. {d}" for i, d in enumerate(corruption_descs))
    example_file = list(corrupted.keys())[0]
    return (
        f"You are an expert debugging agent. The following {library} source code "
        f"has {n_bugs} bug(s) introduced by a recent refactor.\n\n"
        f"Your task: find and fix ALL {n_bugs} bug(s).\n\n"
        f"Each bug is a single-line change. For EACH bug output exactly:\n\n"
        f"FILE: {example_file}\n"
        f"FIND: <exact buggy expression as it appears in the file>\n"
        f"FIX: <corrected expression>\n\n"
        f"{_FEWSHOT_EXAMPLE}\n"
        f"Bug symptom hints:\n{desc_list}\n\n"
        f"Corrupted source code:\n{files_section}"
    )


def extract_findreplace_fixes(text: str) -> list[dict]:
    """Parse FILE/FIND/FIX blocks from model output.

    Handles the canonical form and common variations models produce:
      FILE: ...  FIND: ...  FIX/REPLACE/FIXED: ...
      # FILE: ...  # FIND / ...lines... / # REPLACE / ...lines...  (comment-block style)
    """
    import re
    fixes = []

    # Canonical: FILE / FIND / FIX  (or REPLACE / FIXED / CORRECTED)
    pattern = (
        r'FILE:\s*([^\n`]+)\n'
        r'(?:FIND|BUGGY|WRONG):\s*([^\n]+)\n'
        r'(?:FIX|REPLACE|FIXED|CORRECTED|CORRECT):\s*([^\n]+)'
    )
    for m in re.finditer(pattern, text, re.IGNORECASE):
        fixes.append({
            "file": m.group(1).strip().strip('`').strip(),
            "find": m.group(2).strip().strip('`').strip(),
            "replace": m.group(3).strip().strip('`').strip(),
        })
    if fixes:
        return fixes

    # Fallback: # FILE / # FIND / ... / # REPLACE comment-block style
    pattern2 = (
        r'#\s*FILE:\s*([^\n]+)\n'
        r'#\s*(?:FIND|BUGGY)\s*\n(.*?)'
        r'#\s*(?:REPLACE|FIX|FIXED)\s*\n(.*?)'
        r'(?:#\s*END|\Z)'
    )
    for m in re.finditer(pattern2, text, re.DOTALL | re.IGNORECASE):
        fixes.append({
            "file": m.group(1).strip(),
            "find": m.group(2).strip(),
            "replace": m.group(3).strip(),
        })
    return fixes


def score_findreplace_textmatch(fixes: list[dict], corruptions: list) -> dict:
    """Text-match score: did the model output the exact right find/replace pairs?"""
    fixed = 0
    for c in corruptions:
        for fix in fixes:
            # c.replace is the buggy code injected into the file
            # c.find is the original correct code
            file_match = (fix["file"] == c.source_file or
                          fix["file"].endswith(c.source_file) or
                          c.source_file.endswith(fix["file"]))
            if file_match and (fix["find"].strip() == c.replace.strip() and
                               fix["replace"].strip() == c.find.strip()):
                fixed += 1
                break
    return {"score": fixed / max(1, len(corruptions)), "fixed_count": fixed,
            "total_bugs": len(corruptions)}


def _apply_findreplace_to_sandbox(
    fixes: list[dict],
    corruptions: list,
    sandbox: Path,
) -> dict[str, str]:
    """Produce the model's fixed full files, starting from the corrupted state.

    Workflow:
      1. Read each sandbox file (currently clean/original).
      2. Apply the corruption in-memory (c.find → c.replace) so the text
         matches what the model saw.
      3. Apply the model's fix (fix["find"] → fix["replace"]) to that text.
      4. Return the resulting files — these should match the original if the
         model found the right fix.

    File path matching is lenient (suffix match). Falls back to scanning all
    corruption files for the find string when the model gives a partial path.
    """
    # Step 1+2: build corrupted-state texts for all affected files
    corrupted_texts: dict[str, str] = {}
    for c in corruptions:
        path = sandbox / c.source_file
        if not path.exists():
            continue
        text = path.read_text()
        corrupted_texts[c.source_file] = text.replace(c.find, c.replace, 1)

    # Step 3: apply model fixes to the corrupted texts
    full = dict(corrupted_texts)  # start from corrupted; unfixed files pass through
    for fix in fixes:
        find_str = fix["find"]
        replace_str = fix["replace"]
        applied = False

        # Try file-path match first
        for c in corruptions:
            file_match = (fix["file"] == c.source_file or
                          fix["file"].endswith(c.source_file) or
                          c.source_file.endswith(fix["file"]) or
                          fix["file"].endswith(c.source_file.split("/")[-1]))
            if not file_match:
                continue
            text = full.get(c.source_file, "")
            if find_str in text:
                full[c.source_file] = text.replace(find_str, replace_str, 1)
                applied = True
                break

        if not applied:
            # Fallback: find which corrupted file contains the find string
            for rel_path, text in full.items():
                if find_str in text:
                    full[rel_path] = text.replace(find_str, replace_str, 1)
                    break
    return full


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--domain", default="sklearn", choices=list(DOMAIN_CONFIG))
    ap.add_argument("--catalog", default="hand", choices=["hand", "sgs", "all"],
                    help="hand=hand-crafted only, sgs=SGS-generated only, all=combined")
    ap.add_argument("--tp", type=int, default=2, help="tensor-parallel size for vLLM")
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--max_model_len", type=int, default=16384)
    ap.add_argument("--gpu_util", type=float, default=0.85)
    ap.add_argument("--counts", "--n-list", default="1,3,5,7,10")
    ap.add_argument("--diversity", default="clustered,scattered")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out_dir", "--out", default="results/local_eval",
                    help="Output directory (or full path ending in .json)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--held-out-ids", default=None,
                    help="Path to eval_results.json from train_rl.py; filters catalog to "
                         "held-out corruption IDs only (prevents train/eval leakage)")
    ap.add_argument("--thinking", action="store_true",
                    help="Enable thinking mode (Qwen3 enable_thinking=True in chat template)")
    ap.add_argument("--thinking-budget", type=int, default=4096,
                    help="Max tokens for the thinking block (default 4096). "
                         "Remaining tokens go to the actual answer.")
    ap.add_argument("--format", default="full_file", choices=["full_file", "find_replace"],
                    help="Output format: full_file=complete fixed file, "
                         "find_replace=<<<FIND>>>...<<<REPLACE>>>...<<<END>>> blocks (~100-300 tokens/bug)")
    ap.add_argument("--sanity", type=int, default=0,
                    help="If >0, run only this many prompts and print raw output+scores (de-risk check)")
    args = ap.parse_args()

    counts = [int(x) for x in args.counts.split(",")]
    diversity = args.diversity.split(",")
    cfg = DOMAIN_CONFIG[args.domain]

    # Override module-level paths so load_snippet_files reads from the right sandbox
    pkg = cfg["package_root"]
    _ba.SOURCE_FILES = cfg["source_files"](pkg)

    # Load corruption catalog
    import importlib
    mod_name, attr_name = cfg["catalog_module"]
    mod = importlib.import_module(mod_name)
    hand_catalog = list(getattr(mod, attr_name))

    if args.catalog == "hand":
        catalog = hand_catalog
    elif args.catalog == "sgs":
        catalog = load_sgs_catalog(args.domain)
    else:  # all
        catalog = hand_catalog + load_sgs_catalog(args.domain)

    _out_arg = Path(args.out_dir)
    if not _out_arg.is_absolute():
        _out_arg = ROOT / _out_arg
    if _out_arg.suffix == ".json":
        out_path = _out_arg
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = _out_arg
        out_dir.mkdir(parents=True, exist_ok=True)
        model_slug = args.model.replace("/", "_")
        thinking_tag = "_thinking" if args.thinking else ""
        fmt_tag = "_fr" if args.format == "find_replace" else ""
        out_path = (out_dir /
            f"{model_slug}_domain={args.domain}_cat={args.catalog}"
            f"_n={'_'.join(map(str,counts))}_d={'_'.join(diversity)}_t{args.trials}{thinking_tag}{fmt_tag}.json")

    # ── T_fail filtering ──────────────────────────────────────────────────────
    tfail = load_tfail_cache(catalog, cfg["tfail_cache"])
    healthy = filter_healthy(catalog, tfail)
    print(f"Domain: {args.domain}")
    print(f"Catalog: {len(catalog)} corruptions, {len(healthy)} healthy.")
    dropped = [c.corruption_id for c in catalog if c not in healthy]
    if dropped:
        print(f"Dropped (no T_fail): {dropped}")

    # ── Held-out filtering (prevent train/eval leakage) ───────────────────────
    if args.held_out_ids:
        held_data = json.loads(Path(args.held_out_ids).read_text())
        allowed = set(held_data["held_out_ids"])
        before = len(healthy)
        healthy = [c for c in healthy if c.corruption_id in allowed]
        print(f"Held-out filter: {before} -> {len(healthy)} corruptions "
              f"({before - len(healthy)} in training pool excluded)")

    sandbox = cfg["sandbox"]
    if not sandbox.exists():
        raise SystemExit(f"Sandbox missing: {sandbox}\nRun: python scripts/discover_tfail.py --domain {args.domain}")

    # Override functional_judge's cache path for this domain
    _fj.TFAIL_CACHE = cfg["tfail_cache"]

    # ── Build all prompts ─────────────────────────────────────────────────────
    from scripts.benchmark_ablation import (
        load_snippet_files, apply_corruptions, extract_fixed_files,
        score_fix as score_text_match, select_by_diversity,
    )
    import random
    rng = random.Random(args.seed)
    plans = []

    prompt_builder = (build_findreplace_prompt if args.format == "find_replace"
                      else build_domain_prompt)

    for n in counts:
        if n > len(healthy):
            print(f"  skipping n={n} (only {len(healthy)} healthy bugs)")
            continue
        for div in diversity:
            for trial in range(args.trials):
                pool = list(healthy)
                rng.shuffle(pool)
                selected = select_by_diversity(pool, n, div)
                if not selected:
                    continue
                snippets = load_snippet_files(selected)
                corrupted = apply_corruptions(snippets, selected)
                descs = [c.description for c in selected]
                prompt = prompt_builder(corrupted, n, descs, cfg["library"])
                label = f"n{n}_{div}_t{trial+1}"
                plans.append({
                    "label": label,
                    "n": n,
                    "diversity": div,
                    "trial": trial + 1,
                    "corruption_ids": [c.corruption_id for c in selected],
                    "corrupted_snippets": corrupted,
                    "prompt": prompt,
                    "_selected": selected,
                })
    print(f"Built {len(plans)} prompts.")

    if args.sanity > 0:
        plans = plans[:args.sanity]
        print(f"[sanity] Trimmed to {len(plans)} plans.")

    # ── Load vLLM ─────────────────────────────────────────────────────────────
    print(f"Loading {args.model} (TP={args.tp}, max_len={args.max_model_len})...")
    t0 = time.time()
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        download_dir=os.environ["HF_HOME"],
    )
    print(f"  loaded in {time.time()-t0:.1f}s")
    tokenizer = llm.get_tokenizer()

    chat_prompts = [
        make_chat_prompt(tokenizer, p["prompt"], cfg["system_prompt"], thinking=args.thinking)
        for p in plans
    ]
    # For thinking mode: total budget = thinking_budget + answer tokens.
    # Cap the thinking block so the model has room to write the actual fix.
    effective_max_tokens = (args.thinking_budget + args.max_tokens
                            if args.thinking else args.max_tokens)
    sampling_kwargs: dict = dict(
        temperature=args.temperature,
        max_tokens=effective_max_tokens,
        top_p=1.0,
    )
    if args.thinking:
        import inspect
        from vllm import SamplingParams as _SP
        if "thinking_budget" in inspect.signature(_SP).parameters:
            sampling_kwargs["thinking_budget"] = args.thinking_budget
    sampling_params = SamplingParams(**sampling_kwargs)

    # ── Generate ──────────────────────────────────────────────────────────────
    print(f"Generating {len(chat_prompts)} prompts...")
    t0 = time.time()
    outputs = llm.generate(chat_prompts, sampling_params)
    gen_dt = time.time() - t0
    n_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"  generated {n_tokens} tokens in {gen_dt:.1f}s ({n_tokens/gen_dt:.1f} tok/s)")

    # ── Score: text-match + functional ────────────────────────────────────────
    import subprocess as _sp

    print("Scoring...")
    results = []
    for plan, out in zip(plans, outputs):
        raw = out.outputs[0].text
        git_target = cfg["git_target"]

        if args.format == "find_replace":
            fixes = extract_findreplace_fixes(raw)
            text_score = score_findreplace_textmatch(fixes, plan["_selected"])
            fixed_full_files = _apply_findreplace_to_sandbox(fixes, plan["_selected"], sandbox)

            if args.sanity > 0:
                print(f"\n[sanity] === {plan['label']} ===")
                print(f"[sanity] raw output ({len(out.outputs[0].token_ids)} tokens):\n{raw[:800]}")
                print(f"[sanity] parsed fixes: {fixes}")
                print(f"[sanity] text_score: {text_score}")
                print(f"[sanity] files to apply: {list(fixed_full_files.keys())}")
                for c in plan["_selected"]:
                    print(f"[sanity] corruption {c.corruption_id}: "
                          f"bug_in_file={repr(c.replace[:80])} "
                          f"correct_code={repr(c.find[:80])}")
        else:
            fixed = extract_fixed_files(raw)
            text_score = score_text_match(plan["corrupted_snippets"], fixed, plan["_selected"])
            fixed_full_files = _expand_fixes_to_full_files(fixed, plan["_selected"], sandbox)

        _sp.run(["git", "checkout", "--", git_target], cwd=sandbox, check=True)
        try:
            for c in plan["_selected"]:
                target = sandbox / c.source_file
                if target.exists():
                    txt = target.read_text()
                    if c.find in txt:
                        target.write_text(txt.replace(c.find, c.replace, 1))
            func_result = judge_fix(
                plan["_selected"],
                fixed_full_files,
                sandbox,
                tfail_cache=tfail,
            )
        finally:
            _sp.run(["git", "checkout", "--", git_target], cwd=sandbox, check=True)

        if args.sanity > 0:
            print(f"[sanity] func_result: score={func_result.score:.0%} "
                  f"fixed={func_result.fixed_count}/{func_result.total_bugs} "
                  f"t_fail_passed={func_result.t_fail_passed}")

        # Regression check
        tfail_set = set(func_result.t_fail)
        reg = _regression_check(
            fixed_full_files, plan["_selected"], sandbox,
            tfail_set, args.domain, git_target,
        )

        results.append({
            "label": plan["label"],
            "n": plan["n"],
            "diversity": plan["diversity"],
            "trial": plan["trial"],
            "corruption_ids": plan["corruption_ids"],
            "raw_output": raw,
            "n_output_tokens": len(out.outputs[0].token_ids),
            "format": args.format,
            "text_match": text_score,
            "functional": {
                "score": func_result.score,
                "fixed_count": func_result.fixed_count,
                "total_bugs": func_result.total_bugs,
                "per_bug": func_result.per_bug,
                "t_fail": func_result.t_fail,
                "t_fail_passed": func_result.t_fail_passed,
                "t_fail_still_failing": func_result.t_fail_still_failing,
                "elapsed_sec": func_result.elapsed_sec,
            },
            "regression": reg,
        })
        reg_str = f" regressions={reg['regression_count']}" if reg["regression_count"] is not None else ""
        print(f"  {plan['label']}: text={text_score['score']:.0%} "
              f"func={func_result.score:.0%} "
              f"({func_result.fixed_count}/{func_result.total_bugs}){reg_str}")

    # ── Summarize per n with confidence intervals ─────────────────────────────
    import math
    summary: dict[int, dict] = {}
    for n in counts:
        n_results = [r for r in results if r["n"] == n]
        if not n_results:
            continue
        scores = [r["functional"]["score"] for r in n_results]
        mean = sum(scores) / len(scores)
        # Wilson-style 95% CI using normal approximation (SEM * 1.96)
        if len(scores) > 1:
            variance = sum((s - mean) ** 2 for s in scores) / (len(scores) - 1)
            sem = math.sqrt(variance / len(scores))
            ci95 = 1.96 * sem
        else:
            ci95 = float("nan")
        summary[n] = {"mean": round(mean, 4), "ci95": round(ci95, 4), "n_trials": len(scores)}
        print(f"  n={n}: {mean:.1%} ± {ci95:.1%}  (k={len(scores)})")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path.write_text(json.dumps({
        "model": args.model,
        "domain": args.domain,
        "format": args.format,
        "tp": args.tp,
        "temperature": args.temperature,
        "thinking": args.thinking,
        "counts": counts,
        "diversity": diversity,
        "trials": args.trials,
        "n_healthy_bugs": len(healthy),
        "held_out_ids_source": args.held_out_ids,
        "summary_by_n": summary,
        "results": results,
        "gen_seconds": gen_dt,
        "n_output_tokens_total": n_tokens,
    }, indent=2))
    print(f"Saved {out_path}")


_SOURCE_TO_TESTS = {
    "sklearn": {
        "sklearn/metrics/_classification.py": ["sklearn/metrics/tests/test_classification.py"],
        "sklearn/metrics/_ranking.py":        ["sklearn/metrics/tests/test_ranking.py"],
        "sklearn/preprocessing/_data.py":     ["sklearn/preprocessing/tests/test_data.py"],
        "sklearn/linear_model/_ridge.py":     ["sklearn/linear_model/tests/test_ridge.py"],
        "sklearn/ensemble/_forest.py":        ["sklearn/ensemble/tests/test_forest.py"],
        "sklearn/model_selection/_split.py":  ["sklearn/model_selection/tests/test_split.py"],
    },
    "fastapi": {
        "fastapi/dependencies/utils.py": ["tests/"],
        "fastapi/routing.py":            ["tests/"],
        "fastapi/applications.py":       ["tests/"],
        "fastapi/params.py":             ["tests/"],
    },
}


def _regression_check(
    fixed_full_files: dict[str, str],
    corruptions: list,
    sandbox: Path,
    tfail_set: set[str],
    domain: str,
    git_target: str,
) -> dict:
    import subprocess as _sp
    """Apply fixed files, run full test suite for affected files, count regressions.

    Regressions = tests that newly fail that are NOT in the T_fail set.
    """
    source_to_tests = _SOURCE_TO_TESTS.get(domain, {})
    test_targets = set()
    for c in corruptions:
        for t in source_to_tests.get(c.source_file, []):
            test_targets.add(t)

    if not test_targets or not fixed_full_files:
        return {"regression_count": None, "regression_tests": []}

    _sp.run(["git", "checkout", "--", git_target], cwd=sandbox, check=True,
            capture_output=True)
    try:
        for rel_path, full_text in fixed_full_files.items():
            target = sandbox / rel_path
            if target.exists():
                target.write_text(full_text)

        result = _sp.run(
            ["python", "-m", "pytest", "--tb=no", "-q", "--timeout=60",
             *list(test_targets)],
            cwd=sandbox, capture_output=True, text=True, timeout=300,
        )
        failed = set()
        for line in result.stdout.splitlines():
            if " FAILED" in line:
                node = line.split(" FAILED")[0].strip()
                failed.add(node)

        regressions = [t for t in failed if t not in tfail_set]
        return {
            "regression_count": len(regressions),
            "regression_tests": regressions[:10],
        }
    except Exception as e:
        return {"regression_count": None, "regression_error": str(e)}
    finally:
        _sp.run(["git", "checkout", "--", git_target], cwd=sandbox, check=True,
                capture_output=True)


def _expand_fixes_to_full_files(
    fixed_snippets: dict[str, str],
    corruptions: list,
    sandbox: Path,
) -> dict[str, str]:
    """Expand snippet-level fixes back to full source files in the sandbox."""
    full = {}
    for rel_path, snippet_fix in fixed_snippets.items():
        sandbox_path = sandbox / rel_path
        if not sandbox_path.exists():
            continue
        full_text = sandbox_path.read_text()
        for c in corruptions:
            if c.source_file != rel_path:
                continue
            if c.find in snippet_fix and c.replace not in snippet_fix:
                if c.replace in full_text:
                    full_text = full_text.replace(c.replace, c.find, 1)
        full[rel_path] = full_text
    return full


if __name__ == "__main__":
    main()
