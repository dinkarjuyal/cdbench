"""GPU-powered programmatic corruption generation via SGS Proposer/Guide game.

No API required — runs entirely on local GPUs via vLLM.
Best model: Qwen/Qwen2.5-Coder-32B-Instruct (TP=4 on 4×H100).

Pipeline:
  Phase 1 — Proposer (temperature=0.9, batched):
    For each sklearn source file, generate N candidate corruptions.
    Each is a single-point find/replace pair with a description.

  Phase 2 — Guide (temperature=0.0, batched):
    Score each candidate on non-triviality, realism, semantics.
    Accept/reject with structured verdict.

  Phase 3 — Validate:
    Filter ambiguous finds (find string appears >1x in source).
    Dedup identical find strings.
    Verify find is still present in source after replacement.

Then run T_fail discovery to find which are testable:
  python scripts/discover_tfail.py --domain sklearn --catalog sgs

Usage:
    python scripts/generate_sgs_vllm.py \\
        --model Qwen/Qwen2.5-Coder-32B-Instruct \\
        --tp 4 \\
        --n-per-file 30 \\
        --domain sklearn \\
        --out results/sgs_corruptions_sklearn.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HOME", "/mnt/localssd/cdbench/hf_cache")

from scripts.functional_judge import SANDBOX_ROOT

SKLEARN_SOURCE_FILES = [
    "sklearn/metrics/_classification.py",
    "sklearn/metrics/_ranking.py",
    "sklearn/preprocessing/_data.py",
    "sklearn/linear_model/_ridge.py",
    "sklearn/ensemble/_forest.py",
    "sklearn/model_selection/_split.py",
]

FASTAPI_SOURCE_FILES = [
    "fastapi/dependencies/utils.py",
    "fastapi/routing.py",
    "fastapi/applications.py",
    "fastapi/params.py",
]

DOMAIN_FILES = {
    "sklearn": SKLEARN_SOURCE_FILES,
    "fastapi": FASTAPI_SOURCE_FILES,
}

# ── Prompts ───────────────────────────────────────────────────────────────────

_PROPOSER_SYSTEM = """\
You are an expert code corruption engineer. Given real library source code, propose ONE subtle, single-point bug.

The bug MUST be:
1. A single-point change: swap one operator, flip one condition, change one constant, swap two arguments
2. Semantically wrong but syntactically valid Python
3. Subtle enough that a code reviewer might miss it
4. Verifiable by a test (the bug should change observable behavior)
5. Realistic — the kind of mistake a tired developer makes during refactoring

Respond with EXACTLY these XML tags and nothing else:
<find>exact string to find in the original (must appear exactly once)</find>
<replace>the buggy replacement string</replace>
<description>one sentence: what changed and what breaks</description>
<subtlety>1-5 where 1=very obvious 5=edge-case only</subtlety>"""

_PROPOSER_USER = """\
Source file: {file_path}
Subtlety target: {subtlety_target} (aim for this level)

```python
{snippet}
```

Propose ONE subtle corruption. The <find> string must appear exactly once in the code above."""

_GUIDE_SYSTEM = """\
You are a code corruption quality judge. Evaluate the proposed bug strictly.

Reject if ANY of these are true:
- The change is purely stylistic (no semantic difference)
- The change would cause an obvious syntax error or immediate crash
- The find string could match multiple locations
- The description does not match the actual change
- The bug is trivial to spot (e.g., replacing True with False in an obvious branch)

Score these axes 1-10:
- non_trivial: requires algorithmic understanding to notice
- behavioral: clearly changes output for some valid input
- realistic: a plausible developer mistake

Respond with EXACTLY these tags:
<non_trivial>score</non_trivial>
<behavioral>score</behavioral>
<realistic>score</realistic>
<verdict>accept</verdict> or <verdict>reject</verdict>
<reason>one sentence</reason>"""

_GUIDE_USER = """\
Original:
```python
{original}
```

Proposed change:
  FIND:    {find}
  REPLACE: {replace}
  DESCRIPTION: {description}

Evaluate this corruption."""


# ── Snippet extraction ────────────────────────────────────────────────────────

def extract_snippet(full_code: str, max_lines: int = 80) -> str:
    """Extract a representative snippet: first large function block."""
    lines = full_code.split("\n")
    # Find a good starting function (not too early, not too late)
    func_starts = [i for i, l in enumerate(lines) if l.startswith("def ") or l.startswith("class ")]
    if not func_starts:
        return "\n".join(lines[:max_lines])
    # Start from ~1/3 into the file for variety
    start_idx = func_starts[len(func_starts) // 3]
    return "\n".join(lines[start_idx: start_idx + max_lines])


def extract_snippets_for_file(full_code: str, n_snippets: int, max_lines: int = 80) -> list[str]:
    """Return n evenly-spaced snippets from different parts of the file."""
    lines = full_code.split("\n")
    func_starts = [i for i, l in enumerate(lines) if l.startswith("def ") or l.startswith("class ")]
    if not func_starts:
        return ["\n".join(lines[:max_lines])] * n_snippets

    # Sample n_snippets positions evenly across func_starts
    step = max(1, len(func_starts) // n_snippets)
    positions = func_starts[::step][:n_snippets]
    # Pad if needed
    while len(positions) < n_snippets:
        positions.append(positions[-1])

    snippets = []
    for pos in positions:
        snippets.append("\n".join(lines[pos: pos + max_lines]))
    return snippets


# ── Parsing ───────────────────────────────────────────────────────────────────

def _extract_tag(text: str, tag: str) -> str | None:
    m = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def parse_proposal(text: str, file_path: str, full_source: str) -> dict | None:
    find = _extract_tag(text, "find")
    replace = _extract_tag(text, "replace")
    desc = _extract_tag(text, "description")
    subtlety_str = _extract_tag(text, "subtlety") or "3"

    if not find or not replace:
        return None
    if find == replace:
        return None
    if find not in full_source:
        return None
    if full_source.count(find) != 1:
        return None  # ambiguous — would corrupt wrong location
    # Verify applying doesn't leave find in the result
    applied = full_source.replace(find, replace, 1)
    if find in applied:
        return None

    try:
        subtlety = max(1, min(5, int(subtlety_str.strip())))
    except ValueError:
        subtlety = 3

    family = file_path.split("/")[-2] if "/" in file_path else file_path.split(".")[0]
    return {
        "source_file": file_path,
        "find": find,
        "replace": replace,
        "description": desc or "SGS-generated corruption",
        "subtlety": subtlety,
        "family": family,
    }


def is_accepted(guide_text: str, min_score: int = 5) -> bool:
    verdict = _extract_tag(guide_text, "verdict") or "reject"
    if "accept" not in verdict.lower():
        return False
    for axis in ["non_trivial", "behavioral", "realistic"]:
        val = _extract_tag(guide_text, axis)
        try:
            if int(val.strip()) < min_score:
                return False
        except (ValueError, AttributeError):
            return False
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--gpu_util", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--domain", default="sklearn", choices=["sklearn", "fastapi"])
    ap.add_argument("--n-per-file", type=int, default=30,
                    help="number of proposals to generate per source file")
    ap.add_argument("--out", default=None,
                    help="output JSON path (default: results/sgs_corruptions_{domain}.json)")
    ap.add_argument("--min-guide-score", type=int, default=5,
                    help="minimum score on each Guide axis to accept (1-10)")
    ap.add_argument("--append", action="store_true",
                    help="append to existing output file instead of overwriting")
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else ROOT / "results" / f"sgs_corruptions_{args.domain}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    source_file_list = DOMAIN_FILES[args.domain]
    sandbox = SANDBOX_ROOT / args.domain

    # Read source files
    sources: dict[str, str] = {}
    for rel_path in source_file_list:
        full_path = sandbox / rel_path
        if full_path.exists():
            sources[rel_path] = full_path.read_text()
        else:
            print(f"WARNING: {full_path} not found — skipping")

    if not sources:
        raise SystemExit(f"No source files found in {sandbox}. Run discover_tfail.py first to set up sandbox.")

    print(f"Loaded {len(sources)} source files from {sandbox}")
    print(f"Generating {args.n_per_file} proposals per file × {len(sources)} files = "
          f"{args.n_per_file * len(sources)} total proposals")

    # Load existing if appending
    existing: list[dict] = []
    if args.append and out_path.exists():
        existing = json.loads(out_path.read_text())
        print(f"Appending to {len(existing)} existing corruptions")

    # ── Load vLLM ────────────────────────────────────────────────────────────
    print(f"\nLoading {args.model} (TP={args.tp})...")
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
    tokenizer = llm.get_tokenizer()
    print(f"  loaded in {time.time()-t0:.1f}s")

    def make_chat(system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # ── Phase 1: Generate proposals ───────────────────────────────────────────
    print(f"\nPhase 1: Generating proposals (temperature=0.9)...")
    proposer_meta = []  # (file_path, full_source) per prompt
    proposer_prompts = []

    subtlety_targets = [1, 2, 3, 4, 5]  # cycle through subtlety targets

    for file_path, full_source in sources.items():
        snippets = extract_snippets_for_file(full_source, n_snippets=args.n_per_file)
        for i, snippet in enumerate(snippets):
            subtlety_target = subtlety_targets[i % len(subtlety_targets)]
            user_msg = _PROPOSER_USER.format(
                file_path=file_path,
                subtlety_target=subtlety_target,
                snippet=snippet,
            )
            proposer_prompts.append(make_chat(_PROPOSER_SYSTEM, user_msg))
            proposer_meta.append((file_path, full_source))

    proposer_params = SamplingParams(temperature=0.9, max_tokens=512, top_p=0.95)
    proposer_outputs = llm.generate(proposer_prompts, proposer_params)
    print(f"  generated {len(proposer_outputs)} proposals")

    # ── Phase 2: Parse proposals ──────────────────────────────────────────────
    print("\nPhase 2: Parsing proposals...")
    candidates = []
    parse_failures = 0
    for (file_path, full_source), out in zip(proposer_meta, proposer_outputs):
        text = out.outputs[0].text
        proposal = parse_proposal(text, file_path, full_source)
        if proposal:
            candidates.append(proposal)
        else:
            parse_failures += 1

    print(f"  valid: {len(candidates)} / {len(proposer_outputs)} "
          f"({parse_failures} parse failures)")

    # Dedup by find string before Guide scoring
    seen_finds: set[str] = set()
    deduped = []
    for c in candidates:
        key = (c["source_file"], c["find"])
        if key not in seen_finds:
            seen_finds.add(key)
            deduped.append(c)
    print(f"  after dedup: {len(deduped)}")

    # ── Phase 3: Guide scoring ─────────────────────────────────────────────────
    print(f"\nPhase 3: Guide scoring (temperature=0.0)...")
    guide_prompts = []
    for c in deduped:
        full_source = sources[c["source_file"]]
        # Show 40 lines around the corruption site
        idx = full_source.find(c["find"])
        line_num = full_source[:idx].count("\n")
        lines = full_source.split("\n")
        start = max(0, line_num - 10)
        end = min(len(lines), line_num + 30)
        context = "\n".join(lines[start:end])
        user_msg = _GUIDE_USER.format(
            original=context,
            find=c["find"],
            replace=c["replace"],
            description=c["description"],
        )
        guide_prompts.append(make_chat(_GUIDE_SYSTEM, user_msg))

    guide_params = SamplingParams(temperature=0.0, max_tokens=256)
    guide_outputs = llm.generate(guide_prompts, guide_params)

    # ── Phase 4: Filter accepted ──────────────────────────────────────────────
    print(f"\nPhase 4: Filtering accepted corruptions (min_score={args.min_guide_score})...")
    accepted = []
    reject_reasons: list[str] = []
    for candidate, g_out in zip(deduped, guide_outputs):
        guide_text = g_out.outputs[0].text
        if is_accepted(guide_text, min_score=args.min_guide_score):
            accepted.append(candidate)
        else:
            reason = _extract_tag(guide_text, "reason") or "score too low"
            reject_reasons.append(reason)

    print(f"  accepted: {len(accepted)} / {len(deduped)}")
    if reject_reasons:
        print(f"  sample reject reasons: {reject_reasons[:3]}")

    # ── Assign IDs and save ───────────────────────────────────────────────────
    existing_ids = {c["corruption_id"] for c in existing if "corruption_id" in c}
    next_id = len(existing) + 1

    final = list(existing)
    for c in accepted:
        cid = f"sgs_{args.domain}_{next_id:04d}"
        while cid in existing_ids:
            next_id += 1
            cid = f"sgs_{args.domain}_{next_id:04d}"
        existing_ids.add(cid)
        c["corruption_id"] = cid
        c["broken_test"] = ""
        c["passing_test"] = ""
        final.append(c)
        next_id += 1

    out_path.write_text(json.dumps(final, indent=2))
    print(f"\nSaved {len(final)} corruptions ({len(accepted)} new) to {out_path}")
    print(f"\nNext step:")
    print(f"  python scripts/discover_tfail.py --domain {args.domain} --catalog sgs")


if __name__ == "__main__":
    main()
