#!/usr/bin/env python3
"""Re-run only the clustered conditions that had n_bugs != bugs_total.

Uses the same hand-crafted corruptions and regenerates SGS per model,
then replaces only the broken runs in each result file.

Usage:
    MODEL=deepseek/deepseek-chat python3 scripts/rerun_clustered.py
    # Or run all 5:
    for m in "Qwen/Qwen3.5-2B" "qwen/qwen3-coder" "deepseek/deepseek-chat" \
             "qwen/qwen3-max" "deepseek/deepseek-r1-0528"; do
        MODEL="$m" N_TRIALS=3 python3 scripts/rerun_clustered.py
    done
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_ablation import (
    HAND_CORRUPTIONS, SOURCE_FILES, SKLEARN_ROOT,
    select_by_diversity, load_snippet_files, apply_corruptions,
    build_prompt, extract_fixed_files, score_fix,
    sgs_generate_corruptions, _make_client, CorruptionSpec,
)

PROVIDER = os.environ.get("PROVIDER", "prime")
MODEL = os.environ.get("MODEL", "qwen/qwen3-coder")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "8192"))
N_TRIALS = int(os.environ.get("N_TRIALS", "3"))

model_slug = MODEL.replace("/", "_").replace(" ", "_")
result_file = ROOT / "results" / f"ablation_{model_slug}.json"


def main():
    print(f"{'='*70}")
    print(f"RE-RUN CLUSTERED — {MODEL}")
    print(f"{'='*70}")

    # Load existing results
    if not result_file.exists():
        print(f"ERROR: {result_file} not found")
        sys.exit(1)

    with open(result_file) as f:
        data = json.load(f)

    if isinstance(data, list):
        existing = data
        metadata = {}
    else:
        existing = data.get("results", [])
        metadata = {k: v for k, v in data.items() if k != "results"}

    # Find broken runs (clustered with n_bugs != bugs_total)
    broken = [r for r in existing
              if isinstance(r, dict)
              and r.get("diversity") == "clustered"
              and r.get("bugs_total") is not None
              and r["n_bugs"] != r["bugs_total"]]

    broken_keys = set()
    for r in broken:
        broken_keys.add((r["n_bugs"], r["diversity"], r["source"], r.get("trial", 1)))

    if not broken_keys:
        print("No broken clustered runs found. Nothing to re-run.")
        return

    print(f"Found {len(broken_keys)} broken clustered runs to re-run")

    # Validate hand-crafted corruptions
    full_sources = {}
    for fname, path in SOURCE_FILES.items():
        if path.exists():
            full_sources[fname] = path.read_text()
    print(f"Loaded {len(full_sources)} sklearn source files")

    valid_hand = []
    for c in HAND_CORRUPTIONS:
        if c.find and c.find in full_sources.get(c.source_file, ""):
            valid_hand.append(c)
    print(f"Valid hand-crafted: {len(valid_hand)}")

    # Generate SGS corruptions
    client = _make_client(None, provider=PROVIDER)
    print(f"Generating SGS corruptions via {MODEL}...")
    snippet_pool = load_snippet_files(valid_hand)
    sgs_corruptions = sgs_generate_corruptions(
        client, MODEL, snippet_pool, n_corruptions=20, max_retries=3,
    )
    print(f"SGS generated: {len(sgs_corruptions)}")

    all_corruptions = list(valid_hand) + sgs_corruptions

    # Re-run only broken conditions
    new_results = []
    conditions = sorted(broken_keys)
    total = len(conditions)

    for idx, (n_bugs, diversity, source, trial) in enumerate(conditions, 1):
        if source == "hand":
            pool = valid_hand
        elif source == "sgs":
            pool = sgs_corruptions if sgs_corruptions else valid_hand
        else:
            pool = all_corruptions

        if len(pool) < n_bugs:
            print(f"\n  [{idx}/{total}] SKIP {n_bugs}bug_{diversity}_{source}_t{trial} "
                  f"(only {len(pool)} available)")
            continue

        selected = select_by_diversity(pool, n_bugs, diversity)
        if not selected:
            continue

        base_snippets = load_snippet_files(selected)
        corrupted = apply_corruptions(base_snippets, selected)
        corruption_descs = [c.description for c in selected]
        prompt = build_prompt(corrupted, n_bugs, corruption_descs)

        label = f"{n_bugs}bug_{diversity}_{source}_t{trial}"
        print(f"\n  [{idx}/{total}] {label} (selecting {len(selected)} bugs)...", end="", flush=True)

        start = time.time()
        try:
            resp = client.messages.create(
                model=MODEL, max_tokens=MAX_TOKENS,
                system="You are an expert Python debugging agent specializing in scikit-learn internals. Find and fix bugs. Return the complete fixed code for each affected file in separate ```python blocks with a # FILE: header.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text if hasattr(resp, "content") else str(resp)
        except Exception as e:
            print(f" ERROR: {e}")
            new_results.append({"n_bugs": n_bugs, "diversity": diversity,
                                "source": source, "trial": trial,
                                "score": 0.0, "error": str(e)})
            continue

        elapsed = time.time() - start
        fixed = extract_fixed_files(text)
        scoring = score_fix(corrupted, fixed, selected)
        print(f" {elapsed:.1f}s | {scoring['score']:.0%} "
              f"({scoring['bugs_fixed']}/{scoring['bugs_total']})")

        new_results.append({
            "n_bugs": n_bugs,
            "diversity": diversity,
            "source": source,
            "trial": trial,
            "elapsed_sec": round(elapsed, 1),
            **scoring,
        })

    # Merge: replace broken runs with new ones, keep everything else
    kept = [r for r in existing
            if (r.get("n_bugs"), r.get("diversity"), r.get("source"), r.get("trial", 1)) not in broken_keys]
    merged = kept + new_results
    merged.sort(key=lambda r: (r.get("n_bugs", 0), r.get("diversity", ""), r.get("source", ""), r.get("trial", 0)))

    # Update metadata
    if metadata:
        metadata["results"] = merged
        metadata["n_sgs_corruptions"] = len(sgs_corruptions)
        output = metadata
    else:
        output = merged

    # Backup original
    backup = result_file.with_suffix(".json.bak")
    result_file.rename(backup)
    print(f"\nBacked up original to {backup}")

    result_file.write_text(json.dumps(output, indent=2))
    print(f"Saved {len(merged)} runs to {result_file}")
    print(f"  Replaced: {len(new_results)} runs")
    print(f"  Kept: {len(kept)} runs")

    # Verify no more mismatches
    mismatches = [r for r in merged if isinstance(r, dict)
                  and r.get("bugs_total") is not None
                  and r["n_bugs"] != r["bugs_total"]]
    if mismatches:
        print(f"\n  WARNING: {len(mismatches)} runs still have n_bugs != bugs_total")
    else:
        print(f"\n  All runs now have n_bugs == bugs_total")


if __name__ == "__main__":
    main()
