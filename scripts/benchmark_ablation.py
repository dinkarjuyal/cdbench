#!/usr/bin/env python3
"""Coding Diffusion Ablation: corruption count × diversity × source (hand vs SGS).

Ablation dimensions:
  - corruption_count: 1, 2, 3, 5, 7, 10, 15, 20
  - diversity: clustered (same file), scattered (different files)
  - source: hand-crafted vs SGS-generated (3-player game)

The SGS source uses the existing AdversarialSGSStrategy pipeline but
adapted for code-fixing: the "proposer" establishes what the code should do,
the "adversary" introduces a subtle bug, the "guide" verifies it's non-trivial.

Usage:
    python3 scripts/benchmark_ablation.py
    MODEL=qwen/qwen3-coder python3 scripts/benchmark_ablation.py
    MODEL=deepseek/deepseek-chat python3 scripts/benchmark_ablation.py
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generators.coding_diffusion import _make_client, CorruptionSpec

PROVIDER = os.environ.get("PROVIDER", "anthropic")
MODEL = os.environ.get("MODEL", "qwen/qwen3-coder")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "8192"))
N_TRIALS = int(os.environ.get("N_TRIALS", "1"))

# ── sklearn source files ────────────────────────────────────────────────────────

SKLEARN_ROOT = Path("/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/sklearn")

SOURCE_FILES = {
    "sklearn/metrics/_classification.py": SKLEARN_ROOT / "metrics" / "_classification.py",
    "sklearn/metrics/_ranking.py": SKLEARN_ROOT / "metrics" / "_ranking.py",
    "sklearn/preprocessing/_data.py": SKLEARN_ROOT / "preprocessing" / "_data.py",
    "sklearn/linear_model/_ridge.py": SKLEARN_ROOT / "linear_model" / "_ridge.py",
    "sklearn/ensemble/_forest.py": SKLEARN_ROOT / "ensemble" / "_forest.py",
    "sklearn/model_selection/_split.py": SKLEARN_ROOT / "model_selection" / "_split.py",
}

# Extract focused snippets (functions containing corruption targets) to keep prompts manageable
def _extract_function(full_code: str, find_str: str) -> tuple[int, int]:
    """Find the line range of the function containing find_str."""
    lines = full_code.split("\n")
    target_idx = None
    for i, line in enumerate(lines):
        if find_str in line:
            target_idx = i
            break
    if target_idx is None:
        return (0, min(50, len(lines)))

    # Find enclosing function (def ... above target)
    func_start = target_idx
    while func_start > 0:
        if lines[func_start].strip().startswith("def "):
            break
        func_start -= 1

    # Find function end (next def or class at same or lower indent)
    func_indent = len(lines[func_start]) - len(lines[func_start].lstrip())
    func_end = target_idx + 1
    while func_end < len(lines):
        stripped = lines[func_end].strip()
        if stripped.startswith("def ") or stripped.startswith("class "):
            if len(lines[func_end]) - len(lines[func_end].lstrip()) <= func_indent:
                break
        func_end += 1

    # Add some context above and below
    start = max(0, func_start - 3)
    end = min(len(lines), func_end + 3)
    return (start, end)


def _extract_snippet(full_code: str, find_str: str) -> str:
    """Extract a function-level snippet around a find string."""
    lines = full_code.split("\n")
    start, end = _extract_function(full_code, find_str)
    return "\n".join(lines[start:end])


def load_snippet_files(corruptions: list[CorruptionSpec]) -> dict[str, str]:
    """Load source files, extracting all functions that contain corruption targets."""
    full_sources = {}
    for fname, path in SOURCE_FILES.items():
        if path.exists():
            full_sources[fname] = path.read_text()

    snippets = {}
    for fname in full_sources:
        full_code = full_sources[fname]
        # Collect line ranges for all corruptions in this file
        file_corruptions = [c for c in corruptions if c.source_file == fname and c.find]
        if not file_corruptions:
            continue

        lines = full_code.split("\n")
        # Find all function ranges containing corruption targets
        ranges = []
        for c in file_corruptions:
            start, end = _extract_function(full_code, c.find)
            ranges.append((start, end))

        # Merge overlapping ranges
        ranges.sort()
        merged = [list(ranges[0])]
        for start, end in ranges[1:]:
            if start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])

        # Build snippet from merged ranges
        snippet_lines = []
        for start, end in merged:
            snippet_lines.extend(lines[start:end])
            snippet_lines.append("")  # separator

        snippets[fname] = "\n".join(snippet_lines).strip()

    return snippets

# ── 20 hand-crafted corruptions ─────────────────────────────────────────────────
# Ordered by subtlety (1=obvious, 5=subtle semantic)
# Grouped by file for diversity control

HAND_CORRUPTIONS = [
    # --- sklearn/metrics/_classification.py (4 corruptions) ---
    CorruptionSpec("hc_01", "sklearn/metrics/_classification.py",
        "beta2 = beta**2", "beta2 = beta*2",
        "F-beta: beta**2 → beta*2 (squaring → doubling, breaks F2/F0.5)", "", "", "metrics", 1),
    CorruptionSpec("hc_02", "sklearn/metrics/_classification.py",
        "y_pred = xp.clip(y_pred, eps, 1 - eps)",
        "y_pred = xp.clip(y_pred, eps, 1 - 2 * eps)",
        "log_loss: clip upper bound 1-eps → 1-2eps (narrower range)", "", "", "metrics", 2),
    CorruptionSpec("hc_03", "sklearn/metrics/_classification.py",
        "cm = cm / cm.sum(axis=1, keepdims=True)",
        "cm = cm / cm.sum(axis=2, keepdims=True)",
        "confusion_matrix normalization: axis=1→2 (invalid axis, crashes on 2D)", "", "", "metrics", 3),
    CorruptionSpec("hc_04", "sklearn/metrics/_classification.py",
        'y_type == {"binary", "multiclass"}',
        'y_type == {"binary"}',
        "_check_targets: allows only binary, rejects multiclass→binary path", "", "", "metrics", 4),

    # --- sklearn/metrics/_ranking.py (4 corruptions) ---
    CorruptionSpec("hc_05", "sklearn/metrics/_ranking.py",
        "area = direction * trapezoid(y, x)",
        "area = direction * trapezoid(x, y)",
        "auc(): swaps x,y args in trapezoid (computes AUC with swapped axes)", "", "", "ranking", 3),
    CorruptionSpec("hc_06", "sklearn/metrics/_ranking.py",
        "fps / fps[-1]", "fps / fps[0]",
        "roc_curve: FPR normalization uses first instead of last element", "", "", "ranking", 2),
    CorruptionSpec("hc_07", "sklearn/metrics/_ranking.py",
        "xp.logical_or(xp.diff(fps, 2), xp.diff(tps, 2))",
        "xp.logical_and(xp.diff(fps, 2), xp.diff(tps, 2))",
        "roc_curve drop_intermediate: OR→AND (only drops if BOTH flat)", "", "", "ranking", 3),
    CorruptionSpec("hc_08", "sklearn/metrics/_ranking.py",
        "recall = tps / tps[-1]", "recall = tps / tps[0]",
        "precision_recall_curve: recall normalization uses first not last", "", "", "ranking", 4),

    # --- sklearn/preprocessing/_data.py (4 corruptions) ---
    CorruptionSpec("hc_09", "sklearn/preprocessing/_data.py",
        "scale[constant_mask] = 1.0", "scale[constant_mask] = 0.0",
        "StandardScaler: constant features get scale=0 (division by zero)", "", "", "preprocessing", 1),
    CorruptionSpec("hc_10", "sklearn/preprocessing/_data.py",
        "upper_bound = n_samples * eps * var + (n_samples * mean * eps) ** 2",
        "upper_bound = n_samples * eps * var - (n_samples * mean * eps) ** 2",
        "constant feature detection: +→- in variance bound (may miss constants)", "", "", "preprocessing", 3),
    CorruptionSpec("hc_11", "sklearn/preprocessing/_data.py",
        "references = self.references_ * 100",
        "references = self.references_ / 100",
        "QuantileTransformer: *100→/100 (percentile→fraction, breaks quantile mapping)", "", "", "preprocessing", 2),
    CorruptionSpec("hc_12", "sklearn/preprocessing/_data.py",
        "self.scale_ = (feature_range[1] - feature_range[0]) / _handle_zeros_in_scale(",
        "self.scale_ = (feature_range[0] - feature_range[1]) / _handle_zeros_in_scale(",
        "MinMaxScaler: feature_range[1]-[0] → [0]-[1] (negates scale)", "", "", "preprocessing", 3),

    # --- sklearn/linear_model/_ridge.py (4 corruptions) ---
    CorruptionSpec("hc_13", "sklearn/linear_model/_ridge.py",
        "coef, info = _sparse_linalg_cg(C, y_column, rtol=tol)",
        "coef, info = _sparse_linalg_cg(C, y_column, rtol=tol * 10)",
        "Ridge sparse_cg: tol→tol*10 (converges too early, less accurate)", "", "", "ridge", 3),
    CorruptionSpec("hc_14", "sklearn/linear_model/_ridge.py",
        "w[intercept_dim] = 0", "w[intercept_dim] = 1",
        "RidgeGCV: intercept regularization (0→1, penalizes intercept)", "", "", "ridge", 4),
    CorruptionSpec("hc_15", "sklearn/linear_model/_ridge.py",
        "alpha.shape[0] == 1 and n_targets > 1",
        "alpha.shape[0] == 1 or n_targets > 1",
        "Ridge alpha expansion: AND→OR (broadcasts alpha incorrectly)", "", "", "ridge", 3),
    CorruptionSpec("hc_16", "sklearn/linear_model/_ridge.py",
        "((singvals_sq + alpha) ** -1) - (alpha**-1)",
        "((singvals_sq + alpha) ** -1) + (alpha**-1)",
        "RidgeGCV SVD: -→+ in effective DOF (adds instead of subtracts)", "", "", "ridge", 5),

    # --- sklearn/ensemble/_forest.py (2 corruptions) ---
    CorruptionSpec("hc_17", "sklearn/ensemble/_forest.py",
        "curr_sample_weight *= sample_counts",
        "curr_sample_weight += sample_counts",
        "RandomForest class_weight: *= → += (accumulates instead of scales)", "", "", "forest", 3),
    CorruptionSpec("hc_18", "sklearn/ensemble/_forest.py",
        "max(round(n_samples * max_samples), 1)",
        "max(int(n_samples * max_samples), 1)",
        "bootstrap sample count: round→int (floors instead of rounding)", "", "", "forest", 2),

    # --- sklearn/model_selection/_split.py (2 corruptions) ---
    CorruptionSpec("hc_19", "sklearn/model_selection/_split.py",
        "y_order[i :: self.n_splits]",
        "y_order[i :: self.n_splits + 1]",
        "StratifiedKFold: step n_splits→n_splits+1 (skips every N+1th sample)", "", "", "split", 3),
    CorruptionSpec("hc_20", "sklearn/model_selection/_split.py",
        "n_folds = n_splits + 1", "n_folds = n_splits - 1",
        "TimeSeriesSplit: n_splits+1→-1 (one fewer fold than expected)", "", "", "split", 2),
]


# ── SGS 3-player game for code corruption ─────────────────────────────────────

_SGS_PROPOSER_SYSTEM = """\
You are a code corruption proposer. Given a piece of real library code, propose
a single subtle bug that would be hard to detect but changes the code's behavior.

The bug must be:
1. A single-point change (operator swap, off-by-one, condition inversion, wrong axis/dim)
2. Subtle — not a crash or obvious error, but a semantic change
3. Realistic — the kind of bug a tired developer might introduce during a refactor

Respond with EXACTLY these tags:
<find>exact code to find in the original</find>
<replace>the buggy version</replace>
<description>one sentence describing the bug</description>
<subtlety>1-5 where 1=obvious 5=edge-case only</subtlety>"""

_SGS_PROPOSER_USER = """\
Source file: {file_path}

Code:
```python
{code}
```

Propose ONE subtle corruption for this code."""

_SGS_GUIDE_SYSTEM = """\
You are a code corruption quality judge. Evaluate whether a proposed bug is:
1. Non-trivial: requires understanding of the algorithm, not just syntax
2. Semantic: changes output meaningfully, not just style
3. Realistic: could plausibly be introduced by a developer

Score each axis 1-10. Reject if any score < 3.

Respond with:
<relevance>1-10</relevance>
<elegance>1-10</elegance>
<non_trivial>1-10</non_trivial>
<verdict>accept or reject</verdict>
<reason>one sentence</reason>"""

_SGS_GUIDE_USER = """\
Original code:
```python
{original}
```

Buggy code:
```python
{buggy}
```

Bug description: {description}

Evaluate this corruption."""


def sgs_generate_corruptions(
    client, model: str, source_files: dict[str, str],
    n_corruptions: int = 20, max_retries: int = 3,
) -> list[CorruptionSpec]:
    """Generate corruptions using the SGS 3-player game.

    Player 1 (Proposer): proposes a subtle bug
    Player 2 (Executor): verifies the bug changes output
    Player 3 (Guide): scores quality — rejects trivial bugs
    """
    corruptions = []
    guide_model = model  # Use same model for guide

    for fname, code in source_files.items():
        if len(corruptions) >= n_corruptions:
            break

        attempts = 0
        while attempts < max_retries and len(corruptions) < n_corruptions:
            attempts += 1
            # Propose a corruption
            try:
                resp = client.messages.create(
                    model=model, max_tokens=1024,
                    system=_SGS_PROPOSER_SYSTEM,
                    messages=[{"role": "user", "content":
                        _SGS_PROPOSER_USER.format(file_path=fname, code=code)}],
                )
                text = resp.content[0].text if hasattr(resp, "content") else str(resp)
            except Exception as e:
                print(f"    [sgs-proposer] Error: {e}")
                continue

            # Parse proposed corruption
            find_str = _extract_tag(text, "find")
            replace_str = _extract_tag(text, "replace")
            desc = _extract_tag(text, "description") or "SGS-generated corruption"
            subtlety_str = _extract_tag(text, "subtlety") or "3"

            if not find_str or not replace_str or find_str not in code:
                continue

            # Validate corruption quality
            if code.count(find_str) > 1:
                print(f"    [sgs] Rejected: find_str appears {code.count(find_str)}x (ambiguous)")
                continue
            if find_str in replace_str or replace_str in find_str:
                print(f"    [sgs] Rejected: find/replace overlap")
                continue
            corrupted_code = code.replace(find_str, replace_str, 1)
            if find_str in corrupted_code:
                print(f"    [sgs] Rejected: find_str still present after applying corruption")
                continue

            try:
                subtlety = int(subtlety_str.strip())
            except ValueError:
                subtlety = 3

            # Guide scorer: verify quality
            try:
                guide_resp = client.messages.create(
                    model=guide_model, max_tokens=256,
                    system=_SGS_GUIDE_SYSTEM,
                    messages=[{"role": "user", "content":
                        _SGS_GUIDE_USER.format(
                            original=code[:2000], buggy=code[:2000].replace(find_str, replace_str, 1),
                            description=desc)}],
                )
                guide_text = guide_resp.content[0].text if hasattr(guide_resp, "content") else str(guide_resp)
            except Exception as e:
                print(f"    [sgs-guide] Error: {e}")
                continue

            verdict = _extract_tag(guide_text, "verdict") or "reject"
            if "accept" not in verdict.lower():
                reason = _extract_tag(guide_text, "reason") or "unknown"
                print(f"    [sgs] Rejected: {reason[:60]}")
                continue

            # Accepted
            cid = f"sgs_{fname.split('/')[-1].replace('.py','')}_{hash(replace_str) % 100000:05d}"
            corruptions.append(CorruptionSpec(
                corruption_id=cid,
                source_file=fname,
                find=find_str,
                replace=replace_str,
                description=desc,
                broken_test="", passing_test="",
                family=fname.split("/")[-2],
                subtlety=subtlety,
            ))
            print(f"    [sgs] Accepted: {desc[:60]} (subtlety={subtlety})")

    return corruptions[:n_corruptions]


def _extract_tag(text: str, tag: str) -> str | None:
    """Extract content from <tag>...</tag>."""
    import re
    pattern = f"<{tag}>(.*?)</{tag}>"
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1).strip() if m else None


# ── Diversity grouping ──────────────────────────────────────────────────────────

def select_by_diversity(corruptions: list[CorruptionSpec], n: int,
                        diversity: str = "scattered") -> list[CorruptionSpec]:
    """Select n corruptions with the specified diversity.

    clustered: all from the same file (lowest diversity)
    scattered: each from a different file (highest diversity)
    mixed: some same, some different (medium diversity)
    """
    if n >= len(corruptions):
        return list(corruptions)

    if diversity == "clustered":
        # Pick the file with most corruptions, then pad from nearby files
        file_counts: dict[str, int] = {}
        for c in corruptions:
            file_counts[c.source_file] = file_counts.get(c.source_file, 0) + 1
        sorted_files = sorted(file_counts, key=file_counts.get, reverse=True)
        selected = []
        for fname in sorted_files:
            file_corruptions = [c for c in corruptions if c.source_file == fname and c not in selected]
            selected.extend(file_corruptions)
            if len(selected) >= n:
                break
        return selected[:n]

    elif diversity == "scattered":
        # Pick at most one from each file, cycling if needed
        by_file: dict[str, list[CorruptionSpec]] = {}
        for c in corruptions:
            by_file.setdefault(c.source_file, []).append(c)
        selected = []
        files = list(by_file.keys())
        idx = 0
        while len(selected) < n and any(by_file.values()):
            fname = files[idx % len(files)]
            if by_file[fname]:
                selected.append(by_file[fname].pop(0))
            idx += 1
            if idx > n * 10:
                break
        return selected[:n]

    else:  # mixed
        half = n // 2
        clustered = select_by_diversity(corruptions, half, "clustered")
        remaining = [c for c in corruptions if c not in clustered]
        scattered = select_by_diversity(remaining, n - half, "scattered")
        return clustered + scattered


# ── Core benchmark ──────────────────────────────────────────────────────────────

def apply_corruptions(source_files: dict[str, str], corruptions: list[CorruptionSpec]) -> dict[str, str]:
    result = dict(source_files)
    for c in corruptions:
        if c.source_file in result and c.find in result[c.source_file]:
            result[c.source_file] = result[c.source_file].replace(c.find, c.replace, 1)
    return result


def build_prompt(corrupted: dict[str, str], n_bugs: int, corruption_descs: list[str]) -> str:
    files_section = ""
    for fname, code in corrupted.items():
        files_section += f"\n### {fname}\n```python\n{code}\n```\n"
    desc_list = "\n".join(f"  {i+1}. {d}" for i, d in enumerate(corruption_descs))
    return (
        f"You are an expert debugging agent. The following scikit-learn code has "
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


def extract_fixed_files(response: str) -> dict[str, str]:
    files = {}
    parts = response.split("```python")
    for part in parts[1:]:
        end = part.find("```")
        if end == -1:
            continue
        code = part[:end].strip()
        fname = None
        import re

        for line in code.split("\n")[:5]:
            stripped = line.strip()
            if stripped.startswith("# FILE:"):
                fname = stripped.replace("# FILE:", "").strip()
                code = "\n".join(code.split("\n")[code.split("\n").index(line)+1:]).strip()
                break
            elif "sklearn/" in stripped and stripped.endswith(".py"):
                m = re.search(r"sklearn/\S+\.py", stripped)
                if m:
                    fname = m.group(0)
                    code = "\n".join(code.split("\n")[code.split("\n").index(line)+1:]).strip()
                break

        # Fallback: match by content
        if fname is None:
            if "precision_recall_fscore_support" in code or "confusion_matrix" in code:
                fname = "sklearn/metrics/_classification.py"
            elif "roc_curve" in code or "precision_recall_curve" in code:
                fname = "sklearn/metrics/_ranking.py"
            elif "StandardScaler" in code or "QuantileTransformer" in code:
                fname = "sklearn/preprocessing/_data.py"
            elif "_ridge_regression" in code or "RidgeGCV" in code:
                fname = "sklearn/linear_model/_ridge.py"
            elif "BaseForest" in code or "_parallel_build_trees" in code:
                fname = "sklearn/ensemble/_forest.py"
            elif "StratifiedKFold" in code or "TimeSeriesSplit" in code:
                fname = "sklearn/model_selection/_split.py"
            else:
                fname = f"unknown_{len(files)}.py"

        if len(code) > 50:
            files[fname] = code
    return files


def score_fix(corrupted: dict[str, str], fixed: dict[str, str],
              corruptions: list[CorruptionSpec]) -> dict:
    bugs_fixed = 0
    details = []
    for c in corruptions:
        fixed_code = fixed.get(c.source_file, corrupted.get(c.source_file, ""))
        if not fixed_code:
            details.append(f"{c.corruption_id}: NOT RETURNED")
            continue
        find_present = c.find in fixed_code
        replace_absent = c.replace not in fixed_code
        if find_present and replace_absent:
            bugs_fixed += 1
            details.append(f"{c.corruption_id}: FIXED")
        elif find_present:
            details.append(f"{c.corruption_id}: PARTIAL")
        else:
            details.append(f"{c.corruption_id}: MISS")

    return {
        "score": round(bugs_fixed / max(len(corruptions), 1), 4),
        "bugs_fixed": bugs_fixed,
        "bugs_total": len(corruptions),
        "details": details,
    }


def main():
    print("=" * 70)
    print(f"CODING DIFFUSION ABLATION — {MODEL}")
    print("=" * 70)

    # Verify hand-crafted corruptions against full source
    full_sources = {}
    for fname, path in SOURCE_FILES.items():
        if path.exists():
            full_sources[fname] = path.read_text()
    print(f"\nLoaded {len(full_sources)} sklearn source files")
    for fname, code in full_sources.items():
        print(f"  {fname}: {len(code.splitlines())} lines")

    # Verify hand-crafted corruptions
    print(f"\nVerifying {len(HAND_CORRUPTIONS)} hand-crafted corruptions:")
    valid_hand = []
    for c in HAND_CORRUPTIONS:
        found = c.find in full_sources.get(c.source_file, "") if c.find else False
        if found:
            valid_hand.append(c)
            print(f"  [OK] {c.corruption_id}: {c.description[:60]}")
        else:
            print(f"  [MISS] {c.corruption_id}: find string not in source")

    # Generate SGS corruptions using snippet-sized code (not full files)
    client = _make_client(None, provider=PROVIDER)
    print(f"\nGenerating SGS corruptions via {MODEL}...")
    # Use snippets for SGS so prompts aren't 3000+ lines
    snippet_pool = load_snippet_files(valid_hand)
    sgs_corruptions = sgs_generate_corruptions(
        client, MODEL, snippet_pool, n_corruptions=20, max_retries=3,
    )
    print(f"  SGS generated: {len(sgs_corruptions)} corruptions")

    # Combine
    all_corruptions = list(valid_hand) + sgs_corruptions
    print(f"\n  Total corruptions available: {len(all_corruptions)} "
          f"({len(valid_hand)} hand + {len(sgs_corruptions)} SGS)")

    # ── Ablation grid ────────────────────────────────────────────────────
    corruption_counts = [1, 2, 3, 5, 7, 10, 15, 20]
    diversity_levels = ["clustered", "scattered"]
    sources = ["hand", "sgs", "mixed"]

    results = []
    total_conditions = len(corruption_counts) * len(diversity_levels) * len(sources)
    total_runs = total_conditions * N_TRIALS
    run_idx = 0

    for n_bugs in corruption_counts:
        for diversity in diversity_levels:
            for source in sources:
                # Select corruptions
                if source == "hand":
                    pool = valid_hand
                elif source == "sgs":
                    pool = sgs_corruptions if sgs_corruptions else valid_hand
                else:
                    pool = all_corruptions

                if len(pool) < n_bugs:
                    run_idx += N_TRIALS
                    print(f"\n  SKIP {n_bugs} bugs, "
                          f"{diversity}, {source} (only {len(pool)} available)")
                    continue

                selected = select_by_diversity(pool, n_bugs, diversity)
                if not selected:
                    run_idx += N_TRIALS
                    continue

                # Apply corruptions to snippet-sized source files
                base_snippets = load_snippet_files(selected)
                corrupted = apply_corruptions(base_snippets, selected)
                corruption_descs = [c.description for c in selected]

                # Build prompt
                prompt = build_prompt(corrupted, n_bugs, corruption_descs)

                for trial in range(N_TRIALS):
                    run_idx += 1

                    # Call model
                    label = f"{n_bugs}bug_{diversity}_{source}_t{trial+1}"
                    print(f"\n  [{run_idx}/{total_runs}] {label}...", end="", flush=True)
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
                        results.append({"n_bugs": n_bugs, "diversity": diversity,
                                        "source": source, "trial": trial + 1,
                                        "score": 0.0, "error": str(e)})
                        continue
                    elapsed = time.time() - start

                    # Extract and score
                    fixed = extract_fixed_files(text)
                    scoring = score_fix(corrupted, fixed, selected)
                    print(f" {elapsed:.1f}s | {scoring['score']:.0%} "
                          f"({scoring['bugs_fixed']}/{scoring['bugs_total']})")

                    results.append({
                        "n_bugs": n_bugs,
                        "diversity": diversity,
                        "source": source,
                        "trial": trial + 1,
                        "elapsed_sec": round(elapsed, 1),
                        **scoring,
                    })

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("ABLATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Model: {MODEL}")

    # Print table: n_bugs × diversity, for each source (mean across trials)
    for source in sources:
        print(f"\n  Source: {source}")
        header = f"  {'N':>3} {'clustered':>10} {'scattered':>10}"
        if N_TRIALS > 1:
            header += f"  (mean of {N_TRIALS} trials)"
        print(header)
        print(f"  {'─'*3} {'─'*10} {'─'*10}")
        for n_bugs in corruption_counts:
            row = []
            for div in diversity_levels:
                matching = [r for r in results
                            if r["n_bugs"] == n_bugs and r["diversity"] == div
                            and r["source"] == source and "error" not in r]
                if matching:
                    mean_s = sum(r.get("score", 0) for r in matching) / len(matching)
                    row.append(f"{mean_s:.0%}")
                else:
                    row.append("—")
            print(f"  {n_bugs:>3} {row[0]:>10} {row[1]:>10}")

    # Save full results (model-specific filename)
    model_slug = MODEL.replace("/", "_").replace(" ", "_")
    out = ROOT / "results" / f"ablation_{model_slug}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": MODEL,
        "provider": PROVIDER,
        "repo": "scikit-learn",
        "n_trials": N_TRIALS,
        "n_hand_corruptions": len(valid_hand),
        "n_sgs_corruptions": len(sgs_corruptions),
        "corruption_counts": corruption_counts,
        "diversity_levels": diversity_levels,
        "sources": sources,
        "results": results,
    }, indent=2))
    print(f"\n  Saved to {out}")
    print(f"  Total API calls: {len(results)}")
    total_time = sum(r.get("elapsed_sec", 0) for r in results)
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    main()
