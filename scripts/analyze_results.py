#!/usr/bin/env python3
"""Analyze benchmark results and print summary tables.

Usage:
    python3 scripts/analyze_results.py [--results-dir PATH]

Reads result.json files from results/runs/<task_id>/<run_dir>/
and prints per-model and per-task summary statistics.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "runs"


def load_results(results_dir: Path) -> list[dict]:
    """Load all result.json files recursively."""
    results = []
    for result_file in results_dir.rglob("*.json"):
        if result_file.name not in ("result.json", "runtime_context.json"):
            continue
        if result_file.name != "runtime_context.json":
            try:
                data = json.loads(result_file.read_text())
                if "validation" in data and "agent_name" in data:
                    results.append(data)
            except Exception:
                pass
    return results


def per_model_summary(results: list[dict]) -> None:
    """Print per-model mean score table."""
    by_model: dict[str, list[float]] = defaultdict(list)
    by_model_passed: dict[str, list[bool]] = defaultdict(list)

    for r in results:
        agent = r.get("agent_name", "unknown")
        score = r["validation"].get("score", 0.0)
        passed = r["validation"].get("passed", False)
        by_model[agent].append(score)
        by_model_passed[agent].append(passed)

    print("\n" + "=" * 70)
    print("PER-MODEL SUMMARY")
    print("=" * 70)
    print(f"{'Model':<35} {'Tasks':>6} {'Mean':>7} {'Median':>7} {'Pass%':>7}")
    print("-" * 70)

    for agent in sorted(by_model):
        scores = by_model[agent]
        passed = by_model_passed[agent]
        mean = sum(scores) / len(scores)
        sorted_scores = sorted(scores)
        n = len(sorted_scores)
        median = sorted_scores[n // 2] if n % 2 else (sorted_scores[n//2-1] + sorted_scores[n//2]) / 2
        pass_rate = sum(passed) / len(passed) * 100
        print(f"{agent:<35} {n:>6} {mean:>7.3f} {median:>7.3f} {pass_rate:>6.1f}%")


def per_task_summary(results: list[dict]) -> None:
    """Print per-task scores across all models."""
    by_task: dict[str, dict[str, float]] = defaultdict(dict)

    for r in results:
        task_id = r.get("task_id", "unknown")
        agent = r.get("agent_name", "unknown")
        score = r["validation"].get("score", 0.0)
        by_task[task_id][agent] = score

    all_agents = sorted({agent for scores in by_task.values() for agent in scores})

    print("\n" + "=" * 100)
    print("PER-TASK SCORES")
    print("=" * 100)
    header = f"{'Task ID':<50}" + "".join(f"{a[:12]:>13}" for a in all_agents) + f"{'Mean':>8}"
    print(header)
    print("-" * len(header))

    task_means = []
    for task_id in sorted(by_task):
        scores = by_task[task_id]
        row_scores = [scores.get(a, None) for a in all_agents]
        valid = [s for s in row_scores if s is not None]
        mean = sum(valid) / len(valid) if valid else 0.0
        task_means.append((task_id, mean))

        row = f"{task_id:<50}"
        for s in row_scores:
            row += f"{'N/A':>13}" if s is None else f"{s:>13.3f}"
        row += f"{mean:>8.3f}"
        print(row)

    print("-" * len(header))
    print(f"\nHardest tasks (lowest mean score):")
    for task_id, mean in sorted(task_means, key=lambda x: x[1])[:5]:
        print(f"  {task_id:<50} {mean:.3f}")
    print(f"\nEasiest tasks (highest mean score):")
    for task_id, mean in sorted(task_means, key=lambda x: -x[1])[:5]:
        print(f"  {task_id:<50} {mean:.3f}")


def score_distribution(results: list[dict]) -> None:
    """Print score band distribution per model."""
    bands = ["0.0-0.2", "0.2-0.6", "0.6-0.9", "0.9-1.0"]
    by_model: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_model[r.get("agent_name", "unknown")].append(r["validation"].get("score", 0.0))

    print("\n" + "=" * 70)
    print("SCORE BAND DISTRIBUTION (% of tasks per band)")
    print("=" * 70)
    print(f"{'Model':<35} {'0.0-0.2':>9} {'0.2-0.6':>9} {'0.6-0.9':>9} {'0.9-1.0':>9}")
    print("-" * 70)

    for agent in sorted(by_model):
        scores = by_model[agent]
        n = len(scores)
        b0 = sum(1 for s in scores if s < 0.2) / n * 100
        b1 = sum(1 for s in scores if 0.2 <= s < 0.6) / n * 100
        b2 = sum(1 for s in scores if 0.6 <= s < 0.9) / n * 100
        b3 = sum(1 for s in scores if s >= 0.9) / n * 100
        print(f"{agent:<35} {b0:>8.1f}% {b1:>8.1f}% {b2:>8.1f}% {b3:>8.1f}%")


def task_type_breakdown(results: list[dict], benchmark_dir: Path) -> None:
    """Break down scores by task type (noop/impossible/invariant_recovery/contract_extension)."""
    # Load task metadata
    task_meta: dict[str, dict] = {}
    tasks_dir = benchmark_dir / "tasks"
    for task_json in tasks_dir.rglob("task.json"):
        try:
            d = json.loads(task_json.read_text())
            task_meta[d["id"]] = d.get("_meta", {})
        except Exception:
            pass

    by_type: dict[str, list[float]] = defaultdict(list)
    for r in results:
        task_id = r.get("task_id", "unknown")
        score = r["validation"].get("score", 0.0)
        meta = task_meta.get(task_id, {})
        if meta.get("is_noop"):
            ttype = "no_op"
        elif meta.get("is_impossible"):
            ttype = "impossible"
        else:
            ttype = meta.get("task_type", "unknown")
        by_type[ttype].append(score)

    print("\n" + "=" * 50)
    print("SCORES BY TASK TYPE")
    print("=" * 50)
    print(f"{'Task Type':<30} {'N':>4} {'Mean':>7}")
    print("-" * 50)
    for ttype in sorted(by_type):
        scores = by_type[ttype]
        mean = sum(scores) / len(scores)
        print(f"{ttype:<30} {len(scores):>4} {mean:>7.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze benchmark results.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--benchmark-dir", type=Path, default=ROOT / "benchmarks" / "scrapy_50")
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"No results directory found at {args.results_dir}")
        print("Run the benchmark first with harness/run_tasks.py")
        return 1

    results = load_results(args.results_dir)
    if not results:
        print(f"No result.json files found in {args.results_dir}")
        return 1

    print(f"Loaded {len(results)} result(s) from {args.results_dir}")

    per_model_summary(results)
    score_distribution(results)
    task_type_breakdown(results, args.benchmark_dir)
    per_task_summary(results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
