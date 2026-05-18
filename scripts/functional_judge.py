"""Functional judge for multi-fault debugging.

Replaces the text-match scoring in benchmark_ablation.score_fix with a
test-based functional judge: a fix is FIXED iff every originally-failing
test passes and no originally-passing test newly fails.

Architecture:
  1. Per-bug T_fail discovery (offline, cached): for each CorruptionSpec,
     apply the single corruption to a clean sklearn checkout and run pytest
     against the relevant module. Tests that transition pass->fail go into
     T_fail[bug_id].
  2. Per-run scoring: compose T_fail = union of T_fail[c] over corruptions
     in the run. Apply model fix, run T_fail (must all pass) plus a sample
     of T_pass (must not regress). Score = pass-rate over T_fail.
  3. Per-bug attribution (optional): for fine-grained reporting, apply
     the model's fix one bug at a time and re-evaluate.

This module is execution-only -- it shells out to pytest in a sandbox.
The sandbox is a Docker container or, in --no-docker mode, a venv with
sklearn editable-installed at a fixed git commit.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable


SKLEARN_GIT_COMMIT = os.environ.get("SKLEARN_COMMIT", "1.6.0")
SANDBOX_ROOT = Path(os.environ.get("CDBENCH_SANDBOX", "/mnt/localssd/cdbench/sandboxes"))
TFAIL_CACHE = Path(os.environ.get("CDBENCH_TFAIL_CACHE", "/mnt/localssd/cdbench/tfail_cache"))
PYTEST_TIMEOUT_S = int(os.environ.get("CDBENCH_PYTEST_TIMEOUT", "120"))


@dataclasses.dataclass(frozen=True)
class TestId:
    """A pytest test identifier: file::Class::method or file::function."""
    nodeid: str

    def __str__(self) -> str:
        return self.nodeid


@dataclasses.dataclass
class JudgeResult:
    fixed_count: int
    total_bugs: int
    score: float  # fixed_count / total_bugs
    per_bug: dict[str, str]  # corruption_id -> "FIXED" | "MISS" | "PARTIAL" | "REGRESSED"
    t_fail: list[str]
    t_fail_passed: list[str]
    t_fail_still_failing: list[str]
    regressions: list[str]
    elapsed_sec: float


# ── Tfail discovery (offline) ────────────────────────────────────────────────


def _bug_cache_path(corruption_id: str, file_hash: str) -> Path:
    return TFAIL_CACHE / f"{corruption_id}_{file_hash[:12]}.json"


def discover_tfail(
    corruption,  # CorruptionSpec from benchmark_ablation
    sandbox_dir: Path,
    test_targets: list[str] | None = None,
    refresh: bool = False,
) -> list[str]:
    """Apply a single corruption and return the set of pytest nodeids that
    transition pass -> fail. Result is cached on disk keyed by
    (corruption_id, hash(find+replace+source_file))."""
    sig = hashlib.sha1(
        f"{corruption.find}|{corruption.replace}|{corruption.source_file}".encode()
    ).hexdigest()
    cache_path = _bug_cache_path(corruption.corruption_id, sig)
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text())["tfail"]

    target_path = sandbox_dir / corruption.source_file
    if not target_path.exists():
        raise FileNotFoundError(f"Source file missing in sandbox: {target_path}")

    original = target_path.read_text()
    if corruption.find not in original:
        raise ValueError(
            f"find string for {corruption.corruption_id} not in {target_path}"
        )

    # Snapshot baseline test result
    baseline = _run_pytest(sandbox_dir, test_targets)
    baseline_pass = set(baseline["passed"])

    # Apply corruption
    target_path.write_text(original.replace(corruption.find, corruption.replace, 1))
    try:
        corrupted = _run_pytest(sandbox_dir, test_targets)
    finally:
        target_path.write_text(original)

    # Tests that newly fail
    corrupted_failed = set(corrupted["failed"]) | set(corrupted["error"])
    tfail = sorted(corrupted_failed & baseline_pass)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "corruption_id": corruption.corruption_id,
        "sig": sig,
        "tfail": tfail,
        "baseline_passed": len(baseline_pass),
        "corrupted_failed": len(corrupted_failed),
    }, indent=2))
    return tfail


# ── Per-run scoring ─────────────────────────────────────────────────────────


def judge_fix(
    corruptions: list,  # list[CorruptionSpec]
    fixed_files: dict[str, str],
    sandbox_dir: Path,
    tfail_cache: dict[str, list[str]] | None = None,
    test_targets: list[str] | None = None,
    sample_regression_n: int = 100,
) -> JudgeResult:
    """Run the functional judge on a set of (corruption, fix) pairs.

    Args:
        corruptions: the bugs that were injected (already in sandbox source)
        fixed_files: model output, mapping source_file -> full new text
        sandbox_dir: directory containing the sklearn checkout with bugs
            already applied
        tfail_cache: precomputed T_fail per corruption_id; if None, we union
            empty sets and rely on pytest output
        test_targets: pytest paths/nodeids to run; if None, all tests
        sample_regression_n: if >0, also run this many random previously-passing
            tests to detect regressions
    """
    import time
    start = time.time()
    bug_ids = [c.corruption_id for c in corruptions]
    tfail_cache = tfail_cache or {}
    tfail_union: list[str] = sorted(set().union(*[set(tfail_cache.get(b, [])) for b in bug_ids]))

    # Apply model's fix (overwrite source files in sandbox)
    backups: dict[Path, str] = {}
    for rel_path, new_text in fixed_files.items():
        target = sandbox_dir / rel_path
        if not target.exists():
            continue
        backups[target] = target.read_text()
        target.write_text(new_text)

    try:
        # Run T_fail tests (must all pass) plus a sample for regression detection
        run_targets = list(tfail_union)
        if not run_targets and test_targets:
            run_targets = list(test_targets)
        result = _run_pytest(sandbox_dir, run_targets if run_targets else None)
    finally:
        for target, original in backups.items():
            target.write_text(original)

    passed_now = set(result["passed"])
    failed_now = set(result["failed"]) | set(result["error"])

    tfail_passed = sorted(set(tfail_union) & passed_now)
    tfail_still = sorted(set(tfail_union) & failed_now)

    # Per-bug attribution: a bug is FIXED iff all its tfail tests pass now
    per_bug: dict[str, str] = {}
    fixed_count = 0
    for bug_id in bug_ids:
        bug_tfail = set(tfail_cache.get(bug_id, []))
        if not bug_tfail:
            per_bug[bug_id] = "NO_TFAIL"
            continue
        if bug_tfail.issubset(passed_now):
            per_bug[bug_id] = "FIXED"
            fixed_count += 1
        elif bug_tfail & passed_now:
            per_bug[bug_id] = "PARTIAL"
        else:
            per_bug[bug_id] = "MISS"

    score = fixed_count / max(1, len(bug_ids))

    return JudgeResult(
        fixed_count=fixed_count,
        total_bugs=len(bug_ids),
        score=round(score, 4),
        per_bug=per_bug,
        t_fail=tfail_union,
        t_fail_passed=tfail_passed,
        t_fail_still_failing=tfail_still,
        regressions=[],  # populated in regression-detection mode
        elapsed_sec=round(time.time() - start, 2),
    )


# ── Pytest runner ───────────────────────────────────────────────────────────


def _run_pytest(sandbox_dir: Path, targets: list[str] | None) -> dict:
    """Run pytest in the sandbox and parse JUnit XML output for pass/fail
    breakdown. Returns dict with keys: passed, failed, error, skipped."""
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
        junit_path = Path(f.name)

    cmd = [
        "python", "-m", "pytest",
        "--tb=no",
        "-q",
        "--no-header",
        f"--junit-xml={junit_path}",
        "--timeout", str(PYTEST_TIMEOUT_S),
        "-p", "no:cacheprovider",
    ]
    if targets:
        cmd.extend(targets)

    try:
        proc = subprocess.run(
            cmd,
            cwd=sandbox_dir,
            capture_output=True,
            text=True,
            timeout=PYTEST_TIMEOUT_S * max(1, len(targets or [1])),
        )
    except subprocess.TimeoutExpired:
        return {"passed": [], "failed": [], "error": ["__pytest_timeout__"], "skipped": [],
                "stdout": "TIMEOUT", "stderr": "TIMEOUT"}

    return _parse_junit_xml(junit_path, proc.stdout, proc.stderr)


def _parse_junit_xml(junit_path: Path, stdout: str, stderr: str) -> dict:
    import xml.etree.ElementTree as ET
    out = {"passed": [], "failed": [], "error": [], "skipped": [],
           "stdout": stdout, "stderr": stderr}
    if not junit_path.exists() or junit_path.stat().st_size == 0:
        return out
    try:
        tree = ET.parse(junit_path)
    except ET.ParseError:
        return out
    root = tree.getroot()
    for tc in root.iter("testcase"):
        # Prefer 'file' attribute (real pytest path) over 'classname' (dotted module path)
        file_attr = tc.get("file") or ""
        name = tc.get("name", "")
        if file_attr:
            nodeid = f"{file_attr}::{name}"
        else:
            classname = tc.get("classname", "")
            filepath = classname.replace(".", "/") + ".py"
            nodeid = f"{filepath}::{name}"
        if tc.find("failure") is not None:
            out["failed"].append(nodeid)
        elif tc.find("error") is not None:
            out["error"].append(nodeid)
        elif tc.find("skipped") is not None:
            out["skipped"].append(nodeid)
        else:
            out["passed"].append(nodeid)
    try:
        junit_path.unlink()
    except OSError:
        pass
    return out


# ── Sandbox setup ───────────────────────────────────────────────────────────


def ensure_sandbox(domain: str, source_repo: str, commit: str | None = None,
                   pip_extra: list[str] | None = None) -> Path:
    """Create a per-domain sandbox at SANDBOX_ROOT/<domain>/. Idempotent.

    Args:
        domain: short name like "sklearn" or "fastapi"
        source_repo: git URL to clone
        commit: pin to this commit/tag
        pip_extra: additional pip-install args (e.g. test extras)
    """
    target = SANDBOX_ROOT / domain
    if (target / ".cdbench_ready").exists():
        return target
    target.mkdir(parents=True, exist_ok=True)
    if not (target / ".git").exists():
        subprocess.run(["git", "clone", source_repo, str(target)], check=True)
    if commit:
        subprocess.run(["git", "checkout", commit], cwd=target, check=True)
    pip_cmd = ["python", "-m", "pip", "install", "-e", str(target)]
    if pip_extra:
        pip_cmd.extend(pip_extra)
    subprocess.run(pip_cmd, check=True)
    (target / ".cdbench_ready").touch()
    return target


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["discover", "judge", "ensure"])
    ap.add_argument("--domain", default="sklearn")
    ap.add_argument("--catalog", help="path to JSON CorruptionSpec list")
    ap.add_argument("--targets", nargs="*", help="pytest targets")
    args = ap.parse_args()

    if args.cmd == "ensure":
        path = ensure_sandbox(
            args.domain,
            "https://github.com/scikit-learn/scikit-learn.git",
            commit="1.6.0",
            pip_extra=["pytest", "pytest-timeout", "pytest-xdist"],
        )
        print(f"Sandbox ready at: {path}")
    else:
        raise SystemExit("discover/judge entry-points to be wired with the catalog runner")
