"""Offline T_fail discovery: for each corruption in the catalog, find which
pytest tests transition pass->fail. Cached on disk so the expensive run is
amortized across all evaluation runs.

Usage:
    python discover_tfail.py --catalog hand --domain sklearn
    python discover_tfail.py --catalog all --workers 8

The cache lives at $CDBENCH_TFAIL_CACHE (default /mnt/localssd/cdbench/tfail_cache).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.functional_judge import (
    SANDBOX_ROOT,
    TFAIL_CACHE,
    discover_tfail,
    ensure_sandbox,
)


# Source file -> pytest target mapping. Narrows discovery to the relevant
# test module so we don't run the entire test suite per corruption.
SKLEARN_SOURCE_TO_TESTS = {
    "sklearn/metrics/_classification.py": ["sklearn/metrics/tests/test_classification.py"],
    "sklearn/metrics/_ranking.py": ["sklearn/metrics/tests/test_ranking.py"],
    "sklearn/preprocessing/_data.py": ["sklearn/preprocessing/tests/test_data.py"],
    "sklearn/linear_model/_ridge.py": ["sklearn/linear_model/tests/test_ridge.py"],
    "sklearn/ensemble/_forest.py": ["sklearn/ensemble/tests/test_forest.py"],
    "sklearn/model_selection/_split.py": ["sklearn/model_selection/tests/test_split.py"],
}

FASTAPI_SOURCE_TO_TESTS = {
    "fastapi/dependencies/utils.py": [
        "tests/test_dependency_cache.py",
        "tests/test_dependency_class.py",
        "tests/test_dependency_contextmanager.py",
        "tests/test_dependency_overrides.py",
        "tests/test_dependency_security_overrides.py",
    ],
    "fastapi/routing.py": [
        "tests/test_additional_responses_router.py",
        "tests/test_custom_route_class.py",
        "tests/test_router_prefix.py",
    ],
    "fastapi/applications.py": [
        "tests/test_application.py",
        "tests/test_additional_responses_default_validationerror.py",
    ],
    "fastapi/params.py": [
        "tests/test_param_class.py",
        "tests/test_ambiguous_params.py",
    ],
}

DOMAIN_CONFIG = {
    "sklearn": {
        "sandbox": ("sklearn", "https://github.com/scikit-learn/scikit-learn.git",
                     "1.6.0", ["pytest", "pytest-timeout", "pytest-xdist"]),
        "source_to_tests": SKLEARN_SOURCE_TO_TESTS,
        "full_suite_targets": None,  # None = run all tests
        "corruption_module": ("scripts.benchmark_ablation", "HAND_CORRUPTIONS"),
    },
    "fastapi": {
        "sandbox": ("fastapi", "https://github.com/fastapi/fastapi.git",
                     None, ["pytest", "pytest-asyncio", "httpx"]),
        "source_to_tests": FASTAPI_SOURCE_TO_TESTS,
        "full_suite_targets": ["tests/"],  # FastAPI narrow mapping misses many corruptions
        "corruption_module": ("scripts.fastapi_corruptions", "FASTAPI_CORRUPTIONS"),
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="sklearn", choices=list(DOMAIN_CONFIG))
    ap.add_argument("--catalog", choices=["hand", "sgs", "all"], default="hand")
    ap.add_argument("--refresh", action="store_true",
                    help="ignore cache, re-run discovery")
    ap.add_argument("--refresh-zero", action="store_true",
                    help="re-run discovery only for corruptions with 0 T_fail (uses full suite)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--full-suite", action="store_true",
                    help="use full test suite instead of narrow mapping")
    args = ap.parse_args()

    cfg = DOMAIN_CONFIG[args.domain]
    sandbox_name, repo_url, commit, pip_extra = cfg["sandbox"]
    sandbox = ensure_sandbox(sandbox_name, repo_url, commit=commit, pip_extra=pip_extra)
    source_to_tests = cfg["source_to_tests"]

    # Load corruption catalog
    mod_name, attr_name = cfg["corruption_module"]
    import importlib
    mod = importlib.import_module(mod_name)
    corruptions = list(getattr(mod, attr_name))

    if args.catalog in ("sgs", "all"):
        sgs_path = ROOT / "results" / f"sgs_corruptions_{args.domain}.json"
        if sgs_path.exists():
            from scripts.generators.coding_diffusion import CorruptionSpec
            sgs = [CorruptionSpec(**c) for c in json.loads(sgs_path.read_text())]
            if args.catalog == "all":
                corruptions.extend(sgs)
            else:
                corruptions = sgs

    if args.limit:
        corruptions = corruptions[: args.limit]

    print(f"Discovering T_fail for {len(corruptions)} corruptions in {sandbox} "
          f"(domain={args.domain})")
    TFAIL_CACHE.mkdir(parents=True, exist_ok=True)

    # Set cache dir per domain
    domain_cache = TFAIL_CACHE.parent / f"tfail_cache_{args.domain}" if args.domain != "sklearn" else TFAIL_CACHE
    domain_cache.mkdir(parents=True, exist_ok=True)

    # Override the cache path in functional_judge
    import scripts.functional_judge as fj_mod
    orig_cache = fj_mod.TFAIL_CACHE
    fj_mod.TFAIL_CACHE = domain_cache

    summary = []
    start = time.time()
    for i, c in enumerate(corruptions, 1):
        targets = source_to_tests.get(c.source_file) if not args.full_suite else None
        if not targets and not args.full_suite:
            print(f"  [{i}/{len(corruptions)}] {c.corruption_id}: no test mapping; will use full suite")
            targets = cfg.get("full_suite_targets")

        t0 = time.time()
        try:
            tfail = discover_tfail(c, sandbox, test_targets=targets, refresh=args.refresh)
        except Exception as e:
            print(f"  [{i}/{len(corruptions)}] {c.corruption_id}: ERROR {e}")
            summary.append({"corruption_id": c.corruption_id, "tfail": [], "error": str(e)})
            continue
        dt = time.time() - t0

        # If 0 T_fail with narrow mapping, try full suite
        if len(tfail) == 0 and targets and not args.full_suite and cfg.get("full_suite_targets"):
            print(f"  [{i}/{len(corruptions)}] {c.corruption_id}: 0 T_fail with narrow mapping, "
                  f"trying full suite...")
            try:
                tfail = discover_tfail(
                    c, sandbox,
                    test_targets=cfg["full_suite_targets"],
                    refresh=True,
                )
            except Exception as e:
                print(f"  [{i}/{len(corruptions)}] {c.corruption_id}: full suite ERROR {e}")
                summary.append({"corruption_id": c.corruption_id, "tfail": [], "error": str(e)})
                continue
            dt = time.time() - t0

        print(f"  [{i}/{len(corruptions)}] {c.corruption_id}: {len(tfail)} tests fail "
              f"({dt:.1f}s)")
        summary.append({"corruption_id": c.corruption_id, "tfail": tfail, "elapsed_sec": dt})

    # Restore original cache path
    fj_mod.TFAIL_CACHE = orig_cache

    total = time.time() - start
    out = ROOT / "results" / f"tfail_summary_{args.domain}_{args.catalog}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "domain": args.domain,
        "catalog": args.catalog,
        "total_elapsed_sec": total,
        "corruptions": summary,
    }, indent=2))
    print(f"\nDone in {total:.1f}s. Summary: {out}")
    print(f"Cache: {domain_cache}")


if __name__ == "__main__":
    main()
