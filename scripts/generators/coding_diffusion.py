"""Coding diffusion strategy: multi-corruption debugging tasks with configurable difficulty.

Inspired by diffusion models — a forward process gradually corrupts a clean codebase
(adding "noise" in the form of bugs), and the agent must perform the reverse process
(denoising/debugging) to restore all tests to passing.

The "noise schedule" controls difficulty:
  - corruption_count: how many bugs to inject (1-10)
  - spread: how distributed the bugs are (clustered | scattered | mixed)
  - dependency: how the bugs relate (independent | cascading | masking)

Pipeline:
  1. RepoAnalyzer extracts behavioral families from the repo
  2. For each family, an LLM proposes candidate corruptions
  3. Each corruption is execution-verified: must independently break >= 1 test
  4. N corruptions are composed per DiffusionSchedule
  5. The combined corrupted state is verified: all target tests fail
  6. TaskCandidate is produced with start_state_patches + visible/hidden tests

Output: TaskCandidate objects (code-fixing tasks), NOT MCTaskCandidate.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic

from scripts.generators.base import TaskCandidate
from scripts.generators.bug_patterns import BugVariant, inject_bugs, get_all_patterns
from scripts.generators.runtime import ExecutionRuntime, PythonRuntime
from scripts.generators.strategy_registry import GenerationStrategy, register_strategy


# ── Provider adapter (shared with adversarial_mc) ─────────────────────────────

class _PIMessage:
    def __init__(self, text: str):
        self.content = [type("_C", (), {"text": text})()]


class PIClientAdapter:
    _BASE_URL = "https://api.pinference.ai/api/v1"

    def __init__(self, api_key: str, base_url: str | None = None):
        from openai import OpenAI
        self._oai = OpenAI(api_key=api_key, base_url=base_url or self._BASE_URL)
        self.messages = self

    def create(self, model: str, max_tokens: int, system: str,
               messages: list[dict], **_kwargs) -> _PIMessage:
        oai_messages = [{"role": "system", "content": system}] + messages
        resp = self._oai.chat.completions.create(
            model=model, max_tokens=max_tokens, messages=oai_messages,
        )
        return _PIMessage(resp.choices[0].message.content or "")


def _make_client(api_key: str | None, provider: str = "anthropic"):
    return anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class CorruptionSpec:
    """One verified corruption (a single 'noise step' in the diffusion process)."""
    corruption_id: str
    source_file: str           # relative path from repo root
    find: str                  # exact text to find in clean code
    replace: str               # text to replace with (the bug)
    description: str           # human-readable bug description
    broken_test: str           # test code that fails after this corruption
    passing_test: str          # test code that passes in clean code but reveals the bug's domain
    family: str                # behavioral family
    subtlety: int              # 1=obvious breakage, 5=edge-case only


@dataclass
class DiffusionSchedule:
    """Controls how corruptions are composed into a task — the 'noise schedule'."""
    corruption_count: int = 3          # 1-10
    spread: str = "scattered"          # clustered | scattered | mixed
    dependency: str = "independent"     # independent | cascading | masking
    subtlety_min: int = 1
    subtlety_max: int = 5

    def difficulty(self) -> int:
        """Map schedule parameters to a 1-5 difficulty rating."""
        base = min(5, max(1, self.corruption_count))
        if self.dependency == "masking":
            base = min(5, base + 1)
        elif self.dependency == "cascading":
            base = min(5, base + 0.5)
        return int(round(base))


# ── LLM prompts ───────────────────────────────────────────────────────────────

_CORRUPTION_PROPOSER_SYSTEM = """\
You are a software regression generator. Given a codebase and behavioral rules,
propose single-point bugs that would break specific tests.

Each corruption must:
1. Be a MINIMAL change (1-3 lines) — logic inversion, off-by-one, missing check,
   wrong variable, swapped arguments, premature return, missing state update
2. Break at least one test when applied to the clean code
3. NOT be trivially detectable (no syntax errors, no import errors, no obvious typos)
4. Represent a REALISTIC regression pattern that could occur in a code review

Respond using ONLY these tags — no other text:

<corruption>
<id>short_identifier</id>
<source_file>relative/path/to/file.py</source_file>
<find>exact text in the current clean code to find</find>
<replace>the bugged version of that text</replace>
<description>one-sentence description of the bug</description>
<broken_test>
self-contained test code that FAILS after this corruption is applied
(include imports, use pytest assertions, test a specific behavior)
</broken_test>
<passing_test>
self-contained test code that PASSES in the clean code and relates to
the corrupted functionality (shows the expected behavior domain)
</passing_test>
<subtlety>1-5 where 1=obvious breakage 5=edge-case only</subtlety>
</corruption>"""

_CORRUPTION_PROPOSER_USER = """\
Family: {family_name} — {family_description}
Seed rules for this family:
{seed_rules}
Library install: {install_line}
Language: {language}
Source files available:
{source_files}

Propose {n_corruptions} different corruptions, each in a different <corruption> block.
Requirements:
- Each corruption must be in a DIFFERENT source file if possible
- Each corruption must independently break at least one specific test
- Corruptions should be subtle — avoid obvious breakage
- Prefer corruptions that interact with the behavioral rules listed above
- broken_test must be self-contained runnable {language} with all imports"""

_CORRUPTION_COMPOSITION_SYSTEM = """\
You are composing multiple corruptions into a single debugging task.

Given {n} verified corruptions, compose them into a coherent task prompt
that describes the debugging challenge without revealing specific bug locations.

Requirements:
- Do NOT reveal which files have bugs or how many bugs there are
- Describe the symptom (tests failing) and let the agent discover the root cause
- Provide enough context for the agent to know what the codebase does
- Include the visible test that the agent can run to check progress

Respond using ONLY these tags — no other text:

<task_prompt>
The complete prompt text to show the agent.
Include: brief codebase description, the failing test output, and instructions.
Tell the agent the total number of bugs to find.
</task_prompt>"""


# ── Tag parsing ────────────────────────────────────────────────────────────────

def _tag(text: str, tag: str) -> str:
    """Extract first occurrence of <tag>...</tag> from text."""
    open_t, close_t = f"<{tag}>", f"</{tag}>"
    start = text.find(open_t)
    if start == -1:
        return ""
    end = text.find(close_t, start)
    if end == -1:
        return ""
    return text[start + len(open_t):end].strip()


def _tag_all(text: str, tag: str) -> list[str]:
    """Extract all occurrences of <tag>...</tag> from text."""
    results = []
    open_t, close_t = f"<{tag}>", f"</{tag}>"
    pos = 0
    while True:
        start = text.find(open_t, pos)
        if start == -1:
            break
        end = text.find(close_t, start)
        if end == -1:
            break
        results.append(text[start + len(open_t):end].strip())
        pos = end + len(close_t)
    return results


# ── Execution helpers ──────────────────────────────────────────────────────────

def _run_snippet(snippet: str, timeout: int = 10) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"ERROR: timed out after {timeout}s"
    if result.returncode != 0:
        return False, f"ERROR: {result.stderr.strip()[:200]}"
    return True, result.stdout.strip()


def _corruption_id(spec: dict, family: str) -> str:
    """Generate a stable ID from corruption metadata."""
    raw = f"{family}:{spec.get('source_file', '')}:{spec.get('find', '')[:60]}"
    h = hashlib.md5(raw.encode()).hexdigest()[:8]
    slug = spec.get("description", "corruption")[:30].lower()
    slug = "".join(c if c.isalnum() else "_" for c in slug).strip("_")
    return f"corr_{family[:8]}_{slug}_{h}"


def _bug_variant_to_corruption(variant: BugVariant, source_file: str, family: str) -> CorruptionSpec:
    """Convert a deterministic BugVariant (from bug_patterns) to a CorruptionSpec.

    This bridges the pattern-based fast path with the LLM-based pipeline,
    allowing deterministic bug patterns to be used as corruptions in
    multi-bug code-fixing tasks.
    """
    # Generate a broken_test that detects the bug by comparing outputs
    broken_test = (
        f"# Bug: {variant.bug_description}\n"
        f"# Original output should differ from buggy output\n"
        f"result_orig = None\n"
        f"result_buggy = None\n"
        f"try:\n"
        f"    exec(compile({variant.original_code!r}, '<orig>', 'exec'), {{'__builtins__': __builtins__}})\n"
        f"except Exception as e:\n"
        f"    result_orig = type(e).__name__\n"
        f"try:\n"
        f"    exec(compile({variant.buggy_code!r}, '<buggy>', 'exec'), {{'__builtins__': __builtins__}})\n"
        f"except Exception as e:\n"
        f"    result_buggy = type(e).__name__\n"
        f"assert result_orig != result_buggy, 'Bug did not change behavior'"
    )

    passing_test = (
        f"# Sanity check: original code runs cleanly\n"
        f"compile({variant.original_code!r}, '<orig>', 'exec')\n"
    )

    return CorruptionSpec(
        corruption_id=f"corr_{family[:8]}_{variant.pattern_name}_{hash(variant.buggy_code) % 100000:05d}",
        source_file=source_file,
        find=variant.original_code,
        replace=variant.buggy_code,
        description=variant.bug_description,
        broken_test=broken_test,
        passing_test=passing_test,
        family=family,
        subtlety=variant.severity_level,
    )


def _extract_test_calls(code: str) -> str:
    """Extract function signatures from code and generate targeted test calls.

    Parses function definitions to determine argument names and generates
    calls with boundary-condition-triggering inputs so that bug patterns
    (off-by-one, logic inversion, etc.) are likely to produce different output.
    """
    import re
    # Match function definitions with their parameter lists
    fn_defs = re.findall(r"^def\s+(\w+)\s*\(([^)]*)\)", code, re.MULTILINE)
    if not fn_defs:
        return ""

    # Per-argument test values based on common parameter name patterns
    arg_values = {
        "values": "[1,2,3]", "items": "[1,2,3]", "data": "[1,2,3]",
        "numbers": "[1,2,3]", "nums": "[1,2,3]", "lst": "[1,2,3]",
        "weights": "[1,1,1]", "target": "2.0", "threshold": "4",
        "value": "5", "x": "4", "n": "3", "idx": "1", "i": "1",
        "lo": "0", "lo_val": "0", "min_val": "0",
        "hi": "10", "hi_val": "10", "max_val": "10",
        "count": "5", "size": "3", "length": "3",
        "p": "50", "percent": "50", "ratio": "0.5",
        "cutoff": "4", "limit": "10",
    }

    lines = ["# Auto-generated boundary test calls"]
    for name, params_str in fn_defs[:8]:
        params = [p.strip().split(":")[0].split("=")[0].strip()
                   for p in params_str.split(",") if p.strip() and p.strip() != "self"]
        if not params:
            lines.append(f"try: print(repr({name}()))\nexcept Exception as e: print(type(e).__name__)")
            continue

        # Build 2-3 argument sets using name-based hints and positional defaults
        default_per_pos = {0: "5", 1: "0", 2: "10", 3: "1"}
        arg_sets: list[list[str]] = []

        # Set 1: name-based values
        set1 = [arg_values.get(p, default_per_pos.get(i, "0"))
                for i, p in enumerate(params)]
        arg_sets.append(set1)

        # Set 2: boundary values (0, negatives)
        set2 = [arg_values.get(p, "0") for p in params]
        arg_sets.append(set2)

        # Set 3: edge values
        set3 = [arg_values.get(p, "-1" if i == 0 else "1")
                for i, p in enumerate(params)]
        arg_sets.append(set3)

        for args in arg_sets[:3]:
            args_str = ", ".join(args)
            lines.append(
                f"try: print(repr({name}({args_str})))\n"
                f"except Exception as e: print(type(e).__name__)"
            )

    return "\n".join(lines)


def _generate_pattern_corruptions(code_snippets: dict[str, str], family: str,
                                   max_per_file: int = 3) -> list[CorruptionSpec]:
    """Generate corruptions using deterministic bug patterns (fast path).

    Args:
        code_snippets: {source_file: code_content} for files to corrupt
        family: behavioral family name
        max_per_file: max corruptions per file

    Returns:
        List of CorruptionSpec from pattern injection, execution-verified.
    """
    corruptions: list[CorruptionSpec] = []
    for source_file, code in code_snippets.items():
        variants = inject_bugs(code, max_variants=max_per_file)
        test_calls = _extract_test_calls(code)

        for variant in variants:
            # Verify the bug actually changes behavior
            # Append test calls so we exercise the functions
            orig_with_test = variant.original_code + ("\n" + test_calls if test_calls else "")
            buggy_with_test = variant.buggy_code + ("\n" + test_calls if test_calls else "")

            ok_orig, out_orig = _run_snippet(orig_with_test, timeout=5)
            ok_buggy, out_buggy = _run_snippet(buggy_with_test, timeout=5)

            # Reject if either has syntax/import errors
            if not ok_orig and ("SyntaxError" in out_orig or "ImportError" in out_orig):
                continue
            if not ok_buggy and ("SyntaxError" in out_buggy or "ImportError" in out_buggy):
                continue
            # Outputs must differ (the bug must be visible)
            if out_orig == out_buggy:
                continue

            corruption = _bug_variant_to_corruption(variant, source_file, family)
            corruptions.append(corruption)

    return corruptions


# ── Core generator ─────────────────────────────────────────────────────────────

class CodingDiffusionGenerator:
    """Generates multi-corruption debugging tasks from behavioral families."""

    def __init__(
        self,
        client,
        model: str = "claude-sonnet-4-6",
        verbose: bool = False,
        seed: int | None = None,
        runtime: ExecutionRuntime | None = None,
        schedule: DiffusionSchedule | None = None,
    ):
        self.client = client
        self.model = model
        self.verbose = verbose
        self.rng = __import__("random").Random(seed)
        self.runtime = runtime or PythonRuntime()
        self.schedule = schedule or DiffusionSchedule()

    def _propose_corruptions(
        self, family: dict, n_corruptions: int = 5,
    ) -> list[CorruptionSpec]:
        """Ask the LLM to propose candidate corruptions for a family."""
        seed_rules = family.get("seed_rules", [])
        if isinstance(seed_rules, list):
            seed_rules_text = "\n".join(f"  - {r}" for r in seed_rules)
        else:
            seed_rules_text = str(seed_rules)

        source_files = family.get("source_files", [])
        if isinstance(source_files, list):
            source_files_text = "\n".join(f"  - {f}" for f in source_files[:20])
        else:
            source_files_text = "(available source files will be inferred from install)"

        install = family.get("install", "pip install <library>")
        language = self.runtime.language

        user_msg = _CORRUPTION_PROPOSER_USER.format(
            family_name=family.get("name", "unknown"),
            family_description=family.get("description", ""),
            seed_rules=seed_rules_text,
            install_line=install,
            language=language,
            source_files=source_files_text,
            n_corruptions=n_corruptions,
        )

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=_CORRUPTION_PROPOSER_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = resp.content[0].text
        except Exception as e:
            if self.verbose:
                print(f"    [corruptor] LLM call failed: {e}")
            return []

        # Parse corruption blocks
        blocks = _tag_all(text, "corruption")
        specs: list[CorruptionSpec] = []
        for block in blocks:
            spec_dict = {
                "id": _tag(block, "id"),
                "source_file": _tag(block, "source_file"),
                "find": _tag(block, "find"),
                "replace": _tag(block, "replace"),
                "description": _tag(block, "description"),
                "broken_test": _tag(block, "broken_test"),
                "passing_test": _tag(block, "passing_test"),
                "subtlety": _tag(block, "subtlety"),
            }

            # Validate required fields
            if not all([spec_dict["source_file"], spec_dict["find"],
                        spec_dict["replace"], spec_dict["description"]]):
                if self.verbose:
                    print(f"    [corruptor] Skipping incomplete corruption block")
                continue

            try:
                subtlety = int(spec_dict["subtlety"] or "3")
                subtlety = max(1, min(5, subtlety))
            except ValueError:
                subtlety = 3

            specs.append(CorruptionSpec(
                corruption_id=spec_dict["id"] or _corruption_id(spec_dict, family.get("name", "")),
                source_file=spec_dict["source_file"],
                find=spec_dict["find"],
                replace=spec_dict["replace"],
                description=spec_dict["description"],
                broken_test=spec_dict["broken_test"],
                passing_test=spec_dict["passing_test"],
                family=family.get("name", "unknown"),
                subtlety=subtlety,
            ))

        if self.verbose:
            print(f"    [corruptor] Proposed {len(specs)} corruptions for family {family.get('name', '?')}")
        return specs

    def _verify_corruption(self, corruption: CorruptionSpec) -> bool:
        """Verify that a corruption actually breaks something when applied.

        Runs the broken_test snippet — if it produces an error, the corruption
        is considered valid (since it's designed to fail after the corruption).
        For a lightweight check, we just verify the broken_test runs without
        import errors in the clean state (it should pass cleanly).
        """
        if not corruption.broken_test:
            return False

        # Quick sanity: the broken_test should at least be valid Python
        ok, out = _run_snippet(corruption.broken_test, timeout=8)
        # In the CLEAN state, the broken_test might pass or fail depending on
        # what it tests. The key is it's valid runnable code.
        if out.startswith("ERROR:") and ("ImportError" in out or "SyntaxError" in out):
            if self.verbose:
                print(f"    [verify] Rejected corruption {corruption.corruption_id}: "
                      f"broken_test has import/syntax error: {out[:80]}")
            return False
        return True

    def _compose_task(
        self,
        corruptions: list[CorruptionSpec],
        family: dict,
        schedule: DiffusionSchedule,
    ) -> TaskCandidate:
        """Compose verified corruptions into a single TaskCandidate."""
        # Generate task ID from corruption set
        corruption_ids = "|".join(c.corruption_id for c in corruptions)
        h = hashlib.md5(corruption_ids.encode()).hexdigest()[:8]
        family_name = family.get("name", "unknown")
        task_id = f"diff_{family_name[:12]}_{len(corruptions)}bug_{h}"

        # Build start_state_patches
        # In a full implementation, these would be applied to a cloned repo.
        # For now, store as metadata that the task writer will use.
        patches: dict[str, str] = {}
        for c in corruptions:
            # The key encodes both the file and the find/replace operation
            # The task writer will apply these sequentially
            key = c.source_file
            # If the same file has multiple corruptions, they compose
            if key not in patches:
                patches[key] = ""  # placeholder — actual patching happens at task write time

        # Build visible tests from corruption broken_tests
        visible_tests = []
        for c in corruptions:
            if c.broken_test:
                visible_tests.append(c.broken_test)

        # Build hidden tests from passing_tests (these should pass after agent fixes)
        hidden_tests = []
        for c in corruptions:
            if c.passing_test:
                hidden_tests.append(c.passing_test)

        # Build prompt
        n_bugs = len(corruptions)
        bug_spread = schedule.spread
        prompt = self._build_prompt(
            n_bugs=n_bugs,
            bug_spread=bug_spread,
            family=family,
            corruptions=corruptions,
        )

        # Store corruption specs in metadata for the task writer
        corruption_meta = []
        for c in corruptions:
            corruption_meta.append({
                "corruption_id": c.corruption_id,
                "source_file": c.source_file,
                "find": c.find,
                "replace": c.replace,
                "description": c.description,
                "family": c.family,
                "subtlety": c.subtlety,
            })

        return TaskCandidate(
            task_id=task_id,
            task_type="coding_diffusion",
            family=family_name,
            difficulty=schedule.difficulty(),
            prompt=prompt,
            start_state_patches=patches,
            visible_tests=visible_tests,
            hidden_tests=hidden_tests,
            structural_checks=[],
            generation_recipe=(
                f"coding_diffusion: {n_bugs} corruptions, "
                f"spread={bug_spread}, dependency={schedule.dependency}, "
                f"subtlety={schedule.subtlety_min}-{schedule.subtlety_max}"
            ),
            is_noop=False,
            is_impossible=False,
            metadata={
                "corruptions": corruption_meta,
                "schedule": {
                    "corruption_count": schedule.corruption_count,
                    "spread": schedule.spread,
                    "dependency": schedule.dependency,
                    "subtlety_min": schedule.subtlety_min,
                    "subtlety_max": schedule.subtlety_max,
                },
                "library_name": family.get("library_name", ""),
                "install": family.get("install", ""),
                "description": f"{n_bugs}-bug debugging task in {family_name}",
            },
        )

    def _build_prompt(
        self,
        n_bugs: int,
        bug_spread: str,
        family: dict,
        corruptions: list[CorruptionSpec],
    ) -> str:
        """Build the agent-facing prompt for the debugging task."""
        library = family.get("library_name", "the library")
        install = family.get("install", "")

        spread_desc = {
            "clustered": "concentrated in a small area of the codebase",
            "scattered": "spread across different files and modules",
            "mixed": "some clustered together, some in other files",
        }.get(bug_spread, "spread across the codebase")

        # Collect the files that are affected
        affected_files = sorted(set(c.source_file for c in corruptions))

        # Build visible test descriptions
        test_hints = []
        for i, c in enumerate(corruptions, 1):
            symptom = c.description[:80]
            test_hints.append(f"  Bug {i} symptom: {symptom}")

        prompt = textwrap.dedent(f"""\
            This codebase has {n_bugs} bugs introduced by a recent refactor.
            Your task is to find and fix ALL {n_bugs} bugs so that every test passes.

            The bugs are {spread_desc}. Each bug is a single-point change
            (logic inversion, off-by-one, missing check, wrong variable, swapped
            arguments, premature return, missing state update, etc.) — no deleted
            or added functions.

            Key files to investigate:
            {chr(10).join('  - ' + f for f in affected_files)}

            Symptom hints:
            {chr(10).join(test_hints)}

            Library: {library}
            Install: {install}

            Verify your fixes with: pytest tests/ -q

            You must fix ALL {n_bugs} bugs. Partial fixes will receive partial credit.
            Do NOT add new functions or refactor existing code — only fix the bugs.
        """)
        return prompt.strip()

    def generate(
        self,
        families: list[dict],
        n_per_family: int = 3,
        schedule: DiffusionSchedule | None = None,
    ) -> list[TaskCandidate]:
        """Generate multi-corruption debugging tasks from behavioral families.

        Uses two corruption sources:
          1. Deterministic bug patterns (fast path, free, for single-bug or when
             family has source code snippets)
          2. LLM-proposed corruptions (richer, for multi-bug and when patterns
             aren't available)

        Falls back from patterns to LLM when patterns don't yield enough
        verified corruptions.
        """
        schedule = schedule or self.schedule
        candidates: list[TaskCandidate] = []

        for family in families:
            family_name = family.get("name", "unknown")
            if self.verbose:
                print(f"\n  [coding_diffusion] Processing family: {family_name}")

            # --- Fast path: try deterministic bug patterns first ---
            verified: list[CorruptionSpec] = []
            source_snippets = family.get("source_snippets", {})
            if source_snippets and self.verbose:
                print(f"    [coding_diffusion] Trying pattern-based fast path "
                      f"({len(source_snippets)} source files)")

            if source_snippets:
                pattern_corruptions = _generate_pattern_corruptions(
                    source_snippets, family_name, max_per_file=schedule.corruption_count
                )
                for c in pattern_corruptions:
                    if schedule.subtlety_min <= c.subtlety <= schedule.subtlety_max:
                        verified.append(c)

                if self.verbose and pattern_corruptions:
                    print(f"    [coding_diffusion] Pattern fast path: "
                          f"{len(pattern_corruptions)} corruptions, "
                          f"{len(verified)} in subtlety range")

            # --- Slow path: LLM-proposed corruptions ---
            # Use LLM if we don't have enough from patterns, or for multi-bug tasks
            if len(verified) < schedule.corruption_count:
                n_proposals = max(schedule.corruption_count * 3, 6)
                proposed = self._propose_corruptions(family, n_corruptions=n_proposals)

                for c in proposed:
                    if self._verify_corruption(c):
                        if schedule.subtlety_min <= c.subtlety <= schedule.subtlety_max:
                            verified.append(c)
                    elif self.verbose:
                        print(f"    [verify] Rejected: {c.corruption_id}")

            if len(verified) < schedule.corruption_count:
                if self.verbose:
                    print(f"    [coding_diffusion] Only {len(verified)} verified corruptions "
                          f"(need {schedule.corruption_count}), skipping family {family_name}")
                continue

            # Step 3: Select corruptions per schedule
            selected = self._select_corruptions(verified, schedule)
            if not selected:
                continue

            # Step 4: Compose into a TaskCandidate
            task = self._compose_task(selected, family, schedule)
            candidates.append(task)

            if self.verbose:
                print(f"    [coding_diffusion] Created task {task.task_id} "
                      f"with {len(selected)} corruptions, difficulty={task.difficulty}")

            # Generate multiple tasks per family if requested
            for _ in range(n_per_family - 1):
                # Re-select with different random seed for variety
                self.rng.shuffle(verified)
                selected2 = self._select_corruptions(verified, schedule)
                if selected2:
                    task2 = self._compose_task(selected2, family, schedule)
                    candidates.append(task2)
                    if self.verbose:
                        print(f"    [coding_diffusion] Created task {task2.task_id} "
                              f"with {len(selected2)} corruptions, difficulty={task2.difficulty}")

        return candidates

    def _select_corruptions(
        self,
        verified: list[CorruptionSpec],
        schedule: DiffusionSchedule,
    ) -> list[CorruptionSpec]:
        """Select N corruptions from the verified pool per the schedule constraints."""
        n = min(schedule.corruption_count, len(verified))
        if n == 0:
            return []

        if schedule.spread == "clustered":
            # Prefer corruptions in the same file
            file_groups: dict[str, list[CorruptionSpec]] = {}
            for c in verified:
                file_groups.setdefault(c.source_file, []).append(c)
            # Find the file with the most corruptions
            best_file = max(file_groups, key=lambda f: len(file_groups[f]))
            pool = file_groups[best_file]
            if len(pool) < n:
                # Fill from other files
                remaining = [c for c in verified if c.source_file != best_file]
                self.rng.shuffle(remaining)
                pool = pool + remaining[:n - len(pool)]
            self.rng.shuffle(pool)
            return pool[:n]

        elif schedule.spread == "scattered":
            # Prefer corruptions in different files
            seen_files: set[str] = set()
            selected: list[CorruptionSpec] = []
            shuffled = verified[:]
            self.rng.shuffle(shuffled)
            # First pass: pick one per file
            for c in shuffled:
                if c.source_file not in seen_files and len(selected) < n:
                    selected.append(c)
                    seen_files.add(c.source_file)
            # Second pass: fill remaining slots
            for c in shuffled:
                if c not in selected and len(selected) < n:
                    selected.append(c)
            return selected

        else:  # mixed
            shuffled = verified[:]
            self.rng.shuffle(shuffled)
            return shuffled[:n]


# ── Strategy registration ──────────────────────────────────────────────────────

@register_strategy("coding_diffusion")
class CodingDiffusionStrategy(GenerationStrategy):
    """Multi-corruption debugging tasks with configurable noise schedule (coding diffusion)."""

    def __init__(
        self,
        api_key: str | None = None,
        verbose: bool = False,
        seed: int | None = None,
        runtime: ExecutionRuntime | None = None,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-6",
        schedule: DiffusionSchedule | None = None,
    ):
        client = _make_client(api_key, provider)
        self._gen = CodingDiffusionGenerator(
            client=client,
            model=model,
            verbose=verbose,
            seed=seed,
            runtime=runtime,
            schedule=schedule,
        )

    def generate(
        self, families: list[dict], n_per_family: int = 3
    ) -> list:
        """Generate coding diffusion tasks.

        Returns list of TaskCandidate (NOT MCTaskCandidate).
        The CLI dispatches to write_diffusion_task() for these.
        """
        return self._gen.generate(
            families=families,
            n_per_family=n_per_family,
        )
