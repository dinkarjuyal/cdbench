"""Export benchmark questions as SFT training data (ChatML JSONL).

Converts MCTaskCandidate objects to chain-of-thought training examples that teach
models to reason about non-obvious library behavior before committing to an answer.

Format: one JSON object per line (JSONL), each with a "messages" array in ChatML format.

Usage:
    from scripts.export_sft import export_sft, candidate_to_sft
    from scripts.generators.pandas_mc import MCTaskCandidate

    # Convert a single candidate
    example = candidate_to_sft(cand)

    # Export a list of candidates to a JSONL file
    export_sft(candidates, "/tmp/train.jsonl")

CLI (via benchmark_creator --export-sft):
    python3 -m benchmark_creator \\
        --families benchmarks/pandas_understanding/meta/repo_profile.json \\
        --strategy adversarial --n 1 --max-tasks 5 \\
        --export-sft /tmp/pandas_sft.jsonl --dry-run
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

# Import at call time to avoid circular imports at module level
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.generators.pandas_mc import MCTaskCandidate


def candidate_to_sft(cand: "MCTaskCandidate") -> dict:
    """Convert one MCTaskCandidate to a ChatML training example.

    The assistant turn uses a chain-of-thought structure:
      1. Common assumption (rule the model likely knows)
      2. Boundary case (why the rule breaks here)
      3. Correct output (the answer)
      4. Explanation (from candidate's explanation field)

    This structure teaches the model to notice the boundary before answering,
    targeting the overgeneralization failure mode that the benchmark measures.
    """
    rule = cand.metadata.get("rule", "")
    why_wrong = cand.metadata.get("why_model_gets_it_wrong", "")
    language = cand.metadata.get("language", "python")
    library_name = cand.metadata.get("library_name", "the library")
    generation_date = cand.metadata.get("generation_date", datetime.utcnow().isoformat())

    correct_text = next(
        (c["text"] for c in cand.choices if c["id"] == cand.correct_id),
        "",
    )

    system = (
        f"You are an expert {language} developer specializing in {library_name}. "
        "When shown a code snippet, reason step by step about what it actually outputs, "
        "paying careful attention to non-obvious edge cases and boundary conditions."
    )

    # Build choice listing for the question
    choice_lines = "\n".join(
        f"  {c['id']}. {c['text']}" for c in cand.choices
    )
    user = (
        f"What does the following code print?\n\n"
        f"```{language}\n{cand.snippet}\n```\n\n"
        f"{choice_lines}"
    )

    # Chain-of-thought assistant turn
    cot_parts = []
    if rule:
        cot_parts.append(f"Common assumption: {rule}")
    if why_wrong:
        cot_parts.append(f"Boundary case: {why_wrong}")
    cot_parts.append(f"Answer: **{cand.correct_id}. {correct_text}**")
    if cand.explanation:
        cot_parts.append(cand.explanation)

    assistant = "\n\n".join(cot_parts)

    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "language": language,
        "source_benchmark": cand.family,
        "question_type": cand.question_type,
        "task_id": cand.task_id,
        "generation_date": generation_date,
    }


def export_sft(
    candidates: "list[MCTaskCandidate]",
    output_path: str | Path,
    *,
    append: bool = False,
) -> int:
    """Write candidates as JSONL to output_path.

    Args:
        candidates: List of MCTaskCandidate objects to export.
        output_path: Destination file path (.jsonl).
        append: If True, append to existing file instead of overwriting.

    Returns:
        Number of examples written.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    written = 0
    with open(path, mode, encoding="utf-8") as f:
        for cand in candidates:
            example = candidate_to_sft(cand)
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            written += 1

    return written
