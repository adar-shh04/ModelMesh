"""Scores one model output for one item. Judge-labelled kinds return None until the judge runs."""

from __future__ import annotations

from modelmesh.ml.benchmark.ifeval import check_all
from modelmesh.ml.benchmark.items import BenchItem
from modelmesh.ml.benchmark.metrics import (
    Score,
    classification_score,
    f1_score,
    label_with_schema_score,
    multiple_choice_score,
    numeric_answer_score,
    schema_valid_score,
)

F1_ACCEPTANCE = 0.5
JUDGE_ACCEPTANCE = 4


def score_automatic(item: BenchItem, output: str) -> Score | None:
    reference = item.reference
    match item.label_kind:
        case "classification":
            return classification_score(output, reference["label"])
        case "support_label":
            return label_with_schema_score(output, reference["label"])
        case "numeric":
            return numeric_answer_score(output, reference["gold"])
        case "choice":
            return multiple_choice_score(output, reference["gold"])
        case "schema_valid":
            return schema_valid_score(output, reference["schema"])
        case "ifeval":
            results = check_all(output, reference["instruction_id_list"], reference["kwargs"])
            passed = sum(results) / len(results)
            return Score(passed, all(results), "ifeval_all_instructions")
        case "f1_and_judge":
            return f1_score(output, reference["golds"], F1_ACCEPTANCE)
        case "judge":
            return None
        case kind:
            raise ValueError(f"unknown label kind: {kind}")


def needs_judge(item: BenchItem) -> bool:
    return item.label_kind in ("judge", "f1_and_judge")


def combined_acceptance(item: BenchItem, automatic: Score | None, judge_score: int | None) -> bool | None:
    """Apply the acceptance criterion of ADR-0005. None while a required judge score is missing."""
    if not needs_judge(item):
        return automatic.accepted if automatic is not None else None
    if judge_score is None:
        return None
    judged = judge_score >= JUDGE_ACCEPTANCE
    if item.label_kind == "f1_and_judge":
        return automatic is not None and automatic.accepted and judged
    return judged
