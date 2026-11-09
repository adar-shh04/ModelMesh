"""Automatic quality metrics. Each returns a graded score in [0, 1] and an acceptance verdict (ADR-0005)."""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_FINAL_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True)
class Score:
    value: float
    accepted: bool
    metric: str


def normalize_answer(text: str) -> str:
    """SQuAD-style answer normalisation."""
    lowered = text.lower()
    without_punctuation = "".join(ch for ch in lowered if ch not in set(string.punctuation))
    return " ".join(_ARTICLES.sub(" ", without_punctuation).split())


def token_f1(prediction: str, gold: str) -> float:
    predicted = normalize_answer(prediction).split()
    reference = normalize_answer(gold).split()
    if not predicted or not reference:
        return float(predicted == reference)
    common = Counter(predicted) & Counter(reference)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(reference)
    return 2 * precision * recall / (precision + recall)


def best_token_f1(prediction: str, golds: list[str]) -> float:
    return max((token_f1(prediction, gold) for gold in golds), default=0.0)


def exact_label(prediction: str, gold: str) -> Score:
    matched = normalize_answer(prediction) == normalize_answer(gold)
    return Score(float(matched), matched, "exact_label")


def json_field(prediction: str, field: str) -> str | None:
    try:
        decoded = json.loads(prediction)
    except ValueError:
        return None
    value = decoded.get(field) if isinstance(decoded, dict) else None
    return str(value) if value is not None else None


def classification_score(prediction: str, gold: str) -> Score:
    label = json_field(prediction, "label")
    return exact_label(label if label is not None else prediction, gold)


def gsm8k_gold(answer: str) -> str:
    return answer.rsplit("####", 1)[-1].strip().replace(",", "")


def last_number(text: str) -> str | None:
    found = _FINAL_NUMBER.findall(text)
    return found[-1].replace(",", "").rstrip(".") if found else None


def numeric_answer_score(prediction: str, gold: str) -> Score:
    predicted = last_number(prediction)
    matched = predicted is not None and float(predicted) == float(gold)
    return Score(float(matched), matched, "numeric_exact_match")


def multiple_choice_score(prediction: str, gold_letter: str) -> Score:
    match = re.search(r"answer\s*(?:is)?\s*[:\-]?\s*\(?([A-J])\)?\b", prediction, re.IGNORECASE)
    letter = match.group(1).upper() if match else (prediction.strip()[:1].upper() or None)
    matched = letter == gold_letter
    return Score(float(matched), matched, "choice_accuracy")


def schema_valid_score(prediction: str, schema: dict[str, Any]) -> Score:
    try:
        instance = json.loads(prediction)
    except ValueError:
        return Score(0.0, False, "schema_valid")
    valid = Draft202012Validator(schema).is_valid(instance)
    return Score(float(valid), valid, "schema_valid")


def label_with_schema_score(prediction: str, gold: str, field: str = "label") -> Score:
    """Citation-verification criterion: valid JSON with the correct support label."""
    label = json_field(prediction, field)
    matched = label is not None and label.strip().lower() == gold.strip().lower()
    return Score(float(matched), matched, "support_label")


def f1_score(prediction: str, golds: list[str], threshold: float = 0.5) -> Score:
    value = best_token_f1(prediction, golds)
    return Score(value, value >= threshold, "token_f1")


def cohen_kappa(left: list[bool], right: list[bool]) -> float | None:
    """Cohen's kappa for two binary raters. None when chance agreement is total."""
    if not left or len(left) != len(right):
        return None
    total = len(left)
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / total
    left_yes = sum(left) / total
    right_yes = sum(right) / total
    expected = left_yes * right_yes + (1 - left_yes) * (1 - right_yes)
    if expected == 1.0:
        return None
    return (observed - expected) / (1 - expected)
