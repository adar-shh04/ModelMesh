"""Programmatic checks for IFEval instructions, following google-research/instruction_following_eval.

The language-detection instructions need an extra dependency and are unsupported: items using them are excluded
at load time and counted, so coverage is visible rather than silently reduced.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

Kwargs = dict[str, Any]
Checker = Callable[[str, Kwargs], bool]


def _compare(count: int, relation: str | None, target: int) -> bool:
    if relation == "less than":
        return count < target
    if relation == "at least":
        return count >= target
    return count == target


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text)


def _sentences(text: str) -> list[str]:
    return [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part]


def _number_words(text: str, kw: Kwargs) -> bool:
    return _compare(len(_words(text)), kw.get("relation"), kw["num_words"])


def _number_sentences(text: str, kw: Kwargs) -> bool:
    return _compare(len(_sentences(text)), kw.get("relation"), kw["num_sentences"])


def _number_paragraphs(text: str, kw: Kwargs) -> bool:
    paragraphs = re.split(r"\s?\*\*\*\s?", text)
    if any(not part.strip() for part in paragraphs):
        return False
    return bool(len(paragraphs) == kw["num_paragraphs"])


def _existence(text: str, kw: Kwargs) -> bool:
    return all(re.search(keyword, text, re.IGNORECASE) for keyword in kw["keywords"])


def _forbidden(text: str, kw: Kwargs) -> bool:
    return not any(re.search(r"\b" + word + r"\b", text, re.IGNORECASE) for word in kw["forbidden_words"])


def _frequency(text: str, kw: Kwargs) -> bool:
    return _compare(len(re.findall(kw["keyword"], text, re.IGNORECASE)), kw.get("relation"), kw["frequency"])


def _letter_frequency(text: str, kw: Kwargs) -> bool:
    return _compare(text.lower().count(kw["letter"].lower()), kw.get("let_relation"), kw["let_frequency"])


def _json_format(text: str, kw: Kwargs) -> bool:
    candidate = text.strip().removeprefix("```json").removeprefix("```JSON").removeprefix("```").removesuffix("```")
    try:
        json.loads(candidate.strip())
    except ValueError:
        return False
    return True


def _bullets(text: str, kw: Kwargs) -> bool:
    count = len(re.findall(r"^\s*\*[^\*].*$", text, re.MULTILINE)) + len(re.findall(r"^\s*-.*$", text, re.MULTILINE))
    return bool(count == kw["num_bullets"])


def _title(text: str, kw: Kwargs) -> bool:
    return any(title.strip("<>").strip() for title in re.findall(r"<<[^\n]+>>", text))


def _sections(text: str, kw: Kwargs) -> bool:
    pattern = r"\s?" + kw["section_spliter"] + r"\s?\d+\s?"
    return bool(len(re.split(pattern, text)) - 1 >= kw["num_sections"])


def _highlights(text: str, kw: Kwargs) -> bool:
    count = sum(1 for match in re.findall(r"\*[^\n\*]*\*", text) if match.strip("*").strip())
    count += sum(1 for match in re.findall(r"\*\*[^\n\*]*\*\*", text) if match.strip("*").strip())
    return bool(count >= kw["num_highlights"])


def _constrained(text: str, kw: Kwargs) -> bool:
    return any(option in text for option in ("My answer is yes.", "My answer is no.", "My answer is maybe."))


def _capital_frequency(text: str, kw: Kwargs) -> bool:
    capitals = [word for word in re.findall(r"\b[\w']+\b", text) if word.isupper()]
    return _compare(len(capitals), kw.get("capital_relation"), kw["capital_frequency"])


def _end_checker(text: str, kw: Kwargs) -> bool:
    return text.strip().strip('"').lower().endswith(kw["end_phrase"].strip().lower())


def _quotation(text: str, kw: Kwargs) -> bool:
    stripped = text.strip()
    return len(stripped) > 1 and stripped[0] == '"' and stripped[-1] == '"'


def _two_responses(text: str, kw: Kwargs) -> bool:
    parts = [part for part in text.split("******") if part.strip()]
    return len(parts) == 2 and parts[0].strip() != parts[1].strip()


def _repeat_prompt(text: str, kw: Kwargs) -> bool:
    return text.strip().lower().startswith(kw["prompt_to_repeat"].strip().lower())


def _placeholders(text: str, kw: Kwargs) -> bool:
    return bool(len(re.findall(r"\[.*?\]", text)) >= kw["num_placeholders"])


def _postscript(text: str, kw: Kwargs) -> bool:
    marker = kw["postscript_marker"].strip().lower()
    pattern = (
        r"\s*p\.\s?p\.\s?s.*$"
        if marker == "p.p.s"
        else r"\s*p\.\s?s\..*$"
        if marker == "p.s."
        else r"\s*" + re.escape(marker) + r".*$"
    )
    return bool(re.findall(pattern, text.lower(), re.MULTILINE))


CHECKERS: dict[str, Checker] = {
    "punctuation:no_comma": lambda text, kw: "," not in text,
    "length_constraints:number_words": _number_words,
    "length_constraints:number_sentences": _number_sentences,
    "length_constraints:number_paragraphs": _number_paragraphs,
    "keywords:existence": _existence,
    "keywords:forbidden_words": _forbidden,
    "keywords:frequency": _frequency,
    "keywords:letter_frequency": _letter_frequency,
    "detectable_format:json_format": _json_format,
    "detectable_format:number_bullet_lists": _bullets,
    "detectable_format:title": _title,
    "detectable_format:multiple_sections": _sections,
    "detectable_format:number_highlighted_sections": _highlights,
    "detectable_format:constrained_response": _constrained,
    "change_case:english_lowercase": lambda text, kw: text == text.lower() and bool(text.strip()),
    "change_case:english_capital": lambda text, kw: text == text.upper() and bool(text.strip()),
    "change_case:capital_word_frequency": _capital_frequency,
    "startend:end_checker": _end_checker,
    "startend:quotation": _quotation,
    "combination:two_responses": _two_responses,
    "combination:repeat_prompt": _repeat_prompt,
    "detectable_content:number_placeholders": _placeholders,
    "detectable_content:postscript": _postscript,
}


def is_supported(instruction_ids: list[str]) -> bool:
    return all(instruction in CHECKERS for instruction in instruction_ids)


def check_all(response: str, instruction_ids: list[str], kwargs: list[Kwargs]) -> list[bool]:
    return [
        bool(response.strip()) and CHECKERS[instruction](response, {k: v for k, v in kw.items() if v is not None})
        for instruction, kw in zip(instruction_ids, kwargs, strict=True)
    ]
