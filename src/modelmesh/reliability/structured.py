"""Validation of structured outputs against what the caller asked for (ADR-0007)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

_ALLOWED_PORTABLE_KEYWORDS = {
    "type",
    "properties",
    "required",
    "items",
    "enum",
    "const",
    "description",
    "minItems",
    "maxItems",
    "additionalProperties",
}
_MAX_PORTABLE_DEPTH = 5
_MAX_REPORTED_ERRORS = 10


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    keyword: str


def is_valid_schema(schema: dict[str, Any]) -> bool:
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError:
        return False
    return True


def is_portable_schema(schema: dict[str, Any], *, depth: int = 1) -> bool:
    """Whether a schema is inside the portable subset of contract section 2.5."""
    if depth > _MAX_PORTABLE_DEPTH or set(schema) - _ALLOWED_PORTABLE_KEYWORDS:
        return False
    if depth == 1 and schema.get("type") != "object":
        return False
    kind = schema.get("type")
    if kind == "object" or (isinstance(kind, list) and "object" in kind):
        properties = schema.get("properties", {})
        if set(schema.get("required", [])) != set(properties) or schema.get("additionalProperties") is not False:
            return False
        return all(is_portable_schema(child, depth=depth + 1) for child in properties.values())
    if kind == "array" or (isinstance(kind, list) and "array" in kind):
        items = schema.get("items")
        return isinstance(items, dict) and is_portable_schema(items, depth=depth + 1)
    return True


def schema_property_count(schema: dict[str, Any]) -> int:
    count = 0
    for child in schema.get("properties", {}).values():
        count += 1 + (schema_property_count(child) if isinstance(child, dict) else 0)
    items = schema.get("items")
    if isinstance(items, dict):
        count += schema_property_count(items)
    return count


def validate_output(response_format: dict[str, Any], content: str | None) -> list[ValidationIssue]:
    """Return the problems with `content`. Issues carry a path and keyword only, never values."""
    kind = response_format.get("type")
    if kind not in ("json_object", "json_schema"):
        return []
    try:
        instance = json.loads(content or "")
    except ValueError:
        return [ValidationIssue("", "json")]
    if kind == "json_object":
        return [] if isinstance(instance, dict) else [ValidationIssue("", "type")]
    validator = Draft202012Validator(response_format["json_schema"]["schema"])
    issues = []
    for error in validator.iter_errors(instance):
        pointer = "".join(f"/{part}" for part in error.absolute_path)
        issues.append(ValidationIssue(pointer, str(error.validator)))
        if len(issues) >= _MAX_REPORTED_ERRORS:
            break
    return issues


def emulation_instruction(schema: dict[str, Any]) -> str:
    return (
        "Respond with only a single JSON value and no surrounding text. It must validate against this JSON Schema:\n"
        + json.dumps(schema)
    )


def repair_instruction(issues: list[ValidationIssue]) -> str:
    described = "; ".join(f"{issue.path or '/'} failed '{issue.keyword}'" for issue in issues)
    return (
        f"The previous reply was not valid JSON for the required schema ({described}). "
        "Reply again with only valid JSON."
    )
