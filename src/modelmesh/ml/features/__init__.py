"""Request features for the learned router (ADR-0004), shared by online routing and offline training.

Pure Python on purpose: this runs on the request path, so it may not pull in numpy-heavy or offline code.
The embedding itself is computed by the caller (fastembed bge-small, ADR-0006); this module only decides
which text is embedded and how many tokens of it.

Feature version 2 caps the embedded text at ``EMBEDDING_MAX_TOKENS`` tokens (version 1 embedded up to the
model's 512-token limit, which made the router miss its 25 ms latency budget on long prompts). The cap is
split between the system prompt and the final user message by ``embedding_token_budget`` so that a long
system prompt cannot crowd the user's message out of the embedding.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

FEATURE_VERSION = "2"
CHARS_PER_TOKEN = 0.25
# Embedded text: at most this many tokens, counting the model's [CLS] and [SEP].
EMBEDDING_MAX_TOKENS = 128
EMBEDDING_SPECIAL_TOKENS = 2
# The system prompt keeps at least this many tokens when the user message would otherwise fill the budget.
EMBEDDING_SYSTEM_TOKENS = 32
# Each segment is cut to this many characters before tokenising, which bounds the tokeniser's work. At 16
# characters per token it never binds on natural text; when it does, it binds identically online and offline.
EMBEDDING_SEGMENT_MAX_CHARS = 16 * EMBEDDING_MAX_TOKENS

TASK_TYPES = (
    "query_classification",
    "query_rewriting",
    "research_planning",
    "evidence_assessment",
    "evidence_synthesis",
    "citation_verification",
    "final_response",
    "evaluation_judge",
    "general",
)
RESPONSE_KINDS = ("text", "json_object", "json_schema")

_MATH = re.compile(r"[=+\-*/^<>%]")
_FENCE = re.compile(r"```")

FEATURE_NAMES: tuple[str, ...] = (
    *(f"task={t}" for t in TASK_TYPES),
    "log_input_tokens",
    "log_system_tokens",
    "log_context_tokens",
    "message_count",
    "user_turns",
    "log_max_tokens",
    *(f"response={k}" for k in RESPONSE_KINDS),
    "schema_properties",
    "schema_depth",
    "schema_required",
    "schema_has_enum",
    "tool_count",
    "code_fence_count",
    "digit_fraction",
    "math_fraction",
    "question_count",
    "non_ascii_fraction",
)


def _text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(part.get("text", "")) for part in content if isinstance(part, Mapping))
    return ""


def _tokens(text: str) -> float:
    return math.log1p(len(text) * CHARS_PER_TOKEN)


def schema_stats(schema: Any, depth: int = 1) -> tuple[int, int, int, bool]:
    """(property count, maximum depth, required count, enum present) of a JSON schema, counted recursively."""
    if not isinstance(schema, Mapping):
        return 0, 0, 0, False
    properties = schema.get("properties")
    count = 0
    max_depth = depth if isinstance(properties, Mapping) and properties else 0
    required = len(schema.get("required") or [])
    has_enum = "enum" in schema
    children: list[Any] = []
    if isinstance(properties, Mapping):
        count += len(properties)
        children.extend(properties.values())
    for key in ("items", "additionalProperties"):
        if isinstance(schema.get(key), Mapping):
            children.append(schema[key])
    for key in ("anyOf", "oneOf", "allOf"):
        children.extend(value for value in schema.get(key) or [] if isinstance(value, Mapping))
    for child in children:
        c_count, c_depth, c_required, c_enum = schema_stats(child, depth + 1)
        count += c_count
        max_depth = max(max_depth, c_depth)
        required += c_required
        has_enum = has_enum or c_enum
    return count, max_depth, required, has_enum


def response_kind(response_format: Mapping[str, Any] | None) -> str:
    kind = (response_format or {}).get("type", "text")
    return kind if kind in RESPONSE_KINDS else "text"


def request_features(
    messages: Sequence[Mapping[str, Any]],
    *,
    task_type: str,
    response_format: Mapping[str, Any] | None = None,
    max_tokens: int | None = None,
    tool_count: int = 0,
) -> dict[str, float]:
    """The ADR-0004 request features. Application, tags and end-user id are deliberately not inputs."""
    system = "\n".join(_text(m) for m in messages if m.get("role") in ("system", "developer"))
    final_user = next((_text(m) for m in reversed(messages) if m.get("role") == "user"), "")
    context = "\n".join(_text(m) for m in messages[:-1] if m.get("role") not in ("system", "developer"))
    everything = "\n".join(_text(m) for m in messages)
    schema: Any = None
    if response_format and response_format.get("type") == "json_schema":
        schema = (response_format.get("json_schema") or {}).get("schema")
    properties, depth, required, has_enum = schema_stats(schema)
    length = max(len(final_user), 1)
    features: dict[str, float] = {f"task={t}": float(t == task_type) for t in TASK_TYPES}
    kind = response_kind(response_format)
    features.update(
        {
            "log_input_tokens": _tokens(everything),
            "log_system_tokens": _tokens(system),
            "log_context_tokens": _tokens(context),
            "message_count": float(len(messages)),
            "user_turns": float(sum(1 for m in messages if m.get("role") == "user")),
            "log_max_tokens": math.log1p(max_tokens or 0),
            **{f"response={k}": float(k == kind) for k in RESPONSE_KINDS},
            "schema_properties": float(properties),
            "schema_depth": float(depth),
            "schema_required": float(required),
            "schema_has_enum": float(has_enum),
            "tool_count": float(tool_count),
            "code_fence_count": float(len(_FENCE.findall(final_user)) // 2),
            "digit_fraction": sum(ch.isdigit() for ch in final_user) / length,
            "math_fraction": len(_MATH.findall(final_user)) / length,
            "question_count": float(final_user.count("?")),
            "non_ascii_fraction": sum(ord(ch) > 127 for ch in final_user) / length,
        }
    )
    return features


def feature_vector(features: Mapping[str, float]) -> list[float]:
    return [float(features.get(name, 0.0)) for name in FEATURE_NAMES]


def embedding_segments(
    messages: Sequence[Mapping[str, Any]], *, max_chars: int | None = EMBEDDING_SEGMENT_MAX_CHARS
) -> tuple[str, str]:
    """(system prompt, final user message): the two texts the router embeds (ADR-0004), each pre-cut."""
    system = "\n".join(_text(m) for m in messages if m.get("role") in ("system", "developer")).strip()
    final_user = next((_text(m) for m in reversed(messages) if m.get("role") == "user"), "").strip()
    if max_chars is None:
        return system, final_user
    return system[:max_chars], final_user[:max_chars]


def embedding_token_budget(system_tokens: int, user_tokens: int) -> tuple[int, int]:
    """How many leading tokens of the system prompt and of the final user message are embedded.

    The user message gets everything the system prompt does not need, and the system prompt keeps at least
    ``EMBEDDING_SYSTEM_TOKENS`` (or all of itself, if shorter) when the user message is long.
    """
    content = EMBEDDING_MAX_TOKENS - EMBEDDING_SPECIAL_TOKENS
    system_take = min(system_tokens, max(EMBEDDING_SYSTEM_TOKENS, content - user_tokens))
    return system_take, min(user_tokens, content - system_take)


def join_embedding_text(system: str, user: str) -> str:
    return f"{system}\n{user}".strip()


def embedding_text(messages: Sequence[Mapping[str, Any]]) -> str:
    """System prompt plus final user message, before the token cap. The embedder applies the cap."""
    return join_embedding_text(*embedding_segments(messages))
