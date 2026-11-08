"""Turn a parsed contract request into the normalised form the rest of the pipeline uses."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from modelmesh.contract.schemas import ChatCompletionRequest, ChatMessage, TextContentPart
from modelmesh.core.errors import ApiError
from modelmesh.policy.engine import CapabilityNeeds
from modelmesh.reliability.structured import is_portable_schema, is_valid_schema, schema_property_count
from modelmesh.routing.solver import KNOWN_TASK_TYPES


def validation_error(exc: ValidationError) -> ApiError:
    """Map a Pydantic failure to the contract's 400, pointing at the offending parameter."""
    first = exc.errors(include_url=False, include_input=False, include_context=False)[0]
    path = ".".join(str(part) for part in first["loc"] if not _is_union_tag(part))
    if first["type"] == "extra_forbidden":
        return ApiError("invalid_request", f"Unknown field '{path}'.", code="unknown_parameter", param=path)
    return ApiError("invalid_request", f"Invalid value for '{path}'.", code="invalid_value", param=path or None)


def _is_union_tag(part: str | int) -> bool:
    return isinstance(part, str) and (
        part in {"str", "system", "developer", "user", "assistant", "tool", "text", "json_object", "json_schema"}
        or "[" in part
        or "function-" in part
    )


def parse_chat_request(raw: bytes) -> ChatCompletionRequest:
    try:
        decoded = json.loads(raw)
    except ValueError:
        raise ApiError("invalid_request", "The body is not valid JSON.", code="invalid_value") from None
    if isinstance(decoded, dict):
        _reject_non_text_parts(decoded.get("messages"))
    try:
        return ChatCompletionRequest.model_validate(decoded)
    except ValidationError as exc:
        raise validation_error(exc) from None


def _reject_non_text_parts(messages: Any) -> None:
    if not isinstance(messages, list):
        return
    for position, message in enumerate(messages):
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list) and any(isinstance(p, dict) and p.get("type") != "text" for p in content):
            raise ApiError(
                "invalid_request",
                "Only text content parts are supported in v1.",
                code="unsupported_content_part",
                param=f"messages.{position}.content",
            )


def _flatten_content(content: str | list[TextContentPart] | None) -> str | None:
    if content is None or isinstance(content, str):
        return content
    return "".join(part.text for part in content)


def normalize_message(message: ChatMessage) -> dict[str, Any]:
    result: dict[str, Any] = {
        "role": "system" if message.role == "developer" else message.role,
        "content": _flatten_content(message.content),
    }
    name = getattr(message, "name", None)
    if name:
        result["name"] = name
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        result["tool_calls"] = [call.model_dump() for call in tool_calls]
    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        result["tool_call_id"] = tool_call_id
    return result


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass
class PreparedRequest:
    body: ChatCompletionRequest
    messages: list[dict[str, Any]]
    metadata: dict[str, str]
    task_type: str
    max_tokens: int | None
    stop: list[str] | None
    response_format: dict[str, Any] | None
    tools: list[dict[str, Any]] | None
    tool_choice: str | dict[str, Any] | None
    needs: CapabilityNeeds
    schema_properties: int
    warnings: list[str] = field(default_factory=list)

    @property
    def final_user_text(self) -> str:
        return str(self.messages[-1].get("content") or "")


def prepare(body: ChatCompletionRequest) -> PreparedRequest:
    if (
        body.max_tokens is not None
        and body.max_completion_tokens is not None
        and body.max_tokens != body.max_completion_tokens
    ):
        raise ApiError(
            "invalid_request",
            "max_tokens and max_completion_tokens must be equal when both are sent.",
            code="max_tokens_mismatch",
            param="max_completion_tokens",
        )
    if body.cache is not None and body.cache.mode == "semantic" and body.cache.context_fingerprint is None:
        raise ApiError(
            "invalid_request",
            "Semantic caching requires cache.context_fingerprint.",
            code="missing_context_fingerprint",
            param="cache.context_fingerprint",
        )

    response_format: dict[str, Any] | None = None
    kind: str | None = None
    portable = False
    properties = 0
    if body.response_format is not None and body.response_format.type != "text":
        kind = body.response_format.type
        response_format = body.response_format.model_dump(by_alias=True, exclude_none=True)
        if kind == "json_schema":
            schema = response_format["json_schema"]["schema"]
            if not is_valid_schema(schema):
                raise ApiError(
                    "invalid_request",
                    "response_format.json_schema.schema is not a valid JSON Schema.",
                    code="invalid_value",
                    param="response_format.json_schema.schema",
                )
            portable = is_portable_schema(schema)
            properties = schema_property_count(schema)

    warnings: list[str] = []
    metadata = body.metadata.as_map() if body.metadata is not None else {}
    requested_task = metadata.get("task_type")
    task_type = requested_task or "general"
    if task_type not in KNOWN_TASK_TYPES:
        warnings.append(f"unknown_task_type:{task_type}")
        task_type = "general"

    tools = [tool.model_dump(exclude_none=True) for tool in body.tools] if body.tools else None
    tool_choice: str | dict[str, Any] | None = None
    if body.tool_choice is not None:
        tool_choice = body.tool_choice if isinstance(body.tool_choice, str) else body.tool_choice.model_dump()
    stop = [body.stop] if isinstance(body.stop, str) else body.stop
    required = frozenset(body.routing.required_capabilities or []) if body.routing else frozenset()
    needs = CapabilityNeeds(
        response_format_kind=kind,
        schema_portable=portable,
        has_tools=bool(tools),
        tool_choice_forced=bool(tools) and tool_choice not in (None, "auto", "none"),
        stream=body.stream,
        required=required,
    )
    return PreparedRequest(
        body=body,
        messages=[normalize_message(message) for message in body.messages],
        metadata=metadata,
        task_type=task_type,
        max_tokens=body.max_tokens or body.max_completion_tokens,
        stop=stop,
        response_format=response_format,
        tools=tools,
        tool_choice=tool_choice,
        needs=needs,
        schema_properties=properties,
        warnings=warnings,
    )
