"""Groq adapter (ADR-0003)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, ClassVar

from modelmesh.core.errors import ErrorClass, ProviderError
from modelmesh.providers.openai_compat import OpenAICompatAdapter, error_text
from modelmesh.providers.types import NormalizedChatRequest, RateLimitObservation, TokenUsage

BASE_URL = "https://api.groq.com/openai/v1"
_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_UNIT_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def parse_duration(value: str | None) -> float | None:
    """Parse Groq's reset headers, e.g. `2m59.56s` or `6s`."""
    if not value:
        return None
    parts = _DURATION_PART.findall(value)
    if not parts:
        return None
    return sum(float(amount) * _UNIT_SECONDS[unit] for amount, unit in parts)


def _as_int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


class GroqAdapter(OpenAICompatAdapter):
    provider_id: ClassVar[str] = "groq"

    def build_messages(self, req: NormalizedChatRequest) -> list[dict[str, Any]]:
        # Groq rejects messages[].name with HTTP 400.
        return [{k: v for k, v in message.items() if k != "name"} for message in req.messages]

    def build_payload(self, req: NormalizedChatRequest, *, stream: bool) -> dict[str, Any]:
        payload = super().build_payload(req, stream=stream)
        # Groq accepts only low, medium and high, and rejects anything else with HTTP 400.
        if payload.get("reasoning_effort") in ("none", "minimal"):
            payload["reasoning_effort"] = "low"
        return payload

    def extract_usage(self, obj: Mapping[str, Any]) -> TokenUsage | None:
        usage = super().extract_usage(obj)
        if usage is not None:
            return usage
        extension = obj.get("x_groq")
        if isinstance(extension, Mapping) and isinstance(extension.get("usage"), Mapping):
            return self._usage_from(extension["usage"])
        return None

    def parse_rate_limit(self, headers: Mapping[str, str]) -> RateLimitObservation | None:
        # Documented semantics: the "requests" headers describe RPD, the "tokens" headers describe TPM.
        requests = _as_int(headers.get("x-ratelimit-remaining-requests"))
        tokens = _as_int(headers.get("x-ratelimit-remaining-tokens"))
        if requests is None and tokens is None:
            return None
        return RateLimitObservation(
            remaining_requests=requests,
            remaining_tokens=tokens,
            reset_requests_s=parse_duration(headers.get("x-ratelimit-reset-requests")),
            reset_tokens_s=parse_duration(headers.get("x-ratelimit-reset-tokens")),
        )

    def classify_status(self, status: int, body: Any, headers: Mapping[str, str]) -> ProviderError:
        if status == 498:  # Flex tier at capacity.
            return ProviderError(ErrorClass.RATE_LIMITED, "groq at capacity (498)", http_status=498)
        if status == 400 and "json_validate_failed" in error_text(body):
            # Groq rejects its own output when it does not match the schema; this is a model failure, not ours.
            return ProviderError(ErrorClass.SCHEMA_VALIDATION_FAILED, "groq output failed its schema", http_status=400)
        return super().classify_status(status, body, headers)
