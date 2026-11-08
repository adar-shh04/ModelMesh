"""OpenRouter adapter (ADR-0003)."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar

import httpx
from pydantic import SecretStr

from modelmesh.core.errors import ErrorClass, ProviderError
from modelmesh.providers.openai_compat import OpenAICompatAdapter, error_text
from modelmesh.providers.types import NormalizedChatRequest

BASE_URL = "https://openrouter.ai/api/v1"


def next_utc_midnight(now: datetime | None = None) -> datetime:
    current = now or datetime.now(UTC)
    return (current + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)


class OpenRouterAdapter(OpenAICompatAdapter):
    provider_id: ClassVar[str] = "openrouter"

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        api_key: SecretStr | None,
        *,
        app_url: str | None = None,
        app_title: str | None = None,
    ) -> None:
        super().__init__(client, base_url, api_key)
        self._app_url = app_url
        self._app_title = app_title

    def build_headers(self) -> dict[str, str]:
        headers = super().build_headers()
        if self._app_url:
            headers["HTTP-Referer"] = self._app_url
        if self._app_title:
            headers["X-OpenRouter-Title"] = self._app_title
        return headers

    def build_payload(self, req: NormalizedChatRequest, *, stream: bool) -> dict[str, Any]:
        payload = super().build_payload(req, stream=stream)
        # OpenRouter reports usage on every stream and has deprecated this option.
        payload.pop("stream_options", None)
        routing: dict[str, Any] = {}
        # Without this, OpenRouter may silently ignore parameters an upstream does not support.
        if (
            req.response_format is not None
            or req.tools is not None
            or req.tool_choice is not None
            or req.stop
        ):
            routing["require_parameters"] = True
        if req.require_no_training:
            routing["data_collection"] = "deny"
        if routing:
            payload["provider"] = routing
        return payload

    def extract_cost(self, obj: Mapping[str, Any]) -> float | None:
        usage = obj.get("usage")
        if isinstance(usage, Mapping):
            cost = usage.get("cost")
            if isinstance(cost, int | float) and not isinstance(cost, bool):
                return float(cost)
        return None

    def classify_status(self, status: int, body: Any, headers: Mapping[str, str]) -> ProviderError:
        result = super().classify_status(status, body, headers)
        text = error_text(body)
        if status == 429 and ("free-models-per-day" in text or "per-day" in text):
            result.error_class = ErrorClass.QUOTA_EXHAUSTED
            result.quota_reset_at = next_utc_midnight()
        elif status == 403:
            # OpenRouter uses 403 for guardrail and moderation blocks, not for bad credentials (401).
            result.error_class = ErrorClass.CONTENT_FILTERED
        return result
