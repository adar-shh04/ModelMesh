"""Gemini adapter, using the OpenAI-compatible endpoint of Google AI Studio (ADR-0003)."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar
from zoneinfo import ZoneInfo

from modelmesh.core.errors import ErrorClass, ProviderError
from modelmesh.providers.openai_compat import OpenAICompatAdapter, error_text

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
_PACIFIC = ZoneInfo("America/Los_Angeles")


def next_pacific_midnight(now: datetime | None = None) -> datetime:
    """Gemini's daily quotas reset at midnight Pacific time."""
    current = (now or datetime.now(UTC)).astimezone(_PACIFIC)
    tomorrow = (current + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return tomorrow.astimezone(UTC)


class GeminiAdapter(OpenAICompatAdapter):
    provider_id: ClassVar[str] = "gemini"

    def classify_status(self, status: int, body: Any, headers: Mapping[str, str]) -> ProviderError:
        result = super().classify_status(status, body, headers)
        if status == 429:
            text = error_text(body)
            if "quota_exceeded" in text or "perday" in text or "per day" in text or "daily" in text:
                result.error_class = ErrorClass.QUOTA_EXHAUSTED
                result.quota_reset_at = next_pacific_midnight()
        return result
