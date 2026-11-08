"""Ollama adapter for local models (ADR-0003). The server sets `num_ctx` via `OLLAMA_CONTEXT_LENGTH`."""

from __future__ import annotations

from typing import Any, ClassVar

import httpx

from modelmesh.core.errors import ProviderError
from modelmesh.providers.openai_compat import OpenAICompatAdapter
from modelmesh.providers.types import NormalizedChatRequest, ProviderHealth


class OllamaAdapter(OpenAICompatAdapter):
    provider_id: ClassVar[str] = "ollama"

    def __init__(self, client: httpx.AsyncClient, base_url: str) -> None:
        self._root_url = base_url.rstrip("/")
        super().__init__(client, f"{self._root_url}/v1", None)

    def build_payload(self, req: NormalizedChatRequest, *, stream: bool) -> dict[str, Any]:
        payload = super().build_payload(req, stream=stream)
        # Ollama has no tool_choice. "auto" is its behaviour anyway; other values make the model ineligible.
        payload.pop("tool_choice", None)
        payload.pop("parallel_tool_calls", None)
        return payload

    async def health(self) -> ProviderHealth:
        try:
            response = await self._client.get(f"{self._root_url}/api/ps", timeout=5.0)
        except httpx.HTTPError as exc:
            error: ProviderError = self._transport_error(exc)
            return ProviderHealth(ok=False, detail=str(error.error_class))
        if response.status_code != 200:
            return ProviderHealth(ok=False, detail=f"HTTP {response.status_code}")
        loaded = response.json().get("models", [])
        return ProviderHealth(ok=True, detail=f"{len(loaded)} model(s) loaded")
