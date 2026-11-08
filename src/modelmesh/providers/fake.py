"""Deterministic provider for CI and load tests. Registered only with MODELMESH_ENABLE_FAKE_PROVIDER=true.

Settings refuses that flag when MODELMESH_ENV=production, so this module can never serve production traffic.
"""

from __future__ import annotations

import json
import random
from collections.abc import AsyncIterator
from typing import Any, ClassVar

import anyio

from modelmesh.core.deadline import Deadline
from modelmesh.core.errors import ErrorClass, ProviderError
from modelmesh.providers.types import (
    NormalizedChatRequest,
    NormalizedChatResponse,
    NormalizedChunk,
    ProviderHealth,
    ProviderModelInfo,
    TokenUsage,
)


def instance_for_schema(schema: dict[str, Any]) -> Any:
    """Build a minimal instance satisfying a simple JSON Schema."""
    if "const" in schema:
        return schema["const"]
    if schema.get("enum"):
        return schema["enum"][0]
    kind = schema.get("type", "object")
    if isinstance(kind, list):
        kind = next((item for item in kind if item != "null"), "null")
    if kind == "object":
        properties = schema.get("properties", {})
        return {key: instance_for_schema(properties[key]) for key in schema.get("required", list(properties))}
    if kind == "array":
        return [instance_for_schema(schema.get("items", {})) for _ in range(int(schema.get("minItems", 0)))]
    return {"string": "fake", "integer": 0, "number": 0.0, "boolean": True, "null": None}.get(kind)


def _text_of(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(part.get("text", "")) for part in content)
    return ""


class FakeAdapter:
    provider_id: ClassVar[str] = "fake"

    def __init__(self, *, seed: int = 7, delay_s: float = 0.0, flaky_rate: float = 0.3) -> None:
        self._rng = random.Random(seed)  # noqa: S311 - not security sensitive
        self._delay_s = delay_s
        self._flaky_rate = flaky_rate

    def _maybe_fail(self, req: NormalizedChatRequest) -> None:
        if "unavailable" in req.provider_model:
            raise ProviderError(ErrorClass.UPSTREAM_UNAVAILABLE, "fake unavailable", http_status=503)
        if "flaky" in req.provider_model and self._rng.random() < self._flaky_rate:
            raise ProviderError(ErrorClass.UPSTREAM_UNAVAILABLE, "fake flaky failure", http_status=503)

    def _answer(self, req: NormalizedChatRequest) -> str:
        fmt = req.response_format
        if fmt and fmt.get("type") == "json_schema":
            return json.dumps(instance_for_schema(fmt["json_schema"]["schema"]))
        if fmt and fmt.get("type") == "json_object":
            return json.dumps({"answer": "fake"})
        if req.structured_mode == "emulated":
            return json.dumps({"answer": "fake"})
        last_user = next((m for m in reversed(req.messages) if m.get("role") == "user"), None)
        return "echo: " + (_text_of(last_user) if last_user else "")

    @staticmethod
    def _usage(req: NormalizedChatRequest, answer: str) -> TokenUsage:
        prompt = sum(len(_text_of(m)) for m in req.messages) // 4 + 1
        return TokenUsage(prompt_tokens=prompt, completion_tokens=len(answer) // 4 + 1)

    async def chat(self, req: NormalizedChatRequest, *, deadline: Deadline) -> NormalizedChatResponse:
        self._maybe_fail(req)
        if self._delay_s:
            await anyio.sleep(self._delay_s)
        answer = self._answer(req)
        return NormalizedChatResponse(
            content=answer, tool_calls=None, finish_reason="stop", usage=self._usage(req, answer)
        )

    async def stream(self, req: NormalizedChatRequest, *, deadline: Deadline) -> AsyncIterator[NormalizedChunk]:
        self._maybe_fail(req)
        answer = self._answer(req)
        words = answer.split(" ")
        for position, word in enumerate(words):
            if self._delay_s:
                await anyio.sleep(self._delay_s)
            yield NormalizedChunk(content=word if position == 0 else f" {word}")
        yield NormalizedChunk(finish_reason="stop")
        yield NormalizedChunk(usage=self._usage(req, answer))

    async def list_models(self) -> list[ProviderModelInfo]:
        return [ProviderModelInfo(id=name) for name in FAKE_MODEL_NAMES]

    async def health(self) -> ProviderHealth:
        return ProviderHealth(ok=True)

    async def aclose(self) -> None:
        return None


FAKE_MODEL_NAMES = ("echo-small", "echo-medium", "echo-large", "flaky", "unavailable")
