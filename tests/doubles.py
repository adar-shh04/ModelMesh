"""Test doubles. Kept out of src/ so they can never reach a production path."""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any, ClassVar

import anyio

from modelmesh.core.deadline import Deadline
from modelmesh.core.errors import ProviderError
from modelmesh.providers.catalog import CatalogModel
from modelmesh.providers.types import (
    NormalizedChatRequest,
    NormalizedChatResponse,
    NormalizedChunk,
    ProviderHealth,
    ProviderModelInfo,
    TokenUsage,
)


def reply(text: str = "ok", *, finish: str = "stop", usage: TokenUsage | None = None, **kwargs: Any):
    return NormalizedChatResponse(
        content=text, tool_calls=None, finish_reason=finish, usage=usage or TokenUsage(10, 5), **kwargs
    )


def chunks(*parts: str, usage: TokenUsage | None = None, finish: str = "stop") -> list[NormalizedChunk]:
    items = [NormalizedChunk(content=part) for part in parts]
    items.append(NormalizedChunk(finish_reason=finish))
    if usage is not None:
        items.append(NormalizedChunk(usage=usage))
    return items


class ScriptedAdapter:
    """Replays queued outcomes per model. A queue item is a response, an exception, or a list of stream items."""

    provider_id: ClassVar[str] = "t"

    def __init__(self) -> None:
        self.scripts: dict[str, deque[Any]] = {}
        self.requests: list[NormalizedChatRequest] = []
        self.delay_s = 0.0
        self.stream_closed = False

    def script(self, model: str, *outcomes: Any) -> None:
        self.scripts.setdefault(model, deque()).extend(outcomes)

    def _next(self, req: NormalizedChatRequest) -> Any:
        self.requests.append(req)
        queue = self.scripts.get(req.provider_model)
        if not queue:
            return reply("default")
        return queue.popleft()

    def models_called(self) -> list[str]:
        return [r.provider_model for r in self.requests]

    async def chat(self, req: NormalizedChatRequest, *, deadline: Deadline) -> NormalizedChatResponse:
        outcome = self._next(req)
        if self.delay_s:
            await anyio.sleep(self.delay_s)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def stream(self, req: NormalizedChatRequest, *, deadline: Deadline) -> AsyncIterator[NormalizedChunk]:
        outcome = self._next(req)
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, NormalizedChatResponse):
            outcome = chunks(outcome.content or "", usage=outcome.usage, finish=outcome.finish_reason)
        try:
            for item in outcome:
                if self.delay_s:
                    await anyio.sleep(self.delay_s)
                if isinstance(item, ProviderError):
                    raise item
                yield item
        finally:
            self.stream_closed = True

    async def list_models(self) -> list[ProviderModelInfo]:
        return []

    async def health(self) -> ProviderHealth:
        return ProviderHealth(ok=True)

    async def aclose(self) -> None:
        return None


BASE_MODEL = CatalogModel(
    id="t/base",
    provider="t",
    provider_model="base",
    family="t",
    context_window=100_000,
    max_output_tokens=8_000,
    size_tier="medium",
    data_use="no_training",
    tools=True,
    json_object=True,
    schema_mode="strict",
    schema_stream=True,
    schema_with_tools=True,
    latency_ttft_ms=10,
    latency_tokens_per_s=1000.0,
)


def make_model(name: str, **overrides: Any) -> CatalogModel:
    return replace(BASE_MODEL, id=f"t/{name}", provider_model=name, **overrides)
