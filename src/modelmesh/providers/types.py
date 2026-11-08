"""Provider-neutral request and response types used on either side of an adapter."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol

from modelmesh.core.deadline import Deadline

StructuredMode = Literal["native_strict", "native_best_effort", "emulated"]
UsageSource = Literal["provider_reported", "estimated"]
FinishReason = Literal["stop", "length", "tool_calls", "content_filter"]


@dataclass
class NormalizedChatRequest:
    provider_model: str
    messages: list[dict[str, Any]]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    reasoning_effort: str | None = None
    response_format: dict[str, Any] | None = None
    structured_mode: StructuredMode | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    require_no_training: bool = False


@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    source: UsageSource = "provider_reported"


@dataclass
class RateLimitObservation:
    """What a response's headers said about remaining upstream quota."""

    remaining_requests: int | None = None
    remaining_tokens: int | None = None
    reset_requests_s: float | None = None
    reset_tokens_s: float | None = None


@dataclass
class NormalizedChatResponse:
    content: str | None
    tool_calls: list[dict[str, Any]] | None
    finish_reason: FinishReason
    usage: TokenUsage | None
    provider_reported_cost_usd: float | None = None
    provider_response_id: str | None = None
    http_status: int = 200
    rate_limit: RateLimitObservation | None = None


@dataclass
class ToolCallDelta:
    index: int
    id: str | None = None
    name: str | None = None
    arguments: str | None = None


@dataclass
class NormalizedChunk:
    content: str | None = None
    tool_calls: list[ToolCallDelta] | None = None
    finish_reason: FinishReason | None = None
    usage: TokenUsage | None = None
    provider_reported_cost_usd: float | None = None
    rate_limit: RateLimitObservation | None = None


@dataclass
class ProviderModelInfo:
    id: str
    context_window: int | None = None
    active: bool = True


@dataclass
class ProviderHealth:
    ok: bool
    detail: str = ""


class ProviderAdapter(Protocol):
    provider_id: ClassVar[str]

    async def chat(self, req: NormalizedChatRequest, *, deadline: Deadline) -> NormalizedChatResponse: ...

    def stream(self, req: NormalizedChatRequest, *, deadline: Deadline) -> AsyncIterator[NormalizedChunk]: ...

    async def list_models(self) -> list[ProviderModelInfo]: ...

    async def health(self) -> ProviderHealth: ...

    async def aclose(self) -> None: ...
