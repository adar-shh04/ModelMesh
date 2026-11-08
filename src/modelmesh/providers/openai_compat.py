"""Shared implementation for providers exposing an OpenAI-compatible Chat Completions API (ADR-0003).

Subclasses override only what is provider-specific. Error messages built here never include the upstream body,
which can echo prompt content; the body is used only to classify the failure.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any, ClassVar, cast

import httpx
from pydantic import SecretStr

from modelmesh.core.deadline import Deadline
from modelmesh.core.errors import ErrorClass, ProviderError
from modelmesh.providers.sse import iter_sse_data
from modelmesh.providers.types import (
    FinishReason,
    NormalizedChatRequest,
    NormalizedChatResponse,
    NormalizedChunk,
    ProviderHealth,
    ProviderModelInfo,
    RateLimitObservation,
    TokenUsage,
    ToolCallDelta,
)

_INCOMPATIBLE_HINTS = (
    "not supported",
    "unsupported",
    "does not support",
    "unknown parameter",
    "unrecognized",
    "not available",
    "invalid schema",
)
_CONTEXT_HINTS = ("context length", "context_length", "maximum context", "too many tokens", "too long")
_FILTER_HINTS = ("safety", "recitation", "moderation", "content policy", "content_filter")

_FINISH_MAP: dict[str, FinishReason] = {
    "stop": "stop",
    "eos": "stop",
    "length": "length",
    "max_tokens": "length",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
    "content_filter": "content_filter",
    "safety": "content_filter",
    "recitation": "content_filter",
    "blocklist": "content_filter",
    "prohibited_content": "content_filter",
    "spii": "content_filter",
}


def parse_retry_after(headers: Mapping[str, str]) -> float | None:
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def error_text(body: Any) -> str:
    """Lower-cased text of an error body, for classification only. Never surfaced to clients."""
    if isinstance(body, list) and body:
        body = body[0]
    if isinstance(body, dict):
        error = body.get("error", body)
        return json.dumps(error).lower()
    return str(body).lower()


class OpenAICompatAdapter:
    provider_id: ClassVar[str] = "openai_compat"
    chat_path: ClassVar[str] = "chat/completions"
    models_path: ClassVar[str] = "models"

    def __init__(self, client: httpx.AsyncClient, base_url: str, api_key: SecretStr | None) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/") + "/"
        self._api_key = api_key

    # ------------------------------------------------------------ extension points
    def build_headers(self) -> dict[str, str]:
        if self._api_key is None:
            return {}
        return {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}

    def build_messages(self, req: NormalizedChatRequest) -> list[dict[str, Any]]:
        return req.messages

    def build_payload(self, req: NormalizedChatRequest, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": req.provider_model, "messages": self.build_messages(req)}
        optional: dict[str, Any] = {
            "temperature": req.temperature,
            "top_p": req.top_p,
            "max_tokens": req.max_tokens,
            "stop": req.stop,
            "seed": req.seed,
            "presence_penalty": req.presence_penalty,
            "frequency_penalty": req.frequency_penalty,
            "reasoning_effort": req.reasoning_effort,
            "response_format": req.response_format,
            "tools": req.tools,
            "tool_choice": req.tool_choice,
            "parallel_tool_calls": req.parallel_tool_calls,
        }
        payload.update({key: value for key, value in optional.items() if value is not None})
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        return payload

    def extract_usage(self, obj: Mapping[str, Any]) -> TokenUsage | None:
        usage = obj.get("usage")
        if not isinstance(usage, Mapping):
            return None
        return self._usage_from(usage)

    def extract_cost(self, obj: Mapping[str, Any]) -> float | None:
        return None

    def parse_rate_limit(self, headers: Mapping[str, str]) -> RateLimitObservation | None:
        return None

    def classify_status(self, status: int, body: Any, headers: Mapping[str, str]) -> ProviderError:
        text = error_text(body)
        retry_after = parse_retry_after(headers)
        message = f"{self.provider_id} returned HTTP {status}"

        def make(error_class: ErrorClass, *, billed: bool = False) -> ProviderError:
            return ProviderError(
                error_class, message, http_status=status, retry_after_s=retry_after, possibly_billed=billed
            )

        if status in (401, 403):
            if any(hint in text for hint in _FILTER_HINTS):
                return make(ErrorClass.CONTENT_FILTERED)
            return make(ErrorClass.AUTH_ERROR)
        if status == 402:
            return make(ErrorClass.QUOTA_EXHAUSTED)
        if status == 404:
            return make(ErrorClass.INCOMPATIBLE)
        if status in (408, 504):
            return make(ErrorClass.TIMEOUT, billed=True)
        if status == 413:
            return make(ErrorClass.CONTEXT_LENGTH_EXCEEDED)
        if status == 429:
            return make(ErrorClass.RATE_LIMITED)
        if status == 422:
            return make(ErrorClass.UPSTREAM_UNAVAILABLE, billed=True)
        if status == 400:
            if any(hint in text for hint in _CONTEXT_HINTS):
                return make(ErrorClass.CONTEXT_LENGTH_EXCEEDED)
            if any(hint in text for hint in _FILTER_HINTS):
                return make(ErrorClass.CONTENT_FILTERED)
            if any(hint in text for hint in _INCOMPATIBLE_HINTS):
                return make(ErrorClass.INCOMPATIBLE)
            return make(ErrorClass.INVALID_REQUEST)
        return make(ErrorClass.UPSTREAM_UNAVAILABLE)

    def classify_body_error(self, body: Mapping[str, Any], *, in_stream: bool) -> ProviderError:
        """Handle an `error` object delivered with HTTP 200."""
        error = body.get("error")
        code = error.get("code") if isinstance(error, Mapping) else None
        status = code if isinstance(code, int) and 400 <= code <= 599 else 502
        result = self.classify_status(status, body, {})
        result.possibly_billed = result.possibly_billed or in_stream
        result.http_status = 200
        return result

    # ------------------------------------------------------------ provider calls
    @property
    def _chat_url(self) -> str:
        return self._base_url + self.chat_path

    @staticmethod
    def _usage_from(usage: Mapping[str, Any]) -> TokenUsage | None:
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if not isinstance(prompt, int) or not isinstance(completion, int):
            return None
        prompt_details = usage.get("prompt_tokens_details")
        completion_details = usage.get("completion_tokens_details")
        cached = prompt_details.get("cached_tokens") if isinstance(prompt_details, Mapping) else None
        reasoning = completion_details.get("reasoning_tokens") if isinstance(completion_details, Mapping) else None
        return TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            cached_tokens=cached if isinstance(cached, int) else None,
            reasoning_tokens=reasoning if isinstance(reasoning, int) else None,
        )

    @staticmethod
    def _finish(value: Any) -> FinishReason | None:
        if not isinstance(value, str):
            return None
        return _FINISH_MAP.get(value.lower(), "stop")

    def _transport_error(self, exc: httpx.HTTPError) -> ProviderError:
        if isinstance(exc, httpx.ConnectTimeout):
            return ProviderError(ErrorClass.TIMEOUT, f"{self.provider_id} connect timeout")
        if isinstance(exc, httpx.TimeoutException):
            return ProviderError(ErrorClass.TIMEOUT, f"{self.provider_id} timed out", possibly_billed=True)
        return ProviderError(ErrorClass.UPSTREAM_UNAVAILABLE, f"{self.provider_id} connection failed")

    async def _decode_error_body(self, response: httpx.Response) -> Any:
        await response.aread()
        try:
            return response.json()
        except ValueError:
            return response.text[:2000]

    async def chat(self, req: NormalizedChatRequest, *, deadline: Deadline) -> NormalizedChatResponse:
        timeout = httpx.Timeout(deadline.remaining_s(), connect=min(5.0, deadline.remaining_s()))
        try:
            response = await self._client.post(
                self._chat_url,
                json=self.build_payload(req, stream=False),
                headers=self.build_headers(),
                timeout=timeout,
            )
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from None
        if response.status_code >= 400:
            raise self.classify_status(response.status_code, await self._decode_error_body(response), response.headers)
        try:
            body = response.json()
        except ValueError:
            raise ProviderError(
                ErrorClass.UPSTREAM_UNAVAILABLE, f"{self.provider_id} sent malformed JSON", http_status=200
            ) from None
        if not isinstance(body, dict):
            raise ProviderError(
                ErrorClass.UPSTREAM_UNAVAILABLE,
                f"{self.provider_id} sent an unexpected body",
                http_status=200,
            )
        if body.get("error"):
            raise self.classify_body_error(body, in_stream=False)
        return self._parse_completion(body, response)

    def _parse_completion(self, body: dict[str, Any], response: httpx.Response) -> NormalizedChatResponse:
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ProviderError(
                ErrorClass.UPSTREAM_UNAVAILABLE, f"{self.provider_id} returned no choices", http_status=200
            )
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content")
        tool_calls = message.get("tool_calls") or None
        if not content and not tool_calls:
            raise ProviderError(
                ErrorClass.UPSTREAM_UNAVAILABLE,
                f"{self.provider_id} returned empty content",
                http_status=200,
                possibly_billed=True,
            )
        finish = self._finish(choice.get("finish_reason")) or "stop"
        return NormalizedChatResponse(
            content=content if isinstance(content, str) else None,
            tool_calls=cast("list[dict[str, Any]] | None", tool_calls),
            finish_reason="tool_calls" if tool_calls and finish == "stop" else finish,
            usage=self.extract_usage(body),
            provider_reported_cost_usd=self.extract_cost(body),
            provider_response_id=body.get("id") if isinstance(body.get("id"), str) else None,
            rate_limit=self.parse_rate_limit(response.headers),
        )

    async def stream(self, req: NormalizedChatRequest, *, deadline: Deadline) -> AsyncIterator[NormalizedChunk]:
        # No read timeout here: the executor bounds time-to-first-token by the deadline and later gaps by the
        # idle timeout, which a single static httpx timeout cannot express.
        timeout = httpx.Timeout(None, connect=min(5.0, deadline.remaining_s()))
        try:
            async with self._client.stream(
                "POST",
                self._chat_url,
                json=self.build_payload(req, stream=True),
                headers=self.build_headers(),
                timeout=timeout,
            ) as response:
                if response.status_code >= 400:
                    raise self.classify_status(
                        response.status_code, await self._decode_error_body(response), response.headers
                    )
                rate_limit = self.parse_rate_limit(response.headers)
                finished = False
                async for data in iter_sse_data(response.aiter_lines()):
                    if data.strip() == "[DONE]":
                        return
                    try:
                        obj = json.loads(data)
                    except ValueError:
                        raise ProviderError(
                            ErrorClass.UPSTREAM_UNAVAILABLE,
                            f"{self.provider_id} sent a malformed stream event",
                            http_status=200,
                            possibly_billed=True,
                        ) from None
                    if not isinstance(obj, dict):
                        continue
                    if obj.get("error"):
                        raise self.classify_body_error(obj, in_stream=True)
                    chunk = self._parse_chunk(obj)
                    chunk.rate_limit = rate_limit
                    rate_limit = None
                    if chunk.finish_reason is not None:
                        if finished:
                            chunk.finish_reason = None
                        finished = True
                    yield chunk
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from None

    def _parse_chunk(self, obj: Mapping[str, Any]) -> NormalizedChunk:
        chunk = NormalizedChunk(usage=self.extract_usage(obj), provider_reported_cost_usd=self.extract_cost(obj))
        choices = obj.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            return chunk
        choice = choices[0]
        if choice.get("finish_reason") == "error":
            raise ProviderError(
                ErrorClass.UPSTREAM_UNAVAILABLE,
                f"{self.provider_id} reported a stream error",
                http_status=200,
                possibly_billed=True,
            )
        delta = choice.get("delta") or {}
        content = delta.get("content")
        chunk.content = content if isinstance(content, str) and content else None
        chunk.finish_reason = self._finish(choice.get("finish_reason"))
        raw_tools = delta.get("tool_calls")
        if isinstance(raw_tools, list) and raw_tools:
            chunk.tool_calls = [
                ToolCallDelta(
                    index=int(item.get("index", position)),
                    id=item.get("id"),
                    name=(item.get("function") or {}).get("name"),
                    arguments=(item.get("function") or {}).get("arguments"),
                )
                for position, item in enumerate(raw_tools)
                if isinstance(item, Mapping)
            ]
        return chunk

    async def list_models(self) -> list[ProviderModelInfo]:
        try:
            response = await self._client.get(
                self._base_url + self.models_path, headers=self.build_headers(), timeout=10.0
            )
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from None
        if response.status_code >= 400:
            raise self.classify_status(response.status_code, await self._decode_error_body(response), response.headers)
        return self._parse_models(response.json())

    def _parse_models(self, body: Any) -> list[ProviderModelInfo]:
        data = body.get("data", []) if isinstance(body, dict) else []
        return [
            ProviderModelInfo(
                id=str(item["id"]),
                context_window=item.get("context_window") or item.get("context_length"),
                active=bool(item.get("active", True)),
            )
            for item in data
            if isinstance(item, dict) and "id" in item
        ]

    async def health(self) -> ProviderHealth:
        try:
            await self.list_models()
        except ProviderError as exc:
            return ProviderHealth(ok=False, detail=str(exc.error_class))
        return ProviderHealth(ok=True)

    async def aclose(self) -> None:
        await self._client.aclose()
