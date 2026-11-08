"""Adapter behaviour against recorded or documented provider responses (ADR-0003). No network, no keys."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
import respx
from pydantic import SecretStr

from modelmesh.core.deadline import Deadline
from modelmesh.core.errors import ErrorClass, ProviderError
from modelmesh.providers.gemini import GeminiAdapter, next_pacific_midnight
from modelmesh.providers.groq import GroqAdapter, parse_duration
from modelmesh.providers.ollama import OllamaAdapter
from modelmesh.providers.openai_compat import OpenAICompatAdapter
from modelmesh.providers.openrouter import OpenRouterAdapter, next_utc_midnight
from modelmesh.providers.types import NormalizedChatRequest, NormalizedChunk

pytestmark = pytest.mark.anyio

KEY = SecretStr("test-secret-key")
GROQ = "https://groq.test/openai/v1"
GEMINI = "https://gemini.test/v1beta/openai"
OPENROUTER = "https://openrouter.test/api/v1"
OLLAMA = "http://ollama.test:11434"
SUCCESS = {
    "id": "resp-1",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def client():
    async with httpx.AsyncClient() as http_client:
        yield http_client


def make_request(**overrides: Any) -> NormalizedChatRequest:
    defaults: dict[str, Any] = {
        "provider_model": "m",
        "messages": [{"role": "user", "content": "hi", "name": "alice"}],
        "max_tokens": 16,
    }
    return NormalizedChatRequest(**(defaults | overrides))


def deadline() -> Deadline:
    return Deadline.in_ms(5_000)


def sse(*events: str) -> str:
    return "".join(f"{event}\n\n" for event in events)


def data(obj: dict[str, Any]) -> str:
    return "data: " + json.dumps(obj)


def delta(content: str | None = None, finish: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {} if content is None else {"content": content}
    return {"choices": [{"index": 0, "delta": body, "finish_reason": finish}]}


async def collect(
    adapter: OpenAICompatAdapter, req: NormalizedChatRequest | None = None
) -> list[NormalizedChunk]:
    return [chunk async for chunk in adapter.stream(req or make_request(), deadline=deadline())]


def groq(client: httpx.AsyncClient) -> GroqAdapter:
    return GroqAdapter(client, GROQ, KEY)


def gemini(client: httpx.AsyncClient) -> GeminiAdapter:
    return GeminiAdapter(client, GEMINI, KEY)


def openrouter(client: httpx.AsyncClient) -> OpenRouterAdapter:
    return OpenRouterAdapter(client, OPENROUTER, KEY, app_url="https://example.test", app_title="mesh")


def ollama(client: httpx.AsyncClient) -> OllamaAdapter:
    return OllamaAdapter(client, OLLAMA)


ADAPTERS = [
    pytest.param(groq, f"{GROQ}/chat/completions", id="groq"),
    pytest.param(gemini, f"{GEMINI}/chat/completions", id="gemini"),
    pytest.param(openrouter, f"{OPENROUTER}/chat/completions", id="openrouter"),
    pytest.param(ollama, f"{OLLAMA}/v1/chat/completions", id="ollama"),
]


@pytest.mark.parametrize(("factory", "url"), ADAPTERS)
@respx.mock
async def test_chat_success_is_normalised(client, factory, url) -> None:
    respx.post(url).mock(return_value=httpx.Response(200, json=SUCCESS))
    result = await factory(client).chat(make_request(), deadline=deadline())
    assert result.content == "hello"
    assert result.finish_reason == "stop"
    assert result.usage is not None
    assert (result.usage.prompt_tokens, result.usage.completion_tokens) == (11, 3)
    assert result.usage.source == "provider_reported"


@pytest.mark.parametrize(("factory", "url"), ADAPTERS)
@respx.mock
async def test_stream_success_with_usage_chunk(client, factory, url) -> None:
    body = sse(
        data(delta("he")),
        data(delta("llo")),
        data(delta(None, "stop")),
        data({"choices": [], "usage": SUCCESS["usage"]}),
        "data: [DONE]",
    )
    respx.post(url).mock(return_value=httpx.Response(200, text=body))
    chunks = await collect(factory(client))
    assert "".join(c.content or "" for c in chunks) == "hello"
    assert [c.finish_reason for c in chunks if c.finish_reason] == ["stop"]
    usage = [c.usage for c in chunks if c.usage]
    assert usage
    assert usage[0].completion_tokens == 3


@pytest.mark.parametrize(("factory", "url"), ADAPTERS)
@respx.mock
async def test_stream_without_usage_reports_none(client, factory, url) -> None:
    respx.post(url).mock(return_value=httpx.Response(200, text=sse(data(delta("x", "stop")), "data: [DONE]")))
    chunks = await collect(factory(client))
    assert all(c.usage is None for c in chunks)


STATUS_CASES = [
    (400, {"error": {"message": "something odd"}}, ErrorClass.INVALID_REQUEST),
    (400, {"error": {"message": "parameter logprobs is not supported"}}, ErrorClass.INCOMPATIBLE),
    (400, {"error": {"message": "exceeds maximum context length"}}, ErrorClass.CONTEXT_LENGTH_EXCEEDED),
    (401, {"error": {"message": "invalid key"}}, ErrorClass.AUTH_ERROR),
    (404, {"error": {"message": "no such model"}}, ErrorClass.INCOMPATIBLE),
    (408, {"error": {"message": "timeout"}}, ErrorClass.TIMEOUT),
    (413, {"error": {"message": "too big"}}, ErrorClass.CONTEXT_LENGTH_EXCEEDED),
    (429, {"error": {"message": "slow down"}}, ErrorClass.RATE_LIMITED),
    (500, {"error": {"message": "boom"}}, ErrorClass.UPSTREAM_UNAVAILABLE),
    (502, {"error": {"message": "bad gateway"}}, ErrorClass.UPSTREAM_UNAVAILABLE),
    (503, {"error": {"message": "overloaded"}}, ErrorClass.UPSTREAM_UNAVAILABLE),
]


@pytest.mark.parametrize(("factory", "url"), ADAPTERS)
@pytest.mark.parametrize(("status", "body", "expected"), STATUS_CASES)
@respx.mock
async def test_status_classification(client, factory, url, status, body, expected) -> None:
    respx.post(url).mock(return_value=httpx.Response(status, json=body))
    with pytest.raises(ProviderError) as excinfo:
        await factory(client).chat(make_request(), deadline=deadline())
    assert excinfo.value.error_class == expected
    assert excinfo.value.http_status == status


@pytest.mark.parametrize(("factory", "url"), ADAPTERS)
@respx.mock
async def test_error_message_never_contains_upstream_body(client, factory, url) -> None:
    secret = "PROMPT-CONTENT-CANARY"
    respx.post(url).mock(return_value=httpx.Response(400, json={"error": {"message": secret}}))
    with pytest.raises(ProviderError) as excinfo:
        await factory(client).chat(make_request(), deadline=deadline())
    assert secret not in str(excinfo.value)


@pytest.mark.parametrize(("factory", "url"), ADAPTERS)
@respx.mock
async def test_retry_after_is_parsed(client, factory, url) -> None:
    respx.post(url).mock(return_value=httpx.Response(429, json={}, headers={"retry-after": "7"}))
    with pytest.raises(ProviderError) as excinfo:
        await factory(client).chat(make_request(), deadline=deadline())
    assert excinfo.value.retry_after_s == 7.0


@pytest.mark.parametrize(("factory", "url"), ADAPTERS)
@respx.mock
async def test_missing_retry_after_is_none(client, factory, url) -> None:
    respx.post(url).mock(return_value=httpx.Response(429, json={}))
    with pytest.raises(ProviderError) as excinfo:
        await factory(client).chat(make_request(), deadline=deadline())
    assert excinfo.value.retry_after_s is None


@pytest.mark.parametrize(("factory", "url"), ADAPTERS)
@pytest.mark.parametrize(
    ("response", "label"),
    [
        (httpx.Response(200, text="not json"), "malformed"),
        (httpx.Response(200, text=""), "empty-body"),
        (httpx.Response(200, json={"choices": []}), "no-choices"),
        (httpx.Response(200, json={"choices": [{"message": {"content": ""}}]}), "empty-content"),
    ],
)
@respx.mock
async def test_malformed_or_empty_success_is_a_failure(client, factory, url, response, label) -> None:
    respx.post(url).mock(return_value=response)
    with pytest.raises(ProviderError) as excinfo:
        await factory(client).chat(make_request(), deadline=deadline())
    assert excinfo.value.error_class == ErrorClass.UPSTREAM_UNAVAILABLE, label


@pytest.mark.parametrize(("factory", "url"), ADAPTERS)
@respx.mock
async def test_timeouts_and_connection_errors(client, factory, url) -> None:
    route = respx.post(url)
    route.mock(side_effect=httpx.ReadTimeout("t"))
    with pytest.raises(ProviderError) as excinfo:
        await factory(client).chat(make_request(), deadline=deadline())
    assert excinfo.value.error_class == ErrorClass.TIMEOUT
    assert excinfo.value.possibly_billed is True

    route.mock(side_effect=httpx.ConnectTimeout("t"))
    with pytest.raises(ProviderError) as excinfo:
        await factory(client).chat(make_request(), deadline=deadline())
    assert excinfo.value.error_class == ErrorClass.TIMEOUT
    assert excinfo.value.possibly_billed is False

    route.mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(ProviderError) as excinfo:
        await factory(client).chat(make_request(), deadline=deadline())
    assert excinfo.value.error_class == ErrorClass.UPSTREAM_UNAVAILABLE


@pytest.mark.parametrize(("factory", "url"), ADAPTERS)
@respx.mock
async def test_stream_error_status_raises_before_chunks(client, factory, url) -> None:
    respx.post(url).mock(return_value=httpx.Response(503, json={"error": {"message": "x"}}))
    with pytest.raises(ProviderError) as excinfo:
        await collect(factory(client))
    assert excinfo.value.error_class == ErrorClass.UPSTREAM_UNAVAILABLE


# ---------------------------------------------------------------- Groq
@respx.mock
async def test_groq_sends_bearer_and_strips_message_name(client) -> None:
    route = respx.post(f"{GROQ}/chat/completions").mock(return_value=httpx.Response(200, json=SUCCESS))
    await groq(client).chat(make_request(), deadline=deadline())
    sent = route.calls.last.request
    assert sent.headers["authorization"] == "Bearer test-secret-key"
    assert "name" not in json.loads(sent.content)["messages"][0]


@respx.mock
async def test_groq_reads_usage_from_x_groq(client) -> None:
    body = sse(
        data(delta("x", "stop") | {"x_groq": {"usage": {"prompt_tokens": 5, "completion_tokens": 2}}}),
        "data: [DONE]",
    )
    respx.post(f"{GROQ}/chat/completions").mock(return_value=httpx.Response(200, text=body))
    chunks = await collect(groq(client))
    assert [c.usage.prompt_tokens for c in chunks if c.usage] == [5]


@respx.mock
async def test_groq_rate_limit_headers_are_parsed(client) -> None:
    headers = {
        "x-ratelimit-remaining-requests": "12",
        "x-ratelimit-remaining-tokens": "900",
        "x-ratelimit-reset-requests": "2m59.56s",
        "x-ratelimit-reset-tokens": "6s",
    }
    respx.post(f"{GROQ}/chat/completions").mock(
        return_value=httpx.Response(200, json=SUCCESS, headers=headers)
    )
    result = await groq(client).chat(make_request(), deadline=deadline())
    assert result.rate_limit is not None
    assert result.rate_limit.remaining_requests == 12
    assert result.rate_limit.remaining_tokens == 900
    assert result.rate_limit.reset_requests_s == pytest.approx(179.56)
    assert result.rate_limit.reset_tokens_s == 6.0


def test_groq_duration_parser() -> None:
    assert parse_duration("1h2m3s") == 3723.0
    assert parse_duration("250ms") == 0.25
    assert parse_duration(None) is None
    assert parse_duration("garbage") is None


@respx.mock
async def test_groq_498_is_retryable_capacity(client) -> None:
    respx.post(f"{GROQ}/chat/completions").mock(return_value=httpx.Response(498, json={}))
    with pytest.raises(ProviderError) as excinfo:
        await groq(client).chat(make_request(), deadline=deadline())
    assert excinfo.value.error_class == ErrorClass.RATE_LIMITED


# ---------------------------------------------------------------- Gemini
@respx.mock
async def test_gemini_distinguishes_daily_quota_from_rate(client) -> None:
    route = respx.post(f"{GEMINI}/chat/completions")
    route.mock(
        return_value=httpx.Response(429, json={"error": {"code": "rate_limit_exceeded", "message": "x"}})
    )
    with pytest.raises(ProviderError) as excinfo:
        await gemini(client).chat(make_request(), deadline=deadline())
    assert excinfo.value.error_class == ErrorClass.RATE_LIMITED

    route.mock(return_value=httpx.Response(429, json={"error": {"code": "quota_exceeded", "message": "x"}}))
    with pytest.raises(ProviderError) as excinfo:
        await gemini(client).chat(make_request(), deadline=deadline())
    assert excinfo.value.error_class == ErrorClass.QUOTA_EXHAUSTED
    assert excinfo.value.quota_reset_at is not None


@respx.mock
async def test_gemini_list_wrapped_error_body(client) -> None:
    payload = [{"error": {"code": 400, "message": "request exceeds maximum context length"}}]
    respx.post(f"{GEMINI}/chat/completions").mock(return_value=httpx.Response(400, json=payload))
    with pytest.raises(ProviderError) as excinfo:
        await gemini(client).chat(make_request(), deadline=deadline())
    assert excinfo.value.error_class == ErrorClass.CONTEXT_LENGTH_EXCEEDED


def test_quota_resets() -> None:
    now = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)
    assert next_pacific_midnight(now) == datetime(2026, 11, 9, 8, 0, tzinfo=UTC)
    assert next_utc_midnight(now) == datetime(2026, 11, 9, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------- OpenRouter
@respx.mock
async def test_openrouter_200_with_error_body_is_a_failure(client) -> None:
    body = {"error": {"code": 502, "message": "provider down"}}
    respx.post(f"{OPENROUTER}/chat/completions").mock(return_value=httpx.Response(200, json=body))
    with pytest.raises(ProviderError) as excinfo:
        await openrouter(client).chat(make_request(), deadline=deadline())
    assert excinfo.value.error_class == ErrorClass.UPSTREAM_UNAVAILABLE


@respx.mock
async def test_openrouter_midstream_error_chunk_and_keepalive(client) -> None:
    body = sse(
        ": OPENROUTER PROCESSING",
        data(delta("partial")),
        data(
            {"error": {"code": 502, "message": "dead"}, "choices": [{"delta": {}, "finish_reason": "error"}]}
        ),
    )
    respx.post(f"{OPENROUTER}/chat/completions").mock(return_value=httpx.Response(200, text=body))
    received: list[NormalizedChunk] = []
    with pytest.raises(ProviderError) as excinfo:
        async for chunk in openrouter(client).stream(make_request(), deadline=deadline()):
            received.append(chunk)
    assert [c.content for c in received] == ["partial"]
    assert excinfo.value.possibly_billed is True


@respx.mock
async def test_openrouter_repeated_finish_reason_is_deduplicated_and_cost_kept(client) -> None:
    body = sse(
        data(delta("x", "stop")),
        data(delta(None, "stop") | {"usage": {"prompt_tokens": 2, "completion_tokens": 1, "cost": 0.00012}}),
        "data: [DONE]",
    )
    respx.post(f"{OPENROUTER}/chat/completions").mock(return_value=httpx.Response(200, text=body))
    chunks = await collect(openrouter(client))
    assert [c.finish_reason for c in chunks if c.finish_reason] == ["stop"]
    assert [c.provider_reported_cost_usd for c in chunks if c.provider_reported_cost_usd] == [0.00012]


@respx.mock
async def test_openrouter_hard_parameters_and_data_policy_are_enforced_upstream(client) -> None:
    route = respx.post(f"{OPENROUTER}/chat/completions").mock(return_value=httpx.Response(200, json=SUCCESS))
    req = make_request(response_format={"type": "json_object"}, require_no_training=True)
    await openrouter(client).chat(req, deadline=deadline())
    sent = route.calls.last.request
    payload = json.loads(sent.content)
    assert payload["provider"] == {"require_parameters": True, "data_collection": "deny"}
    assert sent.headers["x-openrouter-title"] == "mesh"
    assert sent.headers["http-referer"] == "https://example.test"


@respx.mock
async def test_openrouter_free_daily_limit_is_quota(client) -> None:
    body = {"error": {"code": 429, "message": "Rate limit exceeded: free-models-per-day"}}
    respx.post(f"{OPENROUTER}/chat/completions").mock(return_value=httpx.Response(429, json=body))
    with pytest.raises(ProviderError) as excinfo:
        await openrouter(client).chat(make_request(), deadline=deadline())
    assert excinfo.value.error_class == ErrorClass.QUOTA_EXHAUSTED


@respx.mock
async def test_openrouter_403_is_moderation_not_credentials(client) -> None:
    respx.post(f"{OPENROUTER}/chat/completions").mock(return_value=httpx.Response(403, json={"error": {}}))
    with pytest.raises(ProviderError) as excinfo:
        await openrouter(client).chat(make_request(), deadline=deadline())
    assert excinfo.value.error_class == ErrorClass.CONTENT_FILTERED


# ---------------------------------------------------------------- Ollama
@respx.mock
async def test_ollama_sends_no_auth_and_drops_tool_choice(client) -> None:
    route = respx.post(f"{OLLAMA}/v1/chat/completions").mock(return_value=httpx.Response(200, json=SUCCESS))
    req = make_request(tools=[{"type": "function", "function": {"name": "f"}}], tool_choice="auto")
    await ollama(client).chat(req, deadline=deadline())
    sent = route.calls.last.request
    assert "authorization" not in sent.headers
    payload = json.loads(sent.content)
    assert "tool_choice" not in payload
    assert payload["tools"]


@respx.mock
async def test_ollama_health_reports_loaded_models(client) -> None:
    respx.get(f"{OLLAMA}/api/ps").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "qwen3:4b"}]})
    )
    health = await ollama(client).health()
    assert health.ok
    assert "1 model" in health.detail


@respx.mock
async def test_ollama_health_when_down(client) -> None:
    respx.get(f"{OLLAMA}/api/ps").mock(side_effect=httpx.ConnectError("refused"))
    assert not (await ollama(client).health()).ok


# ---------------------------------------------------------------- shared streaming details
@respx.mock
async def test_stream_requests_usage_where_supported(client) -> None:
    route = respx.post(f"{GROQ}/chat/completions").mock(
        return_value=httpx.Response(200, text=sse(data(delta("x", "stop")), "data: [DONE]"))
    )
    await collect(groq(client))
    assert json.loads(route.calls.last.request.content)["stream_options"] == {"include_usage": True}


@respx.mock
async def test_tool_call_deltas_are_normalised(client) -> None:
    tool_delta = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [{"index": 0, "id": "c1", "function": {"name": "f", "arguments": "{"}}]
                },
                "finish_reason": None,
            }
        ]
    }
    body = sse(data(tool_delta), data(delta(None, "tool_calls")), "data: [DONE]")
    respx.post(f"{GROQ}/chat/completions").mock(return_value=httpx.Response(200, text=body))
    chunks = await collect(groq(client))
    assert chunks[0].tool_calls is not None
    assert (chunks[0].tool_calls[0].id, chunks[0].tool_calls[0].name) == ("c1", "f")
    assert chunks[1].finish_reason == "tool_calls"


@respx.mock
async def test_provider_specific_finish_reasons_map_to_contract_values(client) -> None:
    payload = {
        "choices": [{"message": {"content": "x"}, "finish_reason": "SAFETY"}],
        "usage": SUCCESS["usage"],
    }
    respx.post(f"{GEMINI}/chat/completions").mock(return_value=httpx.Response(200, json=payload))
    result = await gemini(client).chat(make_request(), deadline=deadline())
    assert result.finish_reason == "content_filter"


@respx.mock
async def test_list_models_parses_context_windows(client) -> None:
    body = {"data": [{"id": "a", "context_window": 1000, "active": True}, {"id": "b", "active": False}]}
    respx.get(f"{GROQ}/models").mock(return_value=httpx.Response(200, json=body))
    models = await groq(client).list_models()
    assert [(m.id, m.context_window, m.active) for m in models] == [("a", 1000, True), ("b", None, False)]
