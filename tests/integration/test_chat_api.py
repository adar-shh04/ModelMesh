"""The unified API, end to end through real Postgres and Redis, against the fake and scripted providers."""

from __future__ import annotations

import json

import httpx
import pytest
from openai import APIError, AsyncOpenAI, BadRequestError

from conftest import Api, assert_matches, sse_events
from doubles import chunks, make_model
from modelmesh.core.errors import ErrorClass, ProviderError
from modelmesh.providers.types import NormalizedChunk, TokenUsage

pytestmark = [pytest.mark.anyio, pytest.mark.integration]

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer"],
    "properties": {"answer": {"type": "string"}},
}
JSON_SCHEMA_FORMAT = {"type": "json_schema", "json_schema": {"name": "r", "schema": SCHEMA, "strict": True}}


def unavailable(status: int = 503) -> ProviderError:
    return ProviderError(ErrorClass.UPSTREAM_UNAVAILABLE, "down", http_status=status)


def sdk(api: Api) -> AsyncOpenAI:
    http_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://mesh.test")
    return AsyncOpenAI(api_key=api.key, base_url="http://mesh.test/v1", http_client=http_client, max_retries=0)


# ---------------------------------------------------------------- contract shape
async def test_non_streaming_response_matches_the_contract(api: Api) -> None:
    response = await api.chat(headers={"X-Request-ID": "client-1"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert_matches("ChatCompletionResponse", body)
    assert body["choices"][0]["message"]["content"] == "echo: hello"
    meta = body["modelmesh"]
    assert meta["routing"]["mode"] == "explicit_model"
    assert meta["client_request_id"] == "client-1"
    assert response.headers["x-modelmesh-request-id"] == meta["request_id"]
    assert response.headers["x-request-id"] == "client-1"
    assert response.headers["x-modelmesh-contract"] == "1.0"
    assert "x-ratelimit-remaining-requests" in response.headers


async def test_request_id_is_generated_when_absent(api: Api) -> None:
    response = await api.chat()
    assert response.headers["x-request-id"]
    assert response.json()["modelmesh"]["client_request_id"] == response.headers["x-request-id"]


async def test_unknown_field_is_a_400_naming_the_parameter(api: Api) -> None:
    response = await api.chat(routing={"max_cost": 1})
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "unknown_parameter"
    assert error["param"] == "routing.max_cost"
    assert_matches("ErrorResponse", response.json())


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({"n": 2}, "invalid_value"),
        ({"logprobs": True}, "invalid_value"),
        ({"max_tokens": 5, "max_completion_tokens": 6}, "max_tokens_mismatch"),
        ({"cache": {"mode": "semantic"}}, "missing_context_fingerprint"),
        ({"cache": {"context_fingerprint": "sha256:abc"}}, "invalid_value"),
        ({"metadata": {"tags": {"a": "b"}}}, "invalid_value"),
        (
            {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]},
            "unsupported_content_part",
        ),
        (
            {"response_format": {"type": "json_schema", "json_schema": {"name": "a", "schema": {"type": "nope"}}}},
            "invalid_value",
        ),
    ],
)
async def test_invalid_requests(api: Api, body: dict, code: str) -> None:
    response = await api.chat(**body)
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == code


async def test_malformed_json_body(api: Api) -> None:
    response = await api.client.post("/v1/chat/completions", content=b"{nope", headers=api.headers)
    assert response.status_code == 400


async def test_authentication_runs_before_validation(api: Api) -> None:
    response = await api.client.post("/v1/chat/completions", content=b"{nope")
    assert response.status_code == 401
    body = response.json()
    assert_matches("ErrorResponse", body)
    assert body["error"]["request_id"].startswith("mmr_")


@pytest.mark.parametrize("header", ["Bearer mm_test_unknown", "Bearer not-a-key", "Basic abc"])
async def test_bad_credentials(api: Api, header: str) -> None:
    response = await api.client.get("/v1/models", headers={"Authorization": header})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"


async def test_body_too_large(api: Api) -> None:
    huge = b'{"model":"auto","messages":[{"role":"user","content":"' + b"x" * (3 * 1024 * 1024) + b'"}]}'
    response = await api.client.post("/v1/chat/completions", content=huge, headers=api.headers)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "body_too_large"


async def test_unknown_model_is_404(api: Api) -> None:
    response = await api.chat(model="fake/nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


async def test_unconfigured_provider_model_is_404(api: Api) -> None:
    response = await api.chat(model="groq/openai/gpt-oss-20b")
    assert response.status_code == 404


async def test_models_endpoint_lists_auto_and_configured_models_only(api: Api) -> None:
    response = await api.client.get("/v1/models", headers=api.headers)
    assert response.status_code == 200
    body = response.json()
    assert_matches("ModelList", body)
    ids = [model["id"] for model in body["data"]]
    assert ids[0] == "auto"
    assert "fake/echo-small" in ids
    assert not any(model.startswith("groq/") for model in ids)


async def test_health_and_contract_are_public(api: Api) -> None:
    health = await api.client.get("/v1/health")
    assert health.status_code == 200
    assert_matches("Health", health.json())
    assert health.json()["status"] == "ok"
    contract = await api.client.get("/v1/contract")
    assert contract.status_code == 200
    assert_matches("ContractInfo", contract.json())
    assert "cache.namespace" in contract.json()["extensions"]
    assert contract.json()["contract_release"] == "1.0.0"


async def test_calibration_reports_uncalibrated_until_m6(api: Api) -> None:
    response = await api.client.get("/v1/routing/calibration", headers=api.headers)
    assert_matches("RoutingCalibration", response.json())
    assert all(task["calibrated"] is False for task in response.json()["task_types"])


async def test_unknown_endpoint_uses_the_error_envelope(api: Api) -> None:
    response = await api.client.get("/v1/nothing", headers=api.headers)
    assert response.status_code == 404
    assert_matches("ErrorResponse", response.json())


async def test_embeddings_not_enabled(api: Api) -> None:
    response = await api.client.post("/v1/embeddings", json={"model": "x", "input": "y"}, headers=api.headers)
    assert response.status_code == 501
    assert response.json()["error"]["code"] == "embeddings_disabled"


# ---------------------------------------------------------------- official OpenAI SDK
async def test_openai_sdk_non_streaming(api: Api) -> None:
    client = sdk(api)
    result = await client.chat.completions.create(
        model="fake/echo-small", messages=[{"role": "user", "content": "from sdk"}]
    )
    assert result.choices[0].message.content == "echo: from sdk"
    assert result.usage is not None
    assert result.usage.total_tokens > 0


async def test_openai_sdk_streaming_without_usage(api: Api) -> None:
    stream = await sdk(api).chat.completions.create(
        model="fake/echo-small", messages=[{"role": "user", "content": "a b c"}], stream=True
    )
    parts = [chunk.choices[0].delta.content or "" async for chunk in stream]
    assert "".join(parts) == "echo: a b c"


async def test_openai_sdk_streaming_with_usage(api: Api) -> None:
    stream = await sdk(api).chat.completions.create(
        model="fake/echo-small",
        messages=[{"role": "user", "content": "x"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    received = [chunk async for chunk in stream]
    assert received[-1].choices == []
    assert received[-1].usage is not None


async def test_openai_sdk_structured_output_and_extensions(api: Api) -> None:
    result = await sdk(api).chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "x"}],
        response_format=JSON_SCHEMA_FORMAT,
        extra_body={"routing": {"policy": "cost_optimized", "required_capabilities": ["json_schema_strict"]}},
    )
    assert json.loads(result.choices[0].message.content) == {"answer": "fake"}


async def test_openai_sdk_tools_roundtrip_is_accepted(api: Api) -> None:
    api.add_models(make_model("toolful"))
    from modelmesh.providers.types import NormalizedChatResponse

    call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    api.scripted.script(
        "toolful",
        NormalizedChatResponse(content=None, tool_calls=[call], finish_reason="tool_calls", usage=TokenUsage(1, 1)),
    )
    result = await sdk(api).chat.completions.create(
        model="t/toolful",
        messages=[{"role": "user", "content": "x"}],
        tools=[{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
        tool_choice="auto",
    )
    assert result.choices[0].finish_reason == "tool_calls"
    assert result.choices[0].message.tool_calls[0].function.name == "f"


async def test_openai_sdk_unknown_field_raises_bad_request(api: Api) -> None:
    with pytest.raises(BadRequestError):
        await sdk(api).chat.completions.create(
            model="fake/echo-small", messages=[{"role": "user", "content": "x"}], extra_body={"bogus": 1}
        )


async def test_openai_sdk_surfaces_an_in_stream_error(api: Api) -> None:
    api.add_models(make_model("breaks"))
    api.scripted.script("breaks", [NormalizedChunk(content="partial"), unavailable()])
    stream = await sdk(api).chat.completions.create(
        model="t/breaks", messages=[{"role": "user", "content": "x"}], stream=True
    )
    with pytest.raises(APIError):
        async for _ in stream:
            pass


# ---------------------------------------------------------------- streaming semantics
async def test_stream_events_with_usage_match_the_contract(api: Api) -> None:
    response = await api.chat(stream=True, stream_options={"include_usage": True})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-modelmesh-request-id"].startswith("mmr_")
    events = sse_events(response.text)
    assert events[-1] == "[DONE]"
    for event in events[:-1]:
        assert_matches("ChatCompletionChunk", event)
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    finishes = [e["choices"][0]["finish_reason"] for e in events[:-1] if e["choices"]]
    assert finishes[-1] == "stop"
    last = events[-2]
    assert last["choices"] == []
    assert last["usage"]["total_tokens"] > 0
    assert last["modelmesh"]["streaming"] == {"mode": "native"}
    assert_matches("ModelMeshMetadata", last["modelmesh"])
    assert last["modelmesh"]["latency_ms"]["ttft"] is not None


async def test_stream_without_usage_puts_metadata_on_the_finish_chunk(api: Api) -> None:
    events = sse_events((await api.chat(stream=True)).text)
    assert events[-1] == "[DONE]"
    finish = events[-2]
    assert finish["choices"][0]["finish_reason"] == "stop"
    assert "usage" not in finish
    assert finish["modelmesh"]["usage"]["source"] == "provider_reported"


async def test_failures_before_first_content_are_real_http_errors(api: Api) -> None:
    api.only_models(make_model("a"))
    api.scripted.script("a", unavailable(), unavailable())
    response = await api.chat(model="t/a", stream=True)
    assert response.status_code == 502
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["code"] == "all_attempts_failed"


async def test_fallback_before_first_token_is_invisible_to_the_client(api: Api) -> None:
    api.only_models(make_model("first"), make_model("second"), prices={"t/first": (0.1, 0.1), "t/second": (1.0, 1.0)})
    api.scripted.script("first", unavailable(), unavailable())
    api.scripted.script("second", chunks("hi", usage=TokenUsage(1, 1)))
    response = await api.chat(model="auto", routing={"policy": "cost_optimized"}, stream=True)
    events = sse_events(response.text)
    assert events[-1] == "[DONE]"
    meta = events[-2]["modelmesh"]
    assert meta["fallback_used"] is True
    assert meta["model"] == "t/second"
    assert [a["outcome"] for a in meta["attempts"]] == ["upstream_unavailable", "upstream_unavailable", "success"]


async def test_midstream_failure_emits_one_error_event_and_no_done(api: Api) -> None:
    api.only_models(make_model("m"))
    api.scripted.script("m", [NormalizedChunk(content="partial"), unavailable()])
    response = await api.chat(model="t/m", stream=True)
    assert response.status_code == 200
    events = sse_events(response.text)
    assert "[DONE]" not in events
    assert_matches("ErrorResponse", events[-1])
    assert events[-1]["error"]["code"] == "stream_interrupted"
    assert events[-1]["modelmesh"]["attempts"][-1]["outcome"] == "stream_interrupted"
    record = (
        await api.client.get(f"/v1/requests/{response.headers['x-modelmesh-request-id']}", headers=api.headers)
    ).json()
    assert record["status"] == "stream_interrupted"


async def test_stream_ending_without_finish_reason_is_incomplete(api: Api) -> None:
    api.only_models(make_model("m"))
    api.scripted.script("m", [NormalizedChunk(content="partial")])
    events = sse_events((await api.chat(model="t/m", stream=True)).text)
    assert "[DONE]" not in events
    assert events[-1]["error"]["code"] == "stream_interrupted"


async def test_stream_without_provider_usage_is_labelled_estimated(api: Api) -> None:
    api.only_models(make_model("m"))
    api.scripted.script("m", chunks("one two"))
    events = sse_events((await api.chat(model="t/m", stream=True, stream_options={"include_usage": True})).text)
    assert events[-2]["modelmesh"]["usage"]["source"] == "estimated"


async def test_keepalive_comments_are_sent_while_waiting_after_commit(migrated_database: str) -> None:
    from conftest import build_api

    api = await build_api(migrated_database, MODELMESH_STREAM_KEEPALIVE_S=0.02, MODELMESH_STREAM_IDLE_TIMEOUT_S=5.0)
    try:
        api.only_models(make_model("slow"))
        api.scripted.script(
            "slow", [NormalizedChunk(content="a"), NormalizedChunk(content="b"), NormalizedChunk(finish_reason="stop")]
        )
        api.scripted.delay_s = 0.08
        response = await api.chat(model="t/slow", stream=True)
        assert ": keep-alive" in response.text
        assert sse_events(response.text)[-1] == "[DONE]"
    finally:
        await api.client.aclose()
        await api.services.aclose()


async def test_idle_upstream_ends_the_stream_with_an_error(migrated_database: str) -> None:
    from conftest import build_api

    api = await build_api(migrated_database, MODELMESH_STREAM_KEEPALIVE_S=0.01, MODELMESH_STREAM_IDLE_TIMEOUT_S=0.05)
    try:
        api.only_models(make_model("hang"))
        api.scripted.script("hang", [NormalizedChunk(content="a"), NormalizedChunk(content="b")])
        api.scripted.delay_s = 0.3
        response = await api.chat(model="t/hang", stream=True, routing={"timeout_ms": 5000})
        assert sse_events(response.text)[-1]["error"]["code"] == "stream_interrupted"
    finally:
        await api.client.aclose()
        await api.services.aclose()


async def test_client_disconnect_cancels_upstream_and_records_usage(api: Api) -> None:
    api.only_models(make_model("long"))
    api.scripted.script("long", [NormalizedChunk(content=str(i)) for i in range(50)])
    api.scripted.delay_s = 0.01
    body = json.dumps({"model": "t/long", "messages": [{"role": "user", "content": "hello"}], "stream": True}).encode()
    sent: list[dict] = []
    delivered = False

    async def receive() -> dict:
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        while len([m for m in sent if m["type"] == "http.response.body"]) < 3:
            import anyio

            await anyio.sleep(0.005)
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"authorization", f"Bearer {api.key}".encode()), (b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1),
        "server": ("mesh.test", 80),
        "scheme": "http",
    }
    await api.app(scope, receive, send)
    assert api.scripted.stream_closed
    request_id = dict(sent[0]["headers"])[b"x-modelmesh-request-id"].decode()
    record = (await api.client.get(f"/v1/requests/{request_id}", headers=api.headers)).json()
    assert record["status"] == "cancelled"
    assert record["modelmesh"]["attempts"][-1]["outcome"] == "client_cancelled"
    assert record["modelmesh"]["attempts"][-1]["possibly_billed"] is True
    assert record["modelmesh"]["usage"]["source"] == "estimated"


# ---------------------------------------------------------------- telemetry and privacy
async def test_span_tree_has_no_content_and_continues_the_caller_trace(api: Api) -> None:
    parent_trace = "4bf92f3577b34da6a3ce929d0e0e4736"
    traceparent = f"00-{parent_trace}-00f067aa0ba902b7-01"
    response = await api.chat(
        headers={"traceparent": traceparent},
        messages=[{"role": "user", "content": "SECRET-PROMPT-CANARY"}],
    )
    meta = response.json()["modelmesh"]
    assert meta["trace_id"] == parent_trace
    spans = api.spans.get_finished_spans()
    names = {span.name for span in spans}
    assert {"mm.policy", "mm.route", "mm.attempt", "mm.accounting"} <= names
    assert any(span.name.startswith("POST /v1/chat/completions") for span in spans)
    assert all(format(span.context.trace_id, "032x") == parent_trace for span in spans)
    for span in spans:
        rendered = json.dumps(dict(span.attributes or {}), default=str)
        assert "SECRET-PROMPT-CANARY" not in rendered
        assert api.key not in rendered
    attempt = next(span for span in spans if span.name == "mm.attempt")
    assert attempt.attributes["gen_ai.request.model"] == "fake/echo-small"


async def test_unsampled_parent_records_no_spans_but_returns_trace_id(api: Api) -> None:
    parent_trace = "5bf92f3577b34da6a3ce929d0e0e4736"
    response = await api.chat(headers={"traceparent": f"00-{parent_trace}-00f067aa0ba902b7-00"})
    assert response.json()["modelmesh"]["trace_id"] == parent_trace
    assert not api.spans.get_finished_spans()


async def test_logs_never_contain_prompts_or_keys(api: Api, caplog: pytest.LogCaptureFixture) -> None:
    import logging

    from modelmesh.observability.logging import JsonFormatter

    with caplog.at_level(logging.DEBUG):
        await api.chat(messages=[{"role": "user", "content": "SECRET-PROMPT-CANARY"}])
        await api.chat(model="fake/unavailable", messages=[{"role": "user", "content": "SECRET-PROMPT-CANARY"}])
    formatter = JsonFormatter()
    rendered = "\n".join(formatter.format(record) for record in caplog.records)
    assert "SECRET-PROMPT-CANARY" not in rendered
    assert api.key not in rendered
