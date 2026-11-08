"""Routing, retries, fallback, structured output and quotas through the whole pipeline (M2)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from conftest import Api, assert_matches, build_api, sse_events
from doubles import chunks, make_model, reply
from modelmesh.core.errors import ErrorClass, ProviderError
from modelmesh.routing.estimates import QualityEstimate

pytestmark = [pytest.mark.anyio, pytest.mark.integration]

STRICT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer"],
    "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
}


def fmt(schema: dict[str, Any] = STRICT_SCHEMA) -> dict[str, Any]:
    return {"type": "json_schema", "json_schema": {"name": "r", "schema": schema}}


def error(cls: ErrorClass, status: int | None = 503, **kwargs: Any) -> ProviderError:
    return ProviderError(cls, "failed", http_status=status, **kwargs)


def outcomes(body: dict[str, Any]) -> list[str]:
    return [attempt["outcome"] for attempt in body["modelmesh"]["attempts"]]


def attempts_in_error(response: Any) -> list[str]:
    return [attempt["outcome"] for attempt in response.json()["error"]["details"]["attempts"]]


CHEAP_THEN_DEAR = {"t/cheap": (0.1, 0.1), "t/dear": (5.0, 5.0)}


@pytest.fixture
def pair(api: Api) -> Api:
    api.only_models(
        make_model("cheap", size_tier="small"), make_model("dear", size_tier="large"), prices=CHEAP_THEN_DEAR
    )
    return api


# ---------------------------------------------------------------- retries and fallback
async def test_transient_error_retries_same_model_once(pair: Api) -> None:
    pair.scripted.script("cheap", error(ErrorClass.UPSTREAM_UNAVAILABLE), reply("recovered"))
    response = await pair.chat(model="t/cheap")
    assert response.status_code == 200
    assert outcomes(response.json()) == ["upstream_unavailable", "success"]
    assert pair.scripted.models_called() == ["cheap", "cheap"]


async def test_explicit_model_does_not_fall_back_by_default(pair: Api) -> None:
    pair.scripted.script("cheap", error(ErrorClass.UPSTREAM_UNAVAILABLE), error(ErrorClass.UPSTREAM_UNAVAILABLE))
    response = await pair.chat(model="t/cheap")
    assert response.status_code == 502
    assert set(pair.scripted.models_called()) == {"cheap"}
    assert_matches("ErrorResponse", response.json())


async def test_explicit_model_may_opt_in_to_fallback(pair: Api) -> None:
    pair.scripted.script("cheap", error(ErrorClass.INCOMPATIBLE, 400))
    response = await pair.chat(model="t/cheap", routing={"fallback": "within_constraints"})
    assert response.status_code == 200
    assert response.json()["modelmesh"]["model"] == "t/dear"
    assert response.json()["modelmesh"]["fallback_used"] is True
    assert response.json()["modelmesh"]["routing"]["mode"] == "explicit_model"


async def test_auto_falls_back_to_the_next_ranked_model(pair: Api) -> None:
    pair.scripted.script("cheap", error(ErrorClass.INCOMPATIBLE, 400))
    response = await pair.chat(model="auto", routing={"policy": "cost_optimized"})
    assert response.json()["modelmesh"]["model"] == "t/dear"
    assert outcomes(response.json()) == ["incompatible", "success"]


async def test_auth_error_takes_the_whole_provider_out(pair: Api) -> None:
    pair.scripted.script("cheap", error(ErrorClass.AUTH_ERROR, 401))
    response = await pair.chat(model="auto", routing={"policy": "cost_optimized"})
    assert response.status_code == 502
    assert attempts_in_error(response) == ["auth_error", "circuit_open"]
    assert pair.scripted.models_called() == ["cheap"]


async def test_fallback_none_stops_after_first_model(pair: Api) -> None:
    pair.scripted.script("cheap", error(ErrorClass.INCOMPATIBLE, 400))
    response = await pair.chat(model="auto", routing={"policy": "cost_optimized", "fallback": "none"})
    assert response.status_code == 502
    assert pair.scripted.models_called() == ["cheap"]


async def test_permanent_errors_are_never_retried(pair: Api) -> None:
    pair.scripted.script("cheap", error(ErrorClass.AUTH_ERROR, 401))
    response = await pair.chat(model="t/cheap")
    assert response.status_code == 502
    assert pair.scripted.models_called() == ["cheap"]
    assert (
        next(
            p["status"]
            for p in (await pair.client.get("/v1/providers", headers=pair.headers)).json()["data"]
            if p["id"] == "t"
        )
        == "misconfigured"
    )


async def test_translation_bug_is_a_500_without_fallback(pair: Api) -> None:
    pair.scripted.script("cheap", error(ErrorClass.INVALID_REQUEST, 400))
    response = await pair.chat(model="auto", routing={"policy": "cost_optimized"})
    assert response.status_code == 500
    assert pair.scripted.models_called() == ["cheap"]


async def test_content_filter_is_returned_without_fallback(pair: Api) -> None:
    pair.scripted.script("cheap", error(ErrorClass.CONTENT_FILTERED, 403))
    response = await pair.chat(model="auto", routing={"policy": "cost_optimized"})
    assert (response.status_code, response.json()["error"]["code"]) == (502, "content_filtered")
    assert pair.scripted.models_called() == ["cheap"]


async def test_context_length_error_falls_back_without_retry(pair: Api) -> None:
    pair.scripted.script("cheap", error(ErrorClass.CONTEXT_LENGTH_EXCEEDED, 400))
    response = await pair.chat(model="auto", routing={"policy": "cost_optimized"})
    assert response.json()["modelmesh"]["model"] == "t/dear"
    assert pair.scripted.models_called() == ["cheap", "dear"]


async def test_retry_after_is_honoured_within_the_deadline(pair: Api) -> None:
    pair.scripted.script("cheap", error(ErrorClass.RATE_LIMITED, 429, retry_after_s=0.01), reply())
    response = await pair.chat(model="t/cheap")
    assert outcomes(response.json()) == ["rate_limited", "success"]


async def test_retry_after_beyond_the_deadline_moves_on_and_blocks_the_model(pair: Api) -> None:
    pair.scripted.script("cheap", error(ErrorClass.RATE_LIMITED, 429, retry_after_s=120))
    first = await pair.chat(model="auto", routing={"policy": "cost_optimized", "timeout_ms": 2000})
    assert first.json()["modelmesh"]["model"] == "t/dear"
    pair.scripted.requests.clear()
    second = await pair.chat(model="auto", routing={"policy": "cost_optimized", "timeout_ms": 2000})
    assert second.json()["modelmesh"]["model"] == "t/dear"
    assert pair.scripted.models_called() == ["dear"]


async def test_daily_quota_marks_the_model_unavailable_until_reset(pair: Api) -> None:
    reset = datetime.now(UTC) + timedelta(hours=1)
    pair.scripted.script("cheap", error(ErrorClass.QUOTA_EXHAUSTED, 429, quota_reset_at=reset))
    first = await pair.chat(model="auto", routing={"policy": "cost_optimized"})
    assert first.json()["modelmesh"]["model"] == "t/dear"
    models = (await pair.client.get("/v1/models", headers=pair.headers)).json()["data"]
    assert next(m for m in models if m["id"] == "t/cheap")["modelmesh"]["status"] == "quota_exhausted"
    pair.scripted.requests.clear()
    await pair.chat(model="auto", routing={"policy": "cost_optimized"})
    assert pair.scripted.models_called() == ["dear"]


async def test_all_upstreams_limited_is_503_with_capacity_code(pair: Api) -> None:
    pair.scripted.script("cheap", error(ErrorClass.QUOTA_EXHAUSTED, 429))
    pair.scripted.script("dear", error(ErrorClass.QUOTA_EXHAUSTED, 429))
    response = await pair.chat(model="auto", routing={"policy": "cost_optimized"})
    assert (response.status_code, response.json()["error"]["code"]) == (503, "upstream_capacity")
    assert attempts_in_error(response) == ["quota_exhausted", "quota_exhausted"]
    # Now both are known exhausted, so the policy layer answers without calling upstream.
    pair.scripted.requests.clear()
    again = await pair.chat(model="auto", routing={"policy": "cost_optimized"})
    assert (again.status_code, again.json()["error"]["code"]) == (503, "upstream_capacity")
    assert pair.scripted.models_called() == []


async def test_deadline_expiry_is_504(pair: Api) -> None:
    pair.only_models(make_model("slow"))
    pair.scripted.delay_s = 0.5
    response = await pair.chat(model="t/slow", routing={"timeout_ms": 50})
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "deadline_exceeded"
    assert attempts_in_error(response)[0] == "upstream_timeout"


async def test_breaker_opens_and_skips_the_model(pair: Api) -> None:
    for _ in range(3):
        pair.scripted.script("cheap", error(ErrorClass.UPSTREAM_UNAVAILABLE), error(ErrorClass.UPSTREAM_UNAVAILABLE))
        await pair.chat(model="t/cheap")
    assert pair.services.breakers.for_model("t/cheap").state == "open"
    pair.scripted.requests.clear()
    response = await pair.chat(model="auto", routing={"policy": "cost_optimized"})
    assert response.json()["modelmesh"]["model"] == "t/dear"
    assert outcomes(response.json())[0] == "circuit_open"
    assert pair.scripted.models_called() == ["dear"]
    providers = (await pair.client.get("/v1/providers", headers=pair.headers)).json()
    assert_matches("ProviderStatusList", providers)
    assert next(p for p in providers["data"] if p["id"] == "t")["circuit_state"] == "open"
    assert (await pair.client.get("/v1/health")).json()["status"] == "degraded"


# ---------------------------------------------------------------- constraints never relaxed
async def test_fallback_never_violates_data_policy(api: Api) -> None:
    api.only_models(
        make_model("private", data_use="no_training", size_tier="small"),
        make_model("trains", data_use="may_train", size_tier="large"),
    )
    api.scripted.script("private", error(ErrorClass.UPSTREAM_UNAVAILABLE), error(ErrorClass.UPSTREAM_UNAVAILABLE))
    response = await api.chat(model="auto", routing={"data_policy": "no_training_providers"})
    assert response.status_code == 502
    assert "trains" not in api.scripted.models_called()


async def test_local_only_routes_only_to_local_models(api: Api) -> None:
    api.only_models(make_model("cloud", data_use="no_training"), make_model("local", local=True, size_tier="small"))
    response = await api.chat(
        model="auto", routing={"data_policy": "local_only"}, metadata={"task_type": "evidence_synthesis"}
    )
    assert response.status_code == 200
    assert response.json()["modelmesh"]["model"] == "t/local"


async def test_request_cannot_loosen_the_keys_data_policy(api: Api) -> None:
    key = await api.new_key(data_policy_ceiling="no_training_providers", limits={"rpm": 1000, "tpm": 10_000_000})
    response = await api.client.post(
        "/v1/chat/completions",
        json={
            "model": "fake/echo-small",
            "messages": [{"role": "user", "content": "x"}],
            "routing": {"data_policy": "any"},
        },
        headers={"Authorization": f"Bearer {key}"},
    )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "data_policy_exceeds_key"


async def test_key_ceiling_applies_when_request_is_silent(api: Api) -> None:
    api.only_models(make_model("trains", data_use="may_train"))
    key = await api.new_key(data_policy_ceiling="no_training_providers", limits={"rpm": 1000, "tpm": 10_000_000})
    response = await api.client.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": "x"}]},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["details"]["violated_constraints"][0]["constraint"] == "data_policy"


async def test_cost_ceiling_is_rechecked_during_fallback(api: Api) -> None:
    api.only_models(
        make_model("a", size_tier="small"), make_model("b", size_tier="large"), prices={"t/a": (1, 1), "t/b": (2, 2)}
    )
    api.scripted.script(
        "a",
        error(ErrorClass.TIMEOUT, None, possibly_billed=True),
        error(ErrorClass.TIMEOUT, None, possibly_billed=True),
    )
    # Long prompt: each possibly-billed attempt consumes most of what remains under the ceiling.
    long_text = "word " * 2000
    estimate = 0.0026
    response = await api.chat(
        model="auto",
        messages=[{"role": "user", "content": long_text}],
        max_tokens=10,
        routing={"policy": "cost_optimized", "max_cost_usd": estimate + 0.0006},
    )
    assert response.status_code == 504 or response.status_code == 502
    assert "b" not in api.scripted.models_called()


async def test_named_model_violating_a_constraint_is_422_even_with_fallback(pair: Api) -> None:
    response = await pair.chat(
        model="t/dear",
        max_tokens=4000,
        routing={"max_cost_usd": 0.0001, "fallback": "within_constraints"},
    )
    assert response.status_code == 422
    assert pair.scripted.models_called() == []


async def test_no_eligible_model_details_match_the_contract(pair: Api) -> None:
    response = await pair.chat(model="auto", max_tokens=1000, routing={"max_cost_usd": 0.0000001})
    assert response.status_code == 422
    body = response.json()
    assert_matches("ErrorResponse", body)
    violated = body["error"]["details"]["violated_constraints"]
    assert violated[0]["constraint"] == "max_cost_usd"
    assert violated[0]["excluded_candidates"] == 2
    assert violated[0]["best_available"] > 0


async def test_provider_lists_filter_candidates(pair: Api) -> None:
    denied = await pair.chat(model="auto", routing={"denied_providers": ["t"]})
    assert denied.status_code == 422
    allowed = await pair.chat(model="auto", routing={"allowed_models": ["t/dear"]})
    assert allowed.json()["modelmesh"]["model"] == "t/dear"


async def test_key_scope_blocks_a_named_model_with_403(api: Api) -> None:
    key = await api.new_key(scopes={"allowed_models": ["fake/echo-medium"]}, limits={"rpm": 1000, "tpm": 10_000_000})
    response = await api.client.post(
        "/v1/chat/completions",
        json={"model": "fake/echo-small", "messages": [{"role": "user", "content": "x"}]},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "model_not_allowed"


async def test_unknown_task_type_warns_and_becomes_general(api: Api) -> None:
    response = await api.chat(metadata={"task_type": "mystery"})
    routing = response.json()["modelmesh"]["routing"]
    assert routing["task_type"] == "general"
    assert "unknown_task_type:mystery" in routing["warnings"]


async def test_soft_param_dropped_is_reported(api: Api) -> None:
    api.only_models(make_model("nop", dropped_params=("presence_penalty",)))
    response = await api.chat(model="t/nop", presence_penalty=0.5)
    assert "param_dropped:presence_penalty" in response.json()["modelmesh"]["routing"]["warnings"]
    assert api.scripted.requests[-1].presence_penalty is None


# ---------------------------------------------------------------- calibration (contract 2.1.1)
@pytest.mark.parametrize(
    ("policy", "warning"),
    [
        ("cost_optimized", "default_min_quality_skipped:uncalibrated"),
        ("quality_balanced", "quality_uncalibrated:general"),
        ("quality_max", "quality_uncalibrated:general"),
        ("latency_optimized", None),
        ("reliability_first", None),
    ],
)
async def test_uncalibrated_requests_succeed_with_the_right_warning(api: Api, policy: str, warning: str | None) -> None:
    response = await api.chat(model="auto", routing={"policy": policy})
    assert response.status_code == 200
    routing = response.json()["modelmesh"]["routing"]
    assert routing["mode"] == "rules_fallback"
    assert routing["predicted_quality"] is None
    if warning:
        assert warning in routing["warnings"]


@pytest.mark.parametrize(
    "policy", ["cost_optimized", "quality_balanced", "quality_max", "latency_optimized", "reliability_first"]
)
async def test_explicit_min_quality_on_uncalibrated_task_is_422(api: Api, policy: str) -> None:
    response = await api.chat(model="auto", routing={"policy": policy, "min_quality": 0.5})
    assert response.status_code == 422
    assert response.json()["error"]["details"]["violated_constraints"][0]["constraint"] == "min_quality"


async def test_latency_budget_uses_prior_on_a_fresh_deployment(api: Api) -> None:
    response = await api.chat(model="auto", routing={"max_latency_ms": 60_000})
    assert response.status_code == 200
    assert "latency_prior_used" in response.json()["modelmesh"]["routing"]["warnings"]


async def test_local_only_with_small_models_routes_under_every_policy(api: Api) -> None:
    api.only_models(make_model("tiny", local=True, size_tier="small"))
    for policy in ("cost_optimized", "quality_balanced", "quality_max"):
        response = await api.chat(
            model="auto",
            routing={"policy": policy, "data_policy": "local_only"},
            metadata={"task_type": "final_response"},
        )
        assert response.status_code == 200


class StubQuality:
    router_version = "stub-1"

    def __init__(self, values: dict[str, float]) -> None:
        self.values = values

    def is_task_calibrated(self, task_type: str) -> bool:
        return True

    def predict(self, model_id: str, task_type: str, features: dict[str, Any]) -> QualityEstimate | None:
        value = self.values.get(model_id)
        return QualityEstimate(value, (value - 0.1, value + 0.1)) if value is not None else None


async def test_calibrated_quality_drives_the_learned_mode(migrated_database: str) -> None:
    api = await build_api(migrated_database, quality=StubQuality({"t/cheap": 0.5, "t/dear": 0.9}))
    try:
        api.only_models(make_model("cheap"), make_model("dear"), prices=CHEAP_THEN_DEAR)
        response = await api.chat(model="auto", routing={"policy": "cost_optimized"})
        routing = response.json()["modelmesh"]["routing"]
        assert routing["mode"] == "learned"
        assert response.json()["modelmesh"]["model"] == "t/dear"
        assert routing["predicted_quality"] == 0.9
        assert routing["predicted_quality_interval"] == pytest.approx([0.8, 1.0])
        assert routing["router_version"] == "stub-1"
        assert "default_min_quality:0.6" in routing["warnings"]
        explicit = await api.chat(model="auto", routing={"policy": "cost_optimized", "min_quality": 0.95})
        assert explicit.status_code == 422
    finally:
        await api.client.aclose()
        await api.services.aclose()


# ---------------------------------------------------------------- structured output
async def test_native_strict_output_is_reported(api: Api) -> None:
    api.only_models(make_model("s"))
    api.scripted.script("s", reply('{"answer": "yes"}'))
    response = await api.chat(model="t/s", response_format=fmt())
    assert response.json()["modelmesh"]["structured_output"] == {
        "mode": "native_strict",
        "validated": True,
        "repair_attempts": 0,
    }
    sent = api.scripted.requests[-1].response_format
    assert sent["json_schema"]["strict"] is True


async def test_best_effort_violation_is_repaired(api: Api) -> None:
    api.only_models(make_model("b", schema_mode="best_effort"))
    api.scripted.script("b", reply('{"answer": "maybe"}'), reply('{"answer": "no"}'))
    response = await api.chat(model="t/b", response_format=fmt())
    assert response.status_code == 200
    meta = response.json()["modelmesh"]
    assert meta["structured_output"] == {"mode": "native_best_effort", "validated": True, "repair_attempts": 1}
    assert [a["outcome"] for a in meta["attempts"]] == ["schema_validation_failed", "success"]
    assert api.scripted.requests[-1].messages[-1]["role"] == "user"
    assert "maybe" not in api.scripted.requests[-1].messages[-1]["content"]


async def test_emulated_mode_injects_schema_and_validates(api: Api) -> None:
    api.only_models(make_model("e", schema_mode=None, json_object=False))
    api.scripted.script("e", reply("not json"), reply('{"answer": "yes"}'))
    response = await api.chat(model="t/e", response_format=fmt())
    assert response.status_code == 200
    assert response.json()["modelmesh"]["structured_output"]["mode"] == "emulated"
    first = api.scripted.requests[0]
    assert first.messages[0]["role"] == "system"
    assert "JSON Schema" in first.messages[0]["content"]
    assert first.response_format is None


async def test_exhausted_repairs_fall_back_then_fail_with_validation_details(api: Api) -> None:
    api.only_models(
        make_model("a", size_tier="small"), make_model("b", size_tier="large"), prices={"t/a": (1, 1), "t/b": (2, 2)}
    )
    for model in ("a", "b"):
        api.scripted.script(model, reply('{"answer": "maybe"}'), reply('{"answer": "maybe"}'))
    response = await api.chat(model="auto", routing={"policy": "cost_optimized"}, response_format=fmt())
    assert response.status_code == 502
    error_body = response.json()["error"]
    assert error_body["code"] == "schema_validation_failed"
    assert error_body["retryable"] is True
    assert error_body["details"]["validation_errors"] == [{"path": "/answer", "keyword": "enum"}]
    assert "maybe" not in response.text
    assert_matches("ErrorResponse", response.json())


async def test_requiring_json_schema_excludes_emulation(api: Api) -> None:
    api.only_models(make_model("e", schema_mode=None))
    response = await api.chat(model="auto", response_format=fmt(), routing={"required_capabilities": ["json_schema"]})
    assert response.status_code == 422


async def test_requiring_strict_excludes_best_effort(api: Api) -> None:
    api.only_models(make_model("b", schema_mode="best_effort"))
    response = await api.chat(
        model="auto", response_format=fmt(), routing={"required_capabilities": ["json_schema_strict"]}
    )
    assert response.status_code == 422


async def test_non_portable_schema_cannot_be_strict(api: Api) -> None:
    api.only_models(make_model("s"))
    schema = {"type": "object", "properties": {"answer": {"type": "string", "minLength": 1}}}
    api.scripted.script("s", reply('{"answer": "x"}'))
    response = await api.chat(model="t/s", response_format=fmt(schema))
    assert response.json()["modelmesh"]["structured_output"]["mode"] == "native_best_effort"
    required = await api.chat(
        model="auto", response_format=fmt(schema), routing={"required_capabilities": ["json_schema_strict"]}
    )
    assert required.status_code == 422


async def test_json_object_format_is_validated(api: Api) -> None:
    api.only_models(make_model("j"))
    api.scripted.script("j", reply("[1,2]"), reply('{"ok": true}'))
    response = await api.chat(model="t/j", response_format={"type": "json_object"})
    assert response.json()["modelmesh"]["structured_output"]["repair_attempts"] == 1


async def test_buffered_streaming_when_model_cannot_stream_structured(api: Api) -> None:
    api.only_models(make_model("b", schema_stream=False))
    api.scripted.script("b", reply('{"answer": "yes"}'))
    response = await api.chat(model="t/b", response_format=fmt(), stream=True, stream_options={"include_usage": True})
    events = sse_events(response.text)
    assert events[-1] == "[DONE]"
    assert events[-2]["modelmesh"]["streaming"] == {"mode": "buffered"}
    text = "".join(e["choices"][0]["delta"].get("content", "") for e in events[:-1] if e["choices"])
    assert json.loads(text) == {"answer": "yes"}
    for event in events[:-1]:
        assert_matches("ChatCompletionChunk", event)


async def test_buffered_mode_excluded_when_streaming_is_required(api: Api) -> None:
    api.only_models(make_model("b", schema_stream=False))
    response = await api.chat(
        model="auto", response_format=fmt(), stream=True, routing={"required_capabilities": ["streaming"]}
    )
    assert response.status_code == 422


async def test_native_structured_stream_invalid_output_ends_with_error_event(api: Api) -> None:
    api.only_models(make_model("n"))
    api.scripted.script("n", chunks('{"answer": "maybe"}'))
    events = sse_events((await api.chat(model="t/n", response_format=fmt(), stream=True)).text)
    assert "[DONE]" not in events
    assert events[-1]["error"]["code"] == "schema_validation_failed"
    assert events[-1]["error"]["details"]["validation_errors"][0]["keyword"] == "enum"


async def test_tool_choice_required_excludes_models_without_it(api: Api) -> None:
    api.only_models(make_model("no", tool_choice=False))
    tools = [{"type": "function", "function": {"name": "f"}}]
    response = await api.chat(model="auto", tools=tools, tool_choice="required")
    assert response.status_code == 422
