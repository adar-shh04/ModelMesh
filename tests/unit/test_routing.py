"""Policies, rules baseline and hard-constraint filtering on a synthetic catalog."""

from __future__ import annotations

from typing import Any

import pytest

from doubles import make_model
from modelmesh.core.errors import ApiError
from modelmesh.observability.stats import ProviderStats
from modelmesh.policy.engine import (
    CapabilityNeeds,
    Constraints,
    effective_data_policy,
    filter_candidates,
    plan_structured,
)
from modelmesh.providers.catalog import PricingTable
from modelmesh.routing.estimates import CandidateEstimate, Estimator, NoCalibration, TokenEstimator
from modelmesh.routing.solver import SolveInput, rank_by_rules, rules_tier, solve


def estimate(
    name: str,
    *,
    cost: float = 0.01,
    p50: int = 100,
    p90: int = 200,
    quality: float | None = None,
    fail: float = 0.0,
    **model: Any,
) -> CandidateEstimate:
    return CandidateEstimate(
        model=make_model(name, **model),
        input_tokens=100,
        predicted_output_tokens=50,
        upper_bound_output_tokens=500,
        predicted_cost_usd=cost,
        upper_bound_cost_usd=cost * 10,
        latency_p50_ms=p50,
        latency_p90_ms=p90,
        latency_prior_used=False,
        quality=quality,
        quality_interval=None,
        failure_probability=fail,
    )


def inputs(policy: str, *, calibrated: bool = True, min_quality: float | None = None, task: str = "general", **kw: Any):
    return SolveInput(
        policy=policy,
        task_type=task,
        calibrated=calibrated,
        explicit_min_quality=min_quality,
        max_latency_ms=kw.get("max_latency_ms"),
        input_tokens=kw.get("input_tokens", 100),
        schema_properties=kw.get("schema_properties", 0),
    )


def names(result: Any) -> list[str]:
    return [e.model.id.split("/")[1] for e in result.ranked]


def test_cost_optimized_applies_default_floor_when_calibrated() -> None:
    pool = [estimate("cheap", cost=0.001, quality=0.5), estimate("mid", cost=0.01, quality=0.7)]
    result = solve(inputs("cost_optimized"), pool)
    assert names(result)[0] == "mid"
    assert "default_min_quality:0.6" in result.warnings
    assert result.mode == "learned"


def test_cost_optimized_default_floor_dropped_when_nothing_meets_it() -> None:
    pool = [estimate("a", cost=0.002, quality=0.3), estimate("b", cost=0.001, quality=0.4)]
    result = solve(inputs("cost_optimized"), pool)
    assert names(result)[0] == "b"
    assert "default_min_quality_unmet:0.6" in result.warnings


def test_cost_optimized_uncalibrated_has_no_floor_and_says_so() -> None:
    pool = [estimate("expensive", cost=0.1), estimate("cheap", cost=0.001)]
    result = solve(inputs("cost_optimized", calibrated=False), pool)
    assert names(result) == ["cheap", "expensive"]
    assert result.warnings == ["default_min_quality_skipped:uncalibrated"]
    assert result.mode == "rules_fallback"


def test_quality_balanced_picks_cheapest_within_delta() -> None:
    pool = [
        estimate("best", cost=0.1, quality=0.9),
        estimate("close", cost=0.01, quality=0.87),
        estimate("far", cost=0.001, quality=0.7),
    ]
    assert names(solve(inputs("quality_balanced"), pool)) == ["close", "best", "far"]


def test_quality_balanced_ranks_models_without_evidence_last() -> None:
    pool = [estimate("noev", cost=0.0001), estimate("ev", cost=0.1, quality=0.8)]
    assert names(solve(inputs("quality_balanced"), pool)) == ["ev", "noev"]


def test_quality_balanced_uses_rules_when_uncalibrated() -> None:
    pool = [
        estimate("s", cost=0.001, size_tier="small"),
        estimate("m", cost=0.01, size_tier="medium"),
        estimate("l", cost=0.1, size_tier="large"),
    ]
    result = solve(inputs("quality_balanced", calibrated=False, task="general"), pool)
    assert names(result)[0] == "m"
    assert result.warnings == ["quality_uncalibrated:general"]
    assert result.mode == "rules_fallback"


def test_quality_max_uncalibrated_takes_largest_tier_cheapest_within() -> None:
    pool = [
        estimate("l1", cost=0.2, size_tier="large"),
        estimate("l2", cost=0.1, size_tier="large"),
        estimate("s", cost=0.001, size_tier="small"),
    ]
    result = solve(inputs("quality_max", calibrated=False), pool)
    assert names(result) == ["l2", "l1", "s"]
    assert result.warnings == ["quality_uncalibrated:general"]


def test_quality_max_calibrated() -> None:
    pool = [estimate("a", quality=0.7), estimate("b", quality=0.9), estimate("c")]
    assert names(solve(inputs("quality_max"), pool)) == ["b", "a", "c"]


def test_latency_optimized() -> None:
    pool = [estimate("slow", p50=900), estimate("fast", p50=50)]
    result = solve(inputs("latency_optimized", calibrated=False), pool)
    assert names(result) == ["fast", "slow"]
    assert result.warnings == []


def test_reliability_first_tie_breaks_on_cost() -> None:
    pool = [estimate("flaky", fail=0.4, cost=0.001), estimate("a", cost=0.05), estimate("b", cost=0.01)]
    assert names(solve(inputs("reliability_first", calibrated=False), pool)) == ["b", "a", "flaky"]


def test_latency_prior_warning_only_when_budget_is_set() -> None:
    prior = [
        CandidateEstimate(
            **{**estimate("a").__dict__, "latency_prior_used": True},
        )
    ]
    assert (
        "latency_prior_used" in solve(inputs("latency_optimized", calibrated=False, max_latency_ms=500), prior).warnings
    )
    assert "latency_prior_used" not in solve(inputs("latency_optimized", calibrated=False), prior).warnings


def test_rules_tier_table_and_escalation() -> None:
    assert rules_tier("query_classification", 100, 0) == "small"
    assert rules_tier("evidence_assessment", 100, 0) == "small"
    assert rules_tier("citation_verification", 100, 0) == "medium"
    assert rules_tier("evidence_synthesis", 100, 0) == "large"
    assert rules_tier("evaluation_judge", 100, 0) == "large"
    assert rules_tier("query_classification", 9000, 0) == "medium"
    assert rules_tier("query_classification", 100, 11) == "medium"
    assert rules_tier("final_response", 9000, 0) == "large"


def test_rules_prefers_nearest_tier_and_larger_first() -> None:
    small = estimate("s", size_tier="small")
    large = estimate("l", size_tier="large")
    assert [e.model.id for e in rank_by_rules([small, large], "medium")] == ["t/l", "t/s"]
    assert [e.model.id for e in rank_by_rules([small], "large")] == ["t/s"]


def test_local_only_with_small_models_still_routes() -> None:
    pool = [estimate("local", size_tier="small", local=True)]
    result = solve(inputs("quality_balanced", calibrated=False, task="evidence_synthesis"), pool)
    assert names(result) == ["local"]


def test_filter_reports_violations_with_best_available() -> None:
    pool = [estimate("a", cost=0.01), estimate("b", cost=0.5, data_use="may_train")]
    with pytest.raises(ApiError) as info:
        filter_candidates(pool, Constraints(max_cost_usd=0.01, data_policy="no_training_providers"))
    assert info.value.error_type == "no_eligible_model"
    violated = {v["constraint"]: v for v in info.value.details["violated_constraints"]}
    assert violated["max_cost_usd"]["excluded_candidates"] == 2
    assert violated["max_cost_usd"]["best_available"] == pytest.approx(0.1)
    assert violated["data_policy"]["best_available"] is None
    assert info.value.details["candidates_considered"] == 2


def test_only_quota_failures_are_a_capacity_error() -> None:
    with pytest.raises(ApiError) as info:
        filter_candidates([estimate("a")], Constraints(quota_unavailable={"t/a"}))
    assert (info.value.status_code, info.value.code) == (503, "upstream_capacity")


def test_only_budget_failures_are_a_402() -> None:
    with pytest.raises(ApiError) as info:
        filter_candidates([estimate("a", cost=1.0)], Constraints(budget_remaining_usd=0.1))
    assert info.value.status_code == 402


def test_only_context_failures_are_a_413() -> None:
    with pytest.raises(ApiError) as info:
        filter_candidates([estimate("a", context_window=10)], Constraints())
    assert (info.value.status_code, info.value.code) == (413, "context_too_large")


@pytest.mark.parametrize(
    ("requested", "ceiling", "expected"),
    [
        (None, "local_only", "local_only"),
        ("local_only", "any", "local_only"),
        ("any", "any", "any"),
    ],
)
def test_effective_data_policy_only_tightens(requested, ceiling, expected) -> None:
    assert effective_data_policy(requested, ceiling) == expected


def test_data_policy_cannot_loosen_the_key_ceiling() -> None:
    with pytest.raises(ApiError) as info:
        effective_data_policy("any", "no_training_providers")
    assert info.value.status_code == 403


def test_local_only_excludes_every_remote_model() -> None:
    pool = [estimate("remote", data_use="no_training"), estimate("local", local=True)]
    assert [e.model.id for e in filter_candidates(pool, Constraints(data_policy="local_only"))] == ["t/local"]


def test_unknown_data_use_is_excluded_under_no_training() -> None:
    pool = [estimate("u", data_use="unknown"), estimate("y", data_use="no_training")]
    assert [e.model.id for e in filter_candidates(pool, Constraints(data_policy="no_training_providers"))] == ["t/y"]


def test_explicit_min_quality_excludes_models_without_evidence() -> None:
    with pytest.raises(ApiError) as info:
        filter_candidates([estimate("a")], Constraints(min_quality=0.5))
    assert info.value.status_code == 422


SCHEMA = CapabilityNeeds(response_format_kind="json_schema", schema_portable=True)


@pytest.mark.parametrize(
    ("model_kwargs", "needs", "expected"),
    [
        ({"schema_mode": "strict"}, SCHEMA, ("native_strict", False)),
        ({"schema_mode": "best_effort"}, SCHEMA, ("native_best_effort", False)),
        ({"schema_mode": None}, SCHEMA, ("emulated", False)),
        ({"schema_mode": None, "json_object": True}, SCHEMA, ("emulated", False)),
        (
            {"schema_mode": "strict"},
            CapabilityNeeds(response_format_kind="json_schema", schema_portable=False),
            ("native_best_effort", False),
        ),
        (
            {"schema_mode": "strict", "schema_stream": False},
            CapabilityNeeds(response_format_kind="json_schema", schema_portable=True, stream=True),
            ("native_strict", True),
        ),
        (
            {"schema_mode": "strict", "schema_stream": True},
            CapabilityNeeds(response_format_kind="json_schema", schema_portable=True, stream=True),
            ("native_strict", False),
        ),
        (
            {"schema_mode": "strict", "schema_with_tools": False},
            CapabilityNeeds(response_format_kind="json_schema", schema_portable=True, has_tools=True),
            ("emulated", False),
        ),
        ({"json_object": True}, CapabilityNeeds(response_format_kind="json_object"), ("native_best_effort", False)),
        ({"json_object": False}, CapabilityNeeds(response_format_kind="json_object"), ("emulated", False)),
    ],
)
def test_structured_plans(model_kwargs, needs, expected) -> None:
    plan = plan_structured(make_model("m", **model_kwargs), needs)
    assert plan is not None
    assert (plan.mode, plan.buffered) == expected


@pytest.mark.parametrize(
    ("model_kwargs", "required"),
    [
        ({"schema_mode": None}, {"json_schema"}),
        ({"schema_mode": "best_effort"}, {"json_schema_strict"}),
    ],
)
def test_requiring_native_modes_excludes_lesser_ones(model_kwargs, required) -> None:
    needs = CapabilityNeeds(response_format_kind="json_schema", schema_portable=True, required=frozenset(required))
    assert plan_structured(make_model("m", **model_kwargs), needs) is None


def test_requiring_streaming_excludes_buffered_mode() -> None:
    needs = CapabilityNeeds(
        response_format_kind="json_schema", schema_portable=True, stream=True, required=frozenset({"streaming"})
    )
    assert plan_structured(make_model("m", schema_stream=False), needs) is None


def test_estimates_use_prior_until_telemetry_exists() -> None:
    stats = ProviderStats()
    estimator = Estimator(PricingTable("v"), stats, TokenEstimator(), NoCalibration())
    model = make_model("m", latency_ttft_ms=100, latency_tokens_per_s=100.0)
    messages = [{"role": "user", "content": "x" * 400}]
    prior = estimator.estimate(model, messages=messages, max_tokens=100, task_type="general")
    assert prior.latency_prior_used
    assert prior.latency_p50_ms == 100 + 1000
    assert prior.input_tokens == 105
    for tokens in range(1, 31):
        stats.record("t", model.id, ok=True, latency_ms=50 + 2 * tokens, output_tokens=tokens)
    measured = estimator.estimate(model, messages=messages, max_tokens=100, task_type="general")
    assert not measured.latency_prior_used
    assert measured.latency_p50_ms == pytest.approx(50 + 2 * 100, abs=1)


def test_token_estimator_learns_from_reported_usage() -> None:
    tokens = TokenEstimator(smoothing=1.0)
    tokens.observe("fam", 1000, 500)
    assert tokens.estimate_text("x" * 100, "fam") == 51
    assert tokens.estimate_text("x" * 100, "other") == 26
