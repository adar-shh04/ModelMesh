"""M6 lifecycle: the promotion gate, allowed transitions, shadow decisions that never change routing."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from modelmesh.core.errors import ApiError
from modelmesh.ml.artifacts import RouterArtifact
from modelmesh.ml.features import request_features
from modelmesh.observability.stats import ProviderStats
from modelmesh.pipeline.services import Services
from modelmesh.providers.catalog import ModelPrice, PricingTable, fake_models
from modelmesh.routing.estimates import Estimator, NoCalibration, TokenEstimator
from modelmesh.routing.learned import (
    EMBEDDING_KEY,
    FEATURES_KEY,
    LearnedQualityPredictor,
    RouterRuntime,
    shadow_decision,
)
from modelmesh.routing.lifecycle import check_transition, promotion_blockers
from modelmesh.routing.solver import SolveInput, solve
from router_fixtures import MODELS, synthetic_artifact


def test_preliminary_or_failing_reports_cannot_promote() -> None:
    assert promotion_blockers({}) == [
        "report_is_preliminary",
        "adr_0004_success_criterion_not_met",
        "ece_above_0.05",
    ]
    assert promotion_blockers({"preliminary": True, "criterion_passed": True, "ece_calibrated": 0.01}) == [
        "report_is_preliminary"
    ]
    assert promotion_blockers({"preliminary": False, "criterion_passed": True, "ece_calibrated": 0.08}) == [
        "ece_above_0.05"
    ]
    assert promotion_blockers({"preliminary": False, "criterion_passed": True, "ece_calibrated": 0.04}) == []


@pytest.mark.parametrize(
    ("current", "target", "allowed"),
    [
        ("candidate", "shadow", True),
        ("candidate", "active", False),  # a shadow period comes first
        ("shadow", "active", True),
        ("active", "shadow", False),
        ("retired", "active", False),
        ("active", "retired", True),
    ],
)
def test_transitions(current: str, target: str, allowed: bool) -> None:
    if allowed:
        check_transition(current, target)
    else:
        with pytest.raises(ApiError) as caught:
            check_transition(current, target)
        assert caught.value.code == "invalid_transition"


def estimates() -> list[Any]:
    catalog = fake_models()
    pricing = PricingTable("t", {"fake/echo-small": ModelPrice(0.0, 0.0), "fake/echo-large": ModelPrice(1.0, 2.0)})
    estimator = Estimator(pricing, ProviderStats(), TokenEstimator(), NoCalibration())
    messages = [{"role": "user", "content": "hello"}]
    return [estimator.estimate(catalog[m], messages=messages, max_tokens=50, task_type="general") for m in MODELS]


def test_shadow_decision_reports_the_learned_choice_without_touching_the_live_ranking() -> None:
    predictor = LearnedQualityPredictor(RouterArtifact(synthetic_artifact()))
    eligible = estimates()
    request = SolveInput("cost_optimized", "general", False, 0.7, None, eligible[0].input_tokens)
    live = solve(request, eligible).ranked[0].model.id
    before = [(e.model.id, e.quality) for e in eligible]
    hard = {FEATURES_KEY: request_features([], task_type="general"), EMBEDDING_KEY: [0.0, 1.0]}
    decision = shadow_decision(predictor, request, hard, eligible, live)
    assert [(e.model.id, e.quality) for e in eligible] == before  # the live estimates are not modified
    assert decision["live_choice"] == live
    assert decision["shadow_choice"] == "fake/echo-large"  # the small model fails in this region
    easy = {FEATURES_KEY: request_features([], task_type="general"), EMBEDDING_KEY: [1.0, 0.0]}
    assert shadow_decision(predictor, request, easy, eligible, live)["shadow_choice"] == "fake/echo-small"
    # Without an explicit floor the default 0.6 floor applies on a calibrated task type.
    default = SolveInput("cost_optimized", "general", False, None, None, eligible[0].input_tokens)
    assert shadow_decision(predictor, default, hard, eligible, live)["shadow_choice"] == "fake/echo-large"
    impossible = SolveInput("cost_optimized", "general", False, 1.01, None, eligible[0].input_tokens)
    assert shadow_decision(predictor, impossible, hard, eligible, live)["would_reject"] == "min_quality"


def test_installing_an_active_router_switches_the_estimator_and_removing_it_restores_rules() -> None:
    artifact = RouterArtifact(synthetic_artifact())
    holder = SimpleNamespace(router_runtime=RouterRuntime(), estimator=SimpleNamespace(quality=NoCalibration()))
    Services.install_router(holder, "shadow", artifact)  # type: ignore[arg-type]
    assert isinstance(holder.estimator.quality, NoCalibration)  # shadow never changes routing
    assert holder.router_runtime.shadow is not None
    Services.install_router(holder, "active", artifact)  # type: ignore[arg-type]
    assert isinstance(holder.estimator.quality, LearnedQualityPredictor)
    Services.install_router(holder, "active", None)  # type: ignore[arg-type]
    assert isinstance(holder.estimator.quality, NoCalibration)
