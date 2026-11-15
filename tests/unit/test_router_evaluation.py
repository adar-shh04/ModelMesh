"""M6 acceptance test on synthetic data: the evaluation code recovers a known Oracle and hand-computed AIQ."""

from __future__ import annotations

import numpy as np
import pytest

from modelmesh.ml.evaluation.metrics import (
    EvalSet,
    aiq,
    bootstrap,
    brier,
    ece,
    min_cost_for_quality,
    oracle_choice,
    route_cost_optimized,
    upper_hull,
)

# 4 items x 3 models. Costs per call: m0 = 0, m1 = 1, m2 = 3.
ACC = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=np.float64)
COST = np.tile(np.array([0.0, 1.0, 3.0]), (4, 1))


def test_oracle_is_the_cheapest_accepted_model_and_cheapest_when_none_is() -> None:
    assert oracle_choice(ACC, COST).tolist() == [0, 1, 2, 0]


def test_aiq_matches_a_hand_computed_hull() -> None:
    # Anchor (0, 0.2) plus (1, 0.6), (2, 0.7), (3, 0.9). (2, 0.7) lies under the chord from (1, 0.6) to (3, 0.9).
    hull = upper_hull(np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.2, 0.6, 0.7, 0.9]))
    assert hull[0].tolist() == [0.0, 1.0, 3.0]
    assert hull[1].tolist() == [0.2, 0.6, 0.9]
    # Over [0, 4]: 0.5*(0.2+0.6)*1 + 0.5*(0.6+0.9)*2 + 0.9*1 = 2.8, divided by 4.
    assert aiq(hull, 0.0, 4.0) == pytest.approx(0.7)
    assert min_cost_for_quality(hull, 0.75) == pytest.approx(2.0)
    assert np.isnan(min_cost_for_quality(hull, 0.95))


def test_dominated_points_never_enter_the_hull() -> None:
    hull = upper_hull(np.array([0.0, 1.0, 2.0]), np.array([0.5, 0.4, 0.8]))
    assert hull[0].tolist() == [0.0, 2.0]


def test_eval_set_recovers_oracle_zero_router_and_perfect_router_aiq() -> None:
    data = EvalSet(ACC, COST, cheapest=0, priciest=2)
    assert data.cost_range() == (0.0, 3.0)
    oracle = oracle_choice(ACC, COST)
    # Oracle point (1.0, 0.75) with anchor (0, 0.25): (0.5*(0.25+0.75)*1 + 0.75*2) / 3 = 2/3.
    assert data.single(0) == (0.0, 0.25)
    assert data.router_aiq(oracle) == pytest.approx(2 / 3)
    # Zero router: (0, .25), (1, .5), (3, .75) -> (0.375 + 1.25) / 3.
    assert aiq(data.zero_router_hull(), 0.0, 3.0) == pytest.approx(1.625 / 3)
    # A router that knows the labels exactly reproduces the Oracle at every threshold.
    sweep = np.stack([route_cost_optimized(ACC, COST, t) for t in (0.0, 0.5, 1.01)])
    assert sweep[0].tolist() == [0, 0, 0, 0]
    assert sweep[1].tolist() == oracle.tolist()
    assert sweep[2].tolist() == oracle.tolist()
    assert data.router_aiq(sweep) == pytest.approx(2 / 3)


def test_weights_reweight_the_operating_points() -> None:
    weights = np.array([1.0, 1.0, 1.0, 5.0])
    data = EvalSet(ACC, COST, cheapest=0, priciest=2, weights=weights)
    assert data.single(2) == (3.0, pytest.approx(3 / 8))


def test_calibration_metrics_match_hand_values() -> None:
    prob = np.array([0.1, 0.1, 0.9, 0.9])
    label = np.array([0.0, 1.0, 1.0, 1.0])
    assert ece(prob, label) == pytest.approx(0.25)
    assert brier(prob, label) == pytest.approx(0.21)


def test_bootstrap_is_deterministic_and_brackets_the_point_estimate() -> None:
    values = np.arange(50, dtype=np.float64)
    first = bootstrap(lambda idx: np.array([values[idx].mean()]), 50, resamples=300, seed=3)
    second = bootstrap(lambda idx: np.array([values[idx].mean()]), 50, resamples=300, seed=3)
    assert first == second
    assert first[0].low < first[0].point < first[0].high


def test_vectorised_outcomes_agree_with_the_reference_path() -> None:
    from modelmesh.ml.evaluation.metrics import Outcomes

    rng = np.random.default_rng(7)
    acc = (rng.random((40, 4)) > 0.4).astype(np.float64)
    cost = rng.random((40, 4)) * np.array([0.0, 1.0, 2.0, 4.0])
    qhat = np.clip(acc * 0.6 + rng.random((40, 4)) * 0.4, 0, 1)
    sweep = np.stack([route_cost_optimized(qhat, cost, t) for t in np.linspace(0, 1, 11)])
    data = EvalSet(acc, cost, cheapest=0, priciest=3)
    outcomes = Outcomes.from_choices(sweep, acc, cost)
    idx = np.arange(40)
    hull = outcomes.hull(idx, data.single(0))
    assert aiq(hull, *data.cost_range()) == pytest.approx(data.router_aiq(sweep))
