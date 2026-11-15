"""Learned-router runtime (ADR-0004): a QualityPredictor backed by a router artifact, plus shadow decisions.

The artifact's calibrated P(acceptable) replaces the missing quality estimate. When the request is
low-confidence (task type unseen in training, or nearest training neighbour below tau) the B0b per-task
lookup is used instead and the decision is reported as ``rules_fallback``. Models without benchmark labels
get no estimate, so the solver ranks them after models with evidence.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

from modelmesh.ml.artifacts import Prediction, RouterArtifact
from modelmesh.ml.features import embedding_text, request_features
from modelmesh.routing.estimates import CandidateEstimate, QualityEstimate
from modelmesh.routing.solver import SolveInput, solve

FEATURES_KEY = "request"
EMBEDDING_KEY = "embedding"


def routing_features(
    messages: Sequence[Mapping[str, Any]],
    *,
    task_type: str,
    response_format: Mapping[str, Any] | None,
    max_tokens: int | None,
    tool_count: int,
    embedding: Sequence[float],
) -> dict[str, Any]:
    """The ``features`` mapping the Estimator passes to the predictor."""
    return {
        FEATURES_KEY: request_features(
            messages, task_type=task_type, response_format=response_format, max_tokens=max_tokens, tool_count=tool_count
        ),
        EMBEDDING_KEY: list(embedding),
    }


def text_to_embed(messages: Sequence[Mapping[str, Any]]) -> str:
    return embedding_text(messages)


class LearnedQualityPredictor:
    """Implements routing.estimates.QualityPredictor on top of a loaded artifact."""

    def __init__(self, artifact: RouterArtifact) -> None:
        self.artifact = artifact
        self.router_version: str | None = artifact.version
        # The features mapping of the request being routed is kept alive, so identity cannot be reused.
        self._last_features: Mapping[str, Any] | None = None
        self._last: tuple[dict[str, Prediction], bool] | None = None

    def is_task_calibrated(self, task_type: str) -> bool:
        return task_type in self.artifact.calibrated_task_types

    def predictions(self, task_type: str, features: Mapping[str, Any]) -> tuple[dict[str, Prediction], bool]:
        """(per-model predictions, low_confidence). Memoised for the features mapping of the current request."""
        if self._last_features is features and self._last is not None:
            return self._last
        embedding = features[EMBEDDING_KEY]
        low = self.artifact.low_confidence(embedding, task_type)
        if low:
            table = self.artifact.lookup_quality(task_type)
            predictions = {m: Prediction(m, q, q, None) for m, q in table.items()}
        else:
            predictions = self.artifact.predict(features[FEATURES_KEY], embedding, task_type)
        self._last_features, self._last = features, (predictions, low)
        return predictions, low

    def predict(self, model_id: str, task_type: str, features: Mapping[str, Any]) -> QualityEstimate | None:
        if EMBEDDING_KEY not in features or FEATURES_KEY not in features:
            return None
        prediction = self.predictions(task_type, features)[0].get(model_id)
        if prediction is None:
            return None
        return QualityEstimate(prediction.quality, prediction.interval)


def with_learned_quality(
    predictor: LearnedQualityPredictor, task_type: str, features: Mapping[str, Any], eligible: list[CandidateEstimate]
) -> tuple[list[CandidateEstimate], bool]:
    predictions, low = predictor.predictions(task_type, features)
    updated = []
    for estimate in eligible:
        prediction = predictions.get(estimate.model.id)
        updated.append(
            dataclasses.replace(
                estimate,
                quality=prediction.quality if prediction else None,
                quality_interval=prediction.interval if prediction else None,
            )
        )
    return updated, low


def shadow_decision(
    predictor: LearnedQualityPredictor,
    request: SolveInput,
    features: Mapping[str, Any],
    eligible: list[CandidateEstimate],
    live_choice: str,
) -> dict[str, Any]:
    """What the learned router would have chosen. Never used to route; logged for comparison only."""
    estimates, low = with_learned_quality(predictor, request.task_type, features, eligible)
    calibrated = predictor.is_task_calibrated(request.task_type)
    solved = solve(dataclasses.replace(request, calibrated=calibrated), estimates)
    chosen = solved.ranked[0]
    return {
        "router_version": predictor.router_version,
        "mode": "rules_fallback" if low else solved.mode,
        "policy": request.policy,
        "task_type": request.task_type,
        "shadow_choice": chosen.model.id,
        "live_choice": live_choice,
        "agrees": chosen.model.id == live_choice,
        "predicted_quality": chosen.quality,
        "reason": solved.reason,
    }
