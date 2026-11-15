"""Router artifacts: schema, loader and the numpy-only runtime predictor (ADR-0001, ADR-0004).

An artifact is one JSON document holding everything the online router needs: the feature spec, the
quality model's parameters (kNN, logistic regression, or LightGBM trees exported from ``dump_model``),
per-task isotonic calibration tables, split-conformal half-widths, the B0b lookup used as the low-confidence
fallback, and the latency, token and cost predictors. Offline evaluation predicts through this same code, so
what was evaluated is what is served.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from modelmesh.ml.features import FEATURE_NAMES, FEATURE_VERSION, feature_vector

Floats = npt.NDArray[np.float64]

ARTIFACT_SCHEMA_VERSION = 1


class ArtifactError(ValueError):
    pass


def artifact_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sigmoid(x: Floats) -> Floats:
    out: Floats = 1.0 / (1.0 + np.exp(-x))
    return out


# --------------------------------------------------------------------------- tree evaluation (LightGBM dump)
def _tree_value(node: Mapping[str, Any], row: Sequence[float]) -> float:
    while "leaf_value" not in node:
        value = row[node["split_feature"]]
        go_left = bool(node["default_left"]) if math.isnan(value) else value <= node["threshold"]
        node = node["left_child"] if go_left else node["right_child"]
    return float(node["leaf_value"])


def trees_raw_score(trees: Sequence[Mapping[str, Any]], rows: Floats) -> Floats:
    """Sum of leaf values over all trees, the same raw score LightGBM computes for numerical splits."""
    out = np.zeros(rows.shape[0])
    as_lists = rows.tolist()
    for tree in trees:
        for r, row in enumerate(as_lists):
            out[r] += _tree_value(tree, row)
    return out


# --------------------------------------------------------------------------- predictor
@dataclass(frozen=True)
class Prediction:
    model: str
    raw: float
    quality: float
    interval: tuple[float, float] | None


class RouterArtifact:
    """Loaded artifact. ``predict`` returns calibrated P(acceptable) for every trained model."""

    def __init__(self, document: Mapping[str, Any]) -> None:
        if document.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ArtifactError(f"unsupported artifact schema {document.get('schema_version')!r}")
        if document.get("feature_version") != FEATURE_VERSION or tuple(document["feature_names"]) != FEATURE_NAMES:
            raise ArtifactError("artifact was trained with a different feature spec")
        self.document = document
        self.version: str = document["version"]
        self.algorithm: str = document["algorithm"]
        self.models: list[str] = list(document["models"])
        self.task_types: set[str] = set(document["task_types"])
        self.calibrated_task_types: set[str] = set(document.get("calibrated_task_types", []))
        params = document["params"]
        self._params = params
        if "pca" in params:
            self._pca_mean = np.asarray(params["pca"]["mean"])
            self._pca_components = np.asarray(params["pca"]["components"])
        if self.algorithm == "knn":
            self._train_unit = np.asarray(params["train_embeddings"])
            self._train_task = np.asarray(params["train_task"])
            self._train_labels = np.asarray(params["train_labels"], dtype=np.float64)  # NaN = no label
            self._priors = np.asarray(params["priors"])
        elif self.algorithm == "logreg":
            self._scale_mean = np.asarray(params["scaler"]["mean"])
            self._scale_std = np.asarray(params["scaler"]["scale"])
            self._coef = np.asarray(params["coef"])
            self._intercept = np.asarray(params["intercept"])
        elif self.algorithm == "lightgbm":
            self._trees: list[dict[str, Any]] = params["trees"]
            self._model_rows = np.asarray(params["model_features"])
            self._target_encoding: dict[str, list[float]] = params["target_encoding"]
            self._te_default = np.asarray(params["target_encoding_default"])
        else:
            raise ArtifactError(f"unknown algorithm {self.algorithm!r}")
        self._train_unit_nn = np.asarray(params["nn_embeddings"]) if "nn_embeddings" in params else None
        self.tau: float = float(document.get("low_confidence_similarity", -1.0))

    @classmethod
    def load(cls, path: Path, *, expected_sha256: str | None = None) -> RouterArtifact:
        if expected_sha256 is not None and artifact_sha256(path) != expected_sha256:
            raise ArtifactError("artifact file does not match its recorded sha256")
        return cls(json.loads(path.read_text(encoding="utf-8")))

    # ---------------------------------------------------------------- raw model
    def _pca(self, embedding: Floats) -> Floats:
        out: Floats = (embedding - self._pca_mean) @ self._pca_components.T
        return out

    def raw_scores(self, features: Mapping[str, float], embedding: Sequence[float], task_type: str) -> Floats:
        """Uncalibrated P(acceptable) per trained model, in ``self.models`` order."""
        vec = np.asarray(embedding, dtype=np.float64)
        vec = vec / max(float(np.linalg.norm(vec)), 1e-12)
        if self.algorithm == "knn":
            return self._knn(vec, task_type)
        request = np.asarray(feature_vector(features))
        if self.algorithm == "logreg":
            x = np.concatenate(((request - self._scale_mean) / self._scale_std, self._pca(vec)))
            return _sigmoid(self._coef @ x + self._intercept)
        base = np.concatenate((request, self._pca(vec)))
        encoding = np.asarray(self._target_encoding.get(task_type, self._te_default))
        rows = np.hstack(
            (np.tile(base, (len(self.models), 1)), self._model_rows, encoding.reshape(-1, 1)),
        )
        return _sigmoid(trees_raw_score(self._trees, rows))

    def _knn(self, unit: Floats, task_type: str) -> Floats:
        p = self._params
        sims = self._train_unit @ unit
        if p["same_task"]:
            sims = np.where(self._train_task == task_type, sims, -np.inf)
        k = min(int(p["k"]), int(np.isfinite(sims).sum()))
        if k <= 0:
            return np.asarray(self._priors, dtype=np.float64)
        top = np.argpartition(-sims, k - 1)[:k]
        weights = np.exp((sims[top] - sims[top].max()) / float(p["temperature"]))
        labels = self._train_labels[top]  # (k, models)
        known = ~np.isnan(labels)
        mass = (weights[:, None] * known).sum(axis=0)
        total = (weights[:, None] * np.nan_to_num(labels)).sum(axis=0)
        prior_weight = float(p.get("prior_weight", 0.0))
        out: Floats = (total + prior_weight * self._priors) / np.maximum(mass + prior_weight, 1e-12)
        return np.where(mass + prior_weight > 0, out, self._priors)

    def nearest_similarity(self, embedding: Sequence[float]) -> float:
        if self._train_unit_nn is None:
            return 1.0
        vec = np.asarray(embedding, dtype=np.float64)
        vec = vec / max(float(np.linalg.norm(vec)), 1e-12)
        return float((self._train_unit_nn @ vec).max())

    # ---------------------------------------------------------------- calibration
    def calibrate(self, raw: Floats, task_type: str) -> Floats:
        tables = self.document["calibration"]
        table = tables.get(task_type) or tables.get("_global")
        if table is None:
            return raw
        out: Floats = np.interp(raw, table["x"], table["y"])
        return out

    def interval_half_width(self, task_type: str) -> float | None:
        width = self.document.get("conformal_half_width", {}).get(task_type)
        return None if width is None else float(width)

    def predict(
        self, features: Mapping[str, float], embedding: Sequence[float], task_type: str
    ) -> dict[str, Prediction]:
        raw = self.raw_scores(features, embedding, task_type)
        calibrated = self.calibrate(raw, task_type)
        width = self.interval_half_width(task_type)
        out: dict[str, Prediction] = {}
        for model, r, q in zip(self.models, raw.tolist(), calibrated.tolist(), strict=True):
            interval = None if width is None else (max(0.0, q - width), min(1.0, q + width))
            out[model] = Prediction(model, r, q, interval)
        return out

    def low_confidence(self, embedding: Sequence[float], task_type: str) -> bool:
        """ADR-0004 safe fallback: unseen task type, or nearest training neighbour below tau."""
        return task_type not in self.task_types or self.nearest_similarity(embedding) < self.tau

    # ---------------------------------------------------------------- B0b fallback and other predictors
    def lookup_quality(self, task_type: str) -> dict[str, float]:
        """B0b: training-plus-validation acceptance rate per model for the task type."""
        table: dict[str, float] = self.document["b0b"].get(task_type, {})
        return table

    def predicted_output_tokens(self, model: str, task_type: str) -> float:
        tokens = self.document["output_tokens"]
        return float(tokens.get(model, {}).get(task_type, tokens.get(model, {}).get("_all", 0.0)))

    def estimated_input_tokens(self, model: str, chars: int) -> float:
        a, b = self.document["token_fit"].get(model, [0.25, 0.0])
        return max(1.0, float(a) * chars + float(b))

    def latency_ms(self, model: str, input_tokens: float, output_tokens: float) -> tuple[float, float]:
        fit = self.document["latency"].get(model)
        if fit is None:
            return math.nan, math.nan
        mean = fit["alpha"] + fit["beta"] * input_tokens / 1000 + fit["gamma"] * output_tokens / 1000
        return math.exp(mean + fit["r50"]), math.exp(mean + fit["r90"])
