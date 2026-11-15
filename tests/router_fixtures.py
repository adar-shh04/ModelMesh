"""A tiny synthetic router artifact over the fake provider's models, shared by unit and integration tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from modelmesh.ml.artifacts import RouterArtifact
from modelmesh.ml.features import feature_vector, request_features
from modelmesh.ml.training.candidates import (
    Candidate,
    Table,
    base_document,
    finish_document,
    knn_params,
    raw_matrix,
)

MODELS = ["fake/echo-small", "fake/echo-large"]


def synthetic_artifact(version: str = "test-router") -> dict[str, Any]:
    """kNN over 2-d embeddings: the small model passes only near (1, 0)."""
    rng = np.random.default_rng(0)

    def table(n: int) -> Table:
        cluster = rng.integers(0, 2, n)
        unit = np.eye(2)[cluster] + rng.normal(scale=0.05, size=(n, 2))
        unit /= np.linalg.norm(unit, axis=1, keepdims=True)
        features = np.array([feature_vector(request_features([], task_type="general")) for _ in range(n)])
        acc = np.column_stack(((cluster == 0).astype(float), np.ones(n)))
        return Table(
            [f"i{i}" for i in range(n)], ["general"] * n, features, unit, np.full(n, 100.0) + np.arange(n), acc,
            np.zeros((n, 2)), np.full((n, 2), 50.0), np.full((n, 2), 30.0) + np.arange(n)[:, None],
            np.full((n, 2), 5.0),
        )  # fmt: skip

    train, validation = table(60), table(40)
    hp = {"k": 5, "temperature": 0.1, "same_task": True}
    doc = base_document(
        version=version, candidate=Candidate("knn", hp, knn_params(train, hp)), models=MODELS, train=train,
        dataset_manifest_sha256="d" * 64, split_manifest_sha256="s" * 64,
    )  # fmt: skip
    doc = finish_document(doc, validation, raw_matrix(RouterArtifact(doc), validation))
    doc["b0b"] = {"general": {"fake/echo-small": 0.5, "fake/echo-large": 1.0}}
    return doc


def write_artifact(path: Path, version: str = "test-router") -> Path:
    path.write_text(json.dumps(synthetic_artifact(version)), encoding="utf-8")
    return path
