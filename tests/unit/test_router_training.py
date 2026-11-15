"""M6: features, leakage-free splits, candidates and the runtime artifact, on synthetic data (offline only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from modelmesh.ml.artifacts import ArtifactError, RouterArtifact
from modelmesh.ml.benchmark.items import BenchItem
from modelmesh.ml.evaluation.metrics import EvalSet, oracle_choice, route_cost_optimized
from modelmesh.ml.features import FEATURE_NAMES, embedding_text, feature_vector, request_features, schema_stats
from modelmesh.ml.training.candidates import (
    Candidate,
    ModelInfo,
    Table,
    base_document,
    finish_document,
    knn_params,
    raw_matrix,
)
from modelmesh.ml.training.splits import assert_no_leakage, build_groups, grouped_split, leakage

MODELS = ["cheap/a", "pricey/b"]


def item(i: int, task: str = "general", source: str = "src", key: str | None = None) -> BenchItem:
    return BenchItem(
        item_id=f"{source}_{task}:{key or i}",
        source=source,
        task_type=task,
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": f"question {i}?"}],
        label_kind="classification",
    )


# --------------------------------------------------------------------------- features
def test_request_features_follow_the_adr_spec_and_ignore_client_identity() -> None:
    schema = {
        "type": "object",
        "required": ["a"],
        "properties": {"a": {"type": "string", "enum": ["x"]}, "b": {"type": "object", "properties": {"c": {}}}},
    }
    features = request_features(
        [{"role": "system", "content": "be brief"}, {"role": "user", "content": "What is 2+2? Really?"}],
        task_type="general",
        response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": schema}},
        max_tokens=64,
    )
    assert set(features) == set(FEATURE_NAMES)
    assert features["task=general"] == 1.0 and features["task=final_response"] == 0.0
    assert features["response=json_schema"] == 1.0
    assert (features["schema_properties"], features["schema_depth"], features["schema_required"]) == (3, 2, 1)
    assert features["schema_has_enum"] == 1.0
    assert features["question_count"] == 2.0
    assert len(feature_vector(features)) == len(FEATURE_NAMES)
    assert schema_stats(None) == (0, 0, 0, False)
    assert "metadata" not in " ".join(FEATURE_NAMES) and "application" not in " ".join(FEATURE_NAMES)


def test_embedding_text_is_system_prompt_plus_final_user_message() -> None:
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "A"},
        {"role": "user", "content": "last"},
    ]
    assert embedding_text(messages) == "S\nlast"


# --------------------------------------------------------------------------- splits
def test_groups_join_near_duplicates_and_shared_source_questions() -> None:
    items = [item(0), item(1), item(2, task="other", key="1"), item(3)]
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(4, 16))
    emb[3] = emb[0] + 1e-4  # near-duplicate of item 0
    groups = build_groups(items, emb)
    assert groups[0] == groups[3]  # cosine > 0.95
    assert groups[1] == groups[2]  # same source question id under two task types
    assert len(set(groups)) == 2


def test_grouped_split_has_no_group_or_near_duplicate_leakage() -> None:
    rng = np.random.default_rng(1)
    items = [item(i, task=("t1", "t2")[i % 2], source=("s1", "s2")[i % 3 == 0]) for i in range(120)]
    emb = rng.normal(size=(120, 32))
    for a, b in [(0, 7), (10, 11), (40, 90)]:
        emb[b] = emb[a] + rng.normal(scale=1e-3, size=32)
    groups = build_groups(items, emb)
    assignment = grouped_split(items, groups, seed=5)
    report = assert_no_leakage(assignment, groups, emb)
    assert report.groups_crossing == 0
    counts = {s: assignment.count(s) for s in ("train", "validation", "test")}
    assert counts["train"] > counts["validation"] > 0 and counts["test"] > 0
    assert grouped_split(items, groups, seed=5) == assignment  # deterministic


def test_leakage_check_catches_a_split_group_and_a_near_duplicate() -> None:
    emb = np.eye(3)
    emb[2] = emb[0]
    assert leakage(["train", "test", "test"], [0, 1, 0], emb).groups_crossing == 1
    with pytest.raises(AssertionError):
        assert_no_leakage(["train", "test", "test"], [0, 1, 2], emb)


def test_zero_fraction_split_gets_nothing() -> None:
    items = [item(i) for i in range(10)]
    assert "test" not in grouped_split(items, list(range(10)), seed=1, fractions=(0.75, 0.25, 0.0))


# --------------------------------------------------------------------------- synthetic end to end
def synthetic(n: int, seed: int) -> tuple[Table, np.ndarray[Any, Any]]:
    """Two clusters of prompts. The cheap model passes only on cluster A; the expensive one always passes."""
    rng = np.random.default_rng(seed)
    cluster = rng.integers(0, 2, n)
    centres = np.array([np.r_[1.0, np.zeros(15)], np.r_[0.0, 1.0, np.zeros(14)]])
    unit = centres[cluster] + rng.normal(scale=0.05, size=(n, 16))
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    acc = np.column_stack(((cluster == 0).astype(float), np.ones(n)))
    cost = np.tile([0.0, 1e-4], (n, 1))
    features = np.array([feature_vector(request_features([], task_type="general")) for _ in range(n)])
    table = Table(
        [f"i{i}" for i in range(n)], ["general"] * n, features, unit, np.full(n, 400.0), acc, cost,
        np.full((n, 2), 500.0), np.full((n, 2), 100.0), np.full((n, 2), 10.0),
    )  # fmt: skip
    return table, cluster


def document(train: Table, validation: Table, candidate: Candidate) -> dict[str, Any]:
    doc = base_document(
        version="synthetic", candidate=candidate, models=MODELS, train=train,
        dataset_manifest_sha256="d", split_manifest_sha256="s",
    )  # fmt: skip
    return finish_document(doc, validation, raw_matrix(RouterArtifact(doc), validation))


def test_knn_router_recovers_the_oracle_on_separable_synthetic_data() -> None:
    train, _ = synthetic(80, 0)
    validation, _ = synthetic(40, 1)
    test, _ = synthetic(60, 2)
    doc = document(
        train, validation, Candidate("knn", {}, knn_params(train, {"k": 5, "temperature": 0.1, "same_task": True}))
    )
    artifact = RouterArtifact(json.loads(json.dumps(doc)))  # survives a JSON round trip
    qhat = np.vstack([artifact.calibrate(row, "general") for row in raw_matrix(artifact, test)])
    choice = route_cost_optimized(qhat, test.cost, 0.6)
    assert choice.tolist() == oracle_choice(test.acc, test.cost).tolist()
    data = EvalSet(test.acc, test.cost, cheapest=0, priciest=1)
    sweep = np.stack([route_cost_optimized(qhat, test.cost, t) for t in np.linspace(0, 1, 21)])
    assert data.router_aiq(sweep) == pytest.approx(data.router_aiq(oracle_choice(test.acc, test.cost)))


def test_lightgbm_runtime_trees_match_lightgbm_and_logreg_matches_sklearn() -> None:
    pytest.importorskip("lightgbm")
    pytest.importorskip("sklearn")
    from modelmesh.ml.training.candidates import lightgbm_params, logreg_params, model_feature_rows

    train, _ = synthetic(80, 3)
    train.acc[::7, 1] = 0.0  # give the expensive model some failures so both classes exist
    infos = [ModelInfo(m, m.split("/")[0], m.split("/")[0], 8192, 0.1 * k, 0.2 * k) for k, m in enumerate(MODELS)]
    hp = {"n_estimators": 30, "num_leaves": 4, "min_child_samples": 5, "learning_rate": 0.1}
    params = lightgbm_params(train, hp, infos, seed=1)
    booster = params["_booster"]
    doc = base_document(
        version="x", candidate=Candidate("lightgbm", hp, params), models=MODELS, train=train,
        dataset_manifest_sha256="d", split_manifest_sha256="s",
    )  # fmt: skip
    artifact = RouterArtifact(doc)
    test, _ = synthetic(10, 4)
    pca = params["pca"]
    projected = (test.unit - np.asarray(pca["mean"])) @ np.asarray(pca["components"]).T
    encoding = np.asarray(params["target_encoding"]["general"])
    rows = [
        np.concatenate((test.features[i], projected[i], model_feature_rows(infos)[j], [encoding[j]]))
        for i in range(10)
        for j in range(2)
    ]
    expected = booster.predict_proba(np.asarray(rows))[:, 1].reshape(10, 2)
    assert np.allclose(raw_matrix(artifact, test), expected, atol=1e-9)

    lr = logreg_params(train, {"C": 1.0}, seed=1)
    lr_doc = base_document(
        version="y", candidate=Candidate("logreg", {"C": 1.0}, lr), models=MODELS, train=train,
        dataset_manifest_sha256="d", split_manifest_sha256="s",
    )  # fmt: skip
    from sklearn.linear_model import LogisticRegression

    x_train = np.hstack(
        (
            (train.features - np.asarray(lr["scaler"]["mean"])) / np.asarray(lr["scaler"]["scale"]),
            (train.unit - np.asarray(lr["pca"]["mean"])) @ np.asarray(lr["pca"]["components"]).T,
        )
    )
    reference = LogisticRegression(C=1.0, max_iter=2000, random_state=1).fit(x_train, train.acc[:, 1])
    x_test = np.hstack(
        (
            (test.features - np.asarray(lr["scaler"]["mean"])) / np.asarray(lr["scaler"]["scale"]),
            (test.unit - np.asarray(lr["pca"]["mean"])) @ np.asarray(lr["pca"]["components"]).T,
        )
    )
    assert np.allclose(raw_matrix(RouterArtifact(lr_doc), test)[:, 1], reference.predict_proba(x_test)[:, 1])


def test_artifact_rejects_a_changed_file_or_feature_spec(tmp_path: Path) -> None:
    train, _ = synthetic(20, 5)
    doc = document(
        train,
        synthetic(10, 6)[0],
        Candidate("knn", {}, knn_params(train, {"k": 3, "temperature": 0.1, "same_task": False})),
    )
    path = tmp_path / "router.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ArtifactError):
        RouterArtifact.load(path, expected_sha256="0" * 64)
    with pytest.raises(ArtifactError):
        RouterArtifact({**doc, "feature_names": ["other"]})


def test_learned_predictor_falls_back_to_the_lookup_and_skips_unlabelled_models() -> None:
    from modelmesh.routing.learned import EMBEDDING_KEY, FEATURES_KEY, LearnedQualityPredictor

    train, _ = synthetic(40, 7)
    doc = document(
        train,
        synthetic(20, 8)[0],
        Candidate("knn", {}, knn_params(train, {"k": 3, "temperature": 0.1, "same_task": True})),
    )
    doc["b0b"] = {"general": {"cheap/a": 0.5, "pricey/b": 1.0}}
    predictor = LearnedQualityPredictor(RouterArtifact(doc))
    features = {FEATURES_KEY: request_features([], task_type="general"), EMBEDDING_KEY: train.unit[0].tolist()}
    confident = predictor.predict("cheap/a", "general", features)
    assert confident is not None and 0.0 <= confident.probability <= 1.0
    assert predictor.predict("unbenchmarked/model", "general", features) is None
    unseen = {FEATURES_KEY: request_features([], task_type="query_rewriting"), EMBEDDING_KEY: train.unit[0].tolist()}
    assert predictor.predict("pricey/b", "query_rewriting", unseen) is None  # no lookup row for an unseen task
    assert predictor.predict("cheap/a", "general", {}) is None  # no features: no learned estimate


def test_numpy_isotonic_matches_scikit_learn() -> None:
    pytest.importorskip("sklearn")
    from sklearn.isotonic import IsotonicRegression

    from modelmesh.ml.training.candidates import isotonic_table

    rng = np.random.default_rng(11)
    raw = np.round(rng.random(200), 2)  # rounding creates tied inputs
    label = (rng.random(200) < raw).astype(float)
    table = isotonic_table(raw, label)
    assert table is not None
    reference = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw, label)
    probe = np.linspace(-0.1, 1.1, 301)
    assert np.allclose(np.interp(probe, table["x"], table["y"]), reference.predict(probe))
