"""The ADR-0004 candidates (kNN, LightGBM, per-model logistic regression), calibration and simple predictors.

Every candidate is exported to the runtime artifact format (``modelmesh.ml.artifacts``) and evaluated through
it, so the offline numbers are produced by the code that serves.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from modelmesh.ml.artifacts import ARTIFACT_SCHEMA_VERSION, RouterArtifact
from modelmesh.ml.evaluation.metrics import Floats
from modelmesh.ml.features import FEATURE_NAMES, FEATURE_VERSION

PCA_DIM = 32
CONFORMAL_MIN_POINTS = 30
CONFORMAL_LEVEL = 0.9
LOW_CONFIDENCE_QUANTILE = 0.05


@dataclass
class Table:
    """Per-item inputs and the label matrix for one subset of items."""

    item_ids: list[str]
    tasks: list[str]
    features: Floats  # (n, len(FEATURE_NAMES))
    unit: Floats  # (n, d) L2-normalised embeddings
    chars: Floats  # (n,) prompt characters, for the token estimate
    acc: Floats  # (n, m), NaN = no label
    cost: Floats
    latency: Floats
    prompt_tokens: Floats
    completion_tokens: Floats

    def rows(self, index: Sequence[int] | np.ndarray[Any, Any]) -> Table:
        idx = np.asarray(index, dtype=np.int64)
        return Table(
            [self.item_ids[i] for i in idx],
            [self.tasks[i] for i in idx],
            self.features[idx],
            self.unit[idx],
            self.chars[idx],
            self.acc[idx],
            self.cost[idx],
            self.latency[idx],
            self.prompt_tokens[idx],
            self.completion_tokens[idx],
        )

    def feature_dicts(self) -> list[dict[str, float]]:
        return [dict(zip(FEATURE_NAMES, row.tolist(), strict=True)) for row in self.features]


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    family: str
    provider: str
    context_window: int
    input_price: float
    output_price: float


# --------------------------------------------------------------------------- shared pieces
def fit_pca(unit: Floats, dim: int = PCA_DIM) -> dict[str, Any]:
    """PCA by SVD on the training embeddings only; sign fixed so the result is deterministic."""
    mean = unit.mean(axis=0)
    _, _, vt = np.linalg.svd(unit - mean, full_matrices=False)
    components = vt[: min(dim, vt.shape[0])]
    signs = np.sign(components[np.arange(components.shape[0]), np.abs(components).argmax(axis=1)])
    components = components * signs[:, None]
    return {"mean": mean.tolist(), "components": components.tolist()}


def task_priors(train: Table) -> Floats:
    with np.errstate(invalid="ignore"):
        priors = np.nanmean(train.acc, axis=0)
    return np.nan_to_num(priors, nan=0.5)


def lookup_table(table: Table, models: Sequence[str]) -> dict[str, dict[str, float]]:
    """B0b: acceptance rate per (task_type, model), from the rows given."""
    out: dict[str, dict[str, float]] = {}
    tasks = np.array(table.tasks)
    for task in sorted(set(table.tasks)):
        block = table.acc[tasks == task]
        out[task] = {m: float(np.nanmean(block[:, j])) for j, m in enumerate(models) if (~np.isnan(block[:, j])).any()}
    return out


def output_token_medians(train: Table, models: Sequence[str]) -> dict[str, dict[str, float]]:
    tasks = np.array(train.tasks)
    out: dict[str, dict[str, float]] = {}
    for j, model in enumerate(models):
        col = train.completion_tokens[:, j]
        entry: dict[str, float] = {}
        if (~np.isnan(col)).any():
            entry["_all"] = float(np.nanmedian(col))
        for task in sorted(set(train.tasks)):
            values = col[(tasks == task) & ~np.isnan(col)]
            if values.size:
                entry[task] = float(np.median(values))
        out[model] = entry
    return out


def token_fit(train: Table, models: Sequence[str]) -> dict[str, list[float]]:
    """ADR-0004 token estimate per model: prompt_tokens = a * chars + b, by least squares."""
    out: dict[str, list[float]] = {}
    for j, model in enumerate(models):
        ok = ~np.isnan(train.prompt_tokens[:, j])
        if ok.sum() < 3:
            out[model] = [0.25, 0.0]
            continue
        if np.ptp(train.chars[ok]) == 0:
            out[model] = [float(train.prompt_tokens[ok, j].mean() / max(train.chars[ok].mean(), 1.0)), 0.0]
            continue
        a, b = np.polyfit(train.chars[ok], train.prompt_tokens[ok, j], 1)
        out[model] = [float(a), float(b)]
    return out


def latency_fit(
    train: Table, models: Sequence[str], tokens: Mapping[str, list[float]], outputs: Mapping[str, Mapping[str, float]]
) -> dict[str, dict[str, float]]:
    """ADR-0004 latency model per model: log latency = alpha + beta*input_k + gamma*predicted_output_k."""
    out: dict[str, dict[str, float]] = {}
    for j, model in enumerate(models):
        lat = train.latency[:, j]
        ok = np.isfinite(lat) & (lat > 0)
        if ok.sum() < 5:
            continue
        a, b = tokens[model]
        inputs = (a * train.chars[ok] + b) / 1000
        predicted = np.array(
            [
                outputs[model].get(t, outputs[model].get("_all", 0.0))
                for t, keep in zip(train.tasks, ok, strict=True)
                if keep
            ]
        )
        design = np.column_stack((np.ones(ok.sum()), inputs, predicted / 1000))
        target = np.log(lat[ok])
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        residual = target - design @ coef
        out[model] = {
            "alpha": float(coef[0]),
            "beta": float(coef[1]),
            "gamma": float(coef[2]),
            "r50": float(np.quantile(residual, 0.5)),
            "r90": float(np.quantile(residual, 0.9)),
        }
    return out


# --------------------------------------------------------------------------- candidates
def knn_params(train: Table, hp: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "train_embeddings": train.unit.tolist(),
        "train_task": list(train.tasks),
        "train_labels": [[None if math.isnan(v) else v for v in row] for row in train.acc.tolist()],
        "priors": task_priors(train).tolist(),
        "k": int(hp["k"]),
        "temperature": float(hp["temperature"]),
        "same_task": bool(hp["same_task"]),
        "prior_weight": float(hp.get("prior_weight", 0.0)),
    }


def _pca_project(unit: Floats, pca: Mapping[str, Any]) -> Floats:
    out: Floats = (unit - np.asarray(pca["mean"])) @ np.asarray(pca["components"]).T
    return out


def logreg_params(train: Table, hp: Mapping[str, Any], seed: int) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression

    pca = fit_pca(train.unit)
    mean = train.features.mean(axis=0)
    scale = train.features.std(axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    x = np.hstack(((train.features - mean) / scale, _pca_project(train.unit, pca)))
    coefs: list[list[float]] = []
    intercepts: list[float] = []
    for j in range(train.acc.shape[1]):
        ok = ~np.isnan(train.acc[:, j])
        y = train.acc[ok, j]
        if ok.sum() == 0 or len(set(y.tolist())) < 2:
            rate = float(y.mean()) if ok.sum() else 0.5
            rate = min(max(rate, 1e-3), 1 - 1e-3)
            coefs.append([0.0] * x.shape[1])
            intercepts.append(math.log(rate / (1 - rate)))
            continue
        model = LogisticRegression(C=float(hp["C"]), max_iter=2000, random_state=seed)
        model.fit(x[ok], y)
        coefs.append(model.coef_[0].tolist())
        intercepts.append(float(model.intercept_[0]))
    return {
        "pca": pca,
        "scaler": {"mean": mean.tolist(), "scale": scale.tolist()},
        "coef": coefs,
        "intercept": intercepts,
    }


def model_feature_rows(models: Sequence[ModelInfo]) -> list[list[float]]:
    """Model one-hot, family one-hot, provider one-hot, log context window and log price tier."""
    families = sorted({m.family for m in models})
    providers = sorted({m.provider for m in models})
    rows = []
    for m in models:
        rows.append(
            [float(m.model_id == other.model_id) for other in models]
            + [float(m.family == f) for f in families]
            + [float(m.provider == p) for p in providers]
            + [math.log(max(m.context_window, 1)), math.log1p(1000 * (m.input_price + m.output_price))]
        )
    return rows


def _target_encoding(train: Table, rows: np.ndarray[Any, Any]) -> dict[str, list[float]]:
    sub = train.rows(rows)
    priors = task_priors(sub)
    tasks = np.array(sub.tasks)
    out: dict[str, list[float]] = {}
    for task in sorted(set(train.tasks)):
        block = sub.acc[tasks == task]
        values = []
        for j in range(train.acc.shape[1]):
            col = block[:, j] if block.size else np.array([])
            col = col[~np.isnan(col)]
            # Shrink towards the model's overall rate so a task with two labels does not encode them exactly.
            values.append(float((col.sum() + 2 * priors[j]) / (col.size + 2)))
        out[task] = values
    return out


def lightgbm_params(train: Table, hp: Mapping[str, Any], models: Sequence[ModelInfo], seed: int) -> dict[str, Any]:
    import lightgbm as lgb

    pca = fit_pca(train.unit)
    base = np.hstack((train.features, _pca_project(train.unit, pca)))
    model_rows = np.asarray(model_feature_rows(models))
    n, m = train.acc.shape
    # Out-of-fold target encoding on train only (5 folds), so a row never sees its own label.
    rng = np.random.default_rng(seed)
    folds = rng.permutation(n) % 5
    oof = np.zeros((n, m))
    for fold in range(5):
        encoding = _target_encoding(train, np.flatnonzero(folds != fold))
        for row in np.flatnonzero(folds == fold).tolist():
            oof[row] = encoding.get(train.tasks[row], task_priors(train).tolist())
    x_rows, y_rows = [], []
    for i in range(n):
        for j in range(m):
            if not math.isnan(train.acc[i, j]):
                x_rows.append(np.concatenate((base[i], model_rows[j], [oof[i, j]])))
                y_rows.append(train.acc[i, j])
    booster = lgb.LGBMClassifier(
        n_estimators=int(hp["n_estimators"]),
        num_leaves=int(hp["num_leaves"]),
        learning_rate=float(hp["learning_rate"]),
        min_child_samples=int(hp["min_child_samples"]),
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=seed,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )
    booster.fit(np.asarray(x_rows), np.asarray(y_rows))
    dump = booster.booster_.dump_model()
    return {
        "pca": pca,
        "trees": [tree["tree_structure"] for tree in dump["tree_info"]],
        "model_features": model_rows.tolist(),
        "target_encoding": _target_encoding(train, np.arange(n)),
        "target_encoding_default": task_priors(train).tolist(),
        "_booster": booster,  # removed before export; kept for the parity check
    }


# --------------------------------------------------------------------------- calibration and conformal
def isotonic_table(raw: Floats, label: Floats) -> dict[str, list[float]] | None:
    """Isotonic regression by pool-adjacent-violators, exported as knots for linear interpolation.

    Equivalent to scikit-learn's IsotonicRegression(out_of_bounds="clip") at and between the knots (a unit test
    checks this), written in numpy so calibration needs no scikit-learn.
    """
    if raw.size < 2 or len(set(raw.tolist())) < 2:
        return None
    xs, inverse, counts = np.unique(raw, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=label)
    blocks: list[list[float]] = []  # [sum, weight, x_low, x_high]
    for x, total, weight in zip(xs.tolist(), sums.tolist(), counts.tolist(), strict=True):
        blocks.append([total, float(weight), x, x])
        while len(blocks) >= 2 and blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]:
            last = blocks.pop()
            blocks[-1][0] += last[0]
            blocks[-1][1] += last[1]
            blocks[-1][3] = last[3]
    knots_x: list[float] = []
    knots_y: list[float] = []
    for total, weight, low, high in blocks:
        value = min(max(total / weight, 0.0), 1.0)
        for x in (low, high) if high > low else (low,):
            knots_x.append(x)
            knots_y.append(value)
    return {"x": knots_x, "y": knots_y}


def conformal_half_width(residuals: Floats, level: float = CONFORMAL_LEVEL) -> float | None:
    n = residuals.size
    if n < CONFORMAL_MIN_POINTS:
        return None
    rank = min(math.ceil((n + 1) * level), n)
    return float(np.sort(residuals)[rank - 1])


@dataclass
class Candidate:
    algorithm: str
    hp: dict[str, Any]
    params: dict[str, Any]
    extras: dict[str, Any] = field(default_factory=dict)


def base_document(
    *,
    version: str,
    candidate: Candidate,
    models: Sequence[str],
    train: Table,
    dataset_manifest_sha256: str,
    split_manifest_sha256: str,
) -> dict[str, Any]:
    params = {k: v for k, v in candidate.params.items() if not k.startswith("_")}
    params["nn_embeddings"] = train.unit.tolist()
    outputs = output_token_medians(train, models)
    tokens = token_fit(train, models)
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "version": version,
        "algorithm": candidate.algorithm,
        "hyperparameters": candidate.hp,
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "models": list(models),
        "task_types": sorted(set(train.tasks)),
        "params": params,
        "calibration": {},
        "conformal_half_width": {},
        "calibrated_task_types": [],
        "low_confidence_similarity": -1.0,
        "b0b": lookup_table(train, models),
        "output_tokens": outputs,
        "token_fit": tokens,
        "latency": latency_fit(train, models, tokens, outputs),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "split_manifest_sha256": split_manifest_sha256,
    }


def raw_matrix(artifact: RouterArtifact, table: Table) -> Floats:
    return np.vstack(
        [
            artifact.raw_scores(features, unit, task)
            for features, unit, task in zip(table.feature_dicts(), table.unit, table.tasks, strict=True)
        ]
    )


def calibrated_matrix(artifact: RouterArtifact, table: Table, raw: Floats | None = None) -> Floats:
    raw = raw_matrix(artifact, table) if raw is None else raw
    return np.vstack([artifact.calibrate(row, task) for row, task in zip(raw, table.tasks, strict=True)])


def finish_document(document: dict[str, Any], validation: Table, raw_validation: Floats) -> dict[str, Any]:
    """Fits per-task isotonic tables and conformal widths on validation, and the low-confidence tau."""
    tasks = np.array(validation.tasks)
    mask = ~np.isnan(validation.acc)
    calibration: dict[str, Any] = {}
    global_table = isotonic_table(raw_validation[mask], validation.acc[mask])
    if global_table is not None:
        calibration["_global"] = global_table
    for task in sorted(set(validation.tasks)):
        rows = tasks == task
        table = isotonic_table(raw_validation[rows][mask[rows]], validation.acc[rows][mask[rows]])
        if table is not None:
            calibration[task] = table
    document["calibration"] = calibration
    artifact = RouterArtifact(document)
    calibrated = calibrated_matrix(artifact, validation, raw_validation)
    widths: dict[str, float | None] = {}
    for task in sorted(set(validation.tasks)):
        rows = tasks == task
        residuals = np.abs(calibrated[rows][mask[rows]] - validation.acc[rows][mask[rows]])
        widths[task] = conformal_half_width(residuals)
    document["conformal_half_width"] = widths
    document["calibrated_task_types"] = sorted(
        task for task in set(validation.tasks) if int(mask[tasks == task].sum()) >= CONFORMAL_MIN_POINTS
    )
    nearest = [artifact.nearest_similarity(unit) for unit in validation.unit]
    document["low_confidence_similarity"] = float(np.quantile(nearest, LOW_CONFIDENCE_QUANTILE))
    return document
