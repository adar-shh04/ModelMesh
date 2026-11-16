"""M6 preliminary router training and evaluation on the P0 pilot observations (offline, no provider calls).

    uv sync --group ml
    uv run python -m modelmesh.ml.evaluation.p0

Writes ``docs/milestones/M6_P0_results.json`` (aggregates only), the selected artifact under ``artifacts/``
(gitignored) and MLflow runs to a local SQLite store on X: (``MODELMESH_MLFLOW_URI`` overrides it).

This is a P0-only experiment. Its split, its model selection and its numbers must not be reused for P1:
P1 freezes its own test manifest before any model selection.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import os
import platform
import random
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from modelmesh.ml.artifacts import RouterArtifact, artifact_sha256
from modelmesh.ml.benchmark.items import BenchItem
from modelmesh.ml.evaluation.metrics import (
    Floats,
    Ints,
    Outcomes,
    aiq,
    bootstrap,
    brier,
    ece,
    hull_at,
    min_cost_for_quality,
    oracle_choice,
    route_cost_optimized,
    upper_hull,
)
from modelmesh.ml.features import (
    EMBEDDING_MAX_TOKENS,
    FEATURE_NAMES,
    FEATURE_VERSION,
    embedding_segments,
    feature_vector,
    join_embedding_text,
    request_features,
)
from modelmesh.ml.training.candidates import (
    Candidate,
    ModelInfo,
    Table,
    base_document,
    calibrated_matrix,
    finish_document,
    knn_params,
    lightgbm_params,
    logreg_params,
    lookup_table,
    raw_matrix,
)
from modelmesh.ml.training.dataset import LabelMatrix, build_matrix, file_sha256, load_pilot
from modelmesh.ml.training.embedding import DEFAULT_MODEL as DEFAULT_EMBEDDING_MODEL
from modelmesh.ml.training.embedding import CachedEmbedder
from modelmesh.ml.training.splits import (
    assert_no_leakage,
    build_groups,
    grouped_split,
    normalise,
    split_manifest,
)
from modelmesh.providers.catalog import TIER_ORDER, load_catalog, load_pricing
from modelmesh.routing.solver import rules_tier

ROOT = Path(__file__).resolve().parents[4]
PILOT = ROOT / "data" / "benchmark" / "pilot"
RUN = "pilot-2026-11-11"
SEED = 20261115
RESAMPLES = int(os.environ.get("MODELMESH_P0_RESAMPLES", "2000"))  # lower only for smoke runs
# Models with complete observations on the auto-labelled items (see the report's coverage section).
POOL = ["gemini/gemini-3.1-flash-lite", "groq/openai/gpt-oss-120b", "ollama/llama3.2:3b", "ollama/qwen3:4b"]
SPARSE = ["groq/openai/gpt-oss-20b", "gemini/gemini-3.6-flash", "openrouter/liquid/lfm-3-1.7b:free",
          "openrouter/nvidia/nemotron-3.5-super-vl:free"]  # fmt: skip
THRESHOLDS = [round(x, 2) for x in np.linspace(0.0, 1.0, 51)] + [1.01]
Q_MINS = (0.5, 0.6, 0.7, 0.8)
L_MAXES = (2000, 10000)
DEFAULT_FLOOR = 0.6
# Leave-one-dataset-out folds: (task_type, source) held out, within task types that have two sources.
LODO = [("query_classification", "banking77"), ("query_classification", "clinc150"),
        ("citation_verification", "scifact"), ("citation_verification", "attributionbench")]  # fmt: skip
LODO_F1 = [("evidence_synthesis", "qasper"), ("evidence_synthesis", "musique")]
# Mixture shift, fixed before any result was computed. Task-type shares of the target workload.
MIXES: dict[str, dict[str, float]] = {
    "mezvuat_like": {"citation_verification": 0.35, "evidence_assessment": 0.25, "evidence_synthesis": 0.2,
                     "query_classification": 0.1, "general": 0.1},
    "chat_like": {"general": 0.7, "query_classification": 0.1, "citation_verification": 0.1,
                  "evidence_assessment": 0.05, "evidence_synthesis": 0.05},
}  # fmt: skip
GRIDS: dict[str, list[dict[str, Any]]] = {
    "knn": [
        {"k": k, "temperature": t, "same_task": s}
        for k, t, s in itertools.product((5, 10, 20), (0.02, 0.1), (True, False))
    ],
    "logreg": [{"C": c} for c in (0.01, 0.1, 1.0)],
    "lightgbm": [
        {"n_estimators": n, "num_leaves": leaves, "min_child_samples": mcs, "learning_rate": 0.05}
        for n, leaves, mcs in itertools.product((50, 150), (4, 8), (5, 20))
    ],
}


def out(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


# --------------------------------------------------------------------------- data
@dataclass
class Variant:
    name: str
    matrix: LabelMatrix
    table: Table
    assignment: list[str]


def chars_of(item: BenchItem) -> int:
    return sum(len(m.get("content") or "") for m in item.messages)


def grouping_text(item: BenchItem) -> str:
    """Text for near-duplicate grouping: system prompt plus final user message, first 4,096 characters (the model
    reads at most 512 tokens of it). This is the feature-version-1 embedding text. It is kept apart from the
    router's own, token-capped, embedding so that the split does not move when the feature spec changes."""
    return join_embedding_text(*embedding_segments(item.messages, max_chars=None))[:4096]


def make_table(matrix: LabelMatrix, embedder: CachedEmbedder) -> Table:
    features = np.array(
        [
            feature_vector(
                request_features(
                    item.messages,
                    task_type=item.task_type,
                    response_format=item.response_format,
                    max_tokens=item.max_tokens,
                )
            )
            for item in matrix.items
        ]
    )
    unit = normalise(embedder.embed_requests([item.messages for item in matrix.items]))
    return Table(
        matrix.item_ids(),
        [item.task_type for item in matrix.items],
        features,
        unit,
        np.array([chars_of(item) for item in matrix.items], dtype=np.float64),
        matrix.acc,
        matrix.cost,
        matrix.latency,
        matrix.prompt_tokens,
        matrix.completion_tokens,
    )


def model_infos() -> list[ModelInfo]:
    catalog = load_catalog(ROOT / "config" / "models.yaml")
    pricing = load_pricing(ROOT / "config" / "pricing.yaml")
    infos = []
    for model_id in POOL:
        model = catalog.models[model_id]
        price = pricing.get(model_id)
        infos.append(
            ModelInfo(
                model_id, model.family, model.provider, model.context_window,
                price.input_per_mtok_usd, price.output_per_mtok_usd,
            )
        )  # fmt: skip
    return infos


# --------------------------------------------------------------------------- routers
@dataclass
class Context:
    """What routers may use: the artifact's cost and latency predictors and the selection-data lookups."""

    infos: list[ModelInfo]
    doc: dict[str, Any]
    selection: Table  # train + validation rows
    lookup_selection: dict[str, dict[str, float]]
    lookup_validation: dict[str, dict[str, float]]
    cheapest: int
    priciest: int
    best_single: int


def predicted_cost(doc: Mapping[str, Any], table: Table, infos: Sequence[ModelInfo]) -> Floats:
    out_matrix = np.zeros((len(table.item_ids), len(infos)))
    for j, info in enumerate(infos):
        a, b = doc["token_fit"][info.model_id]
        tokens = doc["output_tokens"].get(info.model_id, {})
        for i, task in enumerate(table.tasks):
            est_in = max(1.0, a * table.chars[i] + b)
            est_out = tokens.get(task, tokens.get("_all", 0.0))
            out_matrix[i, j] = (est_in * info.input_price + est_out * info.output_price) / 1e6
    return out_matrix


def predicted_latency(doc: Mapping[str, Any], table: Table, infos: Sequence[ModelInfo]) -> tuple[Floats, Floats]:
    artifact = RouterArtifact(doc)
    p50 = np.full((len(table.item_ids), len(infos)), np.nan)
    p90 = np.full_like(p50, np.nan)
    for j, info in enumerate(infos):
        for i, task in enumerate(table.tasks):
            est_in = artifact.estimated_input_tokens(info.model_id, int(table.chars[i]))
            est_out = artifact.predicted_output_tokens(info.model_id, task)
            p50[i, j], p90[i, j] = artifact.latency_ms(info.model_id, est_in, est_out)
    return p50, p90


def lookup_qhat(lookup: Mapping[str, Mapping[str, float]], table: Table, infos: Sequence[ModelInfo]) -> Floats:
    q = np.full((len(table.item_ids), len(infos)), -1.0)
    for i, task in enumerate(table.tasks):
        for j, info in enumerate(infos):
            q[i, j] = lookup.get(task, {}).get(info.model_id, -1.0)
    return q


def sweep(qhat: Floats, cost_pred: Floats) -> Ints:
    return np.stack([route_cost_optimized(qhat, cost_pred, t) for t in THRESHOLDS])


def b0a_choice(ctx: Context, table: Table, cost_pred: Floats, p50: Floats) -> Ints:
    """B0a static rules (ADR-0004), as routing.solver implements them: task tier, escalation, cheapest in tier."""
    catalog = load_catalog(ROOT / "config" / "models.yaml")
    tiers = [catalog.models[info.model_id].size_tier for info in ctx.infos]
    props = table.features[:, FEATURE_NAMES.index("schema_properties")]
    choices = np.empty(len(table.item_ids), dtype=np.int64)
    for i, task in enumerate(table.tasks):
        target = rules_tier(task, int(table.chars[i] * 0.25), int(props[i]))
        index = TIER_ORDER.index(target)
        preference = sorted(TIER_ORDER, key=lambda t: (abs(TIER_ORDER.index(t) - index), TIER_ORDER.index(t) < index))
        for tier in preference:
            members = [j for j, t in enumerate(tiers) if t == tier]
            if members:
                choices[i] = min(members, key=lambda j: (cost_pred[i, j], np.nan_to_num(p50[i, j], nan=1e12)))
                break
    return choices


def learned_qhat(artifact: RouterArtifact, table: Table, ctx: Context) -> tuple[Floats, Floats, Floats]:
    """(raw, calibrated, routed) Q-hat. Routed applies the low-confidence fallback to the B0b lookup."""
    raw = raw_matrix(artifact, table)
    calibrated = calibrated_matrix(artifact, table, raw)
    routed = calibrated.copy()
    fallback = lookup_qhat(artifact.document["b0b"], table, ctx.infos)
    for i, (unit, task) in enumerate(zip(table.unit, table.tasks, strict=True)):
        if artifact.low_confidence(unit, task):
            routed[i] = fallback[i]
    return raw, calibrated, routed


# --------------------------------------------------------------------------- statistics
@dataclass
class EvalInputs:
    acc: Floats  # (n, m + 1); the last column is the Random router's expected outcome
    cost: Floats
    weights: Floats | None
    ctx: Context


def with_random(acc: Floats, cost: Floats) -> tuple[Floats, Floats]:
    return np.hstack((acc, acc.mean(axis=1, keepdims=True))), np.hstack((cost, cost.mean(axis=1, keepdims=True)))


def single_point(data: EvalInputs, idx: Ints, j: int) -> tuple[float, float]:
    a, c = data.acc[idx, j], data.cost[idx, j]
    if data.weights is None:
        return float(c.mean()), float(a.mean())
    w = data.weights[idx]
    return float((c * w).sum() / w.sum()), float((a * w).sum() / w.sum())


def statistics(
    data: EvalInputs, routers: Mapping[str, Outcomes], compare: Sequence[tuple[str, str]], focus: Sequence[str]
) -> Callable[[Ints], dict[str, float]]:
    ctx = data.ctx

    def compute(idx: Ints) -> dict[str, float]:
        cheapest = single_point(data, idx, ctx.cheapest)
        c_min, c_max = cheapest[0], single_point(data, idx, ctx.priciest)[0]
        bs_cost, bs_q = single_point(data, idx, ctx.best_single)
        result: dict[str, float] = {}
        hulls = {}
        for name, outcomes in routers.items():
            hulls[name] = outcomes.hull(idx, cheapest, data.weights)
            result[f"aiq/{name}"] = aiq(hulls[name], c_min, c_max)
        singles = [single_point(data, idx, j) for j in range(len(ctx.infos))]
        zero = upper_hull(np.array([p[0] for p in singles]), np.array([p[1] for p in singles]))
        result["aiq/zero_router"] = aiq(zero, c_min, c_max)
        for a, b in compare:
            result[f"diff/{a}-{b}"] = result[f"aiq/{a}"] - result[f"aiq/{b}"]
        result["diff/learned-zero_router"] = result["aiq/learned"] - result["aiq/zero_router"]
        oracle = hulls["oracle"]
        for name in focus:
            costs, quals = routers[name].points(idx, data.weights)
            for frac in (0.95, 1.0):
                target = frac * bs_q
                feasible = costs[quals >= target - 1e-12]
                best = float(feasible.min()) if feasible.size else math.nan
                result[f"saving@{frac}/{name}"] = 1 - best / bs_cost if bs_cost > 0 else math.nan
            result[f"quality_regret@bs_cost/{name}"] = hull_at(oracle, bs_cost) - hull_at(hulls[name], bs_cost)
            target = 0.95 * bs_q
            result[f"cost_regret@0.95bs/{name}"] = min_cost_for_quality(hulls[name], target) - min_cost_for_quality(
                oracle, target
            )
        result["best_single/cost"], result["best_single/quality"] = bs_cost, bs_q
        return result

    return compute


def bootstrap_named(fn: Callable[[Ints], dict[str, float]], n: int, seed: int) -> dict[str, dict[str, float]]:
    names = list(fn(np.arange(n)).keys())
    intervals = bootstrap(lambda idx: np.array([fn(idx)[k] for k in names]), n, resamples=RESAMPLES, seed=seed)
    return {name: interval.as_dict() for name, interval in zip(names, intervals, strict=True)}


def operating_point_table(data: EvalInputs, routers: Mapping[str, Outcomes]) -> dict[str, Any]:
    """Each router's operating point at the default floor (sweep routers) or its only point."""
    idx = np.arange(data.acc.shape[0])
    t_index = THRESHOLDS.index(DEFAULT_FLOOR)
    table = {}
    for name, outcomes in routers.items():
        costs, quals = outcomes.points(idx, data.weights)
        k = t_index if costs.size == len(THRESHOLDS) else 0
        table[name] = {"mean_cost_usd": float(costs[k]), "acceptance": float(quals[k])}
    return table


# --------------------------------------------------------------------------- training and selection
def fit_candidate(algorithm: str, hp: dict[str, Any], train: Table, infos: list[ModelInfo]) -> Candidate:
    if algorithm == "knn":
        return Candidate(algorithm, hp, knn_params(train, hp))
    if algorithm == "logreg":
        return Candidate(algorithm, hp, logreg_params(train, hp, SEED))
    return Candidate(algorithm, hp, lightgbm_params(train, hp, infos, SEED))


def make_context(infos: list[ModelInfo], doc: dict[str, Any], train: Table, validation: Table) -> Context:
    selection = Table(
        train.item_ids + validation.item_ids,
        train.tasks + validation.tasks,
        np.vstack((train.features, validation.features)),
        np.vstack((train.unit, validation.unit)),
        np.concatenate((train.chars, validation.chars)),
        np.vstack((train.acc, validation.acc)),
        np.vstack((train.cost, validation.cost)),
        np.vstack((train.latency, validation.latency)),
        np.vstack((train.prompt_tokens, validation.prompt_tokens)),
        np.vstack((train.completion_tokens, validation.completion_tokens)),
    )
    models = [i.model_id for i in infos]
    prices = predicted_cost(doc, selection, infos).mean(axis=0)
    rates = np.nanmean(selection.acc, axis=0)
    mean_cost = np.nanmean(selection.cost, axis=0)
    return Context(
        infos=infos,
        doc=doc,
        selection=selection,
        lookup_selection=lookup_table(selection, models),
        lookup_validation=lookup_table(validation, models),
        cheapest=int(min(range(len(models)), key=lambda j: (prices[j], -rates[j]))),
        priciest=int(max(range(len(models)), key=lambda j: (mean_cost[j], -rates[j]))),
        best_single=int(max(range(len(models)), key=lambda j: (rates[j], -prices[j]))),
    )


def validation_aiq(doc: dict[str, Any], train: Table, validation: Table, infos: list[ModelInfo]) -> tuple[float, float]:
    artifact = RouterArtifact(doc)
    raw = raw_matrix(artifact, validation)
    ctx = make_context(infos, doc, train, validation)
    cost_pred = predicted_cost(doc, validation, infos)
    outcomes = Outcomes.from_choices(sweep(raw, cost_pred), validation.acc, validation.cost)
    idx = np.arange(len(validation.item_ids))
    data = EvalInputs(*with_random(validation.acc, validation.cost), None, ctx)
    cheapest = single_point(data, idx, ctx.cheapest)
    value = aiq(outcomes.hull(idx, cheapest), cheapest[0], single_point(data, idx, ctx.priciest)[0])
    mask = ~np.isnan(validation.acc)
    p = np.clip(raw[mask], 1e-6, 1 - 1e-6)
    y = validation.acc[mask]
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return value, logloss


@dataclass
class Trained:
    algorithm: str
    hp: dict[str, Any]
    doc: dict[str, Any]
    validation_aiq: float
    validation_logloss: float


def select(
    train: Table, validation: Table, infos: list[ModelInfo], hashes: tuple[str, str], log: Callable[..., None]
) -> dict[str, Trained]:
    models = [i.model_id for i in infos]
    best: dict[str, Trained] = {}
    for algorithm, grid in GRIDS.items():
        for hp in grid:
            candidate = fit_candidate(algorithm, hp, train, infos)
            doc = base_document(
                version=f"p0-{algorithm}", candidate=candidate, models=models, train=train,
                dataset_manifest_sha256=hashes[0], split_manifest_sha256=hashes[1],
            )  # fmt: skip
            value, logloss = validation_aiq(doc, train, validation, infos)
            log(algorithm, hp, value, logloss)
            current = best.get(algorithm)
            if current is None or (value, -logloss) > (current.validation_aiq, -current.validation_logloss):
                best[algorithm] = Trained(algorithm, hp, doc, value, logloss)
    for trained in best.values():
        raw_validation = raw_matrix(RouterArtifact(trained.doc), validation)
        trained.doc = finish_document(trained.doc, validation, raw_validation)
    return best


def choose(best: Mapping[str, Trained]) -> str:
    return max(best, key=lambda a: (best[a].validation_aiq, -best[a].validation_logloss))


# --------------------------------------------------------------------------- one evaluation set
def evaluate_routers(
    trained: Mapping[str, Trained],
    selected: str,
    test: Table,
    ctx: Context,
    weights: Floats | None,
    seed: int,
) -> dict[str, Any]:
    infos = ctx.infos
    doc = trained[selected].doc
    cost_pred = predicted_cost(doc, test, infos)
    p50, _ = predicted_latency(doc, test, infos)
    m = len(infos)
    choices: dict[str, Ints] = {}
    for algorithm, t in trained.items():
        _, _, routed = learned_qhat(RouterArtifact(t.doc), test, ctx)
        choices[algorithm] = sweep(routed, cost_pred)
    choices["learned"] = choices[selected]
    choices["b0b"] = sweep(lookup_qhat(ctx.lookup_selection, test, infos), cost_pred)
    choices["b0b_validation_only"] = sweep(lookup_qhat(ctx.lookup_validation, test, infos), cost_pred)
    choices["b0a"] = b0a_choice(ctx, test, cost_pred, p50)[None, :]
    n = len(test.item_ids)
    choices["best_single"] = np.full((1, n), ctx.best_single)
    choices["cheapest"] = np.full((1, n), ctx.cheapest)
    choices["random"] = np.full((1, n), m)  # the virtual expected-outcome column
    choices["oracle"] = oracle_choice(test.acc, test.cost)[None, :]
    acc, cost = with_random(test.acc, test.cost)
    data = EvalInputs(acc, cost, weights, ctx)
    routers = {name: Outcomes.from_choices(c, acc, cost) for name, c in choices.items()}
    compare = [("learned", "b0b"), ("learned", "b0a"), ("knn", "b0b"), ("logreg", "b0b"), ("lightgbm", "b0b")]
    intervals = bootstrap_named(statistics(data, routers, compare, focus=["learned", "b0b", "b0a"]), n, seed)
    priciest_share = {
        name: float(np.mean(c[THRESHOLDS.index(DEFAULT_FLOOR) if c.shape[0] == len(THRESHOLDS) else 0] == ctx.priciest))
        for name, c in choices.items()
    }
    return {
        "n_items": n,
        "intervals": intervals,
        "operating_points": operating_point_table(data, routers),
        "share_to_priciest_at_default_floor": priciest_share,
        "singles": {
            info.model_id: dict(zip(("mean_cost_usd", "acceptance"), single_point(data, np.arange(n), j), strict=True))
            for j, info in enumerate(infos)
        },
    }


def constraint_fidelity(trained: Trained, test: Table, ctx: Context, seed: int) -> dict[str, Any]:
    artifact = RouterArtifact(trained.doc)
    _, _, routed = learned_qhat(artifact, test, ctx)
    cost_pred = predicted_cost(trained.doc, test, ctx.infos)
    _, p90 = predicted_latency(trained.doc, test, ctx.infos)
    n = len(test.item_ids)
    rows = np.arange(n)
    result: dict[str, Any] = {"quality": {}, "latency": {}}
    for q_min in Q_MINS:
        eligible = routed >= q_min
        served = eligible.any(axis=1)
        choice = route_cost_optimized(routed, cost_pred, q_min)
        realised = test.acc[rows, choice]

        def stat(idx: Ints, served: Any = served, realised: Floats = realised) -> Floats:
            s = served[idx]
            return np.array([realised[idx][s].mean() if s.any() else math.nan, 1 - s.mean()])

        acc_ci, rejected_ci = bootstrap(stat, n, resamples=RESAMPLES, seed=seed)
        result["quality"][str(q_min)] = {
            "realised_acceptance": acc_ci.as_dict(),
            "rejected_fraction": rejected_ci.as_dict(),
            "violated": bool(acc_ci.point < q_min),
            "served": int(served.sum()),
        }
    for l_max in L_MAXES:
        masked = np.where(np.nan_to_num(p90, nan=np.inf) <= l_max, routed, -np.inf)
        served = np.isfinite(masked).any(axis=1)
        choice = route_cost_optimized(masked, cost_pred, DEFAULT_FLOOR)
        latency = test.latency[rows, choice]
        known = served & ~np.isnan(latency)

        def lat_stat(idx: Ints, known: Any = known, latency: Floats = latency, l_max: int = l_max) -> Floats:
            k = known[idx]
            return np.array([np.mean(latency[idx][k] > l_max) if k.any() else math.nan])

        (exceed,) = bootstrap(lat_stat, n, resamples=RESAMPLES, seed=seed)
        result["latency"][str(l_max)] = {
            "exceed_fraction": exceed.as_dict(),
            "served": int(served.sum()),
            "with_known_latency": int(known.sum()),
        }
    return result


def calibration_report(trained: Mapping[str, Trained], test: Table, ctx: Context) -> dict[str, Any]:
    report: dict[str, Any] = {}
    tasks = np.array(test.tasks)
    mask = ~np.isnan(test.acc)
    for algorithm, t in trained.items():
        raw, calibrated, _ = learned_qhat(RouterArtifact(t.doc), test, ctx)
        entry: dict[str, Any] = {}
        for task in ["_all", *sorted(set(test.tasks))]:
            rows = np.ones(len(tasks), dtype=bool) if task == "_all" else tasks == task
            sel = mask & rows[:, None]
            y = test.acc[sel]
            entry[task] = {
                "n_cells": int(sel.sum()),
                "ece_raw": ece(np.clip(raw[sel], 0, 1), y),
                "ece_calibrated": ece(np.clip(calibrated[sel], 0, 1), y),
                "brier_raw": brier(raw[sel], y),
                "brier_calibrated": brier(calibrated[sel], y),
            }
        report[algorithm] = entry
    return report


def predictor_report(doc: dict[str, Any], test: Table, infos: list[ModelInfo]) -> dict[str, Any]:
    cost_pred = predicted_cost(doc, test, infos)
    p50, p90 = predicted_latency(doc, test, infos)
    report: dict[str, Any] = {}
    for j, info in enumerate(infos):
        ok = np.isfinite(test.latency[:, j]) & (test.latency[:, j] > 0)
        timed_out = np.isinf(test.latency[:, j])
        real_cost = test.cost[:, j]
        report[info.model_id] = {
            "latency_cells": int(ok.sum()),
            "latency_abs_log_error_median": float(np.median(np.abs(np.log(p50[ok, j]) - np.log(test.latency[ok, j]))))
            if ok.any()
            else None,
            "latency_p90_coverage": float(np.mean(test.latency[ok | timed_out, j] <= p90[ok | timed_out, j]))
            if (ok | timed_out).any()
            else None,
            "cost_mean_predicted_usd": float(cost_pred[:, j].mean()),
            "cost_mean_realised_usd": float(real_cost.mean()),
        }
    return report


# --------------------------------------------------------------------------- robustness and overhead
def typo(text: str, rng: random.Random, rate: float = 0.1) -> str:
    """Swaps two inner letters in about ``rate`` of the words of four or more letters."""
    words = text.split(" ")
    for k, word in enumerate(words):
        if len(word) >= 4 and word.isalpha() and rng.random() < rate:
            p = rng.randrange(1, len(word) - 2)
            words[k] = word[:p] + word[p + 1] + word[p] + word[p + 2 :]
    return " ".join(words)


def perturbed_items(items: Sequence[BenchItem]) -> list[BenchItem]:
    out = []
    for item in items:
        rng = random.Random(f"{SEED}:{item.item_id}")
        messages = [dict(m) for m in item.messages]
        last = max(i for i, m in enumerate(messages) if m["role"] == "user")
        messages[last]["content"] = typo(messages[last]["content"], rng)
        out.append(dataclasses.replace(item, messages=messages))
    return out


def robustness(
    trained: Trained, test_items: list[BenchItem], test: Table, ctx: Context, embedder: CachedEmbedder, seed: int
) -> dict[str, Any]:
    perturbed = perturbed_items(test_items)
    features = np.array(
        [
            feature_vector(
                request_features(
                    i.messages, task_type=i.task_type, response_format=i.response_format, max_tokens=i.max_tokens
                )
            )
            for i in perturbed
        ]
    )
    unit = normalise(embedder.embed_requests([i.messages for i in perturbed]))
    noisy = Table(test.item_ids, test.tasks, features, unit, np.array([chars_of(i) for i in perturbed], dtype=float),
                  test.acc, test.cost, test.latency, test.prompt_tokens, test.completion_tokens)  # fmt: skip
    artifact = RouterArtifact(trained.doc)
    cost_clean = predicted_cost(trained.doc, test, ctx.infos)
    cost_noisy = predicted_cost(trained.doc, noisy, ctx.infos)
    _, _, clean_q = learned_qhat(artifact, test, ctx)
    _, _, noisy_q = learned_qhat(artifact, noisy, ctx)
    flips = route_cost_optimized(clean_q, cost_clean, DEFAULT_FLOOR) != route_cost_optimized(
        noisy_q, cost_noisy, DEFAULT_FLOOR
    )
    p50c, _ = predicted_latency(trained.doc, test, ctx.infos)
    p50n, _ = predicted_latency(trained.doc, noisy, ctx.infos)
    b0a_flips = b0a_choice(ctx, test, cost_clean, p50c) != b0a_choice(ctx, noisy, cost_noisy, p50n)
    changed = float(np.mean([a.messages != b.messages for a, b in zip(test_items, perturbed, strict=True)]))
    (flip_ci,) = bootstrap(lambda idx: np.array([flips[idx].mean()]), len(flips), resamples=RESAMPLES, seed=seed)
    return {
        "perturbation": "typos: two inner letters swapped in about 10% of alphabetic words of 4+ letters",
        "items": len(flips),
        "items_changed": changed,
        "learned_flip_rate": flip_ci.as_dict(),
        "b0a_flip_rate": float(b0a_flips.mean()),
        "b0b_flip_rate": 0.0,
        "paraphrase": "not run: needs an LLM call, and this run makes no provider calls",
    }


def overhead(artifact_path: Path, test_items: list[BenchItem], embedder: CachedEmbedder) -> dict[str, Any]:
    """Router p50/p95 on the online path: features, artifact prediction, Estimator and policy solver.

    "Including embedding" uses the gateway's own embedder class with the gateway's default thread count and its
    LRU disabled, so every call runs the model, as for a prompt the gateway has not seen before.
    """
    from modelmesh.cache.embeddings import FastEmbedEmbedder
    from modelmesh.observability.stats import ProviderStats
    from modelmesh.routing.estimates import Estimator, TokenEstimator
    from modelmesh.routing.learned import LearnedQualityPredictor, routing_features, segments_to_embed
    from modelmesh.routing.solver import SolveInput, solve

    catalog = load_catalog(ROOT / "config" / "models.yaml")
    pricing = load_pricing(ROOT / "config" / "pricing.yaml")
    predictor = LearnedQualityPredictor(RouterArtifact.load(artifact_path))
    estimator = Estimator(pricing, ProviderStats(), TokenEstimator(), predictor)
    pool = [catalog.models[m] for m in POOL]
    online = FastEmbedEmbedder(DEFAULT_EMBEDDING_MODEL, cache_dir=os.environ.get("FASTEMBED_CACHE_PATH"), cache_size=0)

    def route(item: BenchItem, embedding: Sequence[float]) -> str:
        features = routing_features(
            item.messages, task_type=item.task_type, response_format=item.response_format,
            max_tokens=item.max_tokens, tool_count=0, embedding=embedding,
        )  # fmt: skip
        estimates = [
            estimator.estimate(
                model, messages=item.messages, max_tokens=item.max_tokens, task_type=item.task_type, features=features
            )
            for model in pool
        ]
        solved = solve(
            SolveInput("cost_optimized", item.task_type, predictor.is_task_calibrated(item.task_type), None, None,
                       estimates[0].input_tokens),
            estimates,
        )  # fmt: skip
        return solved.ranked[0].model.id

    embeddings = [row.tolist() for row in embedder.embed_requests([i.messages for i in test_items])]
    for item, emb in zip(test_items[:5], embeddings, strict=False):  # warm-up
        route(item, emb)
        online.embed_request_sync(*segments_to_embed(item.messages))
    without, with_embedding, lengths = [], [], []
    for _ in range(5):
        for item, emb in zip(test_items, embeddings, strict=True):
            lengths.append(chars_of(item))  # prompt length before any cut, so buckets compare across versions
            start = time.perf_counter()
            route(item, emb)
            without.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            fresh = online.embed_request_sync(*segments_to_embed(item.messages))
            route(item, fresh)
            with_embedding.append((time.perf_counter() - start) * 1000)
    return {
        "calls": len(without),
        "excluding_embedding_ms": {"p50": float(np.percentile(without, 50)), "p95": float(np.percentile(without, 95))},
        "including_embedding_ms": {
            "p50": float(np.percentile(with_embedding, 50)),
            "p95": float(np.percentile(with_embedding, 95)),
        },
        "including_embedding_by_text_chars": {
            label: {"calls": int(mask.sum()), "p95_ms": float(np.percentile(np.array(with_embedding)[mask], 95))}
            for label, mask in (
                ("<=1000", np.array(lengths) <= 1000),
                ("1001-2048", (np.array(lengths) > 1000) & (np.array(lengths) <= 2048)),
                (">2048", np.array(lengths) > 2048),
            )
            if mask.any()
        },
        "targets_ms": {"excluding_embedding_p95": 10, "including_embedding_p95": 25},
        "embedding": f"fastembed {DEFAULT_EMBEDDING_MODEL} (ONNX, CPU, threads={online.threads}, first "
        f"{EMBEDDING_MAX_TOKENS} tokens, feature version {FEATURE_VERSION}), as the gateway uses it",
        "machine": machine(),
    }


def machine() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
    }
    if sys.platform == "win32":
        import ctypes
        import winreg

        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            info["processor_name"] = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong)
                for name in ("total", "avail", "total_page", "avail_page", "total_virtual", "avail_virtual", "ext")
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            info["ram_gb"] = round(status.total / 2**30, 1)
    return info


# --------------------------------------------------------------------------- driver
def coverage(items: list[BenchItem], observations: Sequence[Any]) -> dict[str, Any]:
    cells = {(o.item_id, o.model) for o in observations}
    models = sorted({o.model for o in observations})
    report: dict[str, Any] = {}
    for model in models:
        by_task: dict[str, dict[str, int]] = {}
        for item in items:
            entry = by_task.setdefault(item.task_type, {"items": 0, "observed": 0})
            entry["items"] += 1
            entry["observed"] += int((item.item_id, model) in cells)
        report[model] = by_task
    return report


def sparse_models_report(items: list[BenchItem], observations: Sequence[Any]) -> dict[str, Any]:
    """Descriptive only: acceptance on the query_classification items that every one of the 8 models reached."""
    everyone = POOL + SPARSE
    matrix = build_matrix(items, list(observations), everyone)
    classification = np.array([i.task_type == "query_classification" for i in matrix.items])
    complete = matrix.complete_rows() & classification
    sub = matrix.restrict(complete)
    oracle = oracle_choice(sub.acc, sub.cost)
    rows = np.arange(len(sub.items))
    return {
        "items": len(sub.items),
        "acceptance": {m: float(sub.acc[:, j].mean()) for j, m in enumerate(sub.models)},
        "oracle_acceptance": float(sub.acc[rows, oracle].mean()),
        "oracle_share": {m: float(np.mean(oracle == j)) for j, m in enumerate(sub.models)},
    }


def mlflow_logger() -> Any:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    uri = os.environ.get("MODELMESH_MLFLOW_URI", "sqlite:///X:/modelmesh-cache/mlflow/mlflow.db")
    mlflow.set_tracking_uri(uri)
    name = "modelmesh-m6-p0-preliminary"
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        location = os.environ.get("MODELMESH_MLFLOW_ARTIFACTS", "file:///X:/modelmesh-cache/mlflow/artifacts")
        mlflow.create_experiment(name, artifact_location=location)
    mlflow.set_experiment(name)
    return mlflow


def lodo(
    variant_table: Table,
    items: list[BenchItem],
    groups: list[int],
    folds: Sequence[tuple[str, str]],
    trained: Mapping[str, Trained],
    selected: str,
    infos: list[ModelInfo],
    hashes: tuple[str, str],
    reference: Context,
) -> dict[str, Any]:
    """Leave-one-dataset-out with the configuration selected on held-out-query validation (no re-selection)."""
    models = [i.model_id for i in infos]
    pooled_rows: list[int] = []
    pooled_choices: dict[str, list[Ints]] = {}
    for task, source in folds:
        test_rows = [i for i, item in enumerate(items) if item.task_type == task and item.source == source]
        rest = [i for i in range(len(items)) if i not in set(test_rows)]
        inner = grouped_split(
            [items[i] for i in rest], [groups[i] for i in rest], seed=SEED, fractions=(0.75, 0.25, 0.0)
        )
        train = variant_table.rows([r for r, s in zip(rest, inner, strict=True) if s == "train"])
        validation = variant_table.rows([r for r, s in zip(rest, inner, strict=True) if s == "validation"])
        test = variant_table.rows(test_rows)
        fold_trained: dict[str, Trained] = {}
        for algorithm, t in trained.items():
            candidate = fit_candidate(algorithm, t.hp, train, infos)
            doc = base_document(
                version=f"p0-lodo-{source}-{algorithm}", candidate=candidate, models=models, train=train,
                dataset_manifest_sha256=hashes[0], split_manifest_sha256=hashes[1],
            )  # fmt: skip
            doc = finish_document(doc, validation, raw_matrix(RouterArtifact(doc), validation))
            fold_trained[algorithm] = Trained(algorithm, t.hp, doc, math.nan, math.nan)
        ctx = make_context(infos, fold_trained[selected].doc, train, validation)
        for t in fold_trained.values():
            t.doc["b0b"] = ctx.lookup_selection
        cost_pred = predicted_cost(fold_trained[selected].doc, test, infos)
        p50, _ = predicted_latency(fold_trained[selected].doc, test, infos)
        fold_choices: dict[str, Ints] = {}
        for algorithm, t in fold_trained.items():
            fold_choices[algorithm] = sweep(learned_qhat(RouterArtifact(t.doc), test, ctx)[2], cost_pred)
        fold_choices["learned"] = fold_choices[selected]
        fold_choices["b0b"] = sweep(lookup_qhat(ctx.lookup_selection, test, infos), cost_pred)
        fold_choices["b0a"] = b0a_choice(ctx, test, cost_pred, p50)[None, :]
        n = len(test_rows)
        fold_choices["best_single"] = np.full((1, n), ctx.best_single)
        fold_choices["cheapest"] = np.full((1, n), ctx.cheapest)
        fold_choices["random"] = np.full((1, n), len(infos))
        fold_choices["oracle"] = oracle_choice(test.acc, test.cost)[None, :]
        pooled_rows.extend(test_rows)
        for name, c in fold_choices.items():
            pooled_choices.setdefault(name, []).append(c)
    pooled = variant_table.rows(pooled_rows)
    # Cheapest, priciest and Best-Single for the AIQ range and savings come from the main split's selection data.
    acc, cost = with_random(pooled.acc, pooled.cost)
    data = EvalInputs(acc, cost, None, reference)
    routers = {name: Outcomes.from_choices(np.hstack(parts), acc, cost) for name, parts in pooled_choices.items()}
    compare = [("learned", "b0b"), ("learned", "b0a"), ("knn", "b0b"), ("logreg", "b0b"), ("lightgbm", "b0b")]
    fn = statistics(data, routers, compare, focus=["learned", "b0b", "b0a"])
    return {
        "folds": [f"{t}/{s}" for t, s in folds],
        "n_items": len(pooled_rows),
        "intervals": bootstrap_named(fn, len(pooled_rows), SEED + 2),
        "operating_points": operating_point_table(data, routers),
    }


def mixture_weights(test: Table, mix: Mapping[str, float]) -> Floats:
    tasks = np.array(test.tasks)
    present = {t: float(np.mean(tasks == t)) for t in set(test.tasks)}
    share = {t: mix.get(t, 0.0) for t in present}
    total = sum(share.values())
    return np.array([share[t] / total / present[t] for t in test.tasks])


def run_variant(
    name: str,
    items: list[BenchItem],
    observations: Sequence[Any],
    embedder: CachedEmbedder,
    dataset_sha: str,
    *,
    include_f1_only: bool,
    failure_policy: str,
    mlflow: Any,
    full: bool,
) -> dict[str, Any]:
    infos = model_infos()
    matrix = build_matrix(items, list(observations), POOL, include_f1_only=include_f1_only,
                          failure_policy=failure_policy)  # type: ignore[arg-type]  # fmt: skip
    matrix = matrix.restrict(matrix.complete_rows())
    # Groups and the split are always built over the widest variant, so every variant shares one split.
    wide = build_matrix(items, list(observations), POOL, include_f1_only=True)
    wide = wide.restrict(wide.complete_rows())
    wide_unit = normalise(embedder.embed([grouping_text(i) for i in wide.items]))
    wide_groups = build_groups(wide.items, wide_unit)
    wide_split = grouped_split(wide.items, wide_groups, seed=SEED)
    leak = assert_no_leakage(wide_split, wide_groups, wide_unit)
    position = {item.item_id: k for k, item in enumerate(wide.items)}
    assignment = [wide_split[position[i.item_id]] for i in matrix.items]
    groups = [wide_groups[position[i.item_id]] for i in matrix.items]
    manifest = split_manifest(
        wide.items, wide_split, name="p0-preliminary-2026-11-15", seed=SEED, dataset_manifest_sha256=dataset_sha,
        note=("P0 pilot only. Model selection on this split must not be reused on P1; P1 freezes its own test "
              "manifest before any model selection."),
    )  # fmt: skip
    table = make_table(matrix, embedder)
    split = np.array(assignment)
    train = table.rows(np.flatnonzero(split == "train"))
    validation = table.rows(np.flatnonzero(split == "validation"))
    test = table.rows(np.flatnonzero(split == "test"))
    test_items = [matrix.items[i] for i in np.flatnonzero(split == "test")]
    hashes = (dataset_sha, manifest["split_manifest_sha256"])
    out(f"[{name}] items={len(matrix.items)} train={len(train.item_ids)} val={len(validation.item_ids)} "
        f"test={len(test.item_ids)}")  # fmt: skip

    with mlflow.start_run(run_name=name) as parent:
        mlflow.log_params({"variant": name, "pool": ",".join(POOL), "seed": SEED, "include_f1_only": include_f1_only,
                           "failure_policy": failure_policy, "dataset_manifest_sha256": dataset_sha,
                           "split_manifest_sha256": manifest["split_manifest_sha256"]})  # fmt: skip

        def log(algorithm: str, hp: dict[str, Any], value: float, logloss: float) -> None:
            with mlflow.start_run(run_name=f"{name}/{algorithm}", nested=True):
                mlflow.log_params({"algorithm": algorithm, **{f"hp.{k}": v for k, v in hp.items()}})
                mlflow.log_metrics({"validation_aiq_raw": value, "validation_logloss": logloss})

        trained = select(train, validation, infos, hashes, log)
        selected = choose(trained)
        ctx = make_context(infos, trained[selected].doc, train, validation)
        for t in trained.values():
            t.doc["b0b"] = ctx.lookup_selection  # the runtime fallback uses train + validation acceptance
        out(f"[{name}] selected={selected} {trained[selected].hp} (test touched once from here)")
        result: dict[str, Any] = {
            "variant": name,
            "include_f1_only": include_f1_only,
            "failure_policy": failure_policy,
            "items": len(matrix.items),
            "split_counts": {s: int(np.sum(split == s)) for s in ("train", "validation", "test")},
            "split_manifest_sha256": manifest["split_manifest_sha256"],
            "leakage": {"groups_crossing": leak.groups_crossing, "max_cross_split_cosine": leak.max_cross_split_cosine},
            "selection": {
                a: {"hp": t.hp, "validation_aiq_raw": t.validation_aiq, "validation_logloss": t.validation_logloss}
                for a, t in trained.items()
            },
            "selected": selected,
            "references": {
                "cheapest": POOL[ctx.cheapest],
                "priciest": POOL[ctx.priciest],
                "best_single": POOL[ctx.best_single],
            },
            "held_out_query": evaluate_routers(trained, selected, test, ctx, None, SEED),
        }
        if full:
            result["mixture_shift"] = {
                mix: evaluate_routers(trained, selected, test, ctx, mixture_weights(test, shares), SEED + 1)
                for mix, shares in MIXES.items()
            }
            folds = LODO + (LODO_F1 if include_f1_only else [])
            result["workload_shift_lodo"] = lodo(
                table, matrix.items, groups, folds, trained, selected, infos, hashes, ctx
            )
            result["constraint_fidelity"] = constraint_fidelity(trained[selected], test, ctx, SEED + 3)
            result["calibration"] = calibration_report(trained, test, ctx)
            result["predictors"] = predictor_report(trained[selected].doc, test, infos)
            result["robustness"] = robustness(trained[selected], test_items, test, ctx, embedder, SEED + 4)
            artifacts = ROOT / "artifacts"
            artifacts.mkdir(exist_ok=True)
            path = artifacts / f"router-p0-{name}-{selected}.json"
            trained[selected].doc["version"] = f"p0-{name}-{selected}"
            path.write_text(json.dumps(trained[selected].doc), encoding="utf-8")
            result["artifact"] = {"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                                  "sha256": artifact_sha256(path), "bytes": path.stat().st_size}  # fmt: skip
            result["overhead"] = overhead(path, test_items, embedder)
            result["split_manifest"] = manifest
        intervals = result["held_out_query"]["intervals"]
        mlflow.log_metrics(
            {
                k.replace("@", "_at_").replace("/", ".").replace("-", "_minus_"): v["point"]
                for k, v in intervals.items()
                if v["point"] == v["point"]
            }
        )
        mlflow.log_dict(result, "result.json")
        result["mlflow_run_id"] = parent.info.run_id
    return result


def criterion(primary: Mapping[str, Any]) -> dict[str, Any]:
    """ADR-0004 primary success criterion, evaluated mechanically."""
    held = primary["held_out_query"]["intervals"]["diff/learned-b0b"]
    shift = primary["workload_shift_lodo"]["intervals"]["diff/learned-b0b"]
    beats = held["low"] > 0
    non_inferior = shift["low"] > -0.01
    return {
        "held_out_query_diff": held,
        "held_out_query_ci_excludes_zero_and_positive": beats,
        "workload_shift_diff": shift,
        "workload_shift_non_inferior_at_minus_0.01": non_inferior,
        "passes": bool(beats and non_inferior),
    }


def main() -> None:
    os.environ.setdefault("FASTEMBED_CACHE_PATH", str(ROOT / ".cache" / "fastembed"))
    items_path = PILOT / "items.jsonl"
    # The run directory is still being appended to by later benchmark lanes. To reproduce a report, point this at
    # a read-only snapshot whose sha256 matches the one the report records (see M6_P0_NOTES.md).
    observations_path = Path(os.environ.get("MODELMESH_P0_OBSERVATIONS") or PILOT / RUN / "observations.jsonl")
    items, observations, dataset_sha = load_pilot(
        items_path, observations_path, ROOT / "data" / "manifests" / "pilot.json"
    )
    embedder = CachedEmbedder(PILOT / "derived" / "embeddings.npz")
    mlflow = mlflow_logger()
    started = time.time()
    primary = run_variant("primary", items, observations, embedder, dataset_sha, include_f1_only=False,
                          failure_policy="reject", mlflow=mlflow, full=True)  # fmt: skip
    f1 = run_variant("with_f1_only_synthesis", items, observations, embedder, dataset_sha, include_f1_only=True,
                     failure_policy="reject", mlflow=mlflow, full=True)  # fmt: skip
    drop = run_variant("failures_dropped", items, observations, embedder, dataset_sha, include_f1_only=False,
                       failure_policy="drop", mlflow=mlflow, full=False)  # fmt: skip
    results: dict[str, Any] = {
        "title": "M6 router evaluation, PRELIMINARY, P0 pilot data",
        "generated_by": "python -m modelmesh.ml.evaluation.p0",
        "run": RUN,
        "observations_sha256": file_sha256(observations_path),
        "dataset_manifest_sha256": dataset_sha,
        "feature_version": FEATURE_VERSION,
        "seed": SEED,
        "bootstrap_resamples": RESAMPLES,
        "thresholds": THRESHOLDS,
        "pool": POOL,
        "coverage": coverage(items, observations),
        "sparse_models_query_classification": sparse_models_report(items, observations),
        "variants": {"primary": primary, "with_f1_only_synthesis": f1, "failures_dropped": drop},
        "criterion": {
            "primary": criterion(primary),
            "with_f1_only_synthesis": criterion(f1),
        },
        "elapsed_s": round(time.time() - started, 1),
    }
    target = ROOT / "docs" / "milestones" / "M6_P0_results.json"
    target.write_text(json.dumps(results, indent=1, sort_keys=False) + "\n", encoding="utf-8", newline="\n")
    out(f"wrote {target.relative_to(ROOT)}; criterion passes on primary: {results['criterion']['primary']['passes']}")


if __name__ == "__main__":
    main()
