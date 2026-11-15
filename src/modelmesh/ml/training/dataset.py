"""Builds the (item x model) label matrix from recorded benchmark observations. No provider is called.

Label rules (documented in the M6 report):
- Only cells with a recorded observation are used. A missing cell is never filled in as a label.
- A recorded failure (502, 504, schema failure) is a real outcome for that request: under the default
  ``failure_policy="reject"`` it counts as not accepted. ``"drop"`` treats it as missing instead.
- Only label kinds that are fully automatic under ADR-0005 carry a label (``STRICT_KINDS``). ``f1_and_judge``
  (evidence_synthesis) also needs judge faithfulness, so its F1-only label is opt-in and reported as a
  sensitivity analysis. Judge-only kinds have no label until the judge pipeline runs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from modelmesh.ml.benchmark.items import BenchItem, read_items
from modelmesh.ml.evaluation.metrics import Floats

STRICT_KINDS = frozenset({"classification", "support_label", "numeric", "choice", "schema_valid", "ifeval"})
F1_ONLY_KIND = "f1_and_judge"
FailurePolicy = Literal["reject", "drop"]
Bools = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class Observation:
    item_id: str
    model: str
    ok: bool
    accepted: bool | None
    error_code: str | None
    cost_usd: float | None
    latency_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None


def read_observations(path: Path) -> list[Observation]:
    rows: list[Observation] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw: dict[str, Any] = json.loads(line)
            rows.append(
                Observation(
                    item_id=raw["item_id"],
                    model=raw["model"],
                    ok=bool(raw["ok"]),
                    accepted=raw.get("automatic_accepted"),
                    error_code=raw.get("error_code"),
                    cost_usd=raw.get("estimated_cost_usd"),
                    latency_ms=raw.get("latency_ms"),
                    prompt_tokens=raw.get("prompt_tokens"),
                    completion_tokens=raw.get("completion_tokens"),
                )
            )
    return rows


def verify_manifest(items: list[BenchItem], manifest_path: Path) -> str:
    """Checks that the local items match the committed manifest; returns its hash."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {row["item_id"]: row["sha256"] for row in manifest["items"]}
    actual = {item.item_id: item.content_hash() for item in items}
    if expected != actual:
        changed = sorted(k for k in expected.keys() | actual.keys() if expected.get(k) != actual.get(k))
        raise ValueError(f"items do not match the committed manifest: {changed[:5]} ({len(changed)} differ)")
    return str(manifest["manifest_sha256"])


def has_label(item: BenchItem, *, include_f1_only: bool) -> bool:
    return item.label_kind in STRICT_KINDS or (include_f1_only and item.label_kind == F1_ONLY_KIND)


@dataclass
class LabelMatrix:
    """Outcomes for items x models. NaN marks a cell without a usable observation."""

    items: list[BenchItem]
    models: list[str]
    acc: Floats
    cost: Floats  # realised estimated cost; failed cells use the (model, task_type) median of successes
    latency: Floats  # gateway latency; +inf for deadline_exceeded, NaN for other failures
    prompt_tokens: Floats
    completion_tokens: Floats
    failed: Bools

    def item_ids(self) -> list[str]:
        return [item.item_id for item in self.items]

    def complete_rows(self) -> Bools:
        return np.asarray(~np.isnan(self.acc).any(axis=1))

    def restrict(self, rows: Iterable[int] | Bools, models: list[str] | None = None) -> LabelMatrix:
        index = np.asarray(list(rows) if not isinstance(rows, np.ndarray) else rows)
        if index.dtype == np.bool_:
            index = np.flatnonzero(index)
        cols = [self.models.index(m) for m in models] if models else list(range(len(self.models)))
        return LabelMatrix(
            items=[self.items[i] for i in index],
            models=[self.models[c] for c in cols],
            acc=self.acc[np.ix_(index, cols)],
            cost=self.cost[np.ix_(index, cols)],
            latency=self.latency[np.ix_(index, cols)],
            prompt_tokens=self.prompt_tokens[np.ix_(index, cols)],
            completion_tokens=self.completion_tokens[np.ix_(index, cols)],
            failed=self.failed[np.ix_(index, cols)],
        )


def build_matrix(
    items: list[BenchItem],
    observations: list[Observation],
    models: list[str],
    *,
    include_f1_only: bool = False,
    failure_policy: FailurePolicy = "reject",
) -> LabelMatrix:
    labelled = [item for item in items if has_label(item, include_f1_only=include_f1_only)]
    row = {item.item_id: i for i, item in enumerate(labelled)}
    col = {model: j for j, model in enumerate(models)}
    shape = (len(labelled), len(models))
    acc = np.full(shape, np.nan)
    cost = np.full(shape, np.nan)
    latency = np.full(shape, np.nan)
    prompt = np.full(shape, np.nan)
    completion = np.full(shape, np.nan)
    failed = np.zeros(shape, dtype=np.bool_)
    seen: set[tuple[str, str]] = set()
    for obs in observations:
        if obs.item_id not in row or obs.model not in col:
            continue
        key = (obs.item_id, obs.model)
        if key in seen:
            raise ValueError(f"duplicate observation for {key}")
        seen.add(key)
        i, j = row[obs.item_id], col[obs.model]
        if obs.ok:
            if obs.accepted is None:
                raise ValueError(f"successful observation {key} has no automatic label")
            acc[i, j] = float(obs.accepted)
            cost[i, j] = float(obs.cost_usd or 0.0)
            latency[i, j] = float(obs.latency_ms or 0.0)
            prompt[i, j] = float(obs.prompt_tokens or 0)
            completion[i, j] = float(obs.completion_tokens or 0)
        else:
            failed[i, j] = True
            if failure_policy == "reject":
                acc[i, j] = 0.0
            latency[i, j] = np.inf if obs.error_code == "deadline_exceeded" else np.nan
    # Failed cells have no recorded cost: charge them the median successful cost for that model and task type.
    tasks = np.array([item.task_type for item in labelled])
    for j in range(len(models)):
        for task in set(tasks.tolist()):
            in_task = tasks == task
            known = cost[in_task, j]
            known = known[~np.isnan(known)]
            median = float(np.median(known)) if known.size else 0.0
            fill = in_task & failed[:, j]
            cost[fill, j] = median
    return LabelMatrix(labelled, list(models), acc, cost, latency, prompt, completion, failed)


def load_pilot(
    items_path: Path,
    observations_path: Path,
    manifest_path: Path,
) -> tuple[list[BenchItem], list[Observation], str]:
    items = read_items(items_path)
    manifest_sha = verify_manifest(items, manifest_path)
    return items, read_observations(observations_path), manifest_sha


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
