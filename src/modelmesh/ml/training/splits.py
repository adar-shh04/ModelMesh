"""Grouped, near-duplicate-aware splits (ADR-0004) and their manifests.

- All model outputs for one prompt stay together, because a split assigns items, not cells.
- Items built from the same source question (for example the MuSiQue planning, sufficiency and synthesis
  variants of one question) share a group.
- Prompts whose embeddings have cosine similarity above 0.95 are clustered into one group.
- Groups are assigned 60/20/20 within each (task_type, source) stratum.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from modelmesh.ml.benchmark.items import BenchItem
from modelmesh.ml.evaluation.metrics import Floats

NEAR_DUPLICATE_COSINE = 0.95
SPLITS = ("train", "validation", "test")


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def source_question_key(item: BenchItem) -> str:
    """Items from one source question share this key: the source plus the id after the item-id prefix."""
    return f"{item.source}:{item.item_id.split(':', 1)[-1]}"


def normalise(embeddings: Floats) -> Floats:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def build_groups(items: Sequence[BenchItem], embeddings: Floats, threshold: float = NEAR_DUPLICATE_COSINE) -> list[int]:
    """A group id per item (the smallest member index)."""
    uf = _UnionFind(len(items))
    by_key: dict[str, int] = {}
    for i, item in enumerate(items):
        key = source_question_key(item)
        if key in by_key:
            uf.union(by_key[key], i)
        else:
            by_key[key] = i
    unit = normalise(embeddings)
    similar = np.argwhere(np.triu(unit @ unit.T, k=1) > threshold)
    for a, b in similar:
        uf.union(int(a), int(b))
    return [uf.find(i) for i in range(len(items))]


def grouped_split(
    items: Sequence[BenchItem],
    groups: Sequence[int],
    *,
    seed: int,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> list[str]:
    """Assigns whole groups to train/validation/test, stratified by the (task_type, source) of each group."""
    members: dict[int, list[int]] = defaultdict(list)
    for i, group in enumerate(groups):
        members[group].append(i)
    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    for group, index in sorted(members.items()):
        first = items[index[0]]
        strata[(first.task_type, first.source)].append(group)
    rng = random.Random(seed)
    assignment = [""] * len(items)
    for stratum in sorted(strata):
        stratum_groups = strata[stratum]
        rng.shuffle(stratum_groups)
        total = sum(len(members[g]) for g in stratum_groups)
        targets = [f * total for f in fractions]
        counts = [0.0, 0.0, 0.0]
        for group in stratum_groups:
            # Largest remaining deficit first; ties go to the earlier split (train, then validation).
            deficit = [targets[k] - counts[k] for k in range(3)]
            k = max((s for s in range(3) if fractions[s] > 0), key=lambda s: (deficit[s], -s))
            counts[k] += len(members[group])
            for i in members[group]:
                assignment[i] = SPLITS[k]
    return assignment


@dataclass(frozen=True)
class LeakageReport:
    groups_crossing: int
    max_cross_split_cosine: float

    @property
    def clean(self) -> bool:
        return self.groups_crossing == 0 and self.max_cross_split_cosine <= NEAR_DUPLICATE_COSINE


def leakage(assignment: Sequence[str], groups: Sequence[int], embeddings: Floats) -> LeakageReport:
    split_of: dict[int, set[str]] = defaultdict(set)
    for split, group in zip(assignment, groups, strict=True):
        split_of[group].add(split)
    crossing = sum(1 for splits in split_of.values() if len(splits) > 1)
    unit = normalise(embeddings)
    labels = np.array(assignment)
    worst = -1.0
    for a in SPLITS:
        for b in SPLITS:
            if a >= b:
                continue
            ia, ib = np.flatnonzero(labels == a), np.flatnonzero(labels == b)
            if ia.size and ib.size:
                worst = max(worst, float((unit[ia] @ unit[ib].T).max()))
    return LeakageReport(crossing, worst)


def assert_no_leakage(assignment: Sequence[str], groups: Sequence[int], embeddings: Floats) -> LeakageReport:
    report = leakage(assignment, groups, embeddings)
    if not report.clean:
        raise AssertionError(f"split leakage: {report}")
    return report


def leave_one_source_out(items: Sequence[BenchItem], held_out: str) -> tuple[list[int], list[int]]:
    """(rest, held-out) item indices for one leave-one-dataset-out fold."""
    test = [i for i, item in enumerate(items) if item.source == held_out]
    rest = [i for i, item in enumerate(items) if item.source != held_out]
    return rest, test


def split_manifest(
    items: Sequence[BenchItem],
    assignment: Sequence[str],
    *,
    name: str,
    seed: int,
    dataset_manifest_sha256: str,
    note: str,
) -> dict[str, Any]:
    rows = sorted((item.item_id, item.content_hash(), split) for item, split in zip(items, assignment, strict=True))
    digest = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
    counts: dict[str, int] = defaultdict(int)
    for split in assignment:
        counts[split] += 1
    return {
        "name": name,
        "note": note,
        "seed": seed,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "near_duplicate_cosine": NEAR_DUPLICATE_COSINE,
        "counts": dict(sorted(counts.items())),
        "split_manifest_sha256": digest,
        "items": [{"item_id": i, "sha256": h, "split": s} for i, h, s in rows],
    }
