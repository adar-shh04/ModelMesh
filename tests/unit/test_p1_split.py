"""P1 pre-registration: the frozen split manifests, and the splitter options P1 uses (offline, no embeddings)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from modelmesh.ml.benchmark.items import BenchItem
from modelmesh.ml.p1 import split as p1
from modelmesh.ml.training.splits import build_groups, grouped_split, max_cross_split

MANIFESTS = Path(__file__).resolve().parents[2] / "data" / "manifests"


def test_frozen_test_manifest_hash_matches() -> None:
    split = p1.verify_frozen()
    test = json.loads(p1.TEST_MANIFEST.read_text(encoding="utf-8"))
    assert p1.FROZEN_SPLIT_SHA256 == "89aaf76128e13062547cccde4e05e9cdb12745cb7be93f8b034895009f949641"
    assert p1.FROZEN_TEST_SHA256 == "c604a5889f36dae5f815a84d75433a972bb34c438671f8fa9c3cb9b4078ca36c"
    assert test["test_manifest_sha256"] == p1.FROZEN_TEST_SHA256
    assert p1.rows_sha256(p1.test_rows(split)) == p1.FROZEN_TEST_SHA256
    assert split["counts"] == {"test": 238, "train": 723, "validation": 239}
    assert test["count"] == 238
    assert split["seed"] == p1.SEED and split["feature_version"] == "2"


def test_frozen_split_covers_exactly_the_full_dataset_manifest() -> None:
    full = json.loads((MANIFESTS / "full.json").read_text(encoding="utf-8"))
    split = json.loads(p1.SPLIT_MANIFEST.read_text(encoding="utf-8"))
    assert full["manifest_sha256"] == p1.DATASET_MANIFEST_SHA256 == split["dataset_manifest_sha256"]
    assert {r["item_id"]: r["sha256"] for r in full["items"]} == {r["item_id"]: r["sha256"] for r in split["items"]}
    assert split["leakage"]["groups_crossing"] == 0
    assert split["leakage"]["max_cross_split_near_duplicate_similarity"] <= 0.95


def test_tampered_manifest_is_rejected(tmp_path: Path) -> None:
    split = json.loads(p1.SPLIT_MANIFEST.read_text(encoding="utf-8"))
    moved = next(row for row in split["items"] if row["split"] == "train")
    moved["split"] = "test"
    (tmp_path / "split.json").write_text(json.dumps(split), encoding="utf-8")
    with pytest.raises(AssertionError):
        p1.verify_frozen(tmp_path / "split.json", p1.TEST_MANIFEST)


def item(i: int, *, source: str = "s", schema: dict[str, object] | None = None) -> BenchItem:
    return BenchItem(
        item_id=f"{source}:{i}",
        source=source,
        task_type="general",
        messages=[{"role": "user", "content": "same instruction"}],
        label_kind="schema_valid",
        response_format=None if schema is None else {"type": "json_schema", "json_schema": {"schema": schema}},
    )


def test_identical_prompts_with_different_schemas_are_not_duplicates() -> None:
    items = [item(i, schema={"title": f"s{i}"}) for i in range(10)]
    same_prompt = np.ones((10, 3))
    orthogonal = {p1.response_format_text(it) or "": np.eye(10)[k] for k, it in enumerate(items)}
    formats = p1.response_format_similarity(items, orthogonal)
    assert np.allclose(np.diag(formats), 1.0) and formats[0, 1] == 0.0
    assert len(set(build_groups(items, same_prompt))) == 1  # prompt similarity alone: one "duplicate" group
    groups = build_groups(items, same_prompt, required_similarity=formats)
    assert len(set(groups)) == 10
    assignment = grouped_split(items, groups, seed=3)
    joint = np.minimum(np.ones((10, 10)), formats)
    assert max_cross_split(assignment, joint) <= 0.95


def test_free_text_items_share_a_format_and_mixed_ones_do_not() -> None:
    items = [item(0), item(1), item(2, schema={"a": 1})]
    formats = p1.response_format_similarity(items, {p1.response_format_text(items[2]) or "": np.ones(3)})
    assert formats[0, 1] == 1.0 and formats[0, 2] == 0.0 and formats[2, 2] == 1.0


def test_largest_first_keeps_a_big_group_from_overshooting() -> None:
    items = [item(i) for i in range(100)]
    groups = [0] * 40 + list(range(40, 100))  # one 40-item group and 60 singletons in one stratum
    for seed in range(20):
        counts = Counter(grouped_split(items, groups, seed=seed, largest_first=True))
        assert counts == {"train": 60, "validation": 20, "test": 20}
    assert grouped_split(items, groups, seed=1) == grouped_split(items, groups, seed=1)  # default unchanged
