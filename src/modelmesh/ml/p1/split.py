"""P1 pre-registration: the grouped, near-duplicate-aware split of the full item set (ADR-0004).

The split is built from ``data/manifests/full.json`` and the item contents only. No observation, judgement or
label is read, so the test set cannot depend on any P1 outcome. It was frozen on 2026-11-17, while P1 was
still collecting and before any P1 outcome had been looked at (``docs/milestones/P1_PREREGISTRATION.md``).

- Splitter: ``modelmesh.ml.training.splits`` as in M6 (groups by source question, then 60/20/20 within each
  (task_type, source) stratum).
- Near-duplicates: a pair is grouped when its prompt cosine is above 0.95 in **either** embedding space, the
  feature-version-1 grouping text (512 tokens, what M6 grouped on) or the feature-version-2 router text
  (128 tokens, what the P1 router sees), **and** the response formats the two items require are near-identical
  too (identical, or schema-text cosine above 0.95). The second condition matters for JSONSchemaBench: its 80
  prompts are one identical instruction and the item is the schema, so prompt similarity alone made all 80 one
  "duplicate" group with no test items. It can only split a prompt-based group, never merge two.
- Assignment: M6's seeded random group order, then largest groups first (``largest_first``). Template-similar
  classification prompts chain into groups of 43 and 63 items; in random order such a group can arrive last
  and overshoot train, leaving a source with a handful of test items.

    python -m modelmesh.ml.p1 freeze-split     # rebuilds and checks against the frozen hashes; never rewrites
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from modelmesh.ml.benchmark.items import BenchItem, read_items
from modelmesh.ml.evaluation.metrics import Floats
from modelmesh.ml.features import EMBEDDING_MAX_TOKENS, FEATURE_VERSION
from modelmesh.ml.training.dataset import verify_manifest
from modelmesh.ml.training.splits import (
    NEAR_DUPLICATE_COSINE,
    build_groups,
    grouped_split,
    grouping_text,
    leakage,
    max_cross_split,
    normalise,
    split_manifest,
)

ROOT = Path(__file__).resolve().parents[4]
FULL = ROOT / "data" / "benchmark" / "full"
ITEMS = FULL / "items.jsonl"
MANIFESTS = ROOT / "data" / "manifests"
DATASET_MANIFEST = MANIFESTS / "full.json"
SPLIT_MANIFEST = MANIFESTS / "p1_router_split.json"
TEST_MANIFEST = MANIFESTS / "p1_test.json"
# Gitignored, outside every run directory.
EMBEDDINGS = FULL / "derived" / "embeddings.npz"

SEED = 20261117
SPLIT_NAME = "p1-2026-11-17"
FROZEN_ON = "2026-11-17"
DATASET_MANIFEST_SHA256 = "832ce76c96a5f59c0475073f0564387887401a964d1803612ff572dd54090c7c"
# Pinned when the split was frozen. Changing any of these is a change to the pre-registration: it must be
# recorded in docs/milestones/P1_PREREGISTRATION.md with a reason, and cannot follow from looking at results.
FROZEN_SPLIT_SHA256 = "89aaf76128e13062547cccde4e05e9cdb12745cb7be93f8b034895009f949641"
FROZEN_TEST_SHA256 = "c604a5889f36dae5f815a84d75433a972bb34c438671f8fa9c3cb9b4078ca36c"


def rows_sha256(rows: Sequence[Sequence[str]]) -> str:
    return hashlib.sha256(json.dumps([list(r) for r in rows]).encode()).hexdigest()


def test_rows(split: dict[str, Any]) -> list[tuple[str, str]]:
    return sorted((row["item_id"], row["sha256"]) for row in split["items"] if row["split"] == "test")


def response_format_text(item: BenchItem) -> str | None:
    return None if item.response_format is None else json.dumps(item.response_format, sort_keys=True)


def response_format_similarity(items: Sequence[BenchItem], vectors: Mapping[str, Floats]) -> Floats:
    """Items x items: 1 for the same required format (or both free text), 0 when only one has a format,
    otherwise the cosine of the two schema texts (``vectors`` maps each distinct text to its embedding)."""
    texts = [response_format_text(item) for item in items]
    present = [k for k, t in enumerate(texts) if t is not None]
    sim = np.zeros((len(texts), len(texts)))
    if present:
        unit = normalise(np.vstack([vectors[str(texts[k])] for k in present]))
        sim[np.ix_(present, present)] = unit @ unit.T
    codes = {t: k for k, t in enumerate(dict.fromkeys(texts))}
    code = np.array([codes[t] for t in texts])
    sim[code[:, None] == code[None, :]] = 1.0
    return sim


def build(
    items: Sequence[BenchItem], grouping: Floats, router: Floats, formats: Floats
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(split manifest, test manifest) from the items, their two prompt embeddings and the response-format
    similarity. Deterministic given the seed."""
    groups = build_groups(items, grouping, extra_embeddings=(router,), required_similarity=formats)
    assignment = grouped_split(items, groups, seed=SEED, largest_first=True)
    leak_grouping = leakage(assignment, groups, grouping)
    leak_router = leakage(assignment, groups, router)
    g, r = normalise(grouping), normalise(router)
    joint = np.minimum(np.maximum(g @ g.T, r @ r.T), formats)
    joint_max = max_cross_split(assignment, joint)
    if leak_grouping.groups_crossing or joint_max > NEAR_DUPLICATE_COSINE:
        raise AssertionError(f"P1 split leakage: crossing={leak_grouping.groups_crossing} joint={joint_max}")
    manifest = split_manifest(
        items,
        assignment,
        name=SPLIT_NAME,
        seed=SEED,
        dataset_manifest_sha256=DATASET_MANIFEST_SHA256,
        note=(
            "P1 pre-registered split, built from item contents only before any P1 outcome was read. "
            "See docs/milestones/P1_PREREGISTRATION.md. The test split is touched once per reported router version."
        ),
    )
    strata: Counter[str] = Counter()
    for item, split in zip(items, assignment, strict=True):
        strata[f"{item.task_type}/{item.source}/{split}"] += 1
    sizes = Counter(Counter(groups).values())
    manifest.update(
        {
            "frozen_on": FROZEN_ON,
            "feature_version": FEATURE_VERSION,
            "grouping": {
                "source_question": "items sharing source and id suffix",
                "near_duplicate_spaces": [
                    "feature-version-1 grouping text (system prompt + final user message, first 4,096 chars, "
                    "512 tokens)",
                    f"feature-version-{FEATURE_VERSION} router text ({EMBEDDING_MAX_TOKENS} tokens)",
                ],
                "required_also": "response formats identical, or schema-text cosine above the threshold",
                "near_duplicate_cosine": NEAR_DUPLICATE_COSINE,
                "assignment": "seeded random group order, then largest group first, 60/20/20 per stratum",
                "groups": len(set(groups)),
                "group_sizes": {str(k): v for k, v in sorted(sizes.items())},
            },
            "leakage": {
                "groups_crossing": leak_grouping.groups_crossing,
                "max_cross_split_near_duplicate_similarity": round(joint_max, 6),
                "max_cross_split_prompt_cosine_grouping_text": round(leak_grouping.max_cross_split_cosine, 6),
                "max_cross_split_prompt_cosine_router_text": round(leak_router.max_cross_split_cosine, 6),
                "note": (
                    "near-duplicate similarity = min(max of the two prompt cosines, response-format similarity). "
                    "A prompt-only cosine above 0.95 across splits comes from items whose prompts are (nearly) "
                    "the same and whose required schemas differ."
                ),
            },
            "counts_by_stratum": dict(sorted(strata.items())),
        }
    )
    rows = test_rows(manifest)
    test = {
        "name": f"{SPLIT_NAME}-test",
        "frozen_on": FROZEN_ON,
        "note": "Frozen P1 test set. Do not use for model selection; evaluate once per reported router version.",
        "split_manifest_sha256": manifest["split_manifest_sha256"],
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "feature_version": FEATURE_VERSION,
        "seed": SEED,
        "count": len(rows),
        "test_manifest_sha256": rows_sha256(rows),
        "items": [{"item_id": i, "sha256": h} for i, h in rows],
    }
    return manifest, test


def embeddings(items: Sequence[BenchItem]) -> tuple[Floats, Floats, Floats]:
    """(grouping-text, router-text, response-format similarity), with embeddings cached on disk. Loads
    fastembed; offline, one thread."""
    from modelmesh.ml.training.embedding import CachedEmbedder

    os.environ.setdefault("FASTEMBED_CACHE_PATH", str(ROOT / ".cache" / "fastembed"))
    embedder = CachedEmbedder(EMBEDDINGS)
    grouping = embedder.embed([grouping_text(item) for item in items])
    router = embedder.embed_requests([item.messages for item in items])
    distinct = sorted({t for t in (response_format_text(item) for item in items) if t is not None})
    vectors = dict(zip(distinct, embedder.embed(distinct), strict=True)) if distinct else {}
    return grouping, router, response_format_similarity(items, vectors)


def load_items() -> list[BenchItem]:
    items = read_items(ITEMS)
    if verify_manifest(items, DATASET_MANIFEST) != DATASET_MANIFEST_SHA256:
        raise ValueError("data/manifests/full.json is not the dataset manifest P1 was pre-registered on")
    return items


def write_json(path: Path, doc: dict[str, Any]) -> None:
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")


def freeze(*, write: bool) -> dict[str, Any]:
    """Builds the split. Writes the manifests only when ``write`` is set and nothing is frozen yet."""
    items = load_items()
    manifest, test = build(items, *embeddings(items))
    frozen = FROZEN_SPLIT_SHA256 != "PENDING"
    if frozen:
        rebuilt = (manifest["split_manifest_sha256"], test["test_manifest_sha256"])
        if rebuilt != (FROZEN_SPLIT_SHA256, FROZEN_TEST_SHA256):
            raise AssertionError("the rebuilt P1 split differs from the frozen one")
    elif write:
        write_json(SPLIT_MANIFEST, manifest)
        write_json(TEST_MANIFEST, test)
    return {
        "frozen": frozen,
        "written": write and not frozen,
        "split_manifest_sha256": manifest["split_manifest_sha256"],
        "test_manifest_sha256": test["test_manifest_sha256"],
        "counts": manifest["counts"],
        "groups": manifest["grouping"]["groups"],
        "leakage": manifest["leakage"],
    }


def verify_frozen(split_path: Path = SPLIT_MANIFEST, test_path: Path = TEST_MANIFEST) -> dict[str, Any]:
    """Checks the committed manifests against the pinned hashes without any embedding. Raises on a mismatch."""
    split: dict[str, Any] = json.loads(split_path.read_text(encoding="utf-8"))
    test: dict[str, Any] = json.loads(test_path.read_text(encoding="utf-8"))
    all_rows = sorted((r["item_id"], r["sha256"], r["split"]) for r in split["items"])
    if not rows_sha256(all_rows) == split["split_manifest_sha256"] == FROZEN_SPLIT_SHA256:
        raise AssertionError("p1_router_split.json does not match the frozen split hash")
    rows = test_rows(split)
    listed = sorted((r["item_id"], r["sha256"]) for r in test["items"])
    if listed != rows or not rows_sha256(rows) == test["test_manifest_sha256"] == FROZEN_TEST_SHA256:
        raise AssertionError("p1_test.json does not match the frozen test hash")
    pointers = (split["dataset_manifest_sha256"], test["split_manifest_sha256"])
    if pointers != (DATASET_MANIFEST_SHA256, FROZEN_SPLIT_SHA256):
        raise AssertionError("the P1 manifests do not point at the pre-registered dataset and split")
    return split


def split_of() -> dict[str, str]:
    """item_id -> train / validation / test, from the verified frozen manifest."""
    return {row["item_id"]: row["split"] for row in verify_frozen()["items"]}
