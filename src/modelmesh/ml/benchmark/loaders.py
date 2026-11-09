"""Downloads the public datasets of ADR-0005 and builds benchmark items. Caches live under HF_HOME or data/raw."""

from __future__ import annotations

import io
import json
import random
import tarfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from modelmesh.ml.benchmark import templates
from modelmesh.ml.benchmark.items import BenchItem

FULL_COUNTS: dict[str, int] = {
    "banking77": 100,
    "clinc150": 100,
    "qrecc": 120,
    "musique_plan": 100,
    "musique_suff": 80,
    "qasper": 50,
    "musique_synth": 50,
    "scifact": 90,
    "attributionbench": 90,
    "ifeval": 120,
    "gsm8k": 80,
    "mmlu_pro": 80,
    "mt_bench": 60,
    "jsonschemabench": 80,
}
PILOT_PER_SOURCE = 20

SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
QRECC_URL = "https://github.com/apple/ml-qrecc/raw/main/dataset/qrecc_data.zip"
MT_BENCH_URL = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"


def _load_hf(name: str, config: str | None, split: str) -> Any:
    from datasets import load_dataset

    try:
        return load_dataset(name, config, split=split)
    except (RuntimeError, ValueError):  # Script-based repos are served through their parquet conversion.
        return load_dataset(name, config, split=split, revision="refs/convert/parquet")


def _download(url: str, destination: Path) -> bytes:
    if destination.exists():
        return destination.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = httpx.get(url, follow_redirects=True, timeout=300.0)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return response.content


def _sample[T](rows: list[T], count: int, rng: random.Random) -> list[T]:
    if len(rows) <= count:
        return rows
    return rng.sample(rows, count)


def _build_banking77(count: int, rng: random.Random, raw: Path) -> list[BenchItem]:
    dataset = _load_hf("PolyAI/banking77", None, "test")
    labels: list[str] = dataset.features["label"].names
    rows = _sample(list(enumerate(dataset)), count, rng)
    return [templates.intent_item("banking77", str(i), r["text"], labels[r["label"]], labels, rng) for i, r in rows]


def _build_clinc150(count: int, rng: random.Random, raw: Path) -> list[BenchItem]:
    dataset = _load_hf("clinc/clinc_oos", "plus", "test")
    labels: list[str] = [str(name) for name in dataset.features["intent"].names]
    in_scope = [(i, row) for i, row in enumerate(dataset) if labels[row["intent"]] != "oos"]
    rows = _sample(in_scope, count, rng)
    valid = [label for label in labels if label != "oos"]
    return [templates.intent_item("clinc150", str(i), r["text"], labels[r["intent"]], valid, rng) for i, r in rows]


def _build_qrecc(count: int, rng: random.Random, raw: Path) -> list[BenchItem]:
    archive = zipfile.ZipFile(io.BytesIO(_download(QRECC_URL, raw / "qrecc_data.zip")))
    with archive.open("qrecc-test.json") as handle:
        records = json.load(handle)
    usable = [r for r in records if r.get("Context") and r.get("Truth_rewrite")]
    return [
        templates.rewrite_item(
            "qrecc", f"{r['Conversation_no']}_{r['Turn_no']}", r["Context"], r["Question"], r["Truth_rewrite"]
        )
        for r in _sample(usable, count, rng)
    ]


def _musique(config: str) -> list[dict[str, Any]]:
    return list(_load_hf("bdsaglam/musique", config, "validation"))


def _build_musique_plan(count: int, rng: random.Random, raw: Path) -> list[BenchItem]:
    return [templates.planning_item(r) for r in _sample(_musique("answerable"), count, rng)]


def _build_musique_synth(count: int, rng: random.Random, raw: Path) -> list[BenchItem]:
    return [templates.musique_synthesis_item(r) for r in _sample(_musique("answerable"), count, rng)]


def _build_musique_suff(count: int, rng: random.Random, raw: Path) -> list[BenchItem]:
    rows = _musique("default")
    half = count // 2
    positive = _sample([r for r in rows if r["answerable"]], half, rng)
    negative = _sample([r for r in rows if not r["answerable"]], count - half, rng)
    return [templates.sufficiency_item(r) for r in [*positive, *negative]]


def _build_qasper(count: int, rng: random.Random, raw: Path) -> list[BenchItem]:
    papers = list(_load_hf("allenai/qasper", "qasper", "test"))
    items = [item for paper in papers for item in templates.qasper_items(paper)]
    return _sample(items, count, rng)


def _build_scifact(count: int, rng: random.Random, raw: Path) -> list[BenchItem]:
    blob = _download(SCIFACT_URL, raw / "scifact_data.tar.gz")
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:

        def read_jsonl(member: str) -> list[dict[str, Any]]:
            handle = archive.extractfile(member)
            assert handle is not None
            return [json.loads(line) for line in handle if line.strip()]

        claims = read_jsonl("data/claims_dev.jsonl")
        corpus = {doc["doc_id"]: doc for doc in read_jsonl("data/corpus.jsonl")}
    pairs = [
        (claim, corpus[doc_id]) for claim in claims for doc_id in claim.get("cited_doc_ids", []) if doc_id in corpus
    ]
    return [templates.scifact_item(claim, doc) for claim, doc in _sample(pairs, count, rng)]


def _build_attribution(count: int, rng: random.Random, raw: Path) -> list[BenchItem]:
    rows = list(_load_hf("osunlp/AttributionBench", "full_data", "test_ood"))
    return [templates.attribution_item(r) for r in _sample(rows, count, rng)]


def _build_ifeval(count: int, rng: random.Random, raw: Path) -> list[BenchItem]:
    built = [item for row in _load_hf("google/IFEval", None, "train") if (item := templates.ifeval_item(row))]
    return _sample(built, count, rng)


def _build_gsm8k(count: int, rng: random.Random, raw: Path) -> list[BenchItem]:
    rows = list(enumerate(_load_hf("openai/gsm8k", "main", "test")))
    return [templates.gsm8k_item(str(i), r) for i, r in _sample(rows, count, rng)]


def _build_mmlu(count: int, rng: random.Random, raw: Path) -> list[BenchItem]:
    return [templates.mmlu_item(r) for r in _sample(list(_load_hf("TIGER-Lab/MMLU-Pro", None, "test")), count, rng)]


def _build_mt_bench(count: int, rng: random.Random, raw: Path) -> list[BenchItem]:
    text = _download(MT_BENCH_URL, raw / "mt_bench_questions.jsonl").decode()
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [templates.mt_bench_item(r) for r in _sample(rows, count, rng)]


def _build_json_schema(count: int, rng: random.Random, raw: Path) -> list[BenchItem]:
    rows = list(_load_hf("epfl-dlab/JSONSchemaBench", "default", "test"))
    return [item for row in _sample(rows, count * 2, rng) if (item := templates.json_schema_item(row))][:count]


BUILDERS: dict[str, Callable[[int, random.Random, Path], list[BenchItem]]] = {
    "banking77": _build_banking77,
    "clinc150": _build_clinc150,
    "qrecc": _build_qrecc,
    "musique_plan": _build_musique_plan,
    "musique_suff": _build_musique_suff,
    "musique_synth": _build_musique_synth,
    "qasper": _build_qasper,
    "scifact": _build_scifact,
    "attributionbench": _build_attribution,
    "ifeval": _build_ifeval,
    "gsm8k": _build_gsm8k,
    "mmlu_pro": _build_mmlu,
    "mt_bench": _build_mt_bench,
    "jsonschemabench": _build_json_schema,
}


def counts_for(profile: str) -> dict[str, int]:
    if profile == "pilot":
        return dict.fromkeys(FULL_COUNTS, PILOT_PER_SOURCE)
    if profile == "full":
        return dict(FULL_COUNTS)
    raise ValueError(f"unknown profile: {profile}")


def build_items(counts: dict[str, int], *, seed: int, raw_dir: Path) -> tuple[list[BenchItem], dict[str, int]]:
    items: list[BenchItem] = []
    realised: dict[str, int] = {}
    for source, count in counts.items():
        built = BUILDERS[source](count, random.Random(f"{seed}:{source}"), raw_dir)  # noqa: S311 - sampling, not security
        items.extend(built)
        realised[source] = len(built)
    return items, realised
