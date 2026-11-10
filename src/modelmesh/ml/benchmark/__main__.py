"""Command line for the benchmark harness: prepare, run, judge, review, report.

The harness only talks to ModelMesh's public API. `run` needs a running gateway and a key with explicit limits
(DECISIONS D1), passed through MODELMESH_BENCH_BASE_URL and MODELMESH_BENCH_API_KEY.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

from modelmesh.ml.benchmark.items import manifest_for, read_items, write_items
from modelmesh.ml.benchmark.loaders import build_items, counts_for
from modelmesh.ml.benchmark.review import agreement_report, build_review_set, read_jsonl, serve
from modelmesh.ml.benchmark.runner import DEFERRALS, JUDGEMENTS, OBSERVATIONS, WARMUPS, BenchmarkRunner, Pacing

DATA_ROOT = Path("data/benchmark")
RAW_DIR = Path("data/raw")
MANIFESTS = Path("data/manifests")
SEED = 20261108


def items_path(profile: str) -> Path:
    return DATA_ROOT / profile / "items.jsonl"


def client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=os.environ["MODELMESH_BENCH_BASE_URL"],
        headers={"Authorization": f"Bearer {os.environ['MODELMESH_BENCH_API_KEY']}"},
        timeout=300.0,
    )


async def model_families(http: httpx.AsyncClient) -> dict[str, str]:
    response = await http.get("/v1/models")
    response.raise_for_status()
    return {m["id"]: m["modelmesh"]["family"] for m in response.json()["data"] if m["modelmesh"]["family"]}


def prepare(profile: str, limit: int | None) -> None:
    counts = counts_for(profile)
    if limit is not None:
        counts = {source: min(limit, count) for source, count in counts.items()}
    items, realised = build_items(counts, seed=SEED, raw_dir=RAW_DIR)
    write_items(items_path(profile), items)
    manifest = manifest_for(items, seed=SEED, counts=realised)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    (MANIFESTS / f"{profile}.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({"items": len(items), "counts": realised, "manifest_sha256": manifest["manifest_sha256"]}))


def parse_pacing(specs: list[str], warmup: list[str]) -> dict[str, Pacing]:
    """`provider=rpm[:max_requests[:timeout_ms]]`; empty fields mean no limit. Warm-up providers run models in turn."""
    pacing: dict[str, Pacing] = {}
    for spec in specs:
        provider, _, values = spec.partition("=")
        fields = [*values.split(":"), "", ""][:3]
        pacing[provider] = Pacing(
            rpm=float(fields[0]) if fields[0] else None,
            max_requests=int(fields[1]) if fields[1] else None,
            timeout_ms=int(fields[2]) if fields[2] else None,
        )
    for provider in warmup:
        current = pacing.setdefault(provider, Pacing())
        current.warmup = True
        current.sequential_models = True
    return pacing


def lane_summary(runner: BenchmarkRunner) -> dict[str, Any]:
    return {name: {"sent": lane.sent, "stopped": lane.stopped_reason} for name, lane in runner.lanes.items()}


async def run(profile: str, run_id: str, models: list[str], pacing: dict[str, Pacing]) -> None:
    items = read_items(items_path(profile))
    async with client() as http:
        families = await model_families(http)
        runner = BenchmarkRunner(http, DATA_ROOT / profile / run_id, run_id, families=families, pacing=pacing)
        await runner.collect(items, models)
    print(json.dumps({"observations": len(runner.observations.read()), "lanes": lane_summary(runner)}))


async def judge(profile: str, run_id: str, judges: dict[str, str], pacing: dict[str, Pacing]) -> None:
    items = read_items(items_path(profile))
    async with client() as http:
        families = await model_families(http)
        runner = BenchmarkRunner(http, DATA_ROOT / profile / run_id, run_id, families=families, pacing=pacing)
        skipped = await runner.judge(items, judges)
    print(
        json.dumps(
            {
                "judgements": len(runner.judgements.read()),
                "skipped_no_independent_judge": skipped,
                "lanes": lane_summary(runner),
            }
        )
    )


def review_sample(profile: str, run_id: str) -> None:
    directory = DATA_ROOT / profile / run_id
    review = build_review_set(
        read_items(items_path(profile)),
        read_jsonl(directory / OBSERVATIONS),
        read_jsonl(directory / JUDGEMENTS),
        seed=SEED,
    )
    with (directory / "review.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in review:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"review_items": len(review)}))


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def _summary(rows: list[dict[str, Any]], deferrals: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if row["ok"]]
    latencies = [float(row["latency_ms"]) for row in ok if row["latency_ms"] is not None]
    errors: dict[str, int] = {}
    for row in rows:
        if not row["ok"]:
            key = f"{row['http_status']}:{row['error_code']}"
            errors[key] = errors.get(key, 0) + 1
    deferred: dict[str, int] = {}
    for row in deferrals:
        key = f"{row['http_status']}:{row['error_code']}"
        deferred[key] = deferred.get(key, 0) + 1
    return {
        "requests": len(rows) + len(deferrals),
        "succeeded": len(ok),
        "failed": len(rows) - len(ok),
        "errors": errors,
        "deferred_for_capacity": deferred,
        "prompt_tokens": sum(row["prompt_tokens"] or 0 for row in ok),
        "completion_tokens": sum(row["completion_tokens"] or 0 for row in ok),
        "latency_p50_ms": _percentile(latencies, 0.5),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "automatic_accept_rate": _rate(
            [row["automatic_accepted"] for row in ok if row["automatic_accepted"] is not None]
        ),
    }


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def report(profile: str, run_id: str) -> None:
    directory = DATA_ROOT / profile / run_id
    rows = read_jsonl(directory / OBSERVATIONS)
    deferrals = [row for row in read_jsonl(directory / DEFERRALS) if row["kind"] == "generation"]
    judgements = read_jsonl(directory / JUDGEMENTS)
    models = sorted({row["model"] for row in rows} | {row["model"] for row in deferrals})
    summary: dict[str, Any] = {
        model: _summary([r for r in rows if r["model"] == model], [d for d in deferrals if d["model"] == model])
        for model in models
    }
    review = read_jsonl(directory / "review.jsonl")
    labels = read_jsonl(directory / "labels.jsonl")
    output = {
        "per_model": summary,
        "warmups": read_jsonl(directory / WARMUPS),
        "judgements": {
            "total": len(judgements),
            "scored": sum(1 for row in judgements if row["score"] is not None),
            "by_judge": {
                judge: sum(1 for row in judgements if row["judge_model"] == judge)
                for judge in sorted({row["judge_model"] for row in judgements})
            },
            "deferred": sum(1 for row in read_jsonl(directory / DEFERRALS) if row["kind"] == "judge"),
        },
        "agreement": agreement_report(review, labels) if labels else None,
    }
    print(json.dumps(output, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m modelmesh.ml.benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="download datasets and write items and a manifest")
    prep.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    prep.add_argument("--limit", type=int, help="cap items per source (for smoke-testing loaders)")
    for name, helptext in (("run", "collect outputs"), ("judge", "score open-ended outputs")):
        sub = commands.add_parser(name, help=helptext)
        sub.add_argument("--profile", choices=["pilot", "full"], default="pilot")
        sub.add_argument("--run-id", required=True)
        if name == "run":
            sub.add_argument("--models", required=True, help="comma-separated model ids")
        else:
            sub.add_argument("--judges", required=True, help="comma-separated family=model pairs")
        sub.add_argument("--pace", action="append", default=[], help="provider=rpm[:max_requests[:timeout_ms]]")
        sub.add_argument("--warmup", action="append", default=[], help="provider whose models need a warm-up")
    for name in ("review-sample", "report"):
        sub = commands.add_parser(name)
        sub.add_argument("--profile", choices=["pilot", "full"], default="pilot")
        sub.add_argument("--run-id", required=True)
    label = commands.add_parser("label", help="open the local review page")
    label.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    label.add_argument("--run-id", required=True)
    label.add_argument("--port", type=int, default=8150)
    args = parser.parse_args()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    if args.command == "prepare":
        prepare(args.profile, args.limit)
    elif args.command == "run":
        asyncio.run(run(args.profile, args.run_id, args.models.split(","), parse_pacing(args.pace, args.warmup)))
    elif args.command == "judge":
        judges = dict(pair.split("=", 1) for pair in args.judges.split(","))
        asyncio.run(judge(args.profile, args.run_id, judges, parse_pacing(args.pace, args.warmup)))
    elif args.command == "review-sample":
        review_sample(args.profile, args.run_id)
    elif args.command == "report":
        report(args.profile, args.run_id)
    else:
        directory = DATA_ROOT / args.profile / args.run_id
        serve(directory / "review.jsonl", directory / "labels.jsonl", port=args.port)


if __name__ == "__main__":
    main()
