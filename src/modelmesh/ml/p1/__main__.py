"""P1 command line. Offline and read-only towards the run directories. Only ``judge --execute`` sends requests
(to the gateway's public API, like the benchmark runner); every other command makes no network call.

    python -m modelmesh.ml.p1 freeze-split [--write]   # build the pre-registered split; --write only if unfrozen
    python -m modelmesh.ml.p1 verify-split             # check the committed manifests against the pinned hashes
    python -m modelmesh.ml.p1 merge                    # merged snapshot -> data/benchmark/full/p1-merged/
    python -m modelmesh.ml.p1 status [--json]          # coverage, failures, deferrals, yield, projected finish
    python -m modelmesh.ml.p1 judge [--strategy adr|balanced] [--capacity MODEL=N]   # dry run: plan and quota
    python -m modelmesh.ml.p1 judge --execute [--pace provider=rpm:max_requests:timeout_ms]
    python -m modelmesh.ml.p1 judge-report             # canary, inter-judge kappa, label coverage
    python -m modelmesh.ml.p1 review-sample            # the D3 human-review set (pre-labels hidden by default)
    python -m modelmesh.ml.p1 label [--port 8150]      # local review page
    python -m modelmesh.ml.p1 agreement                # judge-human kappa per task type, low-trust flags
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from typing import Any

# Same client-side pacing as scripts/pilot.ps1. Groq's daily quota is shared with Mezvuat.
DEFAULT_PACE = {"groq": "groq=8:400:120000", "gemini": "gemini=8::120000"}
KAPPA_GATE = 0.6


def _judge_inputs() -> tuple[Any, Any]:
    from modelmesh.ml.p1.data import merge
    from modelmesh.ml.p1.split import load_items

    items = load_items()
    rows, _ = merge(items)
    return items, rows


def _capacity(args_capacity: list[str], judges: dict[str, str], items: Any) -> dict[str, float]:
    """Requests per day per judge model: --capacity, else the model's measured P1 yield per UTC day."""
    from modelmesh.ml.p1.data import status

    given = {model: float(value) for model, _, value in (c.rpartition("=") for c in args_capacity)}
    measured = status(items)["per_model"]
    out: dict[str, float] = {}
    for model in judges.values():
        if model in given:
            out[model] = given[model]
        else:
            rate = measured.get(model, {}).get("projection", {}).get("rate_per_day")
            if rate:
                out[model] = float(rate)
    return out


def _scope(items: Any) -> dict[str, Any]:
    from modelmesh.ml.p1.data import POOL, SPARSE, _pilot_hashes, scope_for

    pilot = _pilot_hashes()
    return {model: scope_for(model, items, pilot) for model in (*POOL, *SPARSE)}


def judge_command(args: argparse.Namespace) -> None:
    from modelmesh.ml.p1 import judging

    judges = dict(pair.split("=", 1) for pair in args.judges.split(",")) if args.judges else judging.DEFAULT_JUDGES
    items, rows = _judge_inputs()
    capacity = _capacity(args.capacity, judges, items)
    tasks, unjudgeable = judging.plan(
        items, rows, judges=judges, families=judging.catalog_families(), strategy=args.strategy, capacity=capacity,
        canary_per_judge=args.canary_per_judge, second_fraction=args.second_fraction,
    )  # fmt: skip
    existing = judging.read_judgements()
    todo = judging.pending(tasks, existing)
    full_rows = judging.full_coverage_rows(items, rows, _scope(items))
    full_tasks, _ = judging.plan(
        items, full_rows, judges=judges, families=judging.catalog_families(), strategy=args.strategy,
        capacity=capacity, canary_per_judge=args.canary_per_judge, second_fraction=args.second_fraction,
    )  # fmt: skip
    summary = {
        "mode": "execute" if args.execute else "dry-run (nothing sent)",
        "strategy": args.strategy,
        "judges": judges,
        "judge_targets": sum(1 for t in tasks if t.kind == "primary") + unjudgeable,
        "no_independent_judge": unjudgeable,
        "planned_requests": len(tasks),
        "already_done": len(tasks) - len(todo),
        "pending_requests": len(todo),
        "per_judge_pending": judging.estimate(todo, items, rows, capacity=capacity),
        "at_full_coverage_upper_bound": {
            "basis": "every in-scope P1 cell succeeds; missing outputs take the measured median output length",
            "planned_requests": len(full_tasks),
            "per_judge": judging.estimate(full_tasks, items, [*full_rows, *rows], capacity=capacity),
        },
    }
    if not args.execute:
        print(json.dumps(summary, indent=1))
        return
    import httpx

    from modelmesh.ml.benchmark.__main__ import parse_pacing

    pace = args.pace or [
        DEFAULT_PACE[p] for p in sorted({m.split("/", 1)[0] for m in judges.values()}) if p in DEFAULT_PACE
    ]

    async def go() -> dict[str, Any]:
        async with httpx.AsyncClient(
            base_url=os.environ["MODELMESH_BENCH_BASE_URL"],
            headers={"Authorization": f"Bearer {os.environ['MODELMESH_BENCH_API_KEY']}"},
            timeout=300.0,
        ) as http:
            run = judging.JudgeRun(http, judging.JUDGE_DIR, items, rows, pacing=parse_pacing(pace, []))
            return await run.run(tasks)

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print(json.dumps({**summary, "result": asyncio.run(go())}, indent=1))


def judge_report() -> None:
    from modelmesh.ml.p1 import judging

    items, rows = _judge_inputs()
    judgements = judging.read_judgements()
    task_of = {item.item_id: item.task_type for item in items}
    primary = judging.primary_scores(judgements)
    targets = judging.judge_targets(items, rows)
    labelled = {(r["item_id"], r["model"]) for r in primary}
    coverage = Counter(
        f"{r['model']} / {task_of[r['item_id']]}" for r in targets if (r["item_id"], r["model"]) in labelled
    )
    totals = Counter(f"{r['model']} / {task_of[r['item_id']]}" for r in targets)
    print(json.dumps({
        "judgements": len(judgements),
        "by_judge_and_kind": dict(sorted(Counter(f"{r['judge_model']} {r['kind']}" for r in judgements).items())),
        "unscored": sum(1 for r in judgements if r["score"] is None),
        "primary_label_coverage": {k: f"{coverage[k]}/{n}" for k, n in sorted(totals.items())},
        "canary": judging.canary_report(judgements),
        "inter_judge": judging.inter_judge_report(judgements),
    }, indent=1))  # fmt: skip


def review_sample() -> None:
    from modelmesh.ml.benchmark.review import build_review_set
    from modelmesh.ml.p1 import judging

    directory = judging.JUDGE_DIR
    if (directory / "labels.jsonl").exists():
        sys.exit("labels.jsonl exists: a new review set would re-number its rows. Move the labels aside first.")
    items, rows = _judge_inputs()
    judgements = judging.read_judgements(directory)
    review = build_review_set(items, rows, judging.primary_scores(judgements), seed=judging.SEED)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "review.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in review:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    shown = sum(1 for row in review if row["prelabel_shown"])
    print(json.dumps({"review_items": len(review), "prelabel_shown": shown, "prelabel_hidden": len(review) - shown}))


def agreement() -> None:
    from modelmesh.ml.benchmark.review import agreement_report, read_jsonl
    from modelmesh.ml.p1 import judging

    report = agreement_report(
        read_jsonl(judging.JUDGE_DIR / "review.jsonl"), read_jsonl(judging.JUDGE_DIR / "labels.jsonl")
    )
    report["low_trust_task_types"] = sorted(
        task for task, row in report["by_task"].items() if row["kappa"] is None or row["kappa"] < KAPPA_GATE
    )
    print(json.dumps(report, indent=1))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m modelmesh.ml.p1")
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-split", help="build the pre-registered P1 split from item contents")
    freeze.add_argument("--write", action="store_true", help="write the manifests (refused once frozen)")
    commands.add_parser("verify-split", help="check the committed split manifests against the frozen hashes")
    commands.add_parser("merge", help="merge the P1 run directories into one snapshot (outside them)")
    status = commands.add_parser("status", help="read-only P1 progress: coverage, failures, yield, projection")
    status.add_argument("--json", action="store_true", help="print the full JSON instead of tables")
    judge = commands.add_parser("judge", help="plan the judge step (dry run by default) or --execute it")
    judge.add_argument("--execute", action="store_true", help="send the requests (default: dry run, sends nothing)")
    judge.add_argument("--strategy", choices=["adr", "balanced"], default="adr")
    judge.add_argument("--judges", help="family=model pairs, comma-separated, in preference order")
    judge.add_argument("--capacity", action="append", default=[], help="MODEL=requests_per_day for planning")
    judge.add_argument("--pace", action="append", default=[], help="provider=rpm[:max_requests[:timeout_ms]]")
    judge.add_argument("--canary-per-judge", type=int, default=20)
    judge.add_argument("--second-fraction", type=float, default=0.2)
    commands.add_parser("judge-report", help="canary, inter-judge kappa and judge-label coverage")
    commands.add_parser("review-sample", help="write the human-review set (pre-labels hidden by default)")
    label = commands.add_parser("label", help="open the local review page")
    label.add_argument("--port", type=int, default=8150)
    commands.add_parser("agreement", help="judge-human agreement and the kappa gate per task type")
    args = parser.parse_args(argv)

    if args.command == "freeze-split":
        from modelmesh.ml.p1.split import freeze as freeze_split

        print(json.dumps(freeze_split(write=args.write), indent=1))
    elif args.command == "verify-split":
        from modelmesh.ml.p1.split import verify_frozen

        split = verify_frozen()
        print(json.dumps({"ok": True, "split_manifest_sha256": split["split_manifest_sha256"], **split["counts"]}))
    elif args.command == "merge":
        from modelmesh.ml.p1.data import merge, write_merged
        from modelmesh.ml.p1.split import load_items

        rows, report = merge(load_items())
        target = write_merged(rows, report)
        print(json.dumps({"written": str(target), **report.as_dict()}, indent=1))
    elif args.command == "status":
        from modelmesh.ml.p1.data import render_status
        from modelmesh.ml.p1.data import status as p1_status
        from modelmesh.ml.p1.split import load_items

        doc = p1_status(load_items())
        print(json.dumps(doc, indent=1) if args.json else render_status(doc))
    elif args.command == "judge":
        judge_command(args)
    elif args.command == "judge-report":
        judge_report()
    elif args.command == "review-sample":
        review_sample()
    elif args.command == "label":
        from modelmesh.ml.benchmark.review import serve
        from modelmesh.ml.p1.judging import JUDGE_DIR

        serve(JUDGE_DIR / "review.jsonl", JUDGE_DIR / "labels.jsonl", port=args.port)
    elif args.command == "agreement":
        agreement()
    else:  # pragma: no cover - argparse rejects unknown commands
        sys.exit(2)


if __name__ == "__main__":
    main()
