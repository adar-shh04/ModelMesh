"""The P1 dataset: several run directories merged into one logical set, and a read-only status view.

P1 is collected by two relaunch loops (``scripts/p1_loop.ps1``) into separate run directories, because two
processes must never append to one file. OpenRouter stays on the pilot run (D2). This module only **reads**
those directories: files are opened read-only, a partially written last line is skipped (the runner may be in
the middle of appending it), and nothing is ever written next to them.

Merge rule: one observation per (item_id, model), the first one recorded (earliest ``at``; ties go to the source
listed first, then to file order). Every dropped row is counted and reported, never silently discarded.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from modelmesh.ml.benchmark.items import BenchItem

ROOT = Path(__file__).resolve().parents[4]
BENCH = ROOT / "data" / "benchmark"
MERGED = BENCH / "full" / "p1-merged"

# The routed P1 pool (D2): every one of these runs on all 1,200 items.
POOL = (
    "gemini/gemini-3.1-flash-lite",
    "groq/openai/gpt-oss-20b",
    "groq/openai/gpt-oss-120b",
    "ollama/llama3.2:3b",
    "ollama/qwen3:4b",
)
# Kept on the 280-item pilot set (D2). Reported descriptively, never routed.
SPARSE = ("openrouter/nvidia/nemotron-3.5-super-vl:free", "openrouter/liquid/lfm-3-1.7b:free")
RATE_WINDOW = timedelta(days=3)
LAST = datetime.max.replace(tzinfo=UTC)


@dataclass(frozen=True)
class Source:
    """One run directory and the models P1 takes from it."""

    name: str
    directory: Path
    models: tuple[str, ...]
    item_scope: str  # "full": P1 items; "pilot": pilot items whose content equals the P1 item's


SOURCES = (
    Source("cloud", BENCH / "full" / "full-2026-11-16", POOL, "full"),
    Source("local", BENCH / "full" / "full-2026-11-16-local", POOL, "full"),
    Source("openrouter-pilot", BENCH / "pilot" / "pilot-2026-11-11", SPARSE, "pilot"),
)


@dataclass
class ReadResult:
    rows: list[dict[str, Any]]
    bytes_read: int
    sha256: str
    partial_last_line: bool
    malformed_lines: int


def read_jsonl_snapshot(path: Path) -> ReadResult:
    """Reads complete lines only. Bytes after the last newline (a row still being appended) are left out."""
    if not path.exists():
        return ReadResult([], 0, hashlib.sha256(b"").hexdigest(), False, 0)
    with path.open("rb") as handle:  # read-only; never opened for writing
        data = handle.read()
    end = data.rfind(b"\n") + 1
    complete, tail = data[:end], data[end:]
    rows: list[dict[str, Any]] = []
    malformed = 0
    for line in complete.splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            malformed += 1
    return ReadResult(rows, end, hashlib.sha256(complete).hexdigest(), bool(tail.strip()), malformed)


@dataclass
class MergeReport:
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    kept: int = 0
    dropped_duplicates: Counter[str] = field(default_factory=Counter)  # "source:model" of the dropped copy
    duplicate_pairs: int = 0
    duplicates_with_different_ok: int = 0
    out_of_scope: Counter[str] = field(default_factory=Counter)  # reason -> rows

    def as_dict(self) -> dict[str, Any]:
        return {
            "sources": self.sources,
            "kept": self.kept,
            "duplicate_pairs": self.duplicate_pairs,
            "dropped_duplicates": dict(sorted(self.dropped_duplicates.items())),
            "duplicates_with_different_ok": self.duplicates_with_different_ok,
            "out_of_scope": dict(sorted(self.out_of_scope.items())),
        }


def _pilot_hashes() -> dict[str, str]:
    path = ROOT / "data" / "manifests" / "pilot.json"
    return {row["item_id"]: row["sha256"] for row in json.loads(path.read_text(encoding="utf-8"))["items"]}


def merge(
    items: Sequence[BenchItem],
    sources: Sequence[Source] = SOURCES,
    *,
    pilot_hashes: Mapping[str, str] | None = None,
    file_name: str = "observations.jsonl",
) -> tuple[list[dict[str, Any]], MergeReport]:
    """Merges the sources' rows of ``file_name`` into one row per (item_id, model)."""
    full_hashes = {item.item_id: item.content_hash() for item in items}
    pilot = dict(pilot_hashes) if pilot_hashes is not None else None
    report = MergeReport()
    candidates: list[tuple[datetime, int, int, dict[str, Any]]] = []
    for rank, source in enumerate(sources):
        read = read_jsonl_snapshot(source.directory / file_name)
        report.sources[source.name] = {
            "path": _display(source.directory / file_name),
            "rows": len(read.rows),
            "bytes_read": read.bytes_read,
            "sha256_of_bytes_read": read.sha256,
            "partial_last_line_skipped": read.partial_last_line,
            "malformed_lines": read.malformed_lines,
        }
        if source.item_scope == "pilot" and pilot is None:
            pilot = _pilot_hashes()
        for position, row in enumerate(read.rows):
            item_id, model = row.get("item_id"), row.get("model")
            if model not in source.models:
                report.out_of_scope[f"{source.name}:model_not_in_p1:{model}"] += 1
                continue
            if item_id not in full_hashes:
                report.out_of_scope[f"{source.name}:item_not_in_p1"] += 1
                continue
            if source.item_scope == "pilot" and (pilot or {}).get(item_id) != full_hashes[item_id]:
                report.out_of_scope[f"{source.name}:pilot_content_differs_from_p1"] += 1
                continue
            at = _parse_at(row.get("at")) or LAST  # a row without a time sorts after every timed one
            candidates.append((at, rank, position, {**row, "p1_source": source.name}))
    candidates.sort(key=lambda c: (c[0], c[1], c[2]))
    kept: dict[tuple[str, str], dict[str, Any]] = {}
    for _, _, _, row in candidates:
        key = (row["item_id"], row["model"])
        first = kept.get(key)
        if first is None:
            kept[key] = row
            continue
        report.dropped_duplicates[f"{row['p1_source']}:{row['model']}"] += 1
        if bool(first.get("ok")) != bool(row.get("ok")):
            report.duplicates_with_different_ok += 1
    counts = Counter((r["item_id"], r["model"]) for *_, r in candidates)
    report.duplicate_pairs = sum(1 for n in counts.values() if n > 1)
    report.kept = len(kept)
    merged = sorted(kept.values(), key=lambda r: (_parse_at(r.get("at")) or LAST, r["item_id"], r["model"]))
    return merged, report


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def write_merged(rows: Iterable[dict[str, Any]], report: MergeReport, directory: Path = MERGED) -> Path:
    """Writes the merged snapshot outside every run directory. Refuses to write into a source directory."""
    resolved = directory.resolve()
    for source in SOURCES:
        if resolved == source.directory.resolve() or source.directory.resolve() in resolved.parents:
            raise ValueError(f"refusing to write into run directory {source.directory}")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "observations.jsonl"
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    doc = {"merged_at": datetime.now(UTC).isoformat(), **report.as_dict()}
    doc["merged_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    (directory / "merge_report.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    return target


# --------------------------------------------------------------------------- status
def _parse_at(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def scope_for(model: str, items: Sequence[BenchItem], pilot_hashes: Mapping[str, str]) -> list[BenchItem]:
    """The items a model is meant to cover in P1: all of them for the pool, the joinable pilot items otherwise."""
    if model in POOL:
        return list(items)
    return [item for item in items if pilot_hashes.get(item.item_id) == item.content_hash()]


# Cloud lanes are bound by daily free quotas that reset at a fixed time, so their rate is the yield per UTC day.
# Ollama has no quota: it runs continuously, one model after another, so its rate is per elapsed time and lane-wide.
CONTINUOUS_PROVIDERS = frozenset({"ollama"})
DAILY_WINDOW_DAYS = 3


def _finish(rate: float, remaining: int, now: datetime, method: str, basis: str) -> dict[str, Any]:
    if rate <= 0:
        return {"method": method, "basis": basis, "rate_per_day": 0.0, "days_left": None, "finish": "stalled"}
    days = remaining / rate
    return {
        "method": method,
        "basis": basis,
        "rate_per_day": round(rate, 1),
        "days_left": round(days, 1),
        "finish": (now + timedelta(days=days)).date().isoformat(),
    }


def projection(times: Sequence[datetime], remaining: int, now: datetime, *, continuous: bool = False) -> dict[str, Any]:
    """Projected finish from the measured rate.

    - daily (quota-bound lanes): mean observations per UTC day over the last ``DAILY_WINDOW_DAYS`` days on
      which the lane recorded anything, today's partial day included as it stands;
    - continuous (local lane): observations per elapsed day over the last ``RATE_WINDOW``, or since the first row.
    """
    method = "continuous" if continuous else "daily"
    if remaining <= 0:
        return {"method": method, "basis": "", "rate_per_day": None, "days_left": 0.0, "finish": "done"}
    if not times:
        return {"method": method, "basis": "no rows yet", "rate_per_day": None, "days_left": None, "finish": "no data"}
    if continuous:
        start = max(now - RATE_WINDOW, min(times))
        window = max((now - start).total_seconds() / 86400, 1 / 24)
        recent = sum(1 for t in times if t >= start)
        return _finish(recent / window, remaining, now, method, f"{recent} rows in {window * 24:.1f} h")
    per_day = Counter(t.date() for t in times)
    active = sorted(per_day)[-DAILY_WINDOW_DAYS:]
    if (now.date() - active[-1]).days >= DAILY_WINDOW_DAYS:
        basis = f"last row {active[-1]}"
        return {"method": method, "basis": basis, "rate_per_day": 0.0, "days_left": None, "finish": "stalled"}
    total = sum(per_day[d] for d in active)
    partial = " (today partial)" if active[-1] == now.date() else ""
    return _finish(total / len(active), remaining, now, method, f"{total} rows on {len(active)} day(s){partial}")


def status(
    items: Sequence[BenchItem],
    *,
    sources: Sequence[Source] = SOURCES,
    now: datetime | None = None,
    pilot_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Coverage, failures, deferrals, daily yield and a projected finish. Reads only; reports no acceptance."""
    now = now or datetime.now(UTC)
    pilot = dict(pilot_hashes) if pilot_hashes is not None else _pilot_hashes()
    rows, report = merge(items, sources, pilot_hashes=pilot)
    task_of = {item.item_id: item.task_type for item in items}
    tasks = sorted(set(task_of.values()))
    models = [*POOL, *SPARSE]
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    deferrals: dict[str, Counter[str]] = defaultdict(Counter)
    for source in sources:
        for row in read_jsonl_snapshot(source.directory / "deferrals.jsonl").rows:
            if row.get("model") in source.models and row.get("kind", "generation") == "generation":
                deferrals[row["model"]][f"{row.get('http_status')}:{row.get('error_code')}"] += 1

    per_model: dict[str, Any] = {}
    days: set[str] = set()
    lane_times: dict[str, list[datetime]] = {}
    for model in models:
        scope = scope_for(model, items, pilot)
        scope_ids = {item.item_id for item in scope}
        model_rows = [row for row in by_model.get(model, []) if row["item_id"] in scope_ids]
        coverage = {
            task: f"{sum(1 for r in model_rows if task_of[r['item_id']] == task)}/"
            f"{sum(1 for i in scope if i.task_type == task)}"
            for task in tasks
        }
        failures = Counter(f"{r.get('http_status')}:{r.get('error_code')}" for r in model_rows if not r.get("ok"))
        times = [t for t in (_parse_at(r.get("at")) for r in model_rows) if t is not None]
        yield_by_day = Counter(t.date().isoformat() for t in times)
        days.update(yield_by_day)
        per_model[model] = {
            "scope": len(scope),
            "observed": len(model_rows),
            "remaining": len(scope) - len(model_rows),
            "coverage_by_task": coverage,
            "failed": sum(failures.values()),
            "failures": dict(failures.most_common()),
            "deferred": sum(deferrals[model].values()),
            "deferrals": dict(deferrals[model].most_common()),
            "yield_by_day": dict(sorted(yield_by_day.items())),
            "projection": projection(times, len(scope) - len(model_rows), now),
        }
        lane_times.setdefault(model.split("/", 1)[0], []).extend(times)
    # A continuous lane runs its models in turn: project the lane's remaining cells at the lane's rate.
    for provider in sorted(CONTINUOUS_PROVIDERS):
        lane_models = [m for m in models if m.split("/", 1)[0] == provider]
        remaining = sum(per_model[m]["remaining"] for m in lane_models)
        lane = projection(lane_times.get(provider, []), remaining, now, continuous=True)
        for m in lane_models:
            per_model[m]["projection"] = {**lane, "basis": f"{provider} lane, {remaining} cells left: {lane['basis']}"}
    pool_finish = [per_model[m]["projection"]["finish"] for m in POOL]
    loops = {}
    for source in sources[:2]:
        log = read_jsonl_lines_text(source.directory / "loop.log")
        loops[source.name] = log[-1] if log else None
    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "merge": report.as_dict(),
        "tasks": tasks,
        "days": sorted(days),
        "per_model": per_model,
        "pool_complete_items": _complete(rows, items),
        "pool_projected_finish": pool_finish_date(pool_finish),
        "last_loop_log_line": loops,
    }


def pool_finish_date(finishes: Sequence[str]) -> str:
    """The last pool lane's finish date, "done", or why there is none."""
    if all(f == "done" for f in finishes):
        return "done"
    dated = sorted(f for f in finishes if f[:1].isdigit())
    if len(dated) + finishes.count("done") == len(finishes):
        return dated[-1]
    return "unknown (a pool lane has no measured rate)"


def read_jsonl_lines_text(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("rb") as handle:
        return [line.decode("utf-8", "replace").strip() for line in handle.read().splitlines() if line.strip()]


def _complete(rows: Sequence[dict[str, Any]], items: Sequence[BenchItem]) -> int:
    have: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["model"] in POOL:
            have[row["item_id"]].add(row["model"])
    return sum(1 for item in items if len(have[item.item_id]) == len(POOL))


def render_status(doc: Mapping[str, Any]) -> str:
    """Plain-text tables for the terminal."""
    lines = [f"P1 status at {doc['generated_at']} (read-only; acceptance is never shown here)", ""]
    tasks: list[str] = doc["tasks"]
    short = {t: t.replace("_", " ")[:12] for t in tasks}
    lines.append("Coverage (observed cells, any outcome / cells in scope):")
    header = f"{'model':46} " + " ".join(f"{short[t]:>12}" for t in tasks) + f" {'total':>11}"
    lines.append(header)
    for model, info in doc["per_model"].items():
        cells = " ".join(f"{info['coverage_by_task'][t]:>12}" for t in tasks)
        lines.append(f"{model:46} {cells} {info['observed']:>5}/{info['scope']:<5}")
    lines += ["", f"Items with an outcome for all {len(POOL)} pool models: {doc['pool_complete_items']}", ""]
    lines.append(f"{'model':46} {'failed':>6} {'deferred':>8}  top failure codes / deferral codes")
    for model, info in doc["per_model"].items():
        top_f = ", ".join(f"{k} x{v}" for k, v in list(info["failures"].items())[:3]) or "-"
        top_d = ", ".join(f"{k} x{v}" for k, v in list(info["deferrals"].items())[:2]) or "-"
        lines.append(f"{model:46} {info['failed']:>6} {info['deferred']:>8}  {top_f} | {top_d}")
    days: list[str] = doc["days"]
    lines += ["", "Yield per UTC day (observations recorded):"]
    lines.append(f"{'model':46} " + " ".join(f"{d[5:]:>6}" for d in days))
    for model, info in doc["per_model"].items():
        lines.append(f"{model:46} " + " ".join(f"{info['yield_by_day'].get(d, 0):>6}" for d in days))
    lines += [
        "",
        f"Projection. Cloud lanes: mean yield per UTC day over the last {DAILY_WINDOW_DAYS} active days. "
        f"Ollama: lane-wide rate over the last {RATE_WINDOW.days} days (its models run in turn).",
    ]
    lines.append(f"{'model':46} {'remaining':>9} {'rate/day':>9} {'days left':>9}  {'finish':10}  basis")
    for model, info in doc["per_model"].items():
        p = info["projection"]
        rate = "-" if p["rate_per_day"] is None else f"{p['rate_per_day']:.1f}"
        left = "-" if p["days_left"] is None else f"{p['days_left']:.1f}"
        lines.append(f"{model:46} {info['remaining']:>9} {rate:>9} {left:>9}  {p['finish']:10}  {p['basis']}")
    lines += ["", f"Pool projected finish: {doc['pool_projected_finish']}", ""]
    merge_doc = doc["merge"]
    lines.append(
        f"Merge: kept {merge_doc['kept']} rows; {merge_doc['duplicate_pairs']} (item, model) pairs seen more than "
        f"once; dropped copies {merge_doc['dropped_duplicates'] or '{}'}; duplicates whose ok differs: "
        f"{merge_doc['duplicates_with_different_ok']}."
    )
    lines.append(f"Out of scope (not P1 rows): {merge_doc['out_of_scope'] or '{}'}")
    for name, src in merge_doc["sources"].items():
        partial = ", partial last line skipped" if src["partial_last_line_skipped"] else ""
        lines.append(f"  {name}: {src['rows']} rows from {src['path']}{partial}; malformed {src['malformed_lines']}")
    for name, line in doc["last_loop_log_line"].items():
        lines.append(f"  loop {name}: {line}")
    return "\n".join(lines)
