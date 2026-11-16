"""Summarises one scenario's JSON Lines output (written by the Locust scenarios) as Markdown.

    uv run python loadtests/summarize.py data/load/fake_overhead-20261201T100000.jsonl

Client latency is what Locust's process saw, end to end, including the response body. Gateway overhead is the
gateway's own ``modelmesh.latency_ms.overhead`` (total minus provider time). Both are reported because the load
generator shares the machine with the gateway in a local run.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear interpolation between closest ranks (numpy's default)."""
    data = sorted(values)
    if not data:
        return None
    position = (len(data) - 1) * q / 100
    low, high = math.floor(position), math.ceil(position)
    return data[low] + (data[high] - data[low]) * (position - low)


def load(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    meta: dict[str, Any] = {}
    samples: list[dict[str, Any]] = []
    providers: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "meta" in row:
            meta = row["meta"]
        elif "sample" in row:
            samples.append(row["sample"])
        elif "providers" in row:
            providers.append(row["providers"])
    return meta, samples, providers


HIDDEN_META = ("scenario", "host", "started_at", "stopped_at")


def _fmt(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def breaker_states(providers: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    """For each provider, how many polls saw each circuit state."""
    states: dict[str, Counter[str]] = {}
    for snapshot in providers:
        data = snapshot.get("data")
        entries = data.get("data") if isinstance(data, dict) else None
        for entry in entries or []:
            if isinstance(entry, dict) and "id" in entry:
                states.setdefault(str(entry["id"]), Counter())[str(entry.get("circuit_state"))] += 1
    return states


def summary(meta: dict[str, Any], samples: list[dict[str, Any]], providers: list[dict[str, Any]]) -> str:
    lines = [f"## {meta.get('scenario', 'scenario')} against {meta.get('host')}", ""]
    if not samples:
        return "\n".join([*lines, "No samples were recorded.", ""])
    if meta.get("started_at") and meta.get("stopped_at"):
        duration = meta["stopped_at"] - meta["started_at"]  # the whole test, ramp-up included
    else:
        duration = max(s["t"] for s in samples) - min(s["t"] for s in samples)
    ok = [s for s in samples if s["ok"]]
    lines += [
        f"- Requests: {len(samples)} over {duration:.0f} s ({len(samples) / duration:.1f} req/s achieved)"
        if duration > 0
        else f"- Requests: {len(samples)}",
        f"- Success rate: {len(ok) / len(samples):.3f} ({len(ok)} of {len(samples)})",
        f"- Fallback used on {sum(1 for s in ok if s.get('fallback_used'))} successful requests",
        f"- Run settings: `{json.dumps({k: v for k, v in meta.items() if k not in HIDDEN_META})}`",
        "",
        "| Kind | n | success | client p50 ms | client p95 ms | overhead p50 ms | overhead p95 ms |",
        "|---|---|---|---|---|---|---|",
    ]
    kinds = sorted({s["kind"] for s in samples})
    for kind in [*kinds, "all"]:
        rows = samples if kind == "all" else [s for s in samples if s["kind"] == kind]
        good = [s for s in rows if s["ok"]]
        client = [s["client_ms"] for s in good]
        overhead = [s["overhead_ms"] for s in good if s.get("overhead_ms") is not None]
        lines.append(
            f"| {kind} | {len(rows)} | {len(good) / len(rows):.3f} | {_fmt(percentile(client, 50))} | "
            f"{_fmt(percentile(client, 95))} | {_fmt(percentile(overhead, 50))} | {_fmt(percentile(overhead, 95))} |"
        )
    attempts = Counter(s.get("attempts") for s in samples)
    outcomes = Counter(o for s in samples for o in s.get("attempt_outcomes") or [])
    errors = Counter(f"{s['status']} {s.get('error_type')}" for s in samples if not s["ok"])
    lines += [
        "",
        "Attempts per request: " + ", ".join(f"{k}: {v}" for k, v in sorted(attempts.items(), key=str)),
        "",
        "Attempt outcomes: " + (", ".join(f"{k}: {v}" for k, v in outcomes.most_common()) or "none recorded"),
        "",
        "Failed requests: " + (", ".join(f"{k}: {v}" for k, v in errors.most_common()) or "none"),
    ]
    states = breaker_states(providers)
    if states:
        lines += ["", "Circuit states seen in `/v1/providers` polls:"]
        lines += [f"- {p}: " + ", ".join(f"{k} x{v}" for k, v in c.most_common()) for p, c in sorted(states.items())]
    return "\n".join([*lines, ""])


def main(argv: Sequence[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    for name in argv:
        print(summary(*load(Path(name))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
