"""P1 merge and status on fake run directories (no real run directory is read or written)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from modelmesh.ml.benchmark.items import BenchItem
from modelmesh.ml.p1 import data as p1

QWEN = "ollama/qwen3:4b"
LITE = "gemini/gemini-3.1-flash-lite"
LFM = "openrouter/liquid/lfm-3-1.7b:free"


def item(i: int, task: str = "general") -> BenchItem:
    return BenchItem(f"src:{i}", "src", task, [{"role": "user", "content": f"q{i}"}], "classification")


def obs(item_id: str, model: str, at: str, *, ok: bool = True, code: str | None = None) -> dict[str, Any]:
    return {"item_id": item_id, "model": model, "ok": ok, "http_status": 200 if ok else 502, "error_code": code,
            "at": at, "output": "x"}  # fmt: skip


def write(path: Path, rows: list[dict[str, Any]], tail: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + tail, encoding="utf-8")


@pytest.fixture
def layout(tmp_path: Path) -> tuple[list[BenchItem], list[p1.Source], dict[str, str]]:
    items = [item(0), item(1), item(2, "query_rewriting")]
    cloud, local, pilot = tmp_path / "cloud", tmp_path / "local", tmp_path / "pilot"
    write(cloud / "observations.jsonl", [
        obs("src:0", LITE, "2026-11-16T01:00:00+00:00"),
        obs("src:0", QWEN, "2026-11-16T02:00:00+00:00", ok=False, code="deadline_exceeded"),  # early qwen3 row
        obs("src:9", LITE, "2026-11-16T03:00:00+00:00"),  # not a P1 item
        obs("src:1", "gemini/gemini-3.6-flash", "2026-11-16T03:00:00+00:00"),  # not a P1 model
    ], tail='{"item_id": "src:1", "model": "gemini/gem')  # fmt: skip
    write(local / "observations.jsonl", [
        obs("src:0", QWEN, "2026-11-16T05:00:00+00:00"),  # later duplicate of the cloud row: dropped
        obs("src:1", QWEN, "2026-11-17T05:00:00+00:00"),
    ])  # fmt: skip
    write(pilot / "observations.jsonl", [obs("src:0", LFM, "2026-11-12T00:00:00+00:00"),
                                         obs("src:1", LFM, "2026-11-12T00:00:00+00:00")])  # fmt: skip
    write(cloud / "deferrals.jsonl", [{"item_id": "src:1", "model": LITE, "kind": "generation", "http_status": 503,
                                       "error_code": "upstream_capacity"}])  # fmt: skip
    sources = [
        p1.Source("cloud", cloud, p1.POOL, "full"),
        p1.Source("local", local, p1.POOL, "full"),
        p1.Source("openrouter-pilot", pilot, p1.SPARSE, "pilot"),
    ]
    # src:1's pilot content differs from P1, so its OpenRouter row must not join.
    pilot_hashes = {"src:0": items[0].content_hash(), "src:1": "different"}
    return items, sources, pilot_hashes


def test_merge_keeps_the_first_observation_and_reports_every_drop(layout: Any) -> None:
    items, sources, pilot_hashes = layout
    rows, report = p1.merge(items, sources, pilot_hashes=pilot_hashes)
    keyed = {(r["item_id"], r["model"]): r for r in rows}
    assert keyed[("src:0", QWEN)]["p1_source"] == "cloud" and keyed[("src:0", QWEN)]["ok"] is False
    assert set(keyed) == {("src:0", LITE), ("src:0", QWEN), ("src:1", QWEN), ("src:0", LFM)}
    assert report.duplicate_pairs == 1
    assert report.dropped_duplicates == {f"local:{QWEN}": 1}
    assert report.duplicates_with_different_ok == 1
    assert report.out_of_scope == {
        "cloud:item_not_in_p1": 1,
        "cloud:model_not_in_p1:gemini/gemini-3.6-flash": 1,
        "openrouter-pilot:pilot_content_differs_from_p1": 1,
    }
    assert report.sources["cloud"]["partial_last_line_skipped"] is True
    assert report.sources["cloud"]["rows"] == 4


def test_reading_never_writes_to_a_run_directory(layout: Any, tmp_path: Path) -> None:
    items, sources, pilot_hashes = layout
    before = {p: p.read_bytes() for s in sources for p in s.directory.iterdir()}
    p1.status(items, sources=sources, pilot_hashes=pilot_hashes, now=datetime(2026, 11, 17, 12, tzinfo=UTC))
    after = {p: p.read_bytes() for s in sources for p in s.directory.iterdir()}
    assert before == after
    rows, report = p1.merge(items, sources, pilot_hashes=pilot_hashes)
    with pytest.raises(ValueError, match="run directory"):
        p1.write_merged(rows, report, p1.SOURCES[0].directory / "merged")
    target = p1.write_merged(rows, report, tmp_path / "merged")
    assert len(target.read_text(encoding="utf-8").splitlines()) == 4
    assert json.loads((tmp_path / "merged" / "merge_report.json").read_text(encoding="utf-8"))["kept"] == 4


def test_status_reports_coverage_failures_deferrals_and_projection(layout: Any) -> None:
    items, sources, pilot_hashes = layout
    now = datetime(2026, 11, 17, 12, tzinfo=UTC)
    doc = p1.status(items, sources=sources, pilot_hashes=pilot_hashes, now=now)
    qwen = doc["per_model"][QWEN]
    assert qwen["coverage_by_task"] == {"general": "2/2", "query_rewriting": "0/1"}
    assert qwen["failed"] == 1 and qwen["failures"] == {"502:deadline_exceeded": 1}
    assert qwen["yield_by_day"] == {"2026-11-16": 1, "2026-11-17": 1}
    assert doc["per_model"][LITE]["deferred"] == 1
    assert doc["per_model"][LFM]["scope"] == 1  # only the pilot item whose content matches P1
    assert doc["per_model"][LFM]["remaining"] == 0 and doc["per_model"][LFM]["projection"]["finish"] == "done"
    assert doc["per_model"]["groq/openai/gpt-oss-20b"]["projection"]["finish"] == "no data"
    # Both Ollama models share the lane's projection: the lane has qwen3's rows and llama's remaining cells.
    lane = doc["per_model"][QWEN]["projection"]
    assert lane == doc["per_model"]["ollama/llama3.2:3b"]["projection"] and lane["method"] == "continuous"
    assert doc["pool_projected_finish"].startswith("unknown")
    text = p1.render_status(doc)
    assert "acceptance is never shown" in text and "accept" not in json.dumps(doc["per_model"])


def test_projection_uses_the_measured_rate() -> None:
    now = datetime(2026, 11, 17, tzinfo=UTC)
    times = [datetime(2026, 11, 14, tzinfo=UTC)] * 10 + [datetime(2026, 11, 16, tzinfo=UTC)] * 20
    p = p1.projection(times, 100, now, continuous=True)
    assert p["rate_per_day"] == 10.0 and p["days_left"] == 10.0 and p["finish"] == "2026-11-27"
    # Daily: a quota-bound lane's burst in the first hours of a day is one day's yield, not an hourly rate.
    burst = [datetime(2026, 11, 16, 21, tzinfo=UTC)] * 40 + [datetime(2026, 11, 17, 1, tzinfo=UTC)] * 20
    daily = p1.projection(burst, 90, datetime(2026, 11, 17, 2, tzinfo=UTC))
    assert daily["rate_per_day"] == 30.0 and daily["days_left"] == 3.0 and "today partial" in daily["basis"]
    assert p1.projection(times, 5, datetime(2026, 11, 30, tzinfo=UTC))["finish"] == "stalled"
    assert p1.projection([], 5, now)["finish"] == "no data"
    assert p1.pool_finish_date(["2026-11-20", "done", "2026-11-25"]) == "2026-11-25"
