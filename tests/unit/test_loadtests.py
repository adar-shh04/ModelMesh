"""M8 load-test helpers: request bodies satisfy the contract, responses are parsed, summaries are correct.

Only the parts that do not need Locust are tested here; the Locust files themselves are exercised by a smoke
run (docs/milestones/M8_LOAD_REPORT.md).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from modelmesh.contract.schemas import ChatCompletionRequest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "loadtests"))

import mmload
import summarize


@pytest.mark.parametrize("task_type", mmload.TASK_TYPES)
@pytest.mark.parametrize("stream", [False, True])
def test_every_scenario_body_is_a_valid_contract_request(task_type: str, stream: bool) -> None:
    body = mmload.chat_body(task_type, allowed_models=["fake/echo-small"], stream=stream)
    ChatCompletionRequest.model_validate(body)  # strict: an unknown field would raise
    assert body["cache"] == {"mode": "off"}
    explicit = mmload.chat_body(task_type, model="fake/echo-small")
    assert "routing" not in explicit
    ChatCompletionRequest.model_validate(explicit)


def test_bodies_never_repeat_a_prompt() -> None:
    first, second = (mmload.chat_body("general")["messages"][0]["content"] for _ in range(2))
    assert first != second


def test_sample_reads_the_modelmesh_object_and_error_attempts() -> None:
    ok = {
        "modelmesh": {
            "model": "fake/echo-small",
            "latency_ms": {"total": 12, "provider": 2, "overhead": 10},
            "attempts": [{"outcome": "upstream_unavailable"}, {"outcome": "success"}],
            "fallback_used": True,
            "cache": {"status": "miss"},
        }
    }
    sample = mmload.sample_from_response("chaos", "chat:general", 200, 15.0, ok)
    assert (sample.ok, sample.overhead_ms, sample.attempts, sample.fallback_used) == (True, 10, 2, True)
    failed = {"error": {"type": "upstream_error", "details": {"attempts": [{"outcome": "upstream_unavailable"}] * 3}}}
    sample = mmload.sample_from_response("chaos", "chat:general", 502, 30.0, failed)
    assert (sample.ok, sample.error_type, sample.attempts) == (False, "upstream_error", 3)


def test_stream_without_done_is_not_a_success() -> None:
    event = {"choices": [], "modelmesh": {"latency_ms": {"overhead": 4}, "attempts": [{"outcome": "success"}]}}
    complete = f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n"
    assert mmload.sample_from_response("s", "stream", 200, 5.0, None, stream_text=complete).ok
    interrupted = f"data: {json.dumps(event)}\n\n"
    sample = mmload.sample_from_response("s", "stream", 200, 5.0, None, stream_text=interrupted)
    assert not sample.ok and sample.stream_complete is False and sample.overhead_ms == 4


def test_api_key_comes_from_the_environment_or_a_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MM_LOAD_API_KEY", raising=False)
    monkeypatch.setenv("MM_LOAD_API_KEY_FILE", str(tmp_path / "missing"))
    with pytest.raises(SystemExit):
        mmload.api_key()
    (tmp_path / "k").write_text("mm_test_abc\n", encoding="utf-8")
    monkeypatch.setenv("MM_LOAD_API_KEY_FILE", str(tmp_path / "k"))
    assert mmload.api_key() == "mm_test_abc"


def test_percentile_matches_linear_interpolation() -> None:
    assert summarize.percentile([], 50) is None
    assert summarize.percentile([1, 2, 3, 4], 50) == 2.5
    assert summarize.percentile(list(range(101)), 95) == 95
    assert summarize.percentile([5], 95) == 5


def test_summary_round_trip(tmp_path: Path) -> None:
    recorder = mmload.Recorder()
    for i in range(10):
        recorder.add(mmload.Sample(1000.0 + i, "chaos", "chat:general", 200, 10.0 + i, True, overhead_ms=2.0))
    recorder.add(mmload.Sample(1010.0, "chaos", "chat:general", 502, 50.0, False, error_type="upstream_error"))
    recorder.add_providers({"data": [{"id": "fake", "circuit_state": "closed"}]})
    recorder.add_providers({"data": [{"id": "fake", "circuit_state": "open"}]})
    path = tmp_path / "run.jsonl"
    recorder.write(path, {"scenario": "chaos", "host": "http://127.0.0.1:1"})
    text = summarize.summary(*summarize.load(path))
    assert "Success rate: 0.909 (10 of 11)" in text
    assert "502 upstream_error: 1" in text
    assert "fake: closed x1, open x1" in text
