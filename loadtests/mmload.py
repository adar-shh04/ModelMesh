"""Shared pieces of the M8 Locust scenarios: configuration, request bodies and a per-request recorder.

Every scenario targets the base URL given to Locust with ``--host`` (there is no default, so a run can never
hit a gateway by accident) and authenticates with the key in ``MM_LOAD_API_KEY`` or in the file named by
``MM_LOAD_API_KEY_FILE`` (default ``.cache/load_key``, written by ``setup_key.py``). Keys are never printed.

The recorder keeps what the client saw (status, client latency) and what the gateway reported in the
``modelmesh`` object (overhead, attempts, fallback, cache), and writes them to ``MM_LOAD_OUT`` as JSON Lines
when the test stops, for ``summarize.py``. This module does not import Locust, so it can be unit-tested.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_KEY_FILE = REPO / ".cache" / "load_key"

TASK_TYPES = ("query_classification", "citation_verification", "evidence_synthesis", "general")
PROMPTS = {
    "query_classification": "Classify this query into one intent: 'I lost my card yesterday, what now?'",
    "citation_verification": "Claim: the sky is green. Passage: the sky appears blue in daylight. Supported?",
    "evidence_synthesis": "Using only the evidence below, answer in two sentences.\nEvidence: [1] A. [2] B.",
    "general": "Write one sentence about load testing.",
}
CITATION_SCHEMA = {
    "type": "object",
    "properties": {"supported": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["supported", "reason"],
    "additionalProperties": False,
}


def api_key() -> str:
    value = os.environ.get("MM_LOAD_API_KEY")
    if value:
        return value.strip()
    path = Path(os.environ.get("MM_LOAD_API_KEY_FILE") or DEFAULT_KEY_FILE)
    if not path.is_file():
        raise SystemExit(f"no API key: set MM_LOAD_API_KEY or create {path} with loadtests/setup_key.py")
    return path.read_text(encoding="utf-8").strip()


def env_list(name: str, default: str) -> list[str]:
    return [part.strip() for part in os.environ.get(name, default).split(",") if part.strip()]


def headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}", "X-Request-ID": str(uuid.uuid4()), "Content-Type": "application/json"}


def chat_body(
    task_type: str,
    *,
    model: str = "auto",
    policy: str = "cost_optimized",
    allowed_models: list[str] | None = None,
    allowed_providers: list[str] | None = None,
    stream: bool = False,
    unique: bool = True,
    max_tokens: int = 64,
    run_id: str = "load",
) -> dict[str, Any]:
    """A chat request. ``unique`` adds a nonce so that no two requests share a prompt."""
    prompt = PROMPTS[task_type] + (f"\n[nonce {uuid.uuid4().hex[:12]}]" if unique else "")
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "cache": {"mode": "off"},
        "metadata": {"application": "modelmesh-load", "task_type": task_type, "correlation_id": run_id},
    }
    if model == "auto":
        routing: dict[str, Any] = {"policy": policy}
        if allowed_models:
            routing["allowed_models"] = allowed_models
        if allowed_providers:
            routing["allowed_providers"] = allowed_providers
        body["routing"] = routing
    if task_type == "citation_verification":
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "verdict", "schema": CITATION_SCHEMA, "strict": True},
        }
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


def modelmesh_from_sse(text: str) -> dict[str, Any] | None:
    """The ``modelmesh`` object from a streamed response (the last event that carries one)."""
    found: dict[str, Any] | None = None
    for line in text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            event = json.loads(line[6:])
        except ValueError:
            continue
        if isinstance(event, dict) and isinstance(event.get("modelmesh"), dict):
            found = event["modelmesh"]
    return found


@dataclass
class Sample:
    t: float
    scenario: str
    kind: str
    status: int
    client_ms: float
    ok: bool
    error_type: str | None = None
    overhead_ms: float | None = None
    total_ms: float | None = None
    provider_ms: float | None = None
    attempts: int | None = None
    attempt_outcomes: list[str] = field(default_factory=list)
    fallback_used: bool | None = None
    model: str | None = None
    cache_status: str | None = None
    stream_complete: bool | None = None


def sample_from_response(
    scenario: str, kind: str, status: int, client_ms: float, body: Any, *, stream_text: str | None = None
) -> Sample:
    sample = Sample(time.time(), scenario, kind, status, round(client_ms, 3), ok=200 <= status < 300)
    mm: dict[str, Any] | None = None
    if stream_text is not None:
        mm = modelmesh_from_sse(stream_text)
        sample.stream_complete = "data: [DONE]" in stream_text
        sample.ok = sample.ok and bool(sample.stream_complete)
    if isinstance(body, dict):
        mm = body.get("modelmesh") if isinstance(body.get("modelmesh"), dict) else mm
        error = body.get("error")
        if isinstance(error, dict):
            sample.error_type = str(error.get("type"))
            details = error.get("details") or {}
            if isinstance(details.get("attempts"), list):
                sample.attempts = len(details["attempts"])
                sample.attempt_outcomes = [str(a.get("outcome")) for a in details["attempts"]]
    if mm:
        latency = mm.get("latency_ms") or {}
        sample.overhead_ms = latency.get("overhead")
        sample.total_ms = latency.get("total")
        sample.provider_ms = latency.get("provider")
        attempts = mm.get("attempts") or []
        sample.attempts = len(attempts)
        sample.attempt_outcomes = [str(a.get("outcome")) for a in attempts]
        sample.fallback_used = mm.get("fallback_used")
        sample.model = mm.get("model")
        sample.cache_status = (mm.get("cache") or {}).get("status")
    return sample


class Recorder:
    """Thread-safe sample store, plus periodic snapshots of ``/v1/providers`` (breaker states)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.samples: list[Sample] = []
        self.providers: list[dict[str, Any]] = []

    def add(self, sample: Sample) -> None:
        with self._lock:
            self.samples.append(sample)

    def add_providers(self, snapshot: Any) -> None:
        with self._lock:
            self.providers.append({"t": time.time(), "data": snapshot})

    def write(self, path: Path, meta: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"meta": meta}) + "\n")
            for sample in self.samples:
                handle.write(json.dumps({"sample": asdict(sample)}) + "\n")
            for snapshot in self.providers:
                handle.write(json.dumps({"providers": snapshot}) + "\n")


def output_path(scenario: str) -> Path:
    default = REPO / "data" / "load" / f"{scenario}-{time.strftime('%Y%m%dT%H%M%S')}.jsonl"
    return Path(os.environ.get("MM_LOAD_OUT") or default)
