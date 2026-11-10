"""Collects model outputs and judge scores by calling ModelMesh's public API, the same way any client does.

Progress is appended to JSONL files under the run directory, so a run can stop (quota, sleep, crash) and resume:
completed (item, model) pairs are skipped.

Two kinds of failure are kept apart:
- *Observations* are real outcomes, recorded once: successes, upstream errors (502), timeouts (504) and other
  non-capacity errors. They are reliability data and are never retried away.
- *Deferrals* are capacity signals (429 from ModelMesh's own limits, 503 when upstream quota is exhausted). They are
  logged, the pair stays pending for a later run, and a provider lane stops after repeated deferrals.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anyio
import httpx

from modelmesh.ml.benchmark.items import BenchItem
from modelmesh.ml.benchmark.judge import RUBRIC_VERSION, build_judge_prompt, parse_judge_score, select_judge
from modelmesh.ml.benchmark.scoring import needs_judge, score_automatic

APPLICATION = "modelmesh-bench"
OBSERVATIONS = "observations.jsonl"
JUDGEMENTS = "judgements.jsonl"
DEFERRALS = "deferrals.jsonl"
WARMUPS = "warmups.jsonl"
CAPACITY_STATUS = {429, 503}
MAX_BACKOFF_S = 300.0
DEFERRALS_BEFORE_STOP = 3
WARMUP_TIMEOUT_MS = 300_000

SleepFn = Callable[[float], Awaitable[None]]
ClockFn = Callable[[], float]


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def provider_of(model: str) -> str:
    return model.split("/", 1)[0]


@dataclass
class Observation:
    item_id: str
    model: str
    run_id: str
    ok: bool
    http_status: int
    request_id: str | None
    output: str | None
    automatic_score: float | None
    automatic_accepted: bool | None
    automatic_metric: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    usage_source: str | None
    estimated_cost_usd: float | None
    latency_ms: int | None
    error_code: str | None
    at: str = field(default_factory=now_iso)
    client_latency_ms: int | None = None
    attempts: list[str] | None = None


@dataclass
class Judgement:
    item_id: str
    model: str
    judge_model: str
    rubric: str
    rubric_version: str
    score: int | None
    request_id: str | None
    at: str = field(default_factory=now_iso)


@dataclass
class Pacing:
    """Client-side pacing for one provider, applied on top of ModelMesh's own quota tracking."""

    rpm: float | None = None
    max_requests: int | None = None
    timeout_ms: int | None = None
    warmup: bool = False
    sequential_models: bool = False


@dataclass
class Lane:
    provider: str
    pacing: Pacing
    sent: int = 0
    last_sent_at: float | None = None
    consecutive_deferrals: int = 0
    stopped_reason: str | None = None

    @property
    def stopped(self) -> bool:
        return self.stopped_reason is not None


class JsonlLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def append(self, row: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def retry_delay(response: httpx.Response, attempt: int) -> float:
    header = response.headers.get("retry-after")
    if header is not None:
        try:
            return min(float(header), MAX_BACKOFF_S)
        except ValueError:
            pass
    # Long enough to cross the gateway's per-minute quota window by the last attempt.
    return min(15.0 * 2.0**attempt, MAX_BACKOFF_S)


def error_code_of(response: httpx.Response) -> str | None:
    try:
        body = response.json()
    except ValueError:
        return None
    error = body.get("error") if isinstance(body, dict) else None
    return error.get("code") if isinstance(error, dict) else None


class BenchmarkRunner:
    def __init__(
        self,
        client: httpx.AsyncClient,
        run_dir: Path,
        run_id: str,
        *,
        families: dict[str, str] | None = None,
        pacing: dict[str, Pacing] | None = None,
        capacity_attempts: int = 4,
        sleep: SleepFn = anyio.sleep,
        clock: ClockFn = time.monotonic,
    ) -> None:
        self._client = client
        self._run_id = run_id
        self._families = families or {}
        self._pacing = pacing or {}
        self._capacity_attempts = capacity_attempts
        self._sleep = sleep
        self._clock = clock
        self._lanes: dict[str, Lane] = {}
        self.observations = JsonlLog(run_dir / OBSERVATIONS)
        self.judgements = JsonlLog(run_dir / JUDGEMENTS)
        self.deferrals = JsonlLog(run_dir / DEFERRALS)
        self.warmups = JsonlLog(run_dir / WARMUPS)

    # ------------------------------------------------------------ pacing
    def lane(self, provider: str) -> Lane:
        if provider not in self._lanes:
            self._lanes[provider] = Lane(provider, self._pacing.get(provider, Pacing()))
        return self._lanes[provider]

    @property
    def lanes(self) -> dict[str, Lane]:
        return self._lanes

    async def _pace(self, lane: Lane) -> bool:
        """Wait for the provider's next slot. False when its request budget for this invocation is spent."""
        pacing = lane.pacing
        if pacing.max_requests is not None and lane.sent >= pacing.max_requests:
            lane.stopped_reason = lane.stopped_reason or "max_requests"
            return False
        if pacing.rpm and lane.last_sent_at is not None:
            wait = lane.last_sent_at + 60.0 / pacing.rpm - self._clock()
            if wait > 0:
                await self._sleep(wait)
        lane.last_sent_at = self._clock()
        lane.sent += 1
        return True

    async def _post(self, lane: Lane, body: dict[str, Any]) -> tuple[httpx.Response | None, int]:
        """Send with pacing. Capacity responses are retried with back-off; anything else is returned as is.

        Returns (None, elapsed) when the lane's budget ran out before a request could be sent.
        """
        response: httpx.Response | None = None
        started = self._clock()
        for attempt in range(self._capacity_attempts):
            if not await self._pace(lane):
                return response, int((self._clock() - started) * 1000)
            response = await self._client.post("/v1/chat/completions", json=body)
            if response.status_code not in CAPACITY_STATUS:
                break
            if attempt < self._capacity_attempts - 1:
                await self._sleep(retry_delay(response, attempt))
        return response, int((self._clock() - started) * 1000)

    def _defer(self, lane: Lane, item_id: str, model: str, response: httpx.Response | None, kind: str) -> None:
        lane.consecutive_deferrals += 1
        self.deferrals.append(
            {
                "item_id": item_id,
                "model": model,
                "kind": kind,
                "http_status": response.status_code if response is not None else None,
                "error_code": error_code_of(response) if response is not None else "lane_budget_spent",
                "retry_after": response.headers.get("retry-after") if response is not None else None,
                "at": now_iso(),
            }
        )
        if lane.consecutive_deferrals >= DEFERRALS_BEFORE_STOP:
            lane.stopped_reason = lane.stopped_reason or "capacity"

    # ------------------------------------------------------------ generation
    def _request_body(self, item: BenchItem, model: str, lane: Lane) -> dict[str, Any]:
        routing: dict[str, Any] = {"fallback": "none"}
        if lane.pacing.timeout_ms is not None:
            routing["timeout_ms"] = lane.pacing.timeout_ms
        body: dict[str, Any] = {
            "model": model,
            "messages": item.messages,
            "temperature": 0,
            "max_tokens": item.max_tokens,
            "cache": {"mode": "off"},
            "routing": routing,
            "metadata": {
                "application": APPLICATION,
                "task_type": item.task_type,
                "correlation_id": self._run_id,
                "bench_item": item.item_id[:200],
            },
        }
        if item.response_format is not None:
            body["response_format"] = item.response_format
        return body

    async def warm_up(self, model: str) -> dict[str, Any]:
        """Load a local model before measuring it, and record how long the cold start took."""
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
            "temperature": 0,
            "max_tokens": 8,
            "cache": {"mode": "off"},
            "routing": {"fallback": "none", "timeout_ms": WARMUP_TIMEOUT_MS},
            "metadata": {"application": APPLICATION, "task_type": "general", "correlation_id": self._run_id},
        }
        started = self._clock()
        try:
            response = await self._client.post("/v1/chat/completions", json=body)
            status, code = response.status_code, error_code_of(response) if response.status_code != 200 else None
        except httpx.HTTPError as exc:
            status, code = None, type(exc).__name__
        row = {
            "model": model,
            "http_status": status,
            "error_code": code,
            "client_latency_ms": int((self._clock() - started) * 1000),
            "at": now_iso(),
        }
        self.warmups.append(row)
        return row

    async def _collect_one(self, item: BenchItem, model: str, lane: Lane) -> None:
        response, elapsed = await self._post(lane, self._request_body(item, model, lane))
        if response is None:
            return
        if response.status_code in CAPACITY_STATUS or response.status_code == 402:
            self._defer(lane, item.item_id, model, response, "generation")
            return
        lane.consecutive_deferrals = 0
        observation = self._to_observation(item, model, response)
        observation.client_latency_ms = elapsed
        self.observations.append(asdict(observation))

    def _to_observation(self, item: BenchItem, model: str, response: httpx.Response) -> Observation:
        request_id = response.headers.get("x-modelmesh-request-id")
        if response.status_code != 200:
            error: dict[str, Any] = {}
            with contextlib.suppress(ValueError):
                error = response.json().get("error", {})
            attempts = [a.get("outcome") for a in (error.get("details") or {}).get("attempts", [])]
            return Observation(
                item.item_id, model, self._run_id, False, response.status_code, request_id, None, None, None, None,
                None, None, None, None, None, error.get("code"), attempts=attempts or None,
            )  # fmt: skip
        body = response.json()
        meta = body["modelmesh"]
        output = body["choices"][0]["message"].get("content") or ""
        score = score_automatic(item, output)
        return Observation(
            item_id=item.item_id,
            model=model,
            run_id=self._run_id,
            ok=True,
            http_status=200,
            request_id=request_id,
            output=output,
            automatic_score=score.value if score else None,
            automatic_accepted=score.accepted if score else None,
            automatic_metric=score.metric if score else None,
            prompt_tokens=meta["usage"]["prompt_tokens"],
            completion_tokens=meta["usage"]["completion_tokens"],
            usage_source=meta["usage"]["source"],
            estimated_cost_usd=meta["cost"]["estimated_usd"],
            latency_ms=meta["latency_ms"]["total"],
            error_code=None,
            attempts=[a["outcome"] for a in meta["attempts"]],
        )

    def _pending(self, items: list[BenchItem], model: str) -> list[BenchItem]:
        completed = {row["item_id"] for row in self.observations.read() if row["model"] == model}
        return [item for item in items if item.item_id not in completed]

    def _order(self, items: list[BenchItem], models: list[str], pacing: Pacing) -> list[tuple[BenchItem, str]]:
        """Local models run one after another (no reloads); remote ones are interleaved to share a budget fairly."""
        pending = {model: self._pending(items, model) for model in models}
        if pacing.sequential_models:
            return [(item, model) for model in models for item in pending[model]]
        order: list[tuple[BenchItem, str]] = []
        for position in range(max((len(rows) for rows in pending.values()), default=0)):
            order.extend((pending[model][position], model) for model in models if position < len(pending[model]))
        return order

    async def collect(self, items: list[BenchItem], models: list[str]) -> None:
        """Run every pending (item, model) pair. Providers run concurrently; each paces its own requests."""
        by_provider: dict[str, list[str]] = {}
        for model in models:
            by_provider.setdefault(provider_of(model), []).append(model)

        async def worker(provider: str, provider_models: list[str]) -> None:
            lane = self.lane(provider)
            warmed: set[str] = set()
            for item, model in self._order(items, provider_models, lane.pacing):
                if lane.stopped:
                    return
                if lane.pacing.warmup and model not in warmed:
                    await self.warm_up(model)
                    warmed.add(model)
                await self._collect_one(item, model, lane)

        async with anyio.create_task_group() as group:
            for provider, provider_models in by_provider.items():
                group.start_soon(worker, provider, provider_models)

    # ------------------------------------------------------------ judging
    def _judge_targets(self, items: list[BenchItem]) -> list[tuple[BenchItem, dict[str, Any]]]:
        by_id = {item.item_id: item for item in items}
        judged = {(row["item_id"], row["model"]) for row in self.judgements.read()}
        return [
            (by_id[row["item_id"]], row)
            for row in self.observations.read()
            if row["ok"]
            and row["item_id"] in by_id
            and needs_judge(by_id[row["item_id"]])
            and (row["item_id"], row["model"]) not in judged
        ]

    async def judge(self, items: list[BenchItem], judges: dict[str, str]) -> int:
        """Judge outputs still unscored. Returns the number skipped for lack of an independent judge."""
        skipped = 0
        for item, row in self._judge_targets(items):
            family = self._families.get(row["model"], provider_of(row["model"]))
            judge_model = select_judge(family, judges)
            if judge_model is None:
                skipped += 1
                continue
            lane = self.lane(provider_of(judge_model))
            if lane.stopped:
                continue
            await self._judge_one(item, row, judge_model, lane)
        return skipped

    async def _judge_one(self, item: BenchItem, row: dict[str, Any], judge_model: str, lane: Lane) -> None:
        reference = item.reference
        rubric = reference["rubric"]
        evidence = reference.get("gold") or reference.get("evidence") or ""
        prompt = build_judge_prompt(rubric, str(evidence), row["output"] or "")
        routing: dict[str, Any] = {"fallback": "none"}
        if lane.pacing.timeout_ms is not None:
            routing["timeout_ms"] = lane.pacing.timeout_ms
        response, _ = await self._post(
            lane,
            {
                "model": judge_model,
                "messages": prompt.messages,
                "temperature": 0,
                "max_tokens": 32,
                "response_format": prompt.response_format,
                "cache": {"mode": "off"},
                "routing": routing,
                "metadata": {
                    "application": APPLICATION,
                    "task_type": "evaluation_judge",
                    "correlation_id": self._run_id,
                },
            },
        )
        if response is None:
            return
        if response.status_code in CAPACITY_STATUS or response.status_code == 402:
            self._defer(lane, item.item_id, row["model"], response, "judge")
            return
        lane.consecutive_deferrals = 0
        score: int | None = None
        if response.status_code == 200:
            score = parse_judge_score(response.json()["choices"][0]["message"].get("content") or "")
        judgement = Judgement(
            item.item_id, row["model"], judge_model, rubric, RUBRIC_VERSION, score,
            response.headers.get("x-modelmesh-request-id"),
        )  # fmt: skip
        self.judgements.append(asdict(judgement))
