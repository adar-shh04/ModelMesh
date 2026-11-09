"""Collects model outputs and judge scores by calling ModelMesh's public API, the same way any client does.

Progress is appended to JSONL files under the run directory, so a run can be stopped and resumed: completed
(item, model) pairs are skipped. Failures are recorded as observations, because they are reliability data.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_BACKOFF_S = 300.0


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


@dataclass
class Judgement:
    item_id: str
    model: str
    judge_model: str
    rubric: str
    rubric_version: str
    score: int | None
    request_id: str | None


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
    return min(2.0**attempt, MAX_BACKOFF_S)


class BenchmarkRunner:
    def __init__(
        self,
        client: httpx.AsyncClient,
        run_dir: Path,
        run_id: str,
        *,
        families: dict[str, str] | None = None,
        max_attempts: int = 4,
        concurrency_per_model: int = 1,
        sleep: Any = anyio.sleep,
    ) -> None:
        self._client = client
        self._run_id = run_id
        self._families = families or {}
        self._max_attempts = max_attempts
        self._concurrency = concurrency_per_model
        self._sleep = sleep
        self.observations = JsonlLog(run_dir / OBSERVATIONS)
        self.judgements = JsonlLog(run_dir / JUDGEMENTS)

    async def _post(self, body: dict[str, Any]) -> httpx.Response:
        response: httpx.Response | None = None
        for attempt in range(self._max_attempts):
            response = await self._client.post("/v1/chat/completions", json=body)
            if response.status_code not in RETRYABLE_STATUS:
                return response
            if attempt < self._max_attempts - 1:
                await self._sleep(retry_delay(response, attempt))
        assert response is not None
        return response

    def _request_body(self, item: BenchItem, model: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": item.messages,
            "temperature": 0,
            "max_tokens": item.max_tokens,
            "cache": {"mode": "off"},
            "routing": {"fallback": "none"},
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

    async def _collect_one(self, item: BenchItem, model: str) -> None:
        response = await self._post(self._request_body(item, model))
        observation = self._to_observation(item, model, response)
        self.observations.append(asdict(observation))

    def _to_observation(self, item: BenchItem, model: str, response: httpx.Response) -> Observation:
        request_id = response.headers.get("x-modelmesh-request-id")
        if response.status_code != 200:
            error = response.json().get("error", {}) if response.content else {}
            return Observation(
                item.item_id, model, self._run_id, False, response.status_code, request_id, None, None, None, None,
                None, None, None, None, None, error.get("code"),
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
        )

    def _pending(self, items: list[BenchItem], model: str) -> list[BenchItem]:
        completed = {row["item_id"] for row in self.observations.read() if row["model"] == model}
        return [item for item in items if item.item_id not in completed]

    async def collect(self, items: list[BenchItem], models: list[str]) -> None:
        """Run every (item, model) pair not yet recorded. Models run concurrently, each at its own pace."""

        async def worker(model: str) -> None:
            limiter = anyio.Semaphore(self._concurrency)

            async def one(item: BenchItem) -> None:
                async with limiter:
                    await self._collect_one(item, model)

            async with anyio.create_task_group() as group:
                for item in self._pending(items, model):
                    group.start_soon(one, item)

        async with anyio.create_task_group() as group:
            for model in models:
                group.start_soon(worker, model)

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
            family = self._families.get(row["model"], row["model"].split("/", 1)[0])
            judge_model = select_judge(family, judges)
            if judge_model is None:
                skipped += 1
                continue
            await self._judge_one(item, row, judge_model)
        return skipped

    async def _judge_one(self, item: BenchItem, row: dict[str, Any], judge_model: str) -> None:
        reference = item.reference
        rubric = reference["rubric"]
        evidence = reference.get("gold") or reference.get("evidence") or ""
        prompt = build_judge_prompt(rubric, str(evidence), row["output"] or "")
        response = await self._post(
            {
                "model": judge_model,
                "messages": prompt.messages,
                "temperature": 0,
                "max_tokens": 32,
                "response_format": prompt.response_format,
                "cache": {"mode": "off"},
                "routing": {"fallback": "none"},
                "metadata": {
                    "application": APPLICATION,
                    "task_type": "evaluation_judge",
                    "correlation_id": self._run_id,
                },
            }
        )
        score: int | None = None
        if response.status_code == 200:
            score = parse_judge_score(response.json()["choices"][0]["message"].get("content") or "")
        judgement = Judgement(
            item.item_id, row["model"], judge_model, rubric, RUBRIC_VERSION, score,
            response.headers.get("x-modelmesh-request-id"),
        )  # fmt: skip
        self.judgements.append(asdict(judgement))
