"""Runs attempts across a ranked candidate list: retries, backoff, breakers, quotas and fallback (ADR-0008)."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any

from opentelemetry import trace

from modelmesh.core.deadline import Deadline
from modelmesh.core.errors import DECISIONS, OUTCOME_BY_CLASS, ApiError, ErrorClass, ProviderError
from modelmesh.observability.stats import ProviderStats
from modelmesh.observability.telemetry import Instruments
from modelmesh.providers.catalog import CatalogModel
from modelmesh.providers.registry import ProviderRegistry
from modelmesh.providers.types import NormalizedChatRequest, NormalizedChatResponse, NormalizedChunk
from modelmesh.reliability.breaker import BreakerRegistry
from modelmesh.reliability.quota import QuotaTracker
from modelmesh.reliability.structured import ValidationIssue, repair_instruction, validate_output

BACKOFF_BASE_S = 0.25
BACKOFF_CAP_S = 2.0
MAX_RETRIES_PER_MODEL = 1


@dataclass
class AttemptRecord:
    provider: str
    model: str
    outcome: str
    latency_ms: int | None
    http_status: int | None
    retryable: bool
    possibly_billed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "outcome": self.outcome,
            "latency_ms": self.latency_ms,
            "http_status": self.http_status,
            "retryable": self.retryable,
            "possibly_billed": self.possibly_billed,
        }


@dataclass
class Candidate:
    model: CatalogModel
    request: NormalizedChatRequest
    estimated_input_tokens: int
    upper_bound_cost_usd: float
    input_cost_estimate_usd: float
    validation_format: dict[str, Any] | None = None


@dataclass
class ExecutionState:
    deadline: Deadline
    allow_fallback: bool
    max_cost_usd: float | None = None
    attempts: list[AttemptRecord] = field(default_factory=list)
    failed_attempts_cost_usd: float = 0.0
    attempts_elapsed_ms: int = 0
    repair_attempts: int = 0
    warnings: list[str] = field(default_factory=list)

    def fallback_used(self, final_model: str | None) -> bool:
        return any(a.model != final_model for a in self.attempts)


@dataclass
class ExecutionResult:
    candidate: Candidate
    response: NormalizedChatResponse
    latency_ms: int


@dataclass
class StreamHandle:
    candidate: Candidate
    iterator: AsyncIterator[NormalizedChunk]
    buffered: list[NormalizedChunk]
    started_at: float
    ttft_ms: int


SleepFn = Callable[[float], Awaitable[None]]


class Executor:
    def __init__(
        self,
        registry: ProviderRegistry,
        breakers: BreakerRegistry,
        stats: ProviderStats,
        quota: QuotaTracker,
        tracer: trace.Tracer,
        instruments: Instruments,
        *,
        rng: random.Random | None = None,
        sleep: SleepFn = asyncio.sleep,
    ) -> None:
        self._registry = registry
        self._breakers = breakers
        self._stats = stats
        self._quota = quota
        self._tracer = tracer
        self._instruments = instruments
        self._rng = rng or random.Random()  # noqa: S311 - backoff jitter is not security sensitive
        self._sleep = sleep

    # ------------------------------------------------------------ shared helpers
    def backoff_seconds(self, attempt: int) -> float:
        return self._rng.uniform(0.0, min(BACKOFF_CAP_S, BACKOFF_BASE_S * (2**attempt)))

    def _admit(self, candidate: Candidate, state: ExecutionState) -> bool:
        """Skip a candidate whose breaker is open or that would break the cost ceiling."""
        model = candidate.model
        ceiling = state.max_cost_usd
        if ceiling is not None and candidate.upper_bound_cost_usd + state.failed_attempts_cost_usd > ceiling:
            state.warnings.append(f"fallback_skipped_cost:{model.id}")
            return False
        if not self._breakers.for_provider(model.provider).allow() or not self._breakers.for_model(model.id).allow():
            state.attempts.append(
                AttemptRecord(
                    model.provider,
                    model.id,
                    "circuit_open",
                    None,
                    None,
                    retryable=True,
                    possibly_billed=False,
                )
            )
            return False
        return True

    async def _on_failure(
        self, candidate: Candidate, error: ProviderError, latency_ms: int, state: ExecutionState
    ) -> None:
        model = candidate.model
        decision = DECISIONS[error.error_class]
        state.attempts.append(
            AttemptRecord(
                model.provider,
                model.id,
                OUTCOME_BY_CLASS[error.error_class],
                latency_ms,
                error.http_status,
                retryable=decision.retry_same_model or decision.fallback,
                possibly_billed=error.possibly_billed,
            )
        )
        state.attempts_elapsed_ms += latency_ms
        if error.possibly_billed:
            state.failed_attempts_cost_usd += candidate.input_cost_estimate_usd
        self._stats.record(model.provider, model.id, ok=False, latency_ms=latency_ms)
        self._instruments.attempt_duration.record(latency_ms, {"provider": model.provider, "outcome": "failure"})
        timed_out = error.error_class is ErrorClass.TIMEOUT
        if decision.counts_for_breaker:
            self._breakers.for_model(model.id).record_failure(timeout=timed_out)
            connection_level = error.http_status is None
            if connection_level:
                self._breakers.for_provider(model.provider).record_failure(timeout=timed_out)
        else:
            self._breakers.for_model(model.id).release_probe()
            self._breakers.for_provider(model.provider).release_probe()
        if error.error_class is ErrorClass.AUTH_ERROR:
            self._breakers.misconfigured.add(model.provider)
            self._breakers.for_provider(model.provider).trip()
        if error.error_class is ErrorClass.QUOTA_EXHAUSTED:
            if error.quota_reset_at is not None:
                await self._quota.block_until(model.id, error.quota_reset_at)
            else:
                await self._quota.block_for(model.id, error.retry_after_s or 3600.0)

    def _on_success(self, candidate: Candidate) -> None:
        model = candidate.model
        self._breakers.for_model(model.id).record_success()
        self._breakers.for_provider(model.provider).record_success()
        self._breakers.misconfigured.discard(model.provider)

    async def _should_retry(
        self, error: ProviderError, tries: int, state: ExecutionState, candidate: Candidate
    ) -> bool:
        if tries >= MAX_RETRIES_PER_MODEL or not DECISIONS[error.error_class].retry_same_model:
            return False
        if error.retry_after_s is not None:
            if error.retry_after_s > state.deadline.remaining_s():
                # Waiting cannot succeed inside this request, so stop other requests from trying too.
                await self._quota.block_for(candidate.model.id, error.retry_after_s)
                return False
            await self._sleep(error.retry_after_s)
            return True
        delay = self.backoff_seconds(tries)
        if delay >= state.deadline.remaining_s():
            return False
        await self._sleep(delay)
        return True

    def _finished_error(self, state: ExecutionState, last_issues: list[ValidationIssue] | None) -> ApiError:
        attempts = [a.as_dict() for a in state.attempts]
        outcomes = {a.outcome for a in state.attempts}
        if state.deadline.expired:
            return ApiError(
                "upstream_timeout",
                "The request deadline expired.",
                code="deadline_exceeded",
                details={"attempts": attempts},
            )
        if outcomes and outcomes <= {"rate_limited", "quota_exhausted", "circuit_open"}:
            return ApiError(
                "service_unavailable",
                "Every eligible upstream is currently rate- or quota-limited.",
                code="upstream_capacity",
                details={"attempts": attempts},
            )
        if last_issues is not None and "schema_validation_failed" in outcomes:
            return ApiError(
                "upstream_error",
                "No eligible model produced output that satisfies the requested format.",
                code="schema_validation_failed",
                details={
                    "attempts": attempts,
                    "validation_errors": [{"path": i.path, "keyword": i.keyword} for i in last_issues],
                },
            )
        return ApiError(
            "upstream_error",
            "All eligible attempts failed.",
            code="all_attempts_failed",
            details={"attempts": attempts},
        )

    @staticmethod
    def _terminal_error(error: ProviderError, state: ExecutionState) -> ApiError:
        attempts = {"attempts": [a.as_dict() for a in state.attempts]}
        if error.error_class is ErrorClass.CONTENT_FILTERED:
            return ApiError(
                "upstream_error",
                "The provider filtered this content.",
                code="content_filtered",
                details=attempts,
            )
        return ApiError(
            "internal_error",
            "ModelMesh produced a request the provider rejected.",
            details=attempts,
            retryable=True,
        )

    # ------------------------------------------------------------ non-streaming
    async def _call_chat(
        self, candidate: Candidate, request: NormalizedChatRequest, state: ExecutionState
    ) -> tuple[NormalizedChatResponse, int]:
        model = candidate.model
        await self._quota.record_call(model, candidate.estimated_input_tokens + (request.max_tokens or 0))
        started = time.perf_counter()
        with self._tracer.start_as_current_span("mm.attempt", kind=trace.SpanKind.CLIENT) as span:
            span.set_attribute("gen_ai.provider.name", model.provider)
            span.set_attribute("gen_ai.request.model", model.id)
            try:
                response = await asyncio.wait_for(
                    self._registry.adapter(model.provider).chat(request, deadline=state.deadline),
                    timeout=state.deadline.remaining_s(),
                )
            except TimeoutError:
                error = ProviderError(ErrorClass.TIMEOUT, "deadline reached during attempt", possibly_billed=True)
                span.set_attribute("mm.attempt.outcome", OUTCOME_BY_CLASS[error.error_class])
                await self._on_failure(candidate, error, int((time.perf_counter() - started) * 1000), state)
                raise error from None
            except ProviderError as error:
                span.set_attribute("mm.attempt.outcome", OUTCOME_BY_CLASS[error.error_class])
                if error.http_status is not None:
                    span.set_attribute("http.response.status_code", error.http_status)
                await self._on_failure(candidate, error, int((time.perf_counter() - started) * 1000), state)
                raise
            span.set_attribute("mm.attempt.outcome", "success")
            span.set_attribute("gen_ai.response.finish_reasons", [response.finish_reason])
        return response, int((time.perf_counter() - started) * 1000)

    async def run(self, candidates: list[Candidate], state: ExecutionState) -> ExecutionResult:
        last_issues: list[ValidationIssue] | None = None
        for position, candidate in enumerate(candidates):
            if position > 0 and not state.allow_fallback:
                break
            if not self._admit(candidate, state):
                continue
            tries = 0
            request = candidate.request
            repaired = False
            while True:
                if state.deadline.expired:
                    raise self._finished_error(state, last_issues)
                try:
                    response, latency_ms = await self._call_chat(candidate, request, state)
                except ProviderError as error:
                    decision = DECISIONS[error.error_class]
                    if await self._should_retry(error, tries, state, candidate):
                        tries += 1
                        continue
                    if not decision.fallback:
                        raise self._terminal_error(error, state) from None
                    break
                fmt = candidate.validation_format
                issues = validate_output(fmt, response.content) if fmt is not None else []
                if not issues:
                    return self._succeed(candidate, response, latency_ms, state)
                last_issues = issues
                state.attempts.append(
                    AttemptRecord(
                        candidate.model.provider,
                        candidate.model.id,
                        "schema_validation_failed",
                        latency_ms,
                        response.http_status,
                        retryable=not repaired,
                        possibly_billed=True,
                    )
                )
                state.attempts_elapsed_ms += latency_ms
                state.failed_attempts_cost_usd += candidate.input_cost_estimate_usd
                self._breakers.for_model(candidate.model.id).release_probe()
                if repaired:
                    break
                repaired = True
                state.repair_attempts += 1
                request = replace(
                    request,
                    messages=[
                        *request.messages,
                        {"role": "assistant", "content": response.content or ""},
                        {"role": "user", "content": repair_instruction(issues)},
                    ],
                )
        raise self._finished_error(state, last_issues)

    def _succeed(
        self, candidate: Candidate, response: NormalizedChatResponse, latency_ms: int, state: ExecutionState
    ) -> ExecutionResult:
        model = candidate.model
        self._on_success(candidate)
        usage = response.usage
        self._stats.record(
            model.provider,
            model.id,
            ok=True,
            latency_ms=latency_ms,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )
        self._instruments.attempt_duration.record(latency_ms, {"provider": model.provider, "outcome": "success"})
        state.attempts.append(
            AttemptRecord(
                model.provider,
                model.id,
                "success",
                latency_ms,
                response.http_status,
                retryable=False,
                possibly_billed=True,
            )
        )
        state.attempts_elapsed_ms += latency_ms
        return ExecutionResult(candidate, response, latency_ms)

    async def observe_rate_limit(self, model_id: str, response: NormalizedChatResponse) -> None:
        await self._quota.observe(model_id, response.rate_limit)

    # ------------------------------------------------------------ streaming
    async def open_stream(self, candidates: list[Candidate], state: ExecutionState) -> StreamHandle:
        """Attempt candidates until one produces content. Failures here are invisible to the client."""
        for position, candidate in enumerate(candidates):
            if position > 0 and not state.allow_fallback:
                break
            if not self._admit(candidate, state):
                continue
            tries = 0
            while True:
                if state.deadline.expired:
                    raise self._finished_error(state, None)
                try:
                    return await self._open_one(candidate, state)
                except ProviderError as error:
                    decision = DECISIONS[error.error_class]
                    if await self._should_retry(error, tries, state, candidate):
                        tries += 1
                        continue
                    if not decision.fallback:
                        raise self._terminal_error(error, state) from None
                    break
        raise self._finished_error(state, None)

    async def _open_one(self, candidate: Candidate, state: ExecutionState) -> StreamHandle:
        model = candidate.model
        await self._quota.record_call(model, candidate.estimated_input_tokens + (candidate.request.max_tokens or 0))
        started = time.perf_counter()
        iterator = self._registry.adapter(model.provider).stream(candidate.request, deadline=state.deadline).__aiter__()
        buffered: list[NormalizedChunk] = []
        with self._tracer.start_as_current_span("mm.attempt", kind=trace.SpanKind.CLIENT) as span:
            span.set_attribute("gen_ai.provider.name", model.provider)
            span.set_attribute("gen_ai.request.model", model.id)
            try:
                while True:
                    chunk = await asyncio.wait_for(anext(iterator), timeout=state.deadline.remaining_s())
                    buffered.append(chunk)
                    if chunk.content or chunk.tool_calls:
                        break
                    if chunk.finish_reason is not None:
                        raise ProviderError(
                            ErrorClass.UPSTREAM_UNAVAILABLE,
                            f"{model.provider} returned an empty stream",
                            http_status=200,
                            possibly_billed=True,
                        )
            except StopAsyncIteration:
                error = ProviderError(
                    ErrorClass.UPSTREAM_UNAVAILABLE,
                    f"{model.provider} returned an empty stream",
                    http_status=200,
                    possibly_billed=True,
                )
                await self._on_failure(candidate, error, int((time.perf_counter() - started) * 1000), state)
                raise error from None
            except TimeoutError:
                error = ProviderError(ErrorClass.TIMEOUT, "deadline reached before first token", possibly_billed=True)
                await self._on_failure(candidate, error, int((time.perf_counter() - started) * 1000), state)
                raise error from None
            except ProviderError as error:
                span.set_attribute("mm.attempt.outcome", OUTCOME_BY_CLASS[error.error_class])
                await self._on_failure(candidate, error, int((time.perf_counter() - started) * 1000), state)
                raise
            span.set_attribute("mm.attempt.outcome", "stream_opened")
        ttft_ms = int((time.perf_counter() - started) * 1000)
        return StreamHandle(candidate, iterator, buffered, started, ttft_ms)

    def finish_stream(
        self,
        handle: StreamHandle,
        state: ExecutionState,
        *,
        outcome: str,
        input_tokens: int,
        output_tokens: int,
    ) -> int:
        """Record the terminal state of the stream that reached the client."""
        model = handle.candidate.model
        latency_ms = int((time.perf_counter() - handle.started_at) * 1000)
        ok = outcome == "success"
        state.attempts.append(
            AttemptRecord(
                model.provider,
                model.id,
                outcome,
                latency_ms,
                200,
                retryable=outcome == "stream_interrupted",
                possibly_billed=True,
            )
        )
        state.attempts_elapsed_ms += latency_ms
        if ok:
            self._on_success(handle.candidate)
        elif outcome == "client_cancelled":
            self._breakers.for_model(model.id).release_probe()
            self._breakers.for_provider(model.provider).release_probe()
        else:
            self._breakers.for_model(model.id).record_failure()
        if outcome != "client_cancelled":
            self._stats.record(
                model.provider,
                model.id,
                ok=ok,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        return latency_ms
