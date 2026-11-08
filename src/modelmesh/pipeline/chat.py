"""The request lifecycle of spec section 2.4 for chat completions."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import anyio

from modelmesh.accounting.records import ExecutionRecord, UsageFigures
from modelmesh.auth.keys import AuthContext
from modelmesh.auth.limits import RateLimitInfo
from modelmesh.cache.base import CacheRequest
from modelmesh.contract.schemas import ChatCompletionResponse
from modelmesh.core.context import RequestContext
from modelmesh.core.deadline import Deadline
from modelmesh.core.errors import ApiError, ErrorClass, ProviderError
from modelmesh.core.ids import new_completion_id
from modelmesh.observability.telemetry import current_trace_id
from modelmesh.pipeline.prepare import PreparedRequest, canonical_hash, parse_chat_request, prepare
from modelmesh.pipeline.services import Services
from modelmesh.policy.engine import Constraints, effective_data_policy, filter_candidates, plan_structured
from modelmesh.providers.catalog import CatalogModel
from modelmesh.providers.types import (
    NormalizedChatRequest,
    NormalizedChatResponse,
    NormalizedChunk,
    TokenUsage,
)
from modelmesh.reliability.executor import Candidate, ExecutionState, StreamHandle
from modelmesh.reliability.structured import emulation_instruction, validate_output
from modelmesh.routing.estimates import CandidateEstimate
from modelmesh.routing.solver import SolveInput, solve

logger = logging.getLogger("modelmesh.pipeline")
KEEPALIVE = b": keep-alive\n\n"
DONE = b"data: [DONE]\n\n"


@dataclass
class JsonResult:
    body: dict[str, Any]
    headers: dict[str, str]


@dataclass
class StreamResult:
    events: AsyncIterator[bytes]
    headers: dict[str, str]


@dataclass
class _Run:
    auth: AuthContext
    ctx: RequestContext
    trace_id: str
    created_at: datetime
    started: float
    stream: bool = False
    metadata: dict[str, str] = field(default_factory=dict)
    task_type: str | None = None
    rate: RateLimitInfo | None = None

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.started) * 1000)


@dataclass
class _Routed:
    candidates: list[Candidate]
    estimates: dict[str, CandidateEstimate]
    plans: dict[str, Any]
    decision: dict[str, Any]
    allow_fallback: bool
    deadline_ms: int
    max_cost_usd: float | None


def sse_event(payload: dict[str, Any]) -> bytes:
    return b"data: " + json.dumps(payload, separators=(",", ":")).encode() + b"\n\n"


class ChatPipeline:
    def __init__(self, services: Services) -> None:
        self._s = services
        self._tracer = services.telemetry.tracer
        self._instruments = services.telemetry.instruments

    # ------------------------------------------------------------ entry point
    async def handle(self, raw: bytes, auth: AuthContext, ctx: RequestContext) -> JsonResult | StreamResult:
        run = _Run(auth, ctx, current_trace_id(), datetime.now(UTC), time.perf_counter())
        try:
            return await self._handle(raw, run)
        except ApiError as error:
            await self._record_failure(run, error)
            raise

    async def _handle(self, raw: bytes, run: _Run) -> JsonResult | StreamResult:
        prepared = prepare(parse_chat_request(raw))
        run.stream, run.metadata, run.task_type = prepared.body.stream, prepared.metadata, prepared.task_type
        estimated_input = self._s.estimator.tokens.estimate_messages(prepared.messages)
        run.rate = await self._s.limiter.acquire(run.auth, estimated_input + (prepared.max_tokens or 256))
        handed_off = False
        try:
            budget_remaining = await self._s.budgets.remaining(run.auth)
            routed, eligible_estimates = await self._policy_and_estimates(prepared, run, budget_remaining)
            cache_request, cache_status = self._cache_request(prepared, run, routed)
            if cache_request is not None:
                eligible_ids = {estimate.model.id for estimate in eligible_estimates}
                with self._tracer.start_as_current_span("mm.cache.lookup") as span:
                    entry = await self._s.cache.lookup(cache_request, eligible_model_ids=eligible_ids)
                    span.set_attribute("mm.cache.status", entry.status if entry else "miss")
                if entry is not None:
                    return await self._serve_cache_hit(prepared, run, entry.payload, entry.status, entry.similarity)
            self._route(prepared, routed, eligible_estimates)
            state = ExecutionState(
                deadline=Deadline.in_ms(routed.deadline_ms),
                allow_fallback=routed.allow_fallback,
                max_cost_usd=routed.max_cost_usd,
                warnings=routed.decision["warnings"],
            )
            if prepared.body.stream:
                plan = routed.plans[routed.candidates[0].model.id] if routed.candidates else None
                if plan is not None and plan.buffered:
                    return await self._buffered_stream(prepared, run, routed, state, cache_request, cache_status)
                handle = await self._s.executor.open_stream(routed.candidates, state)
                handed_off = True
                return StreamResult(
                    self._stream_events(prepared, run, routed, state, handle, cache_request, cache_status),
                    run.rate.headers(),
                )
            result = await self._s.executor.run(routed.candidates, state)
            await self._s.executor.observe_rate_limit(result.candidate.model.id, result.response)
            body = await self._complete(
                prepared,
                run,
                routed,
                state,
                result.candidate,
                result.response,
                result.latency_ms,
                cache_status,
            )
            await self._store_in_cache(cache_request, body, result.response)
            return JsonResult(body, run.rate.headers())
        finally:
            if not handed_off:
                await self._s.limiter.release(run.auth)

    # ------------------------------------------------------------ policy
    def _scoped_models(self, auth: AuthContext) -> list[CatalogModel]:
        models = self._s.registry.enabled_models()
        if auth.allowed_providers is not None:
            models = [m for m in models if m.provider in auth.allowed_providers]
        if auth.allowed_models is not None:
            models = [m for m in models if m.id in auth.allowed_models]
        return models

    async def _policy_and_estimates(
        self, prepared: PreparedRequest, run: _Run, budget_remaining: float | None
    ) -> tuple[_Routed, list[CandidateEstimate]]:
        with self._tracer.start_as_current_span("mm.policy"):
            body = prepared.body
            options = body.routing
            scoped = self._scoped_models(run.auth)
            explicit: CatalogModel | None = None
            pool = scoped
            if body.model != "auto":
                explicit = self._resolve_explicit(body.model, run.auth, scoped)
                wants_fallback = options is not None and options.fallback == "within_constraints"
                pool = [explicit, *(m for m in scoped if m.id != explicit.id)] if wants_fallback else [explicit]
            requested_fallback = options.fallback if options else None
            allow_fallback = (
                requested_fallback == "within_constraints" if explicit is not None else requested_fallback != "none"
            )
            estimated_tokens = self._s.estimator.tokens.estimate_messages(prepared.messages)
            quota_blocked = await self._s.quota.unavailable(pool, estimated_tokens=estimated_tokens)
            estimates = [
                self._s.estimator.estimate(
                    model,
                    messages=prepared.messages,
                    max_tokens=prepared.max_tokens,
                    task_type=prepared.task_type,
                )
                for model in pool
            ]
            constraints = Constraints(
                data_policy=effective_data_policy(
                    options.data_policy if options else None, run.auth.data_policy_ceiling
                ),
                max_cost_usd=options.max_cost_usd if options else None,
                max_latency_ms=options.max_latency_ms if options else None,
                min_quality=options.min_quality if options else None,
                allowed_providers=set(options.allowed_providers) if options and options.allowed_providers else None,
                denied_providers=set(options.denied_providers or []) if options else set(),
                allowed_models=set(options.allowed_models) if options and options.allowed_models else None,
                budget_remaining_usd=budget_remaining,
                quota_unavailable=quota_blocked,
                needs=prepared.needs,
            )
            if explicit is not None:
                # A named model that breaks a hard constraint is an error, even when fallback is allowed.
                filter_candidates(estimates[:1], constraints)
            eligible = filter_candidates(estimates, constraints) if estimates else []
            if not eligible:
                raise ApiError(
                    "no_eligible_model",
                    "No provider is configured for this key.",
                    code="constraints_unsatisfiable",
                    details={"candidates_considered": 0, "violated_constraints": []},
                )
            deadline_ms = self._deadline_ms(
                options.timeout_ms if options else None, options.max_latency_ms if options else None
            )
            routed = _Routed(
                candidates=[],
                estimates={estimate.model.id: estimate for estimate in eligible},
                plans={},
                decision={
                    "explicit": explicit.id if explicit else None,
                    "considered": len(estimates),
                    "eligible": len(eligible),
                    "data_policy": constraints.data_policy,
                    "warnings": list(prepared.warnings),
                },
                allow_fallback=allow_fallback,
                deadline_ms=deadline_ms,
                max_cost_usd=constraints.max_cost_usd,
            )
            return routed, eligible

    def _resolve_explicit(self, model_id: str, auth: AuthContext, scoped: list[CatalogModel]) -> CatalogModel:
        model = self._s.registry.catalog.get(model_id)
        if model is None or not self._s.registry.is_enabled(model.provider):
            raise ApiError("not_found", f"Unknown model '{model_id}'.", code="model_not_found", param="model")
        if all(candidate.id != model.id for candidate in scoped):
            raise ApiError(
                "permission_denied",
                "This key may not use that model.",
                code="model_not_allowed",
                param="model",
            )
        return model

    def _deadline_ms(self, timeout_ms: int | None, max_latency_ms: int | None) -> int:
        settings = self._s.settings
        if timeout_ms is not None:
            return min(timeout_ms, settings.max_timeout_ms)
        if max_latency_ms is not None:
            return min(2 * max_latency_ms, settings.max_timeout_ms)
        return settings.default_timeout_ms

    # ------------------------------------------------------------ cache
    def _cache_request(self, prepared: PreparedRequest, run: _Run, routed: _Routed) -> tuple[CacheRequest | None, str]:
        body = prepared.body
        options = body.cache
        mode = options.mode if options else None
        if mode == "off" or not run.auth.cache_enabled:
            return None, "bypass"
        if mode is None:
            deterministic = body.temperature == 0 or body.seed is not None
            if not deterministic:
                return None, "bypass"
            mode = "exact"
        if mode == "semantic" and prepared.tools:
            mode = "exact"
        constraints: Any = (
            body.routing.model_dump(exclude={"timeout_ms", "fallback"}, exclude_none=True) if body.routing else {}
        )
        namespace = options.namespace if options else None
        fingerprint = options.context_fingerprint if options else None
        partition = canonical_hash(
            {
                "tenant": run.auth.tenant_id,
                "namespace": namespace,
                "target": body.model,
                "constraints": constraints if body.model == "auto" else None,
                "prefix": prepared.messages[:-1],
                "generation": {
                    "temperature": body.temperature,
                    "top_p": body.top_p,
                    "max_tokens": prepared.max_tokens,
                    "stop": prepared.stop,
                    "seed": body.seed,
                    "presence_penalty": body.presence_penalty,
                    "frequency_penalty": body.frequency_penalty,
                },
                "response_format": prepared.response_format,
                "tools": prepared.tools,
                "tool_choice": prepared.tool_choice,
                "context_fingerprint": fingerprint,
                "task_type": prepared.task_type,
                "format": 1,
            }
        )
        settings = self._s.settings
        maximum = run.auth.cache_max_ttl_s or settings.cache_max_ttl_s
        requested = options.ttl_s if options and options.ttl_s else settings.cache_default_ttl_s
        return (
            CacheRequest(
                tenant_id=run.auth.tenant_id,
                partition=partition,
                final_message=prepared.final_user_text,
                mode=mode,
                task_type=prepared.task_type,
                namespace=namespace,
                context_fingerprint=fingerprint,
                has_tools=bool(prepared.tools),
                ttl_s=min(requested, maximum),
            ),
            "miss",
        )

    async def _store_in_cache(
        self, cache_request: CacheRequest | None, body: dict[str, Any], response: NormalizedChatResponse
    ) -> None:
        if cache_request is None or response.finish_reason not in ("stop", "length"):
            return
        try:
            await self._s.cache.store(cache_request, body)
        except Exception:  # A cache failure must never fail an otherwise successful request.
            logger.exception("cache store failed")

    async def _serve_cache_hit(
        self,
        prepared: PreparedRequest,
        run: _Run,
        payload: dict[str, Any],
        status: str,
        similarity: float | None,
    ) -> JsonResult | StreamResult:
        assert run.rate is not None
        original = payload["modelmesh"]
        self._instruments.cache_hits.add(1, {"status": status})
        metadata = {
            **original,
            "request_id": run.ctx.request_id,
            "client_request_id": run.ctx.effective_client_request_id,
            "trace_id": run.trace_id,
            "routing": None,
            "attempts": [],
            "fallback_used": False,
            "cache": {
                "status": status,
                "similarity": similarity,
                "original_request_id": original["request_id"],
                "saved_estimated_usd": original["cost"]["estimated_usd"],
            },
            "cost": {
                **original["cost"],
                "estimated_usd": 0.0,
                "provider_reported_usd": None,
                "failed_attempts_estimated_usd": 0.0,
            },
            "latency_ms": {
                "total": run.elapsed_ms(),
                "provider": None,
                "overhead": run.elapsed_ms(),
                "ttft": None,
            },
            "streaming": {"mode": "buffered"} if prepared.body.stream else None,
        }
        body = {
            **payload,
            "id": new_completion_id(run.ctx.request_id),
            "created": int(time.time()),
            "modelmesh": metadata,
        }
        usage = metadata["usage"]
        await self._persist(
            run,
            status="succeeded",
            http_status=200,
            metadata=metadata,
            usage=UsageFigures(
                usage["prompt_tokens"],
                usage["completion_tokens"],
                usage.get("cached_tokens"),
                usage.get("reasoning_tokens"),
                usage["source"],
                0.0,
                0.0,
                None,
                original["cost"]["pricing_version"],
            ),
            response_message=body["choices"][0]["message"],
            prepared=prepared,
        )
        if not prepared.body.stream:
            return JsonResult(body, run.rate.headers())
        return StreamResult(_replay(body, include_usage=_include_usage(prepared)), run.rate.headers())

    # ------------------------------------------------------------ routing
    def _route(
        self,
        prepared: PreparedRequest,
        routed: _Routed,
        eligible: list[CandidateEstimate],
    ) -> None:
        body = prepared.body
        options = body.routing
        started = time.perf_counter()
        with self._tracer.start_as_current_span("mm.route") as span:
            explicit: str | None = routed.decision["explicit"]
            policy = options.policy if options else "quality_balanced"
            calibrated = self._s.estimator.quality.is_task_calibrated(prepared.task_type)
            solved = solve(
                SolveInput(
                    policy=policy,
                    task_type=prepared.task_type,
                    calibrated=calibrated,
                    explicit_min_quality=options.min_quality if options else None,
                    max_latency_ms=options.max_latency_ms if options else None,
                    input_tokens=eligible[0].input_tokens,
                    schema_properties=prepared.schema_properties,
                ),
                eligible,
            )
            ranked = solved.ranked
            if explicit is not None:
                ranked = [e for e in eligible if e.model.id == explicit] + [e for e in ranked if e.model.id != explicit]
            routed.candidates, routed.plans = self._build_candidates(prepared, ranked, routed)
            chosen = ranked[0]
            interval = chosen.quality_interval
            warnings: list[str] = routed.decision["warnings"]
            warnings.extend(solved.warnings if explicit is None else [])
            mode = "explicit_model" if explicit is not None else solved.mode
            routed.decision.update(
                {
                    "mode": mode,
                    "policy": None if explicit is not None else policy,
                    "router_version": self._s.estimator.quality.router_version,
                    "task_type": prepared.task_type,
                    "candidates_considered": routed.decision["considered"],
                    "candidates_eligible": routed.decision["eligible"],
                    "predicted_quality": chosen.quality,
                    "predicted_quality_interval": list(interval) if interval else None,
                    "predicted_latency_ms": chosen.latency_p50_ms,
                    "predicted_latency_p90_ms": chosen.latency_p90_ms,
                    "estimated_cost_usd": chosen.upper_bound_cost_usd,
                    "reason": "model requested explicitly" if explicit is not None else solved.reason,
                }
            )
            span.set_attribute("mm.routing.mode", mode)
            span.set_attribute("mm.routing.policy", policy)
            span.set_attribute("mm.task_type", prepared.task_type)
        self._instruments.router_latency.record((time.perf_counter() - started) * 1000)

    def _build_candidates(
        self, prepared: PreparedRequest, ranked: list[CandidateEstimate], routed: _Routed
    ) -> tuple[list[Candidate], dict[str, Any]]:
        body = prepared.body
        candidates: list[Candidate] = []
        plans: dict[str, Any] = {}
        warnings: list[str] = routed.decision["warnings"]
        for position, estimate in enumerate(ranked):
            model = estimate.model
            plan = plan_structured(model, prepared.needs)
            assert plan is not None
            plans[model.id] = plan
            soft: dict[str, Any] = {
                "seed": body.seed,
                "top_p": body.top_p,
                "presence_penalty": body.presence_penalty,
                "frequency_penalty": body.frequency_penalty,
                "reasoning_effort": body.reasoning_effort,
            }
            for name, value in list(soft.items()):
                if name in model.dropped_params and value is not None:
                    soft[name] = None
                    if position == 0:
                        warnings.append(f"param_dropped:{name}")
            request = NormalizedChatRequest(
                provider_model=model.provider_model,
                messages=self._messages_for(prepared, plan.mode),
                temperature=body.temperature,
                top_p=soft["top_p"],
                max_tokens=prepared.max_tokens,
                stop=prepared.stop,
                seed=soft["seed"],
                presence_penalty=soft["presence_penalty"],
                frequency_penalty=soft["frequency_penalty"],
                reasoning_effort=soft["reasoning_effort"],
                response_format=_provider_response_format(prepared, model, plan.mode),
                structured_mode=plan.mode,
                tools=prepared.tools,
                tool_choice=prepared.tool_choice,
                parallel_tool_calls=body.parallel_tool_calls,
                require_no_training=routed.decision["data_policy"] != "any",
            )
            candidates.append(
                Candidate(
                    model=model,
                    request=request,
                    estimated_input_tokens=estimate.input_tokens,
                    upper_bound_cost_usd=estimate.upper_bound_cost_usd,
                    input_cost_estimate_usd=self._s.pricing.cost(model.id, estimate.input_tokens, 0),
                    validation_format=prepared.response_format,
                )
            )
        return candidates, plans

    @staticmethod
    def _messages_for(prepared: PreparedRequest, mode: str | None) -> list[dict[str, Any]]:
        if mode != "emulated" or prepared.response_format is None:
            return prepared.messages
        if prepared.response_format["type"] == "json_schema":
            instruction = emulation_instruction(prepared.response_format["json_schema"]["schema"])
        else:
            instruction = "Respond with only a single JSON object and no surrounding text."
        return [{"role": "system", "content": instruction}, *prepared.messages]

    # ------------------------------------------------------------ results
    def _usage(
        self, usage: TokenUsage | None, estimate: CandidateEstimate, output_text: str, model: CatalogModel
    ) -> TokenUsage:
        if usage is not None:
            return usage
        return TokenUsage(
            prompt_tokens=estimate.input_tokens,
            completion_tokens=self._s.estimator.tokens.estimate_text(output_text, model.family),
            source="estimated",
        )

    def _metadata(
        self,
        run: _Run,
        routed: _Routed,
        state: ExecutionState,
        model: CatalogModel,
        usage: TokenUsage,
        provider_cost: float | None,
        provider_latency_ms: int | None,
        cache_status: str,
        plan: Any,
        streaming_mode: str | None,
        ttft_ms: int | None,
    ) -> dict[str, Any]:
        decision = {
            key: value
            for key, value in routed.decision.items()
            if key not in ("explicit", "considered", "eligible", "data_policy")
        }
        total = run.elapsed_ms()
        structured = (
            {"mode": plan.mode, "validated": True, "repair_attempts": state.repair_attempts}
            if plan is not None and plan.mode is not None
            else None
        )
        return {
            "request_id": run.ctx.request_id,
            "client_request_id": run.ctx.effective_client_request_id,
            "trace_id": run.trace_id,
            "provider": model.provider,
            "model": model.id,
            "routing": decision,
            "attempts": [attempt.as_dict() for attempt in state.attempts],
            "fallback_used": state.fallback_used(model.id),
            "cache": {
                "status": cache_status,
                "similarity": None,
                "original_request_id": None,
                "saved_estimated_usd": None,
            },
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "cached_tokens": usage.cached_tokens,
                "reasoning_tokens": usage.reasoning_tokens,
                "source": usage.source,
            },
            "cost": {
                "estimated_usd": self._s.pricing.cost(model.id, usage.prompt_tokens, usage.completion_tokens),
                "provider_reported_usd": provider_cost,
                "failed_attempts_estimated_usd": state.failed_attempts_cost_usd,
                "pricing_version": self._s.pricing.version,
            },
            "latency_ms": {
                "total": total,
                "provider": provider_latency_ms,
                "overhead": max(0, total - state.attempts_elapsed_ms),
                "ttft": ttft_ms,
            },
            "structured_output": structured,
            "streaming": {"mode": streaming_mode} if streaming_mode else None,
        }

    def _observe_usage(self, prepared: PreparedRequest, model: CatalogModel, usage: TokenUsage, run: _Run) -> None:
        if usage.source == "provider_reported":
            chars = sum(len(str(m.get("content") or "")) for m in prepared.messages)
            self._s.estimator.tokens.observe(model.family, chars, usage.prompt_tokens)
        labels = {"provider": model.provider, "model": model.id}
        self._instruments.tokens.add(usage.prompt_tokens, {**labels, "direction": "input", "source": usage.source})
        self._instruments.tokens.add(usage.completion_tokens, {**labels, "direction": "output", "source": usage.source})

    async def _complete(
        self,
        prepared: PreparedRequest,
        run: _Run,
        routed: _Routed,
        state: ExecutionState,
        candidate: Candidate,
        response: NormalizedChatResponse,
        latency_ms: int,
        cache_status: str,
    ) -> dict[str, Any]:
        model = candidate.model
        usage = self._usage(response.usage, routed.estimates[model.id], response.content or "", model)
        self._observe_usage(prepared, model, usage, run)
        metadata = self._metadata(
            run,
            routed,
            state,
            model,
            usage,
            response.provider_reported_cost_usd,
            latency_ms,
            cache_status,
            routed.plans[model.id],
            None,
            None,
        )
        message: dict[str, Any] = {"role": "assistant", "content": response.content}
        if response.tool_calls:
            message["tool_calls"] = response.tool_calls
        body = ChatCompletionResponse.model_validate(
            {
                "id": new_completion_id(run.ctx.request_id),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model.id,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": response.finish_reason,
                        "logprobs": None,
                    }
                ],
                "usage": _openai_usage(usage),
                "modelmesh": metadata,
            }
        ).to_json()
        await self._finish_success(run, prepared, metadata, usage, response.provider_reported_cost_usd, message)
        return body

    async def _finish_success(
        self,
        run: _Run,
        prepared: PreparedRequest,
        metadata: dict[str, Any],
        usage: TokenUsage,
        provider_cost: float | None,
        message: dict[str, Any] | None,
        status: str = "succeeded",
        http_status: int = 200,
        error: dict[str, Any] | None = None,
    ) -> None:
        cost = metadata["cost"]
        figures = UsageFigures(
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.cached_tokens,
            usage.reasoning_tokens,
            usage.source,
            cost["estimated_usd"],
            cost["failed_attempts_estimated_usd"],
            provider_cost,
            cost["pricing_version"],
        )
        await self._persist(
            run,
            status=status,
            http_status=http_status,
            metadata=metadata,
            usage=figures,
            response_message=message,
            prepared=prepared,
            error=error,
        )
        spent = cost["estimated_usd"] + cost["failed_attempts_estimated_usd"]
        await self._s.budgets.add(run.auth, spent)
        if run.rate is not None:
            await self._s.limiter.reconcile_tokens(
                run.auth, run.rate.reserved_tokens, usage.prompt_tokens + usage.completion_tokens
            )
        self._instruments.cost.add(spent, {"provider": metadata["provider"]})
        if metadata["fallback_used"]:
            self._instruments.fallbacks.add(1)

    async def _persist(
        self,
        run: _Run,
        *,
        status: str,
        http_status: int | None,
        metadata: dict[str, Any] | None,
        usage: UsageFigures | None,
        response_message: dict[str, Any] | None,
        prepared: PreparedRequest | None,
        error: dict[str, Any] | None = None,
    ) -> None:
        retained = run.auth.content_retention and prepared is not None
        record = ExecutionRecord(
            request_id=run.ctx.request_id,
            tenant_id=run.auth.tenant_id,
            api_key_id=run.auth.key_id,
            client_request_id=run.ctx.effective_client_request_id,
            trace_id=run.trace_id,
            endpoint="chat.completions",
            stream=run.stream,
            status=status,
            http_status=http_status,
            created_at=run.created_at,
            completed_at=datetime.now(UTC),
            metadata=run.metadata,
            task_type=run.task_type,
            error=error,
            model_id=metadata["model"] if metadata else None,
            provider=metadata["provider"] if metadata else None,
            execution=metadata,
            cache_status=metadata["cache"]["status"] if metadata else None,
            fallback_used=bool(metadata and metadata["fallback_used"]),
            latency_total_ms=metadata["latency_ms"]["total"] if metadata else run.elapsed_ms(),
            latency_provider_ms=metadata["latency_ms"]["provider"] if metadata else None,
            ttft_ms=metadata["latency_ms"]["ttft"] if metadata else None,
            attempts=metadata["attempts"] if metadata else [],
            usage=usage,
            retained_messages=prepared.messages if retained and prepared is not None else None,
            retained_response=response_message if retained else None,
            retention_days=self._s.settings.content_retention_ttl_days,
        )
        with self._tracer.start_as_current_span("mm.accounting"):
            try:
                await self._s.records.write(record)
            except Exception:  # The client already has an answer; losing a record must not lose it.
                logger.exception("failed to write the execution record")
        self._instruments.requests.add(1, {"status": status, "provider": metadata["provider"] if metadata else "none"})
        self._instruments.request_duration.record(run.elapsed_ms())

    async def _record_failure(self, run: _Run, error: ApiError) -> None:
        attempts = error.details.get("attempts", [])
        envelope = {
            "type": error.error_type,
            "code": error.code,
            "message": error.message,
            "param": error.param,
            "request_id": run.ctx.request_id,
            "client_request_id": run.ctx.client_request_id,
            "retryable": error.retryable,
            "details": error.details,
        }
        status = "failed" if error.status_code >= 500 else "rejected"
        record = ExecutionRecord(
            request_id=run.ctx.request_id,
            tenant_id=run.auth.tenant_id,
            api_key_id=run.auth.key_id,
            client_request_id=run.ctx.effective_client_request_id,
            trace_id=run.trace_id,
            endpoint="chat.completions",
            stream=run.stream,
            status=status,
            http_status=error.status_code,
            created_at=run.created_at,
            completed_at=datetime.now(UTC),
            metadata=run.metadata,
            task_type=run.task_type,
            error=envelope,
            attempts=attempts,
            latency_total_ms=run.elapsed_ms(),
        )
        try:
            await self._s.records.write(record)
        except Exception:
            logger.exception("failed to write the rejection record")
        self._instruments.requests.add(1, {"status": status, "provider": "none"})

    # ------------------------------------------------------------ streaming
    async def _buffered_stream(
        self,
        prepared: PreparedRequest,
        run: _Run,
        routed: _Routed,
        state: ExecutionState,
        cache_request: CacheRequest | None,
        cache_status: str,
    ) -> StreamResult:
        assert run.rate is not None
        result = await self._s.executor.run(routed.candidates, state)
        await self._s.executor.observe_rate_limit(result.candidate.model.id, result.response)
        model = result.candidate.model
        usage = self._usage(result.response.usage, routed.estimates[model.id], result.response.content or "", model)
        self._observe_usage(prepared, model, usage, run)
        metadata = self._metadata(
            run,
            routed,
            state,
            model,
            usage,
            result.response.provider_reported_cost_usd,
            result.latency_ms,
            cache_status,
            routed.plans[model.id],
            "buffered",
            result.latency_ms,
        )
        message: dict[str, Any] = {"role": "assistant", "content": result.response.content}
        if result.response.tool_calls:
            message["tool_calls"] = result.response.tool_calls
        body = {
            "id": new_completion_id(run.ctx.request_id),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model.id,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": result.response.finish_reason,
                    "logprobs": None,
                }
            ],
            "usage": _openai_usage(usage),
            "modelmesh": metadata,
        }
        await self._finish_success(run, prepared, metadata, usage, result.response.provider_reported_cost_usd, message)
        await self._store_in_cache(cache_request, body, result.response)
        return StreamResult(_replay(body, include_usage=_include_usage(prepared)), run.rate.headers())

    async def _stream_events(
        self,
        prepared: PreparedRequest,
        run: _Run,
        routed: _Routed,
        state: ExecutionState,
        handle: StreamHandle,
        cache_request: CacheRequest | None,
        cache_status: str,
    ) -> AsyncIterator[bytes]:
        settings = self._s.settings
        model = handle.candidate.model
        completion_id = new_completion_id(run.ctx.request_id)
        created = int(time.time())
        include_usage = _include_usage(prepared)
        text: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        usage: TokenUsage | None = None
        provider_cost: float | None = None
        finish: str | None = None
        role_sent = False
        final_usage: TokenUsage | None = None
        metadata: dict[str, Any] | None = None
        outcome: str | None = None
        http_error: dict[str, Any] | None = None
        pending: asyncio.Future[NormalizedChunk] | None = None

        def chunk_event(delta: dict[str, Any], finish_reason: str | None = None) -> bytes:
            nonlocal role_sent
            if not role_sent:
                delta = {"role": "assistant", **delta}
                role_sent = True
            return sse_event(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model.id,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
                }
            )

        def conclude(result: str) -> dict[str, Any]:
            nonlocal final_usage, outcome
            outcome = result
            final_usage = self._usage(usage, routed.estimates[model.id], "".join(text), model)
            latency = self._s.executor.finish_stream(
                handle,
                state,
                outcome=result,
                input_tokens=final_usage.prompt_tokens,
                output_tokens=final_usage.completion_tokens,
            )
            return self._metadata(
                run,
                routed,
                state,
                model,
                final_usage,
                provider_cost,
                latency,
                cache_status,
                routed.plans[model.id],
                "native",
                handle.ttft_ms,
            )

        def absorb(chunk: NormalizedChunk) -> dict[str, Any] | None:
            nonlocal usage, provider_cost, finish
            usage = chunk.usage or usage
            provider_cost = (
                chunk.provider_reported_cost_usd if chunk.provider_reported_cost_usd is not None else provider_cost
            )
            finish = chunk.finish_reason or finish
            delta: dict[str, Any] = {}
            if chunk.content:
                text.append(chunk.content)
                delta["content"] = chunk.content
            if chunk.tool_calls:
                delta["tool_calls"] = []
                for call in chunk.tool_calls:
                    slot = tool_calls.setdefault(call.index, {"id": call.id, "name": call.name, "arguments": ""})
                    slot["id"] = call.id or slot["id"]
                    slot["name"] = call.name or slot["name"]
                    slot["arguments"] += call.arguments or ""
                    item: dict[str, Any] = {"index": call.index}
                    if call.id:
                        item["id"] = call.id
                        item["type"] = "function"
                    function = {k: v for k, v in (("name", call.name), ("arguments", call.arguments)) if v is not None}
                    if function:
                        item["function"] = function
                    delta["tool_calls"].append(item)
            return delta or None

        try:
            for chunk in handle.buffered:
                delta = absorb(chunk)
                if delta:
                    yield chunk_event(delta)
            idle_s = 0.0
            while finish is None or usage is None:
                if pending is None:
                    pending = asyncio.ensure_future(anext(handle.iterator))
                wait_s = min(settings.stream_keepalive_s, settings.stream_idle_timeout_s - idle_s)
                done, _ = await asyncio.wait({pending}, timeout=max(wait_s, 0.001))
                if not done:
                    idle_s += wait_s
                    if idle_s >= settings.stream_idle_timeout_s:
                        raise ProviderError(ErrorClass.TIMEOUT, "upstream idle timeout", possibly_billed=True)
                    yield KEEPALIVE
                    continue
                current, pending = pending, None
                try:
                    chunk = current.result()
                except StopAsyncIteration:
                    break
                idle_s = 0.0
                delta = absorb(chunk)
                if delta:
                    yield chunk_event(delta)
            if finish is None:
                raise ProviderError(
                    ErrorClass.UPSTREAM_UNAVAILABLE, "upstream stream ended early", possibly_billed=True
                )
            if prepared.response_format is not None:
                issues = validate_output(prepared.response_format, "".join(text))
                if issues:
                    http_error = {
                        "type": "upstream_error",
                        "code": "schema_validation_failed",
                        "message": "The streamed output did not satisfy the requested format.",
                        "details": {"validation_errors": [{"path": i.path, "keyword": i.keyword} for i in issues]},
                    }
                    raise ProviderError(
                        ErrorClass.SCHEMA_VALIDATION_FAILED, "stream validation failed", possibly_billed=True
                    )
            metadata = conclude("success")
            final_finish = "tool_calls" if tool_calls and finish == "stop" else finish
            finish_event = chunk_event({}, final_finish)
            if include_usage:
                yield finish_event
                assert final_usage is not None
                yield sse_event(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model.id,
                        "choices": [],
                        "usage": _openai_usage(final_usage),
                        "modelmesh": metadata,
                    }
                )
            else:
                payload = json.loads(finish_event[len(b"data: ") :])
                payload["modelmesh"] = metadata
                yield sse_event(payload)
            yield DONE
        except ProviderError as error:
            logger.warning("upstream stream failed", extra={"error_class": str(error.error_class)})
            if pending is not None:
                pending.cancel()
                pending = None
            metadata = conclude("stream_interrupted")
            envelope: dict[str, Any] = {
                "type": "upstream_error",
                "code": "stream_interrupted",
                "message": "The upstream stream failed before completing.",
                "details": {},
                **(http_error or {}),
                "param": None,
                "request_id": run.ctx.request_id,
                "client_request_id": run.ctx.client_request_id,
                "retryable": False,
            }
            http_error = envelope
            yield sse_event({"error": envelope, "modelmesh": metadata})
        finally:
            with anyio.CancelScope(shield=True):
                if pending is not None:
                    pending.cancel()
                closer = getattr(handle.iterator, "aclose", None)
                if closer is not None:
                    try:
                        await closer()
                    except Exception:
                        logger.debug("closing the upstream stream failed", exc_info=True)
                if outcome is None:
                    metadata = conclude("client_cancelled")
                assert metadata is not None
                assert final_usage is not None
                message: dict[str, Any] = {"role": "assistant", "content": "".join(text) or None}
                status = {"success": "succeeded", "stream_interrupted": "stream_interrupted"}.get(
                    outcome or "", "cancelled"
                )
                await self._finish_success(
                    run,
                    prepared,
                    metadata,
                    final_usage,
                    provider_cost,
                    message,
                    status=status,
                    error=http_error,
                )
                if outcome == "success" and cache_request is not None and finish in ("stop", "length"):
                    stored = _body_from_stream(completion_id, created, model.id, message, finish, final_usage, metadata)
                    try:
                        await self._s.cache.store(cache_request, stored)
                    except Exception:
                        logger.exception("cache store failed")
                await self._s.limiter.release(run.auth)


def _provider_response_format(
    prepared: PreparedRequest, model: CatalogModel, mode: str | None
) -> dict[str, Any] | None:
    fmt = prepared.response_format
    if fmt is None:
        return None
    if mode == "emulated":
        return {"type": "json_object"} if model.json_object else None
    if fmt["type"] == "json_object":
        return {"type": "json_object"}
    spec = {**fmt["json_schema"], "strict": mode == "native_strict"}
    return {"type": "json_schema", "json_schema": spec}


def _include_usage(prepared: PreparedRequest) -> bool:
    options = prepared.body.stream_options
    return bool(options and options.include_usage)


def _openai_usage(usage: TokenUsage) -> dict[str, Any]:
    result: dict[str, Any] = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.prompt_tokens + usage.completion_tokens,
    }
    if usage.cached_tokens is not None:
        result["prompt_tokens_details"] = {"cached_tokens": usage.cached_tokens}
    if usage.reasoning_tokens is not None:
        result["completion_tokens_details"] = {"reasoning_tokens": usage.reasoning_tokens}
    return result


def _body_from_stream(
    completion_id: str,
    created: int,
    model_id: str,
    message: dict[str, Any],
    finish: str,
    usage: TokenUsage,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_id,
        "choices": [{"index": 0, "message": message, "finish_reason": finish, "logprobs": None}],
        "usage": _openai_usage(usage),
        "modelmesh": metadata,
    }


async def _replay(body: dict[str, Any], *, include_usage: bool) -> AsyncIterator[bytes]:
    """Emit a complete answer as stream chunks (buffered mode and cache hits)."""
    choice = body["choices"][0]
    message = choice["message"]
    base = {
        "id": body["id"],
        "object": "chat.completion.chunk",
        "created": body["created"],
        "model": body["model"],
    }
    delta: dict[str, Any] = {"role": "assistant"}
    if message.get("content"):
        delta["content"] = message["content"]
    if message.get("tool_calls"):
        delta["tool_calls"] = [{"index": i, **call} for i, call in enumerate(message["tool_calls"])]
    yield sse_event({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
    finish = {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}]}
    if include_usage:
        yield sse_event(finish)
        yield sse_event({**base, "choices": [], "usage": body["usage"], "modelmesh": body["modelmesh"]})
    else:
        yield sse_event({**finish, "modelmesh": body["modelmesh"]})
    yield DONE
