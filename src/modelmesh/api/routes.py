"""Public /v1 endpoints. Request bodies are parsed by hand so that authentication always comes first."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ValidationError
from redis.exceptions import RedisError
from sqlalchemy import text

from modelmesh.accounting.queries import AccountingQueries, UsageQuery
from modelmesh.api.errors import invalid_query
from modelmesh.auth.keys import AuthContext
from modelmesh.contract.schemas import (
    CacheInvalidateRequest,
    CacheInvalidateResponse,
    ContractInfo,
    FeedbackBatchItem,
    FeedbackBatchRequest,
    FeedbackBatchResponse,
    FeedbackDeleteResponse,
    FeedbackRequest,
    FeedbackResponse,
    Health,
    Model,
    ModelList,
    ProviderStatusList,
    RequestRecord,
    RequestRecordList,
    RoutingCalibration,
    UsageReport,
)
from modelmesh.core.context import RequestContext
from modelmesh.core.errors import ApiError
from modelmesh.core.settings import CONTRACT_RELEASE, CONTRACT_VERSION
from modelmesh.pipeline.chat import ChatPipeline, JsonResult
from modelmesh.pipeline.prepare import validation_error
from modelmesh.pipeline.services import Services
from modelmesh.pipeline.status import acceptance_criteria
from modelmesh.routing.solver import KNOWN_TASK_TYPES

router = APIRouter(prefix="/v1")

EXTENSIONS = [
    "routing.policy",
    "routing.max_cost_usd",
    "routing.max_latency_ms",
    "routing.timeout_ms",
    "routing.min_quality",
    "routing.allowed_providers",
    "routing.denied_providers",
    "routing.allowed_models",
    "routing.required_capabilities",
    "routing.data_policy",
    "routing.fallback",
    "cache.mode",
    "cache.ttl_s",
    "cache.context_fingerprint",
    "cache.namespace",
]
CAPABILITIES = ["json_object", "json_schema", "json_schema_strict", "tools", "streaming", "long_context:<n>"]
POLICIES = ["cost_optimized", "latency_optimized", "quality_balanced", "quality_max", "reliability_first"]
FEEDBACK_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "citation_validity",
    "citation_support_rate",
    "temporal_accuracy",
    "schema_valid",
    "task_success",
    "user_thumbs",
]
MAX_USAGE_RANGE = timedelta(days=92)


def services(request: Request) -> Services:
    result: Services = request.app.state.services
    return result


ServicesDep = Annotated[Services, Depends(services)]


async def authenticated(request: Request, svc: ServicesDep) -> AuthContext:
    try:
        return await svc.authenticator.authenticate(request.headers.get("authorization"))
    except RedisError:
        raise ApiError(
            "service_unavailable", "A required dependency is unavailable.", code="dependency_unavailable"
        ) from None


AuthDep = Annotated[AuthContext, Depends(authenticated)]


def context_of(request: Request) -> RequestContext:
    ctx: RequestContext = request.state.request_context
    return ctx


async def read_body(request: Request) -> bytes:
    body = await request.body()
    if request.state.body_limit_exceeded():
        raise ApiError("request_too_large", "The request body is too large.", code="body_too_large")
    return body


def parse_body[T: BaseModel](model: type[T], raw: bytes) -> T:
    try:
        decoded = json.loads(raw)
    except ValueError:
        raise ApiError("invalid_request", "The body is not valid JSON.", code="invalid_value") from None
    try:
        return model.model_validate(decoded)
    except ValidationError as exc:
        raise validation_error(exc) from None


@router.post("/chat/completions")
async def chat_completions(request: Request, svc: ServicesDep, auth: AuthDep) -> Any:
    raw = await read_body(request)
    result = await ChatPipeline(svc).handle(raw, auth, context_of(request))
    if isinstance(result, JsonResult):
        return JSONResponse(result.body, headers=result.headers)
    return StreamingResponse(result.events, media_type="text/event-stream", headers=result.headers)


@router.post("/embeddings")
async def embeddings(auth: AuthDep) -> Any:
    raise ApiError("not_implemented", "Embeddings are not enabled on this server.", code="embeddings_disabled")


@router.get("/models")
async def list_models(svc: ServicesDep, auth: AuthDep) -> Any:
    from modelmesh.pipeline.status import model_entries

    entries = await model_entries(svc, auth)
    return ModelList(object="list", data=[Model.model_validate(entry) for entry in entries]).to_json()


@router.get("/health")
async def health(svc: ServicesDep) -> Any:
    postgres = "ok"
    redis_status = "ok"
    try:
        async with svc.engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
    except Exception:
        postgres = "down"
    try:
        await svc.redis.ping()
    except RedisError:
        redis_status = "down"
    providers = {"ok": 0, "degraded": 0, "down": 0}
    for provider in svc.registry.adapters:
        if not svc.registry.is_enabled(provider):
            continue
        models = [m.id for m in svc.registry.enabled_models() if m.provider == provider]
        if provider in svc.breakers.misconfigured:
            providers["down"] += 1
        elif svc.breakers.provider_state(provider, models) == "closed":
            providers["ok"] += 1
        else:
            providers["degraded"] += 1
    if postgres == "down" or redis_status == "down":
        status = "down"
    elif providers["degraded"] or providers["down"]:
        status = "degraded"
    else:
        status = "ok"
    body = Health.model_validate(
        {
            "status": status,
            "contract_version": CONTRACT_VERSION,
            "components": {
                "postgres": {"status": postgres},
                "redis": {"status": redis_status},
                "providers": providers,
            },
        }
    ).to_json()
    return JSONResponse(body, status_code=503 if status == "down" else 200)


@router.get("/contract")
async def contract() -> Any:
    endpoints = [
        "POST /v1/chat/completions",
        "POST /v1/embeddings",
        "GET /v1/models",
        "GET /v1/health",
        "GET /v1/contract",
        "GET /v1/providers",
        "GET /v1/usage",
        "GET /v1/requests",
        "GET /v1/requests/{request_id}",
        "POST /v1/feedback",
        "POST /v1/feedback/batch",
        "DELETE /v1/feedback/{request_id}",
        "POST /v1/cache/invalidate",
        "GET /v1/routing/calibration",
    ]
    return ContractInfo(
        contract_version=CONTRACT_VERSION,
        contract_release=CONTRACT_RELEASE,
        endpoints=endpoints,
        extensions=EXTENSIONS,
        task_types=list(KNOWN_TASK_TYPES),
        capabilities=CAPABILITIES,
        routing_policies=POLICIES,
        feedback_metrics=FEEDBACK_METRICS,
    ).to_json()


@router.get("/routing/calibration")
async def routing_calibration(svc: ServicesDep, auth: AuthDep) -> Any:
    quality = svc.estimator.quality
    tasks: list[dict[str, Any]] = [
        {
            "task_type": task,
            "calibrated": quality.is_task_calibrated(task),
            "acceptance_criterion": acceptance_criteria(task),
            "acceptance_criterion_version": "v1",
            "benchmark_manifest": None,
            "calibration_samples": 0,
            "ece": None,
            "brier": None,
            "judge_human_kappa": None,
            "models": [],
        }
        for task in KNOWN_TASK_TYPES
    ]
    return RoutingCalibration.model_validate(
        {"router_version": quality.router_version, "generated_at": datetime.now(UTC), "task_types": tasks}
    ).to_json()


@router.get("/providers")
async def providers_status(svc: ServicesDep, auth: AuthDep) -> Any:
    data = []
    for provider in sorted({m.provider for m in svc.registry.catalog.models.values()} | set(svc.registry.adapters)):
        models = [m.id for m in svc.registry.enabled_models() if m.provider == provider]
        p50, p95 = svc.stats.latency_percentiles(provider)
        state = svc.breakers.provider_state(provider, models)
        if not svc.registry.is_enabled(provider):
            status = "disabled"
        elif provider in svc.breakers.misconfigured:
            status = "misconfigured"
        elif state == "closed":
            status = "ok"
        else:
            status = "degraded" if models else "down"
        data.append(
            {
                "id": provider,
                "status": status,
                "circuit_state": state,
                "error_rate_5m": svc.stats.error_rate(provider),
                "p50_latency_ms": p50,
                "p95_latency_ms": p95,
                "models_available": len(models),
            }
        )
    return ProviderStatusList.model_validate({"data": data}).to_json()


def parse_timestamp(name: str, value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ApiError("invalid_request", f"Invalid value for '{name}'.", code="invalid_value", param=name) from None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


@router.get("/usage")
async def usage(
    svc: ServicesDep,
    auth: AuthDep,
    start: Annotated[str, Query(alias="from")],
    end: Annotated[str, Query(alias="to")],
    group_by: Annotated[str, Query(pattern="^(model|provider|task_type|application|day)$")] = "model",
    api_key_id: Annotated[str | None, Query(pattern="^key_[A-Za-z0-9]{8,40}$")] = None,
) -> Any:
    start_at = parse_timestamp("from", start)
    end_at = parse_timestamp("to", end)
    assert start_at is not None
    assert end_at is not None
    if end_at <= start_at or end_at - start_at > MAX_USAGE_RANGE:
        raise ApiError(
            "invalid_request",
            "The range must be positive and at most 92 days.",
            code="invalid_value",
            param="to",
        )
    rows, totals = await AccountingQueries(svc.sessions).usage(
        UsageQuery(auth.tenant_id, start_at, end_at, group_by, api_key_id)
    )
    return UsageReport.model_validate(
        {"from": start_at, "to": end_at, "group_by": group_by, "data": rows, "totals": totals}
    ).to_json()


@router.get("/requests")
async def list_requests(
    svc: ServicesDep,
    auth: AuthDep,
    correlation_id: Annotated[str | None, Query(max_length=128, pattern=r"^[A-Za-z0-9_:.-]+$")] = None,
    start: Annotated[str | None, Query(alias="from")] = None,
    end: Annotated[str | None, Query(alias="to")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    cursor: Annotated[str | None, Query(max_length=256)] = None,
) -> Any:
    rows, next_cursor = await AccountingQueries(svc.sessions).list_requests(
        auth.tenant_id,
        correlation_id=correlation_id,
        start=parse_timestamp("from", start),
        end=parse_timestamp("to", end),
        limit=limit,
        cursor=cursor,
    )
    return RequestRecordList.model_validate({"data": rows, "next_cursor": next_cursor}).to_json()


@router.get("/requests/{request_id}")
async def get_request(request_id: str, svc: ServicesDep, auth: AuthDep) -> Any:
    record = await AccountingQueries(svc.sessions).get_request(auth.tenant_id, request_id)
    if record is None:
        raise ApiError("not_found", "Unknown request id.", code="request_not_found", param="request_id")
    return RequestRecord.model_validate(record).to_json()


def _feedback_service(svc: Services) -> Any:
    from modelmesh.accounting.feedback import FeedbackService

    return FeedbackService(svc.sessions)


@router.post("/feedback")
async def create_feedback(request: Request, svc: ServicesDep, auth: AuthDep) -> Any:
    item = parse_body(FeedbackRequest, await read_body(request))
    result = await _feedback_service(svc).submit(auth.tenant_id, item)
    body = FeedbackResponse.model_validate(result).to_json()
    return JSONResponse(body, status_code=201 if result["status"] == "stored" else 200)


@router.post("/feedback/batch")
async def create_feedback_batch(request: Request, svc: ServicesDep, auth: AuthDep) -> Any:
    raw = await read_body(request)
    envelope = parse_body(FeedbackBatchRequest, raw)
    ctx = context_of(request)
    service = _feedback_service(svc)
    results: list[FeedbackBatchItem] = []
    status_by_type = {"conflict": "conflict", "not_found": "not_found"}
    for index, item in enumerate(envelope.items):
        try:
            stored = await service.submit(auth.tenant_id, item)
            results.append(
                FeedbackBatchItem.model_validate(
                    {"index": index, "status": stored["status"], "feedback": stored, "error": None}
                )
            )
        except ApiError as error:
            results.append(
                FeedbackBatchItem.model_validate(
                    {
                        "index": index,
                        "status": status_by_type.get(error.error_type, "invalid"),
                        "feedback": None,
                        "error": {
                            "type": error.error_type,
                            "code": error.code,
                            "message": error.message,
                            "param": error.param,
                            "request_id": ctx.request_id,
                            "client_request_id": ctx.client_request_id,
                            "retryable": error.retryable,
                            "details": error.details,
                        },
                    }
                )
            )
    return FeedbackBatchResponse(results=results).to_json()


@router.delete("/feedback/{request_id}")
async def delete_feedback(
    request_id: str,
    svc: ServicesDep,
    auth: AuthDep,
    metric: Annotated[str | None, Query(pattern=r"^[a-z0-9_.:-]{1,64}$")] = None,
    evaluator: Annotated[str | None, Query(max_length=128)] = None,
    metric_version: Annotated[str | None, Query(max_length=64)] = None,
) -> Any:
    deleted = await _feedback_service(svc).delete(
        auth.tenant_id, request_id, metric=metric, evaluator=evaluator, metric_version=metric_version
    )
    return FeedbackDeleteResponse(deleted=deleted).to_json()


@router.post("/cache/invalidate")
async def invalidate_cache(request: Request, svc: ServicesDep, auth: AuthDep) -> Any:
    body = parse_body(CacheInvalidateRequest, await read_body(request))
    try:
        deleted = await svc.cache.invalidate(auth.tenant_id, body.namespace, body.context_fingerprint)
    except RedisError:
        raise ApiError(
            "service_unavailable", "A required dependency is unavailable.", code="dependency_unavailable"
        ) from None
    return CacheInvalidateResponse(deleted_entries=deleted).to_json()


__all__ = ["invalid_query", "router"]
