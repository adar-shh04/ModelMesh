"""Durable execution records: one per request, including rejected ones (ADR-0010)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from modelmesh.persistence import models as db


@dataclass
class UsageFigures:
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int | None
    reasoning_tokens: int | None
    source: str
    estimated_cost_usd: float
    failed_attempts_estimated_usd: float
    provider_reported_cost_usd: float | None
    pricing_version: str


@dataclass
class ExecutionRecord:
    request_id: str
    tenant_id: str
    api_key_id: str
    client_request_id: str
    trace_id: str
    endpoint: str
    stream: bool
    status: str
    http_status: int | None
    created_at: datetime
    completed_at: datetime
    metadata: dict[str, str] = field(default_factory=dict)
    task_type: str | None = None
    error: dict[str, Any] | None = None
    model_id: str | None = None
    provider: str | None = None
    execution: dict[str, Any] | None = None
    cache_status: str | None = None
    fallback_used: bool = False
    latency_total_ms: int | None = None
    latency_provider_ms: int | None = None
    ttft_ms: int | None = None
    attempts: list[dict[str, Any]] = field(default_factory=list)
    usage: UsageFigures | None = None
    retained_messages: list[dict[str, Any]] | None = None
    retained_response: dict[str, Any] | None = None
    retention_days: int = 7


class RecordStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def write(self, record: ExecutionRecord) -> None:
        async with self._sessions.begin() as session:
            error = record.error or {}
            session.add(
                db.RequestRecord(
                    id=record.request_id,
                    tenant_id=record.tenant_id,
                    api_key_id=record.api_key_id,
                    client_request_id=record.client_request_id,
                    trace_id=record.trace_id,
                    correlation_id=record.metadata.get("correlation_id"),
                    application=record.metadata.get("application"),
                    task_type=record.task_type,
                    end_user_hash=record.metadata.get("end_user_id"),
                    endpoint=record.endpoint,
                    stream=record.stream,
                    status=record.status,
                    http_status=record.http_status,
                    error_type=error.get("type"),
                    error_code=error.get("code"),
                    error=record.error,
                    model_id=record.model_id,
                    provider=record.provider,
                    request_metadata=record.metadata,
                    execution=record.execution,
                    cache_status=record.cache_status,
                    fallback_used=record.fallback_used,
                    latency_total_ms=record.latency_total_ms,
                    latency_provider_ms=record.latency_provider_ms,
                    ttft_ms=record.ttft_ms,
                    created_at=record.created_at,
                    completed_at=record.completed_at,
                )
            )
            await session.flush()
            for seq, attempt in enumerate(record.attempts):
                session.add(
                    db.Attempt(
                        request_id=record.request_id,
                        seq=seq,
                        provider=attempt["provider"],
                        model_id=attempt["model"],
                        outcome=attempt["outcome"],
                        http_status=attempt["http_status"],
                        retryable=attempt["retryable"],
                        latency_ms=attempt["latency_ms"],
                        possibly_billed=attempt["possibly_billed"],
                    )
                )
            if record.usage is not None:
                usage = record.usage
                session.add(
                    db.UsageRecord(
                        request_id=record.request_id,
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                        cached_tokens=usage.cached_tokens,
                        reasoning_tokens=usage.reasoning_tokens,
                        usage_source=usage.source,
                        estimated_cost_usd=usage.estimated_cost_usd,
                        failed_attempts_estimated_usd=usage.failed_attempts_estimated_usd,
                        provider_reported_cost_usd=usage.provider_reported_cost_usd,
                        pricing_version=usage.pricing_version,
                    )
                )
            if record.retained_messages is not None:
                session.add(
                    db.RequestContent(
                        request_id=record.request_id,
                        messages=record.retained_messages,
                        response=record.retained_response,
                        expires_at=datetime.now(UTC) + timedelta(days=record.retention_days),
                    )
                )

    async def spend_since(self, key_id: str, since: datetime) -> float:
        statement = (
            select(
                func.coalesce(
                    func.sum(db.UsageRecord.estimated_cost_usd + db.UsageRecord.failed_attempts_estimated_usd),
                    0.0,
                )
            )
            .join(db.RequestRecord, db.RequestRecord.id == db.UsageRecord.request_id)
            .where(db.RequestRecord.api_key_id == key_id, db.RequestRecord.created_at >= since)
        )
        async with self._sessions() as session:
            return float((await session.execute(statement)).scalar_one())
