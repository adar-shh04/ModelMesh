"""Read side of accounting: usage aggregates and execution records, always scoped to a tenant."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import Select, and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from modelmesh.persistence import models as db

GROUP_COLUMNS: dict[str, Any] = {
    "model": db.RequestRecord.model_id,
    "provider": db.RequestRecord.provider,
    "task_type": db.RequestRecord.task_type,
    "application": db.RequestRecord.application,
    "day": func.to_char(func.timezone("UTC", db.RequestRecord.created_at), "YYYY-MM-DD"),
}


def _sum_if(condition: Any, column: Any) -> Any:
    return func.coalesce(func.sum(column).filter(condition), 0)


def _usage_columns() -> list[Any]:
    usage = db.UsageRecord
    request = db.RequestRecord
    provider_reported = usage.usage_source == "provider_reported"
    estimated = usage.usage_source == "estimated"
    return [
        func.count(request.id).label("requests"),
        func.count(request.id).filter(request.status == "succeeded").label("succeeded"),
        func.count(request.id).filter(request.status != "succeeded").label("failed"),
        func.count(request.id).filter(request.cache_status.in_(["hit_exact", "hit_semantic"])).label("cache_hits"),
        func.count(request.id).filter(request.fallback_used.is_(True)).label("fallback_events"),
        _sum_if(provider_reported, usage.prompt_tokens).label("prompt_reported"),
        _sum_if(provider_reported, usage.completion_tokens).label("completion_reported"),
        _sum_if(estimated, usage.prompt_tokens).label("prompt_estimated"),
        _sum_if(estimated, usage.completion_tokens).label("completion_estimated"),
        func.coalesce(func.sum(usage.estimated_cost_usd + usage.failed_attempts_estimated_usd), 0.0).label("cost"),
        func.sum(usage.provider_reported_cost_usd).label("reported_cost"),
        func.count(usage.provider_reported_cost_usd).label("reported_cost_count"),
        func.percentile_cont(0.5).within_group(request.latency_total_ms).label("p50"),
        func.percentile_cont(0.95).within_group(request.latency_total_ms).label("p95"),
    ]


def _usage_row(group: str | None, row: Any) -> dict[str, Any]:
    return {
        "group": group,
        "requests": int(row.requests),
        "succeeded": int(row.succeeded),
        "failed": int(row.failed),
        "cache_hits": int(row.cache_hits),
        "fallback_events": int(row.fallback_events),
        "prompt_tokens_provider_reported": int(row.prompt_reported),
        "completion_tokens_provider_reported": int(row.completion_reported),
        "prompt_tokens_estimated": int(row.prompt_estimated),
        "completion_tokens_estimated": int(row.completion_estimated),
        "estimated_cost_usd": float(row.cost),
        "provider_reported_cost_usd": float(row.reported_cost) if row.reported_cost is not None else None,
        "provider_reported_cost_coverage": int(row.reported_cost_count) / int(row.requests) if row.requests else 0.0,
        "latency_p50_ms": int(row.p50) if row.p50 is not None else None,
        "latency_p95_ms": int(row.p95) if row.p95 is not None else None,
    }


@dataclass
class UsageQuery:
    tenant_id: str
    start: datetime
    end: datetime
    group_by: str
    api_key_id: str | None = None


class AccountingQueries:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def usage(self, query: UsageQuery) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        request = db.RequestRecord
        conditions = [
            request.tenant_id == query.tenant_id,
            request.created_at >= query.start,
            request.created_at < query.end,
        ]
        if query.api_key_id is not None:
            conditions.append(request.api_key_id == query.api_key_id)
        joined = db.UsageRecord.request_id == request.id
        group = GROUP_COLUMNS[query.group_by]
        grouped = (
            select(group.label("group_value"), *_usage_columns())
            .select_from(request)
            .outerjoin(db.UsageRecord, joined)
            .where(and_(*conditions))
            .group_by(group)
            .order_by(group)
        )
        totals = (
            select(*_usage_columns()).select_from(request).outerjoin(db.UsageRecord, joined).where(and_(*conditions))
        )
        async with self._sessions() as session:
            rows = (await session.execute(grouped)).all()
            total = (await session.execute(totals)).one()
        return [_usage_row(row.group_value or "unknown", row) for row in rows], _usage_row(None, total)

    @staticmethod
    def _detail_query() -> Select[Any]:
        feedback_count = (
            select(func.count(db.Feedback.id))
            .where(db.Feedback.request_id == db.RequestRecord.id)
            .correlate(db.RequestRecord)
            .scalar_subquery()
        )
        return select(db.RequestRecord, db.RequestContent, feedback_count.label("feedback_count")).outerjoin(
            db.RequestContent,
            and_(db.RequestContent.request_id == db.RequestRecord.id, db.RequestContent.expires_at > func.now()),
        )

    @staticmethod
    def _record_json(row: Any) -> dict[str, Any]:
        record: db.RequestRecord = row[0]
        content: db.RequestContent | None = row[1]
        return {
            "id": record.id,
            "client_request_id": record.client_request_id,
            "created_at": record.created_at,
            "completed_at": record.completed_at,
            "status": record.status,
            "http_status": record.http_status,
            "api_key_id": record.api_key_id,
            "endpoint": record.endpoint,
            "stream": record.stream,
            "metadata": record.request_metadata,
            "modelmesh": record.execution,
            "error": record.error,
            "feedback_count": int(row[2]),
            "content": (
                {"messages": content.messages, "response": content.response, "expires_at": content.expires_at}
                if content is not None
                else None
            ),
        }

    async def get_request(self, tenant_id: str, request_id: str) -> dict[str, Any] | None:
        statement = self._detail_query().where(
            db.RequestRecord.id == request_id, db.RequestRecord.tenant_id == tenant_id
        )
        async with self._sessions() as session:
            row = (await session.execute(statement)).first()
        return self._record_json(row) if row is not None else None

    async def list_requests(
        self,
        tenant_id: str,
        *,
        correlation_id: str | None,
        start: datetime | None,
        end: datetime | None,
        limit: int,
        cursor: str | None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        statement = self._detail_query().where(db.RequestRecord.tenant_id == tenant_id)
        if correlation_id is not None:
            statement = statement.where(db.RequestRecord.correlation_id == correlation_id)
        if start is not None:
            statement = statement.where(db.RequestRecord.created_at >= start)
        if end is not None:
            statement = statement.where(db.RequestRecord.created_at < end)
        if cursor is not None:
            statement = statement.where(db.RequestRecord.id < decode_cursor(cursor))
        statement = statement.order_by(db.RequestRecord.id.desc()).limit(limit + 1)
        async with self._sessions() as session:
            rows = (await session.execute(statement)).all()
        page = rows[:limit]
        next_cursor = encode_cursor(page[-1][0].id) if len(rows) > limit else None
        return [self._record_json(row) for row in page], next_cursor

    async def trace_ids(self, tenant_id: str, request_ids: list[str]) -> dict[str, str]:
        """Trace ids for request records, for links to the trace viewer. Records of other tenants are ignored."""
        if not request_ids:
            return {}
        statement = select(db.RequestRecord.id, db.RequestRecord.trace_id).where(
            db.RequestRecord.tenant_id == tenant_id, db.RequestRecord.id.in_(request_ids)
        )
        async with self._sessions() as session:
            rows = (await session.execute(statement)).all()
        return {row.id: row.trace_id for row in rows}


def encode_cursor(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode()).decode()


def decode_cursor(value: str) -> str:
    try:
        return base64.urlsafe_b64decode(value.encode()).decode()
    except ValueError:
        return ""
