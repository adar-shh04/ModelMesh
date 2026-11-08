"""SQLAlchemy models (ADR-0010). The model catalog and pricing are versioned YAML in config/."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    settings: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    hmac: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    prefix: Mapped[str] = mapped_column(String(32), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    scopes: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    limits: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    data_policy_ceiling: Mapped[str] = mapped_column(String(32), nullable=False, default="any")
    content_retention: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class RequestRecord(Base):
    __tablename__ = "requests"
    __table_args__ = (
        Index("ix_requests_tenant_created", "tenant_id", "created_at"),
        Index("ix_requests_correlation", "tenant_id", "correlation_id"),
        Index("ix_requests_trace", "trace_id"),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    api_key_id: Mapped[str] = mapped_column(String(64), nullable=False)
    client_request_id: Mapped[str] = mapped_column(String(128), nullable=False)
    trace_id: Mapped[str] = mapped_column(String(32), nullable=False)
    correlation_id: Mapped[str | None] = mapped_column(String(128))
    application: Mapped[str | None] = mapped_column(String(64))
    task_type: Mapped[str | None] = mapped_column(String(64))
    end_user_hash: Mapped[str | None] = mapped_column(String(128))
    endpoint: Mapped[str] = mapped_column(String(32), nullable=False)
    stream: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    http_status: Mapped[int | None] = mapped_column(Integer)
    error_type: Mapped[str | None] = mapped_column(String(64))
    error_code: Mapped[str | None] = mapped_column(String(64))
    error: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    model_id: Mapped[str | None] = mapped_column(String(200))
    provider: Mapped[str | None] = mapped_column(String(32))
    request_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, default=dict)
    execution: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    cache_status: Mapped[str | None] = mapped_column(String(16))
    fallback_used: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    latency_total_ms: Mapped[int | None] = mapped_column(Integer)
    latency_provider_ms: Mapped[int | None] = mapped_column(Integer)
    ttft_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Attempt(Base):
    __tablename__ = "attempts"

    request_id: Mapped[str] = mapped_column(ForeignKey("requests.id", ondelete="CASCADE"), primary_key=True)
    seq: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    model_id: Mapped[str] = mapped_column(String(200), nullable=False)
    outcome: Mapped[str] = mapped_column(String(40), nullable=False)
    http_status: Mapped[int | None] = mapped_column(Integer)
    retryable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    possibly_billed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class UsageRecord(Base):
    __tablename__ = "usage_records"

    request_id: Mapped[str] = mapped_column(ForeignKey("requests.id", ondelete="CASCADE"), primary_key=True)
    prompt_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    cached_tokens: Mapped[int | None] = mapped_column(BigInteger)
    reasoning_tokens: Mapped[int | None] = mapped_column(BigInteger)
    usage_source: Mapped[str] = mapped_column(String(32), nullable=False)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    failed_attempts_estimated_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    provider_reported_cost_usd: Mapped[float | None] = mapped_column(Float)
    pricing_version: Mapped[str] = mapped_column(String(32), nullable=False)


class RequestContent(Base):
    """Prompt and response content. Written only when the key enabled content retention."""

    __tablename__ = "request_content"

    request_id: Mapped[str] = mapped_column(ForeignKey("requests.id", ondelete="CASCADE"), primary_key=True)
    messages: Mapped[list[Any]] = mapped_column(JSONB, nullable=False)
    response: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)


class Feedback(Base):
    __tablename__ = "feedback"
    __table_args__ = (UniqueConstraint("request_id", "metric", "metric_version", "evaluator", name="uq_feedback_key"),)

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    request_id: Mapped[str] = mapped_column(ForeignKey("requests.id", ondelete="CASCADE"), nullable=False, index=True)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    label_source: Mapped[str] = mapped_column(String(32), nullable=False)
    metric: Mapped[str] = mapped_column(String(64), nullable=False)
    metric_version: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    score: Mapped[float] = mapped_column(Float, nullable=False)
    higher_is_better: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    accepted: Mapped[bool | None] = mapped_column(Boolean)
    attribution: Mapped[str] = mapped_column(String(16), nullable=False, default="direct")
    evaluator: Mapped[str] = mapped_column(String(128), nullable=False)
    eval_run_id: Mapped[str | None] = mapped_column(String(128))
    dataset_version: Mapped[str | None] = mapped_column(String(128))
    labelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    task_type: Mapped[str | None] = mapped_column(String(64))
    notes: Mapped[str | None] = mapped_column(Text)
    training_eligible: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class ProviderSetting(Base):
    """Operator enable/disable switch per provider (admin API)."""

    __tablename__ = "provider_settings"

    provider: Mapped[str] = mapped_column(String(32), primary_key=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
