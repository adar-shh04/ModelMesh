"""Contract error model and the internal upstream-error taxonomy (ADR-0008)."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

HTTP_STATUS_BY_TYPE: dict[str, int] = {
    "invalid_request": 400,
    "authentication_error": 401,
    "budget_exceeded": 402,
    "permission_denied": 403,
    "not_found": 404,
    "conflict": 409,
    "request_too_large": 413,
    "no_eligible_model": 422,
    "rate_limited": 429,
    "internal_error": 500,
    "not_implemented": 501,
    "upstream_error": 502,
    "service_unavailable": 503,
    "upstream_timeout": 504,
}

RETRYABLE_TYPES = {
    "rate_limited",
    "internal_error",
    "upstream_error",
    "service_unavailable",
    "upstream_timeout",
}


class ApiError(Exception):
    """An error that is reported to the client using the contract's error envelope."""

    def __init__(
        self,
        error_type: str,
        message: str,
        *,
        code: str | None = None,
        param: str | None = None,
        details: dict[str, Any] | None = None,
        retry_after_s: int | None = None,
        retryable: bool | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.code = code
        self.param = param
        self.details = details or {}
        self.retry_after_s = retry_after_s
        self.retryable = error_type in RETRYABLE_TYPES if retryable is None else retryable
        self.headers = headers or {}

    @property
    def status_code(self) -> int:
        return HTTP_STATUS_BY_TYPE[self.error_type]


class ErrorClass(enum.StrEnum):
    """Normalised classification of an upstream failure."""

    AUTH_ERROR = "auth_error"
    INVALID_REQUEST = "invalid_request"
    INCOMPATIBLE = "incompatible"
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
    CONTENT_FILTERED = "content_filtered"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXHAUSTED = "quota_exhausted"
    TIMEOUT = "timeout"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"
    SCHEMA_VALIDATION_FAILED = "schema_validation_failed"


@dataclass(frozen=True)
class RetryDecision:
    retry_same_model: bool
    fallback: bool
    counts_for_breaker: bool


DECISIONS: dict[ErrorClass, RetryDecision] = {
    ErrorClass.AUTH_ERROR: RetryDecision(False, True, True),
    ErrorClass.INVALID_REQUEST: RetryDecision(False, False, False),
    ErrorClass.INCOMPATIBLE: RetryDecision(False, True, False),
    ErrorClass.CONTEXT_LENGTH_EXCEEDED: RetryDecision(False, True, False),
    ErrorClass.CONTENT_FILTERED: RetryDecision(False, False, False),
    ErrorClass.RATE_LIMITED: RetryDecision(True, True, False),
    ErrorClass.QUOTA_EXHAUSTED: RetryDecision(False, True, False),
    ErrorClass.TIMEOUT: RetryDecision(True, True, True),
    ErrorClass.UPSTREAM_UNAVAILABLE: RetryDecision(True, True, True),
    ErrorClass.SCHEMA_VALIDATION_FAILED: RetryDecision(False, True, False),
}

OUTCOME_BY_CLASS: dict[ErrorClass, str] = {
    ErrorClass.AUTH_ERROR: "auth_error",
    ErrorClass.INVALID_REQUEST: "upstream_error",
    ErrorClass.INCOMPATIBLE: "incompatible",
    ErrorClass.CONTEXT_LENGTH_EXCEEDED: "context_length_exceeded",
    ErrorClass.CONTENT_FILTERED: "content_filtered",
    ErrorClass.RATE_LIMITED: "rate_limited",
    ErrorClass.QUOTA_EXHAUSTED: "quota_exhausted",
    ErrorClass.TIMEOUT: "upstream_timeout",
    ErrorClass.UPSTREAM_UNAVAILABLE: "upstream_unavailable",
    ErrorClass.SCHEMA_VALIDATION_FAILED: "schema_validation_failed",
}


@dataclass
class ProviderError(Exception):
    """Raised by adapters. `message` must never contain prompt content or credentials."""

    error_class: ErrorClass
    message: str
    http_status: int | None = None
    retry_after_s: float | None = None
    quota_reset_at: datetime | None = None
    possibly_billed: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.error_class}: {self.message}"
