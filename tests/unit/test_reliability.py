"""Breaker state machine, backoff bounds, the taxonomy table and validation helpers (ADR-0007, ADR-0008)."""

from __future__ import annotations

import random

import pytest

from modelmesh.core.errors import DECISIONS, ErrorClass
from modelmesh.reliability.breaker import CircuitBreaker
from modelmesh.reliability.executor import BACKOFF_CAP_S, Executor
from modelmesh.reliability.structured import is_portable_schema, schema_property_count, validate_output


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_breaker_opens_at_failure_rate_with_minimum_calls() -> None:
    clock = Clock()
    breaker = CircuitBreaker(clock=clock)
    for _ in range(4):
        breaker.record_failure()
    assert breaker.state == "closed"
    breaker.record_failure()
    assert breaker.state == "open"
    assert not breaker.allow()


def test_breaker_needs_half_failing_over_the_window() -> None:
    breaker = CircuitBreaker(clock=Clock())
    for _ in range(11):
        breaker.record_success()
    for _ in range(9):
        breaker.record_failure()
    assert breaker.state == "closed"


def test_breaker_opens_on_three_consecutive_timeouts() -> None:
    breaker = CircuitBreaker(clock=Clock())
    for _ in range(10):
        breaker.record_success()
    for _ in range(3):
        breaker.record_failure(timeout=True)
    assert breaker.state == "open"


def test_breaker_half_open_allows_one_probe_then_closes_on_success() -> None:
    clock = Clock()
    breaker = CircuitBreaker(clock=clock)
    breaker.trip()
    clock.now += 30
    assert breaker.state == "half_open"
    assert breaker.allow()
    assert not breaker.allow()
    breaker.record_success()
    assert breaker.state == "closed"
    assert breaker.allow()


def test_breaker_half_open_reopens_on_failure() -> None:
    clock = Clock()
    breaker = CircuitBreaker(clock=clock)
    breaker.trip()
    clock.now += 30
    assert breaker.allow()
    breaker.record_failure()
    assert breaker.state == "open"
    clock.now += 29
    assert not breaker.allow()
    clock.now += 1
    assert breaker.allow()


def test_breaker_probe_can_be_released_without_a_verdict() -> None:
    clock = Clock()
    breaker = CircuitBreaker(clock=clock)
    breaker.trip()
    clock.now += 30
    assert breaker.allow()
    breaker.release_probe()
    assert breaker.allow()


@pytest.mark.parametrize(
    ("error_class", "retry", "fallback", "breaker"),
    [
        (ErrorClass.AUTH_ERROR, False, True, True),
        (ErrorClass.INVALID_REQUEST, False, False, False),
        (ErrorClass.INCOMPATIBLE, False, True, False),
        (ErrorClass.CONTEXT_LENGTH_EXCEEDED, False, True, False),
        (ErrorClass.CONTENT_FILTERED, False, False, False),
        (ErrorClass.RATE_LIMITED, True, True, False),
        (ErrorClass.QUOTA_EXHAUSTED, False, True, False),
        (ErrorClass.TIMEOUT, True, True, True),
        (ErrorClass.UPSTREAM_UNAVAILABLE, True, True, True),
        (ErrorClass.SCHEMA_VALIDATION_FAILED, False, True, False),
    ],
)
def test_taxonomy_decisions(error_class, retry, fallback, breaker) -> None:
    decision = DECISIONS[error_class]
    assert (decision.retry_same_model, decision.fallback, decision.counts_for_breaker) == (retry, fallback, breaker)


def test_every_error_class_has_a_decision() -> None:
    assert set(DECISIONS) == set(ErrorClass)


def test_backoff_uses_full_jitter_within_the_cap() -> None:
    executor = Executor.__new__(Executor)
    executor._rng = random.Random(1)
    samples = [executor.backoff_seconds(attempt) for attempt in (0, 1, 2, 3, 10) for _ in range(200)]
    assert min(samples) >= 0
    assert max(samples) <= BACKOFF_CAP_S
    assert max(executor.backoff_seconds(0) for _ in range(500)) <= 0.25


PORTABLE = {
    "type": "object",
    "additionalProperties": False,
    "required": ["a", "b"],
    "properties": {
        "a": {"type": ["string", "null"]},
        "b": {"type": "array", "items": {"type": "integer"}, "maxItems": 3},
    },
}


def test_portable_schema_detection() -> None:
    assert is_portable_schema(PORTABLE)
    assert not is_portable_schema({**PORTABLE, "required": ["a"]})
    assert not is_portable_schema({k: v for k, v in PORTABLE.items() if k != "additionalProperties"})
    assert not is_portable_schema({"type": "array", "items": {"type": "string"}})
    assert not is_portable_schema(
        {**PORTABLE, "properties": {"a": {"type": "string", "pattern": "x"}, "b": {"type": "string"}}}
    )


def test_portable_schema_rejects_excess_depth() -> None:
    schema: dict = {"type": "string"}
    for _ in range(6):
        schema = {"type": "object", "additionalProperties": False, "required": ["x"], "properties": {"x": schema}}
    assert not is_portable_schema(schema)


def test_schema_property_count() -> None:
    assert schema_property_count(PORTABLE) == 2


def test_validate_output_reports_only_path_and_keyword() -> None:
    fmt = {"type": "json_schema", "json_schema": {"schema": PORTABLE}}
    issues = validate_output(fmt, '{"a": 5, "b": []}')
    assert [(i.path, i.keyword) for i in issues] == [("/a", "type")]
    assert "5" not in repr(issues)


def test_validate_output_for_each_format() -> None:
    assert validate_output({"type": "json_object"}, '{"x": 1}') == []
    assert validate_output({"type": "json_object"}, "[1]")[0].keyword == "type"
    assert validate_output({"type": "json_object"}, "not json")[0].keyword == "json"
    assert validate_output({"type": "text"}, "anything") == []
