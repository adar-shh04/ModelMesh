"""Upstream quota tracking, shared across workers through Redis (ADR-0008)."""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from modelmesh.providers.catalog import CatalogModel
from modelmesh.providers.types import RateLimitObservation


def _blocked_key(model_id: str) -> str:
    return f"mm:quota:{model_id}:blocked"


def _counter_key(model_id: str, unit: str, window: str, stamp: int) -> str:
    return f"mm:quota:{model_id}:{unit}:{window}:{stamp}"


class QuotaTracker:
    def __init__(self, redis: Any) -> None:
        self._redis = redis

    async def block_until(self, model_id: str, when: datetime) -> None:
        seconds = max(1, int((when - datetime.now(UTC)).total_seconds()))
        await self._redis.set(_blocked_key(model_id), "1", ex=seconds)

    async def block_for(self, model_id: str, seconds: float) -> None:
        await self._redis.set(_blocked_key(model_id), "1", ex=max(1, int(seconds)))

    async def observe(self, model_id: str, observation: RateLimitObservation | None) -> None:
        if observation is None:
            return
        exhausted = [
            reset
            for remaining, reset in (
                (observation.remaining_requests, observation.reset_requests_s),
                (observation.remaining_tokens, observation.reset_tokens_s),
            )
            if remaining is not None and remaining <= 0
        ]
        if exhausted:
            await self.block_for(model_id, max((reset or 60.0) for reset in exhausted))

    async def unavailable(self, models: Sequence[CatalogModel], *, estimated_tokens: int = 0) -> set[str]:
        """Model ids with no remaining quota, so that the free tier is never hammered."""
        if not models:
            return set()
        now = int(time.time())
        pipe = self._redis.pipeline(transaction=False)
        # (model id, highest counter value that still admits this call); None marks the "blocked" flag.
        checks: list[tuple[str, int | None]] = []
        for model in models:
            pipe.exists(_blocked_key(model.id))
            checks.append((model.id, None))
            for unit, window, seconds, limit, cost in self._windows(model, estimated_tokens):
                pipe.get(_counter_key(model.id, unit, window, now // seconds))
                checks.append((model.id, limit - cost))
        results = await pipe.execute()
        out: set[str] = set()
        for (model_id, allowed), value in zip(checks, results, strict=True):
            if allowed is None:
                if value:
                    out.add(model_id)
            elif int(value or 0) > allowed:
                out.add(model_id)
        return out

    @staticmethod
    def _windows(model: CatalogModel, tokens: int) -> list[tuple[str, str, int, int, int]]:
        windows = [
            ("req", "min", 60, model.limit_rpm, 1),
            ("req", "day", 86_400, model.limit_rpd, 1),
            ("tok", "min", 60, model.limit_tpm, tokens),
            ("tok", "day", 86_400, model.limit_tpd, tokens),
        ]
        return [(unit, window, seconds, limit, cost) for unit, window, seconds, limit, cost in windows if limit]

    async def record_call(self, model: CatalogModel, tokens: int) -> None:
        windows = self._windows(model, tokens)
        if not windows:
            return
        now = int(time.time())
        pipe = self._redis.pipeline(transaction=False)
        for unit, window, seconds, _limit, cost in windows:
            key = _counter_key(model.id, unit, window, now // seconds)
            pipe.incrby(key, max(cost, 0))
            pipe.expire(key, seconds * 2)
        await pipe.execute()
