"""Per-key rate limits (GCRA in Redis) and estimated-cost budgets (ADR-0010).

GCRA with a burst equal to the limit: a key can spend its whole per-minute allowance at once, and the allowance
then refills smoothly. Redis is required for this path, so a Redis outage fails closed.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from redis.exceptions import RedisError

from modelmesh.auth.keys import AuthContext
from modelmesh.core.errors import ApiError

_GCRA = """
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + tonumber(t[2]) / 1000
local interval = tonumber(ARGV[1])
local cost = tonumber(ARGV[2])
local burst = tonumber(ARGV[3])
local tat = tonumber(redis.call('GET', KEYS[1]) or now)
if tat < now then tat = now end
local new_tat = tat + cost * interval
local tolerance = burst * interval
if new_tat - now > tolerance then
  return {0, tostring(new_tat - tolerance - now), tostring(tat - now)}
end
redis.call('SET', KEYS[1], tostring(new_tat), 'PX', math.ceil(new_tat - now) + 1000)
return {1, '0', tostring(new_tat - now)}
"""

_ADJUST = """
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + tonumber(t[2]) / 1000
local tat = tonumber(redis.call('GET', KEYS[1]) or now)
local new_tat = math.max(now, tat + tonumber(ARGV[1]))
redis.call('SET', KEYS[1], tostring(new_tat), 'PX', math.ceil(new_tat - now) + 1000)
return 1
"""

CONCURRENCY_TTL_S = 900


def dependency_unavailable() -> ApiError:
    return ApiError("service_unavailable", "A required dependency is unavailable.", code="dependency_unavailable")


@dataclass
class RateLimitInfo:
    limit: int
    remaining: int
    reset_s: float
    reserved_tokens: int = 0

    def headers(self) -> dict[str, str]:
        return {
            "X-RateLimit-Limit-Requests": str(self.limit),
            "X-RateLimit-Remaining-Requests": str(self.remaining),
            "X-RateLimit-Reset-Requests": f"{self.reset_s:.3f}",
        }


class RateLimiter:
    def __init__(self, redis: Any) -> None:
        self._redis = redis
        self._gcra = redis.register_script(_GCRA)
        self._adjust = redis.register_script(_ADJUST)

    @staticmethod
    def _rejected(code: str, wait_ms: float, limit: int, period: str = "minute") -> ApiError:
        retry_after = max(1, math.ceil(wait_ms / 1000))
        resets_at = (datetime.now(UTC) + timedelta(seconds=retry_after)).isoformat()
        return ApiError(
            "rate_limited",
            f"This key exceeded its {code} limit.",
            code=code,
            retry_after_s=retry_after,
            details={"scope": "key", "period": period, "resets_at": resets_at},
            headers={
                "X-RateLimit-Limit-Requests": str(limit),
                "X-RateLimit-Remaining-Requests": "0",
                "X-RateLimit-Reset-Requests": str(retry_after),
            },
        )

    async def acquire(self, auth: AuthContext, estimated_tokens: int) -> RateLimitInfo:
        limits = auth.limits
        rpm_interval = 60_000.0 / limits.rpm
        try:
            allowed, wait_ms, used_ms = await self._gcra(
                keys=[f"mm:rl:{auth.key_id}:rpm"], args=[rpm_interval, 1, limits.rpm]
            )
            if not int(allowed):
                raise self._rejected("rpm", float(wait_ms), limits.rpm)
            reserved = min(estimated_tokens, limits.tpm)
            tpm_interval = 60_000.0 / limits.tpm
            token_allowed, token_wait, _ = await self._gcra(
                keys=[f"mm:rl:{auth.key_id}:tpm"], args=[tpm_interval, reserved, limits.tpm]
            )
            if not int(token_allowed):
                raise self._rejected("tpm", float(token_wait), limits.rpm)
            concurrency_key = f"mm:rl:{auth.key_id}:conc"
            current = await self._redis.incr(concurrency_key)
            await self._redis.expire(concurrency_key, CONCURRENCY_TTL_S)
            if int(current) > limits.concurrency:
                await self._redis.decr(concurrency_key)
                raise self._rejected("concurrency", 1000.0, limits.rpm)
        except RedisError:
            raise dependency_unavailable() from None
        remaining = max(0, int((limits.rpm * rpm_interval - float(used_ms)) // rpm_interval))
        return RateLimitInfo(limits.rpm, remaining, float(used_ms) / 1000, reserved_tokens=reserved)

    async def release(self, auth: AuthContext) -> None:
        try:
            value = await self._redis.decr(f"mm:rl:{auth.key_id}:conc")
            if int(value) < 0:
                await self._redis.set(f"mm:rl:{auth.key_id}:conc", 0, ex=CONCURRENCY_TTL_S)
        except RedisError:
            return

    async def reconcile_tokens(self, auth: AuthContext, reserved: int, actual: int) -> None:
        """Replace the pre-call token reservation with what the call really used."""
        difference = actual - reserved
        if difference == 0:
            return
        delta_ms = difference * 60_000.0 / auth.limits.tpm
        try:
            await self._adjust(keys=[f"mm:rl:{auth.key_id}:tpm"], args=[delta_ms])
        except RedisError:
            return


SpendLoader = Callable[[str, datetime], Awaitable[float]]


def _day_start(now: datetime) -> datetime:
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _month_start(now: datetime) -> datetime:
    return _day_start(now).replace(day=1)


def _next_month(now: datetime) -> datetime:
    return (_month_start(now) + timedelta(days=32)).replace(day=1)


class BudgetTracker:
    """Fast spend counters in Redis. Postgres usage is the source of truth, and rebuilds lost counters."""

    def __init__(self, redis: Any, loader: SpendLoader) -> None:
        self._redis = redis
        self._loader = loader

    @staticmethod
    def _keys(auth: AuthContext, now: datetime) -> tuple[str, str]:
        return (f"mm:bud:{auth.key_id}:d:{now:%Y%m%d}", f"mm:bud:{auth.key_id}:m:{now:%Y%m}")

    async def _spent(self, key: str, key_id: str, since: datetime, ttl_s: int) -> float:
        value = await self._redis.get(key)
        if value is not None:
            return float(value)
        total = await self._loader(key_id, since)
        await self._redis.set(key, str(total), ex=ttl_s, nx=True)
        return total

    async def remaining(self, auth: AuthContext) -> float | None:
        """Remaining estimated-USD allowance across the key's budgets, or None when unlimited."""
        limits = auth.limits
        if limits.daily_budget_usd is None and limits.monthly_budget_usd is None:
            return None
        now = datetime.now(UTC)
        daily_key, monthly_key = self._keys(auth, now)
        allowances: list[float] = []
        try:
            if limits.daily_budget_usd is not None:
                spent = await self._spent(daily_key, auth.key_id, _day_start(now), 2 * 86_400)
                allowances.append(limits.daily_budget_usd - spent)
                if spent >= limits.daily_budget_usd:
                    raise ApiError(
                        "budget_exceeded",
                        "This key's daily budget is exhausted.",
                        code="daily_budget",
                        details={
                            "scope": "key",
                            "period": "day",
                            "resets_at": (_day_start(now) + timedelta(days=1)).isoformat(),
                        },
                    )
            if limits.monthly_budget_usd is not None:
                spent = await self._spent(monthly_key, auth.key_id, _month_start(now), 35 * 86_400)
                allowances.append(limits.monthly_budget_usd - spent)
                if spent >= limits.monthly_budget_usd:
                    raise ApiError(
                        "budget_exceeded",
                        "This key's monthly budget is exhausted.",
                        code="monthly_budget",
                        details={
                            "scope": "key",
                            "period": "month",
                            "resets_at": _next_month(now).isoformat(),
                        },
                    )
        except RedisError:
            raise dependency_unavailable() from None
        return min(allowances)

    async def add(self, auth: AuthContext, amount_usd: float) -> None:
        if amount_usd <= 0 or (auth.limits.daily_budget_usd is None and auth.limits.monthly_budget_usd is None):
            return
        now = datetime.now(UTC)
        daily_key, monthly_key = self._keys(auth, now)
        try:
            # Only touch counters that already exist; a missing counter is rebuilt from Postgres, which by then
            # holds this request's usage record.
            for key in (daily_key, monthly_key):
                if await self._redis.exists(key):
                    await self._redis.incrbyfloat(key, amount_usd)
        except RedisError:
            return
