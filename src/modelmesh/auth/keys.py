"""API-key format, hashing and verification (ADR-0010)."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from modelmesh.core.errors import ApiError
from modelmesh.persistence.models import ApiKey

KEY_CACHE_TTL_S = 60
PREFIX_LENGTH = 12


@dataclass(frozen=True)
class KeyLimits:
    rpm: int
    tpm: int
    concurrency: int
    daily_budget_usd: float | None
    monthly_budget_usd: float | None = None


@dataclass(frozen=True)
class AuthContext:
    tenant_id: str
    key_id: str
    limits: KeyLimits
    data_policy_ceiling: str = "any"
    allowed_providers: list[str] | None = None
    allowed_models: list[str] | None = None
    content_retention: bool = False
    cache_enabled: bool = True
    cache_max_ttl_s: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def generate_key(env: str = "live") -> str:
    return f"mm_{env}_{secrets.token_urlsafe(32)}"


def hash_key(pepper: SecretStr, key: str) -> str:
    return hmac.new(pepper.get_secret_value().encode(), key.encode(), hashlib.sha256).hexdigest()


def _context_from_row(row: ApiKey) -> AuthContext:
    limits = row.limits
    scopes = row.scopes
    return AuthContext(
        tenant_id=row.tenant_id,
        key_id=row.id,
        limits=KeyLimits(
            rpm=int(limits["rpm"]),
            tpm=int(limits["tpm"]),
            concurrency=int(limits["concurrency"]),
            daily_budget_usd=limits.get("daily_budget_usd"),
            monthly_budget_usd=limits.get("monthly_budget_usd"),
        ),
        data_policy_ceiling=row.data_policy_ceiling,
        allowed_providers=scopes.get("allowed_providers"),
        allowed_models=scopes.get("allowed_models"),
        content_retention=row.content_retention,
        cache_enabled=bool(scopes.get("cache_enabled", True)),
        cache_max_ttl_s=scopes.get("cache_max_ttl_s"),
    )


def _cache_key(digest: str) -> str:
    return f"mm:kv:{digest}"


class KeyAuthenticator:
    def __init__(self, sessions: async_sessionmaker[AsyncSession], redis: Any, pepper: SecretStr) -> None:
        self._sessions = sessions
        self._redis = redis
        self._pepper = pepper

    async def authenticate(self, authorization: str | None) -> AuthContext:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise ApiError("authentication_error", "Missing bearer token.", code="invalid_api_key")
        presented = authorization[7:].strip()
        if not presented.startswith("mm_"):
            raise ApiError("authentication_error", "Invalid API key.", code="invalid_api_key")
        digest = hash_key(self._pepper, presented)
        cached = await self._redis.get(_cache_key(digest))
        if cached is not None:
            data = json.loads(cached)
            data["limits"] = KeyLimits(**data["limits"])
            return AuthContext(**data)
        async with self._sessions() as session:
            row = (await session.execute(select(ApiKey).where(ApiKey.hmac == digest))).scalar_one_or_none()
        if row is None:
            raise ApiError("authentication_error", "Invalid API key.", code="invalid_api_key")
        if row.status != "active":
            raise ApiError("authentication_error", "This API key has been revoked.", code="revoked_api_key")
        context = _context_from_row(row)
        await self._redis.set(_cache_key(digest), json.dumps(asdict(context)), ex=KEY_CACHE_TTL_S)
        return context

    async def forget(self, digest: str) -> None:
        await self._redis.delete(_cache_key(digest))
