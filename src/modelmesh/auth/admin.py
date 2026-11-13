"""Operator actions on tenants, API keys and provider enablement (ADR-0010)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from modelmesh.auth.keys import PREFIX_LENGTH, KeyAuthenticator, generate_key, hash_key
from modelmesh.core.errors import ApiError
from modelmesh.core.ids import new_api_key_id, new_tenant_id
from modelmesh.persistence import models as db


@dataclass(frozen=True)
class KeyDefaults:
    rpm: int
    tpm: int
    concurrency: int
    daily_budget_usd: float | None


class AdminService:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        authenticator: KeyAuthenticator,
        pepper: SecretStr,
        defaults: KeyDefaults,
    ) -> None:
        self._sessions = sessions
        self._authenticator = authenticator
        self._pepper = pepper
        self._defaults = defaults

    async def create_tenant(self, name: str, settings: dict[str, Any] | None = None) -> dict[str, Any]:
        tenant = db.Tenant(id=new_tenant_id(), name=name, settings=settings or {})
        async with self._sessions.begin() as session:
            session.add(tenant)
        return {"id": tenant.id, "name": tenant.name, "status": tenant.status, "settings": tenant.settings}

    async def update_tenant_settings(self, tenant_id: str, settings: dict[str, Any]) -> dict[str, Any]:
        async with self._sessions.begin() as session:
            tenant = await session.get(db.Tenant, tenant_id)
            if tenant is None:
                raise ApiError("not_found", "Unknown tenant.", code="tenant_not_found")
            tenant.settings = {**tenant.settings, **settings}
            return {
                "id": tenant.id,
                "name": tenant.name,
                "status": tenant.status,
                "settings": tenant.settings,
            }

    async def delete_tenant(self, tenant_id: str) -> None:
        """Delete a tenant. Keys, records, usage, feedback and retained content go with it (ON DELETE CASCADE)."""
        async with self._sessions.begin() as session:
            tenant = await session.get(db.Tenant, tenant_id)
            if tenant is None:
                raise ApiError("not_found", "Unknown tenant.", code="tenant_not_found")
            digests = list(
                (await session.execute(select(db.ApiKey.hmac).where(db.ApiKey.tenant_id == tenant_id))).scalars()
            )
            await session.delete(tenant)
        for digest in digests:
            await self._authenticator.forget(digest)

    async def create_key(
        self,
        tenant_id: str,
        *,
        name: str,
        env: str = "live",
        limits: dict[str, Any] | None = None,
        scopes: dict[str, Any] | None = None,
        data_policy_ceiling: str = "any",
        content_retention: bool = False,
    ) -> dict[str, Any]:
        """Create a key. The plaintext is returned once and is never stored."""
        defaults = self._defaults
        merged_limits = {
            "rpm": defaults.rpm,
            "tpm": defaults.tpm,
            "concurrency": defaults.concurrency,
            "daily_budget_usd": defaults.daily_budget_usd,
            "monthly_budget_usd": None,
            **(limits or {}),
        }
        secret = generate_key(env)
        key = db.ApiKey(
            id=new_api_key_id(),
            tenant_id=tenant_id,
            hmac=hash_key(self._pepper, secret),
            prefix=secret[:PREFIX_LENGTH],
            name=name,
            scopes=scopes or {},
            limits=merged_limits,
            data_policy_ceiling=data_policy_ceiling,
            content_retention=content_retention,
        )
        async with self._sessions.begin() as session:
            if await session.get(db.Tenant, tenant_id) is None:
                raise ApiError("not_found", "Unknown tenant.", code="tenant_not_found")
            session.add(key)
        return {"id": key.id, "key": secret, "prefix": key.prefix, "limits": merged_limits}

    async def revoke_key(self, key_id: str) -> dict[str, Any]:
        async with self._sessions.begin() as session:
            key = await session.get(db.ApiKey, key_id)
            if key is None:
                raise ApiError("not_found", "Unknown key.", code="key_not_found")
            key.status = "revoked"
            key.revoked_at = datetime.now(UTC)
            digest = key.hmac
        await self._authenticator.forget(digest)
        return {"id": key_id, "status": "revoked"}

    async def rotate_key(self, key_id: str) -> dict[str, Any]:
        """Replace a key's secret, keeping its limits and scopes. The old secret stops working immediately."""
        secret: str
        async with self._sessions.begin() as session:
            key = await session.get(db.ApiKey, key_id)
            if key is None or key.status != "active":
                raise ApiError("not_found", "Unknown key.", code="key_not_found")
            old_digest = key.hmac
            prefix = key.prefix.split("_")
            secret = generate_key(prefix[1] if len(prefix) > 1 else "live")
            key.hmac = hash_key(self._pepper, secret)
            key.prefix = secret[:PREFIX_LENGTH]
        await self._authenticator.forget(old_digest)
        return {"id": key_id, "key": secret, "prefix": secret[:PREFIX_LENGTH]}

    async def list_keys(self, tenant_id: str) -> list[dict[str, Any]]:
        async with self._sessions() as session:
            rows = (await session.execute(select(db.ApiKey).where(db.ApiKey.tenant_id == tenant_id))).scalars().all()
        return [
            {
                "id": k.id,
                "name": k.name,
                "prefix": k.prefix,
                "status": k.status,
                "limits": k.limits,
                "scopes": k.scopes,
                "data_policy_ceiling": k.data_policy_ceiling,
                "content_retention": k.content_retention,
                "created_at": k.created_at,
                "revoked_at": k.revoked_at,
            }
            for k in sorted(rows, key=lambda key: key.created_at)
        ]

    async def list_tenants(self) -> list[dict[str, Any]]:
        async with self._sessions() as session:
            rows = (await session.execute(select(db.Tenant).order_by(db.Tenant.created_at, db.Tenant.id))).scalars()
            return [{"id": t.id, "name": t.name, "status": t.status, "settings": t.settings} for t in rows]

    async def set_provider_enabled(self, provider: str, *, enabled: bool) -> None:
        statement = (
            insert(db.ProviderSetting)
            .values(provider=provider, enabled=enabled)
            .on_conflict_do_update(
                index_elements=["provider"], set_={"enabled": enabled, "updated_at": datetime.now(UTC)}
            )
        )
        async with self._sessions.begin() as session:
            await session.execute(statement)

    async def disabled_providers(self) -> set[str]:
        async with self._sessions() as session:
            rows = await session.execute(
                select(db.ProviderSetting.provider).where(db.ProviderSetting.enabled.is_(False))
            )
        return {row[0] for row in rows}
