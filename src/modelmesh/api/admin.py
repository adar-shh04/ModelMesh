"""Operator API under /admin/v1. Not part of the client inference contract; see docs/contract/admin-api.yaml."""

from __future__ import annotations

import hmac
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field

from modelmesh.auth.admin import AdminService, KeyDefaults
from modelmesh.core.errors import ApiError
from modelmesh.pipeline.services import Services

admin_router = APIRouter(prefix="/admin/v1")


def _services(request: Request) -> Services:
    result: Services = request.app.state.services
    return result


def require_admin(request: Request, svc: Annotated[Services, Depends(_services)]) -> AdminService:
    header = request.headers.get("authorization") or ""
    expected = f"Bearer {svc.settings.admin_token.get_secret_value()}"
    if not hmac.compare_digest(header.encode(), expected.encode()):
        raise ApiError("authentication_error", "Invalid admin token.", code="invalid_api_key")
    settings = svc.settings
    return AdminService(
        svc.sessions,
        svc.authenticator,
        settings.api_key_pepper,
        KeyDefaults(
            settings.default_key_rpm,
            settings.default_key_tpm,
            settings.default_key_concurrency,
            settings.default_key_daily_budget_usd,
        ),
    )


AdminDep = Annotated[AdminService, Depends(require_admin)]


class TenantCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Annotated[str, Field(min_length=1, max_length=200)]
    settings: dict[str, Any] = Field(default_factory=dict)


class TenantSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    settings: dict[str, Any]


class KeyLimitsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rpm: Annotated[int, Field(ge=1)] | None = None
    tpm: Annotated[int, Field(ge=1)] | None = None
    concurrency: Annotated[int, Field(ge=1)] | None = None
    daily_budget_usd: Annotated[float, Field(ge=0)] | None = None
    monthly_budget_usd: Annotated[float, Field(ge=0)] | None = None


class KeyScopes(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed_providers: list[str] | None = None
    allowed_models: list[str] | None = None
    cache_enabled: bool = True
    cache_max_ttl_s: Annotated[int, Field(ge=1)] | None = None


class KeyCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Annotated[str, Field(min_length=1, max_length=200)]
    env: Literal["live", "test"] = "live"
    limits: KeyLimitsInput = Field(default_factory=KeyLimitsInput)
    scopes: KeyScopes = Field(default_factory=KeyScopes)
    data_policy_ceiling: Literal["any", "no_training_providers", "local_only"] = "any"
    content_retention: bool = False


class ProviderToggle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool


@admin_router.post("/tenants", status_code=201)
async def create_tenant(body: TenantCreate, admin: AdminDep) -> Any:
    return await admin.create_tenant(body.name, body.settings)


@admin_router.patch("/tenants/{tenant_id}")
async def update_tenant(tenant_id: str, body: TenantSettings, admin: AdminDep) -> Any:
    return await admin.update_tenant_settings(tenant_id, body.settings)


@admin_router.post("/tenants/{tenant_id}/keys", status_code=201)
async def create_key(tenant_id: str, body: KeyCreate, admin: AdminDep) -> Any:
    return await admin.create_key(
        tenant_id,
        name=body.name,
        env=body.env,
        limits=body.limits.model_dump(exclude_none=True),
        scopes=body.scopes.model_dump(exclude_none=True),
        data_policy_ceiling=body.data_policy_ceiling,
        content_retention=body.content_retention,
    )


@admin_router.get("/tenants/{tenant_id}/keys")
async def list_keys(tenant_id: str, admin: AdminDep) -> Any:
    return {"data": await admin.list_keys(tenant_id)}


@admin_router.post("/keys/{key_id}/revoke")
async def revoke_key(key_id: str, admin: AdminDep) -> Any:
    return await admin.revoke_key(key_id)


@admin_router.post("/keys/{key_id}/rotate")
async def rotate_key(key_id: str, admin: AdminDep) -> Any:
    return await admin.rotate_key(key_id)


@admin_router.put("/providers/{provider}")
async def set_provider(provider: str, body: ProviderToggle, request: Request, admin: AdminDep) -> Any:
    await admin.set_provider_enabled(provider, enabled=body.enabled)
    svc: Services = request.app.state.services
    if body.enabled:
        svc.registry.disabled.discard(provider)
    else:
        svc.registry.disabled.add(provider)
    return {"provider": provider, "enabled": body.enabled}
