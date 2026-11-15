"""Operator API under /admin/v1. Not part of the client inference contract; see docs/contract/admin-api.yaml."""

from __future__ import annotations

import hmac
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field

from modelmesh.auth.admin import AdminService, KeyDefaults
from modelmesh.core.errors import ApiError
from modelmesh.pipeline.services import Services
from modelmesh.routing.lifecycle import RouterModelService

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


@admin_router.delete("/tenants/{tenant_id}")
async def delete_tenant(tenant_id: str, request: Request, admin: AdminDep) -> Any:
    await admin.delete_tenant(tenant_id)
    svc: Services = request.app.state.services
    purged = await svc.cache.invalidate(tenant_id, None, None)
    return {"id": tenant_id, "status": "deleted", "cache_entries_deleted": purged}


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


# --------------------------------------------------------------------------- learned router lifecycle (ADR-0004)
class RouterModelRegister(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Annotated[str, Field(min_length=1, max_length=128)]
    artifact_uri: Annotated[str, Field(min_length=1, description="Path to the artifact JSON on the server")]
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Offline report summary: preliminary, criterion_passed, ece_calibrated, report path",
    )
    mlflow_run_id: str | None = None


class RouterPromotion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved_by: Annotated[str, Field(min_length=1, max_length=200)]
    confirm: Literal[True]


def _router_models(request: Request, _: AdminDep) -> RouterModelService:
    svc: Services = request.app.state.services
    return RouterModelService(svc.sessions)


RouterModelsDep = Annotated[RouterModelService, Depends(_router_models)]


@admin_router.get("/router-models")
async def list_router_models(models: RouterModelsDep) -> Any:
    return {"data": await models.list_models()}


@admin_router.post("/router-models", status_code=201)
async def register_router_model(body: RouterModelRegister, models: RouterModelsDep) -> Any:
    return await models.register(body.version, body.artifact_uri, body.metrics, body.mlflow_run_id)


async def _transition(
    request: Request, models: RouterModelService, version: str, target: str, approved_by: str | None = None
) -> Any:
    svc: Services = request.app.state.services
    previous = {status: getattr(svc.router_runtime, status) for status in ("active", "shadow")}
    row, artifact = await models.transition(version, target, approved_by=approved_by)
    if target in ("active", "shadow"):
        svc.install_router(target, artifact)
        if target == "active" and previous["shadow"] and previous["shadow"].router_version == version:
            svc.install_router("shadow", None)
    else:
        for status, predictor in previous.items():
            if predictor is not None and predictor.router_version == version:
                svc.install_router(status, None)
    return row


@admin_router.post("/router-models/{version}/shadow")
async def shadow_router_model(version: str, request: Request, models: RouterModelsDep) -> Any:
    """Computes and logs this router's decisions next to live traffic. Traffic is not changed."""
    return await _transition(request, models, version, "shadow")


@admin_router.post("/router-models/{version}/promote")
async def promote_router_model(version: str, body: RouterPromotion, request: Request, models: RouterModelsDep) -> Any:
    """Human approval to route traffic with this version. Refused unless the offline gate is met."""
    return await _transition(request, models, version, "active", approved_by=body.approved_by)


@admin_router.post("/router-models/{version}/retire")
async def retire_router_model(version: str, request: Request, models: RouterModelsDep) -> Any:
    return await _transition(request, models, version, "retired")
