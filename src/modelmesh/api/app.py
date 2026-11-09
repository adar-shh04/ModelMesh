"""ASGI application factory."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from modelmesh.accounting.retention import retention_loop
from modelmesh.api.admin import admin_router
from modelmesh.api.errors import install_error_handlers
from modelmesh.api.middleware import RequestIdentityMiddleware
from modelmesh.api.routes import router
from modelmesh.auth.admin import AdminService, KeyDefaults
from modelmesh.core.settings import Settings, get_settings
from modelmesh.observability.logging import configure_logging
from modelmesh.pipeline.bootstrap import build_services
from modelmesh.pipeline.services import Services

CONTRACT_PATH = Path("docs/contract/openapi-v1.yaml")
logger = logging.getLogger("modelmesh.api")


def create_app(services: Services | None = None, settings: Settings | None = None) -> FastAPI:
    active_settings = settings or (services.settings if services else get_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        svc: Services = app.state.services
        admin = AdminService(
            svc.sessions,
            svc.authenticator,
            svc.settings.api_key_pepper,
            KeyDefaults(1, 1, 1, None),
        )
        try:
            svc.registry.disabled = await admin.disabled_providers()
        except Exception:  # The database may still be starting; /v1/health reports it.
            svc.registry.disabled = set()
        try:
            await svc.cache.prepare()
        except Exception:  # Redis may still be starting; exact and semantic lookups degrade safely.
            logger.warning("cache preparation failed")
        sweeper = asyncio.create_task(retention_loop(svc.sessions))
        yield
        sweeper.cancel()
        await svc.aclose()

    # The published contract is authoritative, so FastAPI's generated documents are not served.
    app = FastAPI(title="ModelMesh", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.services = services or build_services(active_settings)
    configure_logging(active_settings.log_level)
    install_error_handlers(app)
    app.include_router(router)
    app.include_router(admin_router)
    FastAPIInstrumentor.instrument_app(app, tracer_provider=app.state.services.telemetry.tracer_provider)
    app.add_middleware(RequestIdentityMiddleware, max_body_bytes=active_settings.max_body_bytes)

    contract_document_body = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8")) if CONTRACT_PATH.exists() else {}

    @app.get("/openapi.yaml", include_in_schema=False)
    async def contract_document() -> Any:
        return contract_document_body

    return app


def app_factory() -> FastAPI:
    return create_app()
