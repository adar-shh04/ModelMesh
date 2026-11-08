"""Shared fixtures. Integration tests use real Postgres and Redis from docker-compose (or CI services)."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import jsonschema
import psycopg
import pytest
import yaml
from alembic import command
from alembic.config import Config
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import SecretStr
from sqlalchemy import text

from doubles import ScriptedAdapter
from modelmesh.api.app import create_app
from modelmesh.auth.admin import AdminService, KeyDefaults
from modelmesh.core.settings import Settings
from modelmesh.observability.telemetry import build_telemetry
from modelmesh.pipeline.bootstrap import build_services
from modelmesh.pipeline.services import Services
from modelmesh.providers.catalog import ModelPrice
from modelmesh.providers.fake import FakeAdapter

REPO = Path(__file__).resolve().parents[1]
PG_ADMIN_DSN = os.environ.get("TEST_PG_ADMIN_DSN", "postgresql://modelmesh:modelmesh@127.0.0.1:55432/modelmesh")
TEST_DB = "modelmesh_test"
TEST_DB_URL = os.environ.get("TEST_DATABASE_URL", f"postgresql+psycopg://modelmesh:modelmesh@127.0.0.1:55432/{TEST_DB}")
TEST_REDIS_URL = os.environ.get("TEST_REDIS_URL", "redis://127.0.0.1:56379/0")

CONTRACT = yaml.safe_load((REPO / "docs/contract/openapi-v1.yaml").read_text(encoding="utf-8"))
TABLES = "tenants, api_keys, requests, attempts, usage_records, request_content, feedback, provider_settings"


def assert_matches(schema_name: str, instance: Any) -> None:
    """Validate a real payload against the published contract."""
    schema = {"$ref": f"#/components/schemas/{schema_name}", "components": CONTRACT["components"]}
    jsonschema.Draft202012Validator(schema).validate(instance)


@pytest.fixture
def anyio_backend() -> tuple[str, dict[str, Any]]:
    # psycopg's async mode cannot run on Windows' default ProactorEventLoop.
    return ("asyncio", {"loop_factory": asyncio.SelectorEventLoop})


def alembic_config(url: str) -> Config:
    os.environ["MODELMESH_DATABASE_URL"] = url
    config = Config(str(REPO / "alembic.ini"))
    config.set_main_option("script_location", str(REPO / "migrations"))
    return config


@pytest.fixture(scope="session")
def migrated_database() -> str:
    with psycopg.connect(PG_ADMIN_DSN, autocommit=True, connect_timeout=5) as connection:
        exists = connection.execute("SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,)).fetchone()
        if not exists:
            connection.execute(f'CREATE DATABASE "{TEST_DB}"')
    command.upgrade(alembic_config(TEST_DB_URL), "head")
    return TEST_DB_URL


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "MODELMESH_ENV": "test",
        "MODELMESH_API_KEY_PEPPER": "test-pepper",
        "MODELMESH_ADMIN_TOKEN": "test-admin-token",
        "MODELMESH_DATABASE_URL": TEST_DB_URL,
        "MODELMESH_REDIS_URL": TEST_REDIS_URL,
        "MODELMESH_ENABLE_FAKE_PROVIDER": True,
        "MODELMESH_MODEL_CATALOG_PATH": REPO / "config/models.yaml",
        "MODELMESH_PRICING_PATH": REPO / "config/pricing.yaml",
        "MODELMESH_CACHE_THRESHOLDS_PATH": REPO / "config/cache_thresholds.yaml",
    } | overrides
    return Settings(_env_file=None, **values)


@dataclass
class Api:
    client: httpx.AsyncClient
    services: Services
    admin: AdminService
    tenant_id: str
    key_id: str
    key: str
    scripted: ScriptedAdapter
    spans: InMemorySpanExporter
    app: Any

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.key}"}

    async def chat(self, *, headers: dict[str, str] | None = None, **body: Any) -> httpx.Response:
        payload = {"model": "fake/echo-small", "messages": [{"role": "user", "content": "hello"}]} | body
        return await self.client.post("/v1/chat/completions", json=payload, headers={**self.headers, **(headers or {})})

    async def new_key(self, tenant_id: str | None = None, **kwargs: Any) -> str:
        created = await self.admin.create_key(tenant_id or self.tenant_id, name="extra", **kwargs)
        return created["key"]

    def add_models(self, *models: Any, prices: dict[str, tuple[float, float]] | None = None) -> None:
        for model in models:
            self.services.registry.catalog.models[model.id] = model
            input_price, output_price = (prices or {}).get(model.id, (1.0, 2.0))
            self.services.pricing.prices[model.id] = ModelPrice(input_price, output_price)

    def only_models(self, *models: Any, prices: dict[str, tuple[float, float]] | None = None) -> None:
        """Replace the catalog so that routing considers exactly these models."""
        self.services.registry.catalog.models.clear()
        self.add_models(*models, prices=prices)


ROOMY_LIMITS = {"rpm": 100_000, "tpm": 100_000_000, "concurrency": 1000, "daily_budget_usd": None}


async def build_api(migrated: str, cache: Any = None, quality: Any = None, **settings: Any) -> Api:
    configured = make_settings(MODELMESH_DATABASE_URL=migrated, **settings)
    exporter = InMemorySpanExporter()
    telemetry = build_telemetry(
        service_name="test", otlp_endpoint=None, span_processors=[SimpleSpanProcessor(exporter)]
    )
    scripted = ScriptedAdapter()
    services = build_services(
        configured,
        adapters={"fake": FakeAdapter(), "t": scripted},
        telemetry=telemetry,
        cache=cache,
        quality=quality,
    )
    async with services.engine.begin() as connection:
        await connection.execute(text(f"TRUNCATE {TABLES} CASCADE"))
    await services.redis.flushdb()
    await services.cache.prepare()
    app = create_app(services)
    admin = AdminService(
        services.sessions, services.authenticator, SecretStr("test-pepper"), KeyDefaults(20, 50_000, 10, 0.25)
    )
    tenant = await admin.create_tenant("test")
    key = await admin.create_key(tenant["id"], name="main", env="test", limits=ROOMY_LIMITS)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://mesh.test", timeout=30)
    return Api(client, services, admin, tenant["id"], key["id"], key["key"], scripted, exporter, app)


@pytest.fixture
async def api(migrated_database: str) -> AsyncIterator[Api]:
    instance = await build_api(migrated_database)
    yield instance
    await instance.client.aclose()
    await instance.services.aclose()


def sse_events(body: str) -> list[Any]:
    """Decode an SSE body into JSON objects and the [DONE] sentinel."""
    events: list[Any] = []
    for block in body.split("\n\n"):
        if block.startswith("data: "):
            data = block[len("data: ") :]
            events.append("[DONE]" if data == "[DONE]" else json.loads(data))
    return events
