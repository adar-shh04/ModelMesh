"""The published admin API document and the catalog files must match the code."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx
import yaml
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from modelmesh.api.admin import admin_router
from modelmesh.cli import _catalog_verify
from modelmesh.core.settings import Settings
from modelmesh.providers.catalog import load_catalog, load_pricing

REPO = Path(__file__).resolve().parents[2]


def test_admin_api_document_is_current() -> None:
    app = FastAPI()
    app.include_router(admin_router)
    spec = get_openapi(title="ModelMesh Admin API", version="1.0.0", routes=app.routes)
    published = yaml.safe_load((REPO / "docs/contract/admin-api.yaml").read_text(encoding="utf-8"))
    assert set(spec["paths"]) == set(published["paths"])


def test_every_catalog_model_has_a_price_and_a_valid_id() -> None:
    catalog = load_catalog(REPO / "config/models.yaml")
    pricing = load_pricing(REPO / "config/pricing.yaml")
    assert set(catalog.models) == set(pricing.prices)
    for model in catalog.models.values():
        assert model.id == f"{model.provider}/{model.provider_model}"


def test_data_use_stays_conservative_until_reviewed() -> None:
    """DECISIONS D4: only local models count as no-training until someone reviews the provider terms."""
    catalog = load_catalog(REPO / "config/models.yaml")
    for model in catalog.models.values():
        assert (model.data_use == "no_training") == (model.provider == "ollama")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
@respx.mock
async def test_catalog_verify_reports_drift(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(REPO)
    # Never read the developer's real .env here: only the Groq adapter is configured, with a dummy key.
    settings = Settings(_env_file=None, GROQ_API_KEY="not-a-real-key", MODELMESH_ENABLE_FAKE_PROVIDER=False)
    listed = {"data": [{"id": "openai/gpt-oss-20b", "context_window": 131072, "active": True}]}
    respx.get("https://api.groq.com/openai/v1/models").mock(return_value=httpx.Response(200, json=listed))
    code = await _catalog_verify(settings)
    captured = capsys.readouterr().out
    assert code == 1
    assert "groq/openai/gpt-oss-120b: not offered" in captured
    assert "1 configured providers" in captured
