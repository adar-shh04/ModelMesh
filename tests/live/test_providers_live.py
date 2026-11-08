"""Real-provider smoke tests. Skipped when credentials or the runtime are missing; a skip is NOT a pass."""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from modelmesh.core.deadline import Deadline
from modelmesh.providers import gemini as gemini_module
from modelmesh.providers import groq as groq_module
from modelmesh.providers import openrouter as openrouter_module
from modelmesh.providers.catalog import load_catalog
from modelmesh.providers.gemini import GeminiAdapter
from modelmesh.providers.groq import GroqAdapter
from modelmesh.providers.ollama import OllamaAdapter
from modelmesh.providers.openrouter import OpenRouterAdapter
from modelmesh.providers.types import NormalizedChatRequest

pytestmark = [pytest.mark.anyio, pytest.mark.live]

CATALOG = load_catalog(Path(__file__).resolve().parents[2] / "config/models.yaml")
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer"],
    "properties": {"answer": {"type": "string"}},
}


def _adapter(provider: str, client: httpx.AsyncClient):
    if provider == "groq":
        return GroqAdapter(client, groq_module.BASE_URL, SecretStr(os.environ["GROQ_API_KEY"]))
    if provider == "gemini":
        return GeminiAdapter(client, gemini_module.BASE_URL, SecretStr(os.environ["GEMINI_API_KEY"]))
    if provider == "openrouter":
        return OpenRouterAdapter(client, openrouter_module.BASE_URL, SecretStr(os.environ["OPENROUTER_API_KEY"]))
    return OllamaAdapter(client, os.environ["OLLAMA_BASE_URL"])


REQUIRED_ENV = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "ollama": "OLLAMA_BASE_URL",
}
MODELS = [pytest.param(model, id=model.id) for model in CATALOG.models.values()]


def requirement(model) -> None:
    variable = REQUIRED_ENV[model.provider]
    if not os.environ.get(variable):
        pytest.skip(f"{variable} is not set")


def request_for(model, **kwargs) -> NormalizedChatRequest:
    return NormalizedChatRequest(
        provider_model=model.provider_model,
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        max_tokens=256,
        temperature=0,
        **kwargs,
    )


@pytest.mark.parametrize("model", MODELS)
async def test_non_streaming_call(model) -> None:
    requirement(model)
    async with httpx.AsyncClient() as client:
        result = await _adapter(model.provider, client).chat(request_for(model), deadline=Deadline.in_ms(60_000))
    assert result.content or result.tool_calls


@pytest.mark.parametrize("model", MODELS)
async def test_streaming_call(model) -> None:
    requirement(model)
    async with httpx.AsyncClient() as client:
        chunks = [
            c
            async for c in _adapter(model.provider, client).stream(request_for(model), deadline=Deadline.in_ms(60_000))
        ]
    assert any(c.content for c in chunks)
    assert any(c.finish_reason for c in chunks)


@pytest.mark.parametrize("model", [p for p in MODELS if p.values[0].schema_mode is not None])
async def test_json_schema_where_the_catalog_claims_support(model) -> None:
    requirement(model)
    fmt = {
        "type": "json_schema",
        "json_schema": {"name": "r", "schema": SCHEMA, "strict": model.schema_mode == "strict"},
    }
    async with httpx.AsyncClient() as client:
        result = await _adapter(model.provider, client).chat(
            request_for(model, response_format=fmt), deadline=Deadline.in_ms(60_000)
        )
    assert "answer" in json.loads(result.content or "{}")
