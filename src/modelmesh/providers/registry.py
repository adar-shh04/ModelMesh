"""Builds the adapters that have credentials, and answers which models are usable."""

from __future__ import annotations

import httpx

from modelmesh.core.settings import Settings
from modelmesh.providers import gemini, groq, openrouter
from modelmesh.providers.catalog import Catalog, CatalogModel
from modelmesh.providers.fake import FakeAdapter
from modelmesh.providers.gemini import GeminiAdapter
from modelmesh.providers.groq import GroqAdapter
from modelmesh.providers.ollama import OllamaAdapter
from modelmesh.providers.openrouter import OpenRouterAdapter
from modelmesh.providers.types import ProviderAdapter


class ProviderRegistry:
    def __init__(self, catalog: Catalog, adapters: dict[str, ProviderAdapter]) -> None:
        self.catalog = catalog
        self.adapters = adapters
        self.disabled: set[str] = set()

    def adapter(self, provider: str) -> ProviderAdapter:
        return self.adapters[provider]

    def is_enabled(self, provider: str) -> bool:
        return provider in self.adapters and provider not in self.disabled

    def enabled_models(self) -> list[CatalogModel]:
        return [model for model in self.catalog.models.values() if self.is_enabled(model.provider)]

    async def aclose(self) -> None:
        for adapter in self.adapters.values():
            await adapter.aclose()


def build_adapters(settings: Settings) -> dict[str, ProviderAdapter]:
    """An adapter exists only when its credential or endpoint is configured (ADR-0003)."""
    adapters: dict[str, ProviderAdapter] = {}
    limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)

    def client() -> httpx.AsyncClient:
        return httpx.AsyncClient(limits=limits)

    if settings.groq_api_key is not None and settings.groq_api_key.get_secret_value():
        adapters["groq"] = GroqAdapter(client(), groq.BASE_URL, settings.groq_api_key)
    if settings.gemini_api_key is not None and settings.gemini_api_key.get_secret_value():
        adapters["gemini"] = GeminiAdapter(client(), gemini.BASE_URL, settings.gemini_api_key)
    if settings.openrouter_api_key is not None and settings.openrouter_api_key.get_secret_value():
        adapters["openrouter"] = OpenRouterAdapter(
            client(),
            openrouter.BASE_URL,
            settings.openrouter_api_key,
            app_url=settings.openrouter_app_url,
            app_title=settings.openrouter_app_title,
        )
    if settings.ollama_base_url:
        adapters["ollama"] = OllamaAdapter(client(), settings.ollama_base_url)
    if settings.enable_fake_provider:
        adapters["fake"] = FakeAdapter(
            delay_s=settings.fake_provider_delay_ms / 1000, failure_rate=settings.fake_provider_failure_rate
        )
    return adapters
