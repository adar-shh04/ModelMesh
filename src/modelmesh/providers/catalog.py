"""The versioned model catalog and pricing tables (config/*.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from modelmesh.providers.fake import FAKE_MODEL_NAMES

DataUse = Literal["no_training", "may_train", "unknown"]
SchemaMode = Literal["strict", "best_effort"]
SizeTier = Literal["small", "medium", "large"]
TIER_ORDER: tuple[SizeTier, ...] = ("small", "medium", "large")


@dataclass(frozen=True)
class CatalogModel:
    id: str
    provider: str
    provider_model: str
    family: str
    context_window: int
    max_output_tokens: int
    size_tier: SizeTier
    data_use: DataUse
    local: bool = False
    free_tier: bool = False
    streaming: bool = True
    tools: bool = False
    tool_choice: bool = False
    json_object: bool = False
    schema_mode: SchemaMode | None = None
    schema_stream: bool = False
    schema_with_tools: bool = False
    dropped_params: tuple[str, ...] = ()
    latency_ttft_ms: int = 800
    latency_tokens_per_s: float = 50.0

    def capability_tokens(self) -> list[str]:
        tokens = []
        if self.json_object:
            tokens.append("json_object")
        if self.schema_mode is not None:
            tokens.append("json_schema")
        if self.schema_mode == "strict":
            tokens.append("json_schema_strict")
        if self.tools:
            tokens.append("tools")
        if self.streaming:
            tokens.append("streaming")
        tokens.append(f"long_context:{self.context_window}")
        return tokens


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok_usd: float
    output_per_mtok_usd: float


@dataclass
class PricingTable:
    version: str
    prices: dict[str, ModelPrice] = field(default_factory=dict)

    def get(self, model_id: str) -> ModelPrice:
        return self.prices.get(model_id, ModelPrice(0.0, 0.0))

    def cost(self, model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
        price = self.get(model_id)
        return (
            prompt_tokens * price.input_per_mtok_usd + completion_tokens * price.output_per_mtok_usd
        ) / 1_000_000


@dataclass
class Catalog:
    version: str
    models: dict[str, CatalogModel]

    def get(self, model_id: str) -> CatalogModel | None:
        return self.models.get(model_id)


def _model_from_entry(entry: dict[str, Any]) -> CatalogModel:
    schema = entry.get("json_schema") or {}
    latency = entry.get("latency_prior") or {}
    return CatalogModel(
        id=entry["id"],
        provider=entry["provider"],
        provider_model=entry["id"].split("/", 1)[1],
        family=entry["family"],
        context_window=int(entry["context_window"]),
        max_output_tokens=int(entry["max_output_tokens"]),
        size_tier=entry["size_tier"],
        data_use=entry["data_use"],
        local=bool(entry.get("local", False)),
        free_tier=bool(entry.get("free_tier", False)),
        streaming=bool(entry.get("streaming", True)),
        tools=bool(entry.get("tools", False)),
        tool_choice=bool(entry.get("tool_choice", entry.get("tools", False))),
        json_object=bool(entry.get("json_object", False)),
        schema_mode=schema.get("mode"),
        schema_stream=bool(schema.get("stream", False)),
        schema_with_tools=bool(schema.get("with_tools", False)),
        dropped_params=tuple(entry.get("dropped_params", ())),
        latency_ttft_ms=int(latency.get("ttft_ms", 800)),
        latency_tokens_per_s=float(latency.get("tokens_per_s", 50.0)),
    )


def fake_models() -> dict[str, CatalogModel]:
    """Catalog entries for the isolated fake provider."""
    tiers: dict[str, SizeTier] = {"echo-small": "small", "echo-medium": "medium", "echo-large": "large"}
    return {
        f"fake/{name}": CatalogModel(
            id=f"fake/{name}",
            provider="fake",
            provider_model=name,
            family="fake",
            context_window=128_000,
            max_output_tokens=8_192,
            size_tier=tiers.get(name, "medium"),
            data_use="no_training",
            local=True,
            tools=True,
            json_object=True,
            schema_mode="strict",
            schema_stream=True,
            schema_with_tools=True,
            latency_ttft_ms=5,
            latency_tokens_per_s=10_000.0,
        )
        for name in FAKE_MODEL_NAMES
    }


def load_catalog(path: Path, *, include_fake: bool = False) -> Catalog:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    models = {entry["id"]: _model_from_entry(entry) for entry in raw["models"]}
    if include_fake:
        models.update(fake_models())
    return Catalog(version=str(raw["version"]), models=models)


def load_pricing(path: Path, *, include_fake: bool = False) -> PricingTable:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    prices = {
        model_id: ModelPrice(float(entry["input_per_mtok_usd"]), float(entry["output_per_mtok_usd"]))
        for model_id, entry in raw["prices"].items()
    }
    if include_fake:
        small = ModelPrice(0.05, 0.1)
        medium = ModelPrice(0.2, 0.6)
        large = ModelPrice(1.0, 3.0)
        prices.update(
            {
                "fake/echo-small": small,
                "fake/echo-medium": medium,
                "fake/echo-large": large,
                "fake/flaky": small,
                "fake/unavailable": small,
            }
        )
    return PricingTable(version=str(raw["version"]), prices=prices)
