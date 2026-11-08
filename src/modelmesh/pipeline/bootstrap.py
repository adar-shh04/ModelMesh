"""Wires the production object graph."""

from __future__ import annotations

import yaml
from redis.asyncio import Redis

from modelmesh.accounting.records import RecordStore
from modelmesh.auth.keys import KeyAuthenticator
from modelmesh.auth.limits import BudgetTracker, RateLimiter
from modelmesh.cache.base import ResponseCache
from modelmesh.cache.embeddings import FastEmbedEmbedder
from modelmesh.cache.redis_cache import RedisResponseCache
from modelmesh.core.settings import Settings
from modelmesh.observability.stats import ProviderStats
from modelmesh.observability.telemetry import Telemetry, build_telemetry
from modelmesh.persistence.db import make_engine, make_session_factory
from modelmesh.pipeline.services import Services
from modelmesh.providers.catalog import load_catalog, load_pricing
from modelmesh.providers.registry import ProviderRegistry, build_adapters
from modelmesh.providers.types import ProviderAdapter
from modelmesh.reliability.breaker import BreakerRegistry
from modelmesh.reliability.executor import Executor
from modelmesh.reliability.quota import QuotaTracker
from modelmesh.routing.estimates import Estimator, NoCalibration, QualityPredictor, TokenEstimator


def build_cache(settings: Settings, redis: Redis) -> ResponseCache:
    embedder = (
        FastEmbedEmbedder(settings.embedding_model, cache_dir=settings.fastembed_cache_path)
        if settings.semantic_cache_enabled
        else None
    )
    raw = yaml.safe_load(settings.cache_thresholds_path.read_text(encoding="utf-8"))
    return RedisResponseCache(
        redis, embedder, thresholds=raw.get("task_types") or {}, default_threshold=float(raw["default"])
    )


def build_services(
    settings: Settings,
    *,
    adapters: dict[str, ProviderAdapter] | None = None,
    telemetry: Telemetry | None = None,
    quality: QualityPredictor | None = None,
    cache: ResponseCache | None = None,
) -> Services:
    settings.require_runtime_secrets()
    include_fake = settings.enable_fake_provider
    catalog = load_catalog(settings.model_catalog_path, include_fake=include_fake)
    pricing = load_pricing(settings.pricing_path, include_fake=include_fake)
    engine = make_engine(settings.database_url)
    sessions = make_session_factory(engine)
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    registry = ProviderRegistry(catalog, adapters if adapters is not None else build_adapters(settings))
    built_telemetry = telemetry or build_telemetry(
        service_name=settings.otel_service_name,
        otlp_endpoint=settings.otlp_endpoint,
        metrics_enabled=settings.otlp_metrics_enabled,
    )
    stats = ProviderStats()
    breakers = BreakerRegistry()
    quota = QuotaTracker(redis)
    estimator = Estimator(pricing, stats, TokenEstimator(), quality or NoCalibration())
    executor = Executor(registry, breakers, stats, quota, built_telemetry.tracer, built_telemetry.instruments)
    records = RecordStore(sessions)
    return Services(
        settings=settings,
        engine=engine,
        sessions=sessions,
        redis=redis,
        registry=registry,
        pricing=pricing,
        stats=stats,
        breakers=breakers,
        quota=quota,
        estimator=estimator,
        executor=executor,
        authenticator=KeyAuthenticator(sessions, redis, settings.api_key_pepper),
        limiter=RateLimiter(redis),
        budgets=BudgetTracker(redis, records.spend_since),
        records=records,
        cache=cache or build_cache(settings, redis),
        telemetry=built_telemetry,
    )
