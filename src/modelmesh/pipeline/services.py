"""The set of collaborators a request needs. Built once at start-up (or by tests)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from modelmesh.accounting.records import RecordStore
from modelmesh.auth.keys import KeyAuthenticator
from modelmesh.auth.limits import BudgetTracker, RateLimiter
from modelmesh.cache.base import ResponseCache
from modelmesh.core.settings import Settings
from modelmesh.ml.artifacts import RouterArtifact
from modelmesh.observability.stats import ProviderStats
from modelmesh.observability.telemetry import Telemetry
from modelmesh.providers.catalog import PricingTable
from modelmesh.providers.registry import ProviderRegistry
from modelmesh.reliability.breaker import BreakerRegistry
from modelmesh.reliability.executor import Executor
from modelmesh.reliability.quota import QuotaTracker
from modelmesh.routing.estimates import Estimator, NoCalibration
from modelmesh.routing.learned import RouterRuntime


@dataclass
class Services:
    settings: Settings
    engine: AsyncEngine
    sessions: async_sessionmaker[AsyncSession]
    redis: Any
    registry: ProviderRegistry
    pricing: PricingTable
    stats: ProviderStats
    breakers: BreakerRegistry
    quota: QuotaTracker
    estimator: Estimator
    executor: Executor
    authenticator: KeyAuthenticator
    limiter: RateLimiter
    budgets: BudgetTracker
    records: RecordStore
    cache: ResponseCache
    telemetry: Telemetry
    router_runtime: RouterRuntime = field(default_factory=RouterRuntime)

    def install_router(self, status: str, artifact: RouterArtifact | None) -> None:
        """Serves an artifact as the active router or computes it in shadow. None removes it."""
        self.router_runtime.install(status, artifact)
        if status == "active":
            self.estimator.quality = self.router_runtime.active or NoCalibration()

    async def aclose(self) -> None:
        await self.registry.aclose()
        await self.redis.aclose()
        await self.engine.dispose()
        self.telemetry.shutdown()
