"""The cache boundary. The request pipeline depends only on this interface, not on a particular store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class CacheRequest:
    """Everything that decides whether a cached answer may be reused (ADR-0006)."""

    tenant_id: str
    partition: str
    final_message: str
    mode: str  # "exact" or "semantic"
    task_type: str
    namespace: str | None
    context_fingerprint: str | None
    has_tools: bool
    ttl_s: int


@dataclass(frozen=True)
class CachedEntry:
    payload: dict[str, Any]
    status: str  # "hit_exact" or "hit_semantic"
    similarity: float | None


class ResponseCache(Protocol):
    async def prepare(self) -> None: ...

    async def lookup(self, request: CacheRequest, *, eligible_model_ids: set[str]) -> CachedEntry | None: ...

    async def store(self, request: CacheRequest, payload: dict[str, Any]) -> None: ...

    async def invalidate(self, tenant_id: str, namespace: str | None, context_fingerprint: str | None) -> int: ...


class NullCache:
    """Used when caching is disabled for a deployment."""

    async def prepare(self) -> None:
        return None

    async def lookup(self, request: CacheRequest, *, eligible_model_ids: set[str]) -> CachedEntry | None:
        return None

    async def store(self, request: CacheRequest, payload: dict[str, Any]) -> None:
        return None

    async def invalidate(self, tenant_id: str, namespace: str | None, context_fingerprint: str | None) -> int:
        return 0
