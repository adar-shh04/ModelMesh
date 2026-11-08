"""Exact and semantic response cache on Redis (ADR-0006).

Entries are hashes under `mm:c:{tenant}:`, so a tenant's entries can never be addressed from another tenant's
request. Semantic lookups are KNN queries pre-filtered by tenant and partition. Query Engine failures degrade to
exact-only caching rather than failing requests.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import struct
from collections import Counter
from collections.abc import Mapping
from typing import Any, Protocol

from redis.exceptions import RedisError, ResponseError

from modelmesh.cache.base import CachedEntry, CacheRequest

logger = logging.getLogger("modelmesh.cache")

INDEX_NAME = "mm_cache_idx"
KEY_PREFIX = "mm:c:"
EMBEDDING_DIMENSIONS = 384
KNN_CANDIDATES = 5
DEFAULT_THRESHOLD = 0.97

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_QUOTED = re.compile(r"\"([^\"]+)\"|'([^']+)'")
_WORD = re.compile(r"[a-z']+")
_NEGATIONS = {
    "not",
    "no",
    "never",
    "without",
    "none",
    "nor",
    "neither",
    "cannot",
    "except",
    "unless",
}


def lexical_guard(text: str) -> str:
    """A canonical form of the tokens that must match exactly for a semantic hit."""
    lowered = text.lower()
    numbers = _NUMBER.findall(lowered)
    quoted = [left or right for left, right in _QUOTED.findall(lowered)]
    negations = [word for word in _WORD.findall(lowered) if word in _NEGATIONS or word.endswith("n't")]
    payload = {
        "numbers": sorted(Counter(numbers).items()),
        "quoted": sorted(Counter(quoted).items()),
        "negations": sorted(Counter(negations).items()),
    }
    return json.dumps(payload, sort_keys=True)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _tag(value: str | None) -> str:
    """TAG values are hashed so they need no query escaping."""
    return _digest(value or "")[:32]


def _entry_key(request: CacheRequest) -> str:
    return f"{KEY_PREFIX}{request.tenant_id}:x:{_digest(request.partition + chr(0) + request.final_message)}"


def _vector_bytes(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


class Embedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...


class RedisResponseCache:
    def __init__(
        self,
        redis: Any,
        embedder: Embedder | None,
        *,
        thresholds: Mapping[str, float] | None = None,
        default_threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self._redis = redis
        self._embedder = embedder
        self._thresholds = dict(thresholds or {})
        self._default_threshold = default_threshold
        self._semantic_ready = False

    @property
    def semantic_ready(self) -> bool:
        return self._semantic_ready

    def threshold_for(self, task_type: str) -> float:
        return self._thresholds.get(task_type, self._default_threshold)

    async def prepare(self) -> None:
        await self.ensure_index()

    async def ensure_index(self) -> bool:
        """Create the vector index if needed. Returns whether semantic lookups can run."""
        if self._embedder is None:
            return False
        try:
            await self._redis.execute_command("FT.INFO", INDEX_NAME)
        except ResponseError:
            try:
                await self._redis.execute_command(
                    "FT.CREATE", INDEX_NAME, "ON", "HASH", "PREFIX", "1", KEY_PREFIX, "SCHEMA",
                    "tenant", "TAG",
                    "partition", "TAG",
                    "embedding", "VECTOR", "FLAT", "6", "TYPE", "FLOAT32", "DIM", str(EMBEDDING_DIMENSIONS),
                    "DISTANCE_METRIC", "COSINE",
                )  # fmt: skip
            except ResponseError as exc:
                logger.warning("semantic cache disabled: could not create index", extra={"detail": str(exc)})
                return False
        except RedisError:
            return False
        self._semantic_ready = True
        return True

    async def lookup(self, request: CacheRequest, *, eligible_model_ids: set[str]) -> CachedEntry | None:
        try:
            entry = await self._lookup_exact(request, eligible_model_ids)
            if entry is not None or request.mode != "semantic":
                return entry
            return await self._lookup_semantic(request, eligible_model_ids)
        except RedisError:
            logger.warning("cache lookup failed")
            return None

    async def _lookup_exact(self, request: CacheRequest, eligible: set[str]) -> CachedEntry | None:
        stored = await self._redis.hmget(_entry_key(request), "payload", "model")
        if stored[0] is None or stored[1] not in eligible:
            return None
        return CachedEntry(json.loads(stored[0]), "hit_exact", None)

    async def _lookup_semantic(self, request: CacheRequest, eligible: set[str]) -> CachedEntry | None:
        if self._embedder is None or not self._semantic_ready or request.has_tools:
            return None
        threshold = self.threshold_for(request.task_type)
        if threshold >= 1.0:
            return None
        vector = _vector_bytes(await self._embedder.embed(request.final_message))
        query = f"(@tenant:{{{_tag(request.tenant_id)}}} @partition:{{{_tag(request.partition)}}})"
        query += f"=>[KNN {KNN_CANDIDATES} @embedding $vector AS distance]"
        raw = await self._redis.execute_command(
            "FT.SEARCH", INDEX_NAME, query, "PARAMS", "2", "vector", vector, "SORTBY", "distance", "ASC",
            "RETURN", "4", "distance", "payload", "model", "guard", "DIALECT", "2",
        )  # fmt: skip
        guard = lexical_guard(request.final_message)
        for fields in _search_documents(raw):
            similarity = 1.0 - float(fields["distance"])
            if similarity < threshold:
                break
            if fields.get("guard") == guard and fields.get("model") in eligible:
                return CachedEntry(json.loads(fields["payload"]), "hit_semantic", similarity)
        return None

    async def store(self, request: CacheRequest, payload: dict[str, Any]) -> None:
        mapping: dict[str, Any] = {
            "payload": json.dumps(payload),
            "model": payload["model"],
            "tenant": _tag(request.tenant_id),
            "partition": _tag(request.partition),
            "namespace": _tag(request.namespace),
            "fingerprint": _tag(request.context_fingerprint),
            "guard": lexical_guard(request.final_message),
        }
        semantic = request.mode == "semantic" and not request.has_tools and self._semantic_ready
        if semantic and self._embedder is not None:
            mapping["embedding"] = _vector_bytes(await self._embedder.embed(request.final_message))
        key = _entry_key(request)
        pipe = self._redis.pipeline(transaction=True)
        pipe.hset(key, mapping=mapping)
        pipe.expire(key, request.ttl_s)
        await pipe.execute()

    async def invalidate(self, tenant_id: str, namespace: str | None, context_fingerprint: str | None) -> int:
        """Delete this tenant's entries matching every supplied selector."""
        wanted_namespace = _tag(namespace) if namespace is not None else None
        wanted_fingerprint = _tag(context_fingerprint) if context_fingerprint is not None else None
        deleted = 0
        pattern = f"{KEY_PREFIX}{tenant_id}:*"
        async for key in self._redis.scan_iter(match=pattern, count=500):
            stored_namespace, stored_fingerprint = await self._redis.hmget(key, "namespace", "fingerprint")
            if wanted_namespace is not None and stored_namespace != wanted_namespace:
                continue
            if wanted_fingerprint is not None and stored_fingerprint != wanted_fingerprint:
                continue
            deleted += int(await self._redis.delete(key))
        return deleted


def _search_documents(raw: Any) -> list[dict[str, str]]:
    """Decode an FT.SEARCH reply, in either the RESP2 list form or the RESP3 map form."""
    if isinstance(raw, dict):
        return [dict(result.get("extra_attributes", {})) for result in raw.get("results", [])]
    documents: list[dict[str, str]] = []
    for position in range(2, len(raw), 2):
        pairs = raw[position]
        documents.append({str(pairs[i]): pairs[i + 1] for i in range(0, len(pairs), 2)})
    return documents
