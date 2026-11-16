"""Offline embeddings with the same model the gateway uses (fastembed bge-small, ADR-0006), cached on disk.

Request texts go through the gateway's own ``FastEmbedEmbedder.request_text``, so training embeds exactly the
text the router embeds online (feature spec ``modelmesh.ml.features``).
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from modelmesh.cache.embeddings import FastEmbedEmbedder
from modelmesh.ml.evaluation.metrics import Floats
from modelmesh.ml.features import embedding_segments

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# Offline batches run on one thread: they are not latency-bound and should not crowd the machine.
OFFLINE_THREADS = 1

Messages = Sequence[Mapping[str, Any]]


class CachedEmbedder:
    """Embeds texts once; the cache is keyed by the SHA-256 of model name and text and kept in a gitignored .npz."""

    def __init__(self, cache_path: Path, model_name: str = DEFAULT_MODEL) -> None:
        self.cache_path = cache_path
        self.model_name = model_name
        self._embedder = FastEmbedEmbedder(
            model_name,
            cache_dir=os.environ.get("FASTEMBED_CACHE_PATH"),
            threads=OFFLINE_THREADS,
            max_chars=None,
            cache_size=0,
        )
        self._cache: dict[str, Floats] = {}
        if cache_path.exists():
            with np.load(cache_path) as data:
                self._cache = {key: np.asarray(data[key]) for key in data.files}

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model_name}\n{text}".encode()).hexdigest()

    def request_texts(self, requests: Sequence[Messages]) -> list[str]:
        """The texts the router embeds for these requests, under the feature spec's token budget."""
        return [self._embedder.request_text(*embedding_segments(messages)) for messages in requests]

    def embed_requests(self, requests: Sequence[Messages]) -> Floats:
        return self.embed(self.request_texts(requests))

    def embed(self, texts: Sequence[str]) -> Floats:
        """Embeds texts as given (no token budget). Use ``embed_requests`` for router features."""
        keys = [self._key(t) for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts, strict=True) if k not in self._cache]
        if missing:
            vectors = self._embedder.embed_many_sync([t for _, t in missing])
            for (key, _), vector in zip(missing, vectors, strict=True):
                self._cache[key] = np.asarray(vector, dtype=np.float64)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.cache_path, **self._cache)  # type: ignore[arg-type]
        return np.vstack([self._cache[k] for k in keys])
