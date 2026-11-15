"""Offline embeddings with the same model the gateway uses (fastembed bge-small, ADR-0006), cached on disk."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from modelmesh.ml.evaluation.metrics import Floats

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def _load(model_name: str) -> Any:
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model_name, cache_dir=os.environ.get("FASTEMBED_CACHE_PATH"), threads=1)


class CachedEmbedder:
    """Embeds texts once; the cache is keyed by the SHA-256 of model name and text and kept in a gitignored .npz."""

    def __init__(self, cache_path: Path, model_name: str = DEFAULT_MODEL) -> None:
        self.cache_path = cache_path
        self.model_name = model_name
        self._model: Any = None
        self._cache: dict[str, Floats] = {}
        if cache_path.exists():
            with np.load(cache_path) as data:
                self._cache = {key: np.asarray(data[key]) for key in data.files}

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model_name}\n{text}".encode()).hexdigest()

    def embed(self, texts: Sequence[str]) -> Floats:
        keys = [self._key(t) for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts, strict=True) if k not in self._cache]
        if missing:
            if self._model is None:
                self._model = _load(self.model_name)
            vectors = list(self._model.embed([t for _, t in missing]))
            for (key, _), vector in zip(missing, vectors, strict=True):
                self._cache[key] = np.asarray(vector, dtype=np.float64)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.cache_path, **self._cache)  # type: ignore[arg-type]
        return np.vstack([self._cache[k] for k in keys])

    def embed_one_uncached(self, text: str) -> Floats:
        """For latency measurement: always runs the model."""
        if self._model is None:
            self._model = _load(self.model_name)
        return np.asarray(next(iter(self._model.embed([text]))), dtype=np.float64)
