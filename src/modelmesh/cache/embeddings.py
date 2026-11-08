"""CPU text embeddings through fastembed (ONNX, no torch). The model is loaded on first use."""

from __future__ import annotations

import threading
from typing import Any

import anyio

MAX_CHARS = 2048


class FastEmbedEmbedder:
    def __init__(self, model_name: str, *, cache_dir: str | None = None) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(model_name=self._model_name, cache_dir=self._cache_dir, threads=1)
            return self._model

    def embed_sync(self, text: str) -> list[float]:
        vector = next(iter(self._load().embed([text[:MAX_CHARS]])))
        return [float(value) for value in vector]

    async def embed(self, text: str) -> list[float]:
        return await anyio.to_thread.run_sync(self.embed_sync, text)
