"""CPU text embeddings through fastembed (ONNX, no torch). The model is loaded on first use.

One class serves both users of embeddings, with different inputs:

- the semantic cache embeds the final message, cut to ``MAX_CHARS`` characters (``embed``);
- the learned router embeds the system prompt and final user message under the token budget that
  ``modelmesh.ml.features`` defines (``embed_request``). Offline training goes through the same
  ``request_text``, so the router sees the same text in training and in serving.

Results are kept in a small LRU keyed by the SHA-256 of the embedded text, so a text embedded twice in a
short time (a semantic-cache lookup followed by its store, a shadow decision after the live one, a retried
request) runs the model once.
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict
from typing import Any

import anyio

from modelmesh.ml.features import embedding_token_budget, join_embedding_text

MAX_CHARS = 2048
DEFAULT_CACHE_SIZE = 1024


def default_threads() -> int:
    return max(1, min(4, os.cpu_count() or 1))


def _head(text: str, offsets: list[tuple[int, int]], tokens: int) -> str:
    """The first ``tokens`` tokens of ``text``, cut at the end of the last kept token."""
    if tokens <= 0:
        return ""
    if tokens >= len(offsets):
        return text
    return text[: offsets[tokens - 1][1]]


class FastEmbedEmbedder:
    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: str | None = None,
        threads: int | None = None,
        max_chars: int | None = MAX_CHARS,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self.threads = threads or default_threads()
        self._max_chars = max_chars
        self._cache_size = cache_size
        self._model: Any = None
        self._tokenizer: Any = None
        self._lock = threading.Lock()
        self._cache: OrderedDict[bytes, tuple[float, ...]] = OrderedDict()
        self._cache_lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(
                    model_name=self._model_name, cache_dir=self._cache_dir, threads=self.threads
                )
            return self._model

    def tokenizer(self) -> Any:
        """The model's own tokenizer, without truncation or padding, for counting and cutting tokens."""
        if self._tokenizer is not None:
            return self._tokenizer
        model = self._load()
        with self._lock:
            if self._tokenizer is None:
                from tokenizers import Tokenizer

                inner = getattr(getattr(model, "model", None), "tokenizer", None)
                if inner is None:  # pragma: no cover - depends on the fastembed version
                    raise RuntimeError("fastembed no longer exposes its tokenizer; pin fastembed or update this code")
                tokenizer = Tokenizer.from_str(inner.to_str())
                tokenizer.no_truncation()
                tokenizer.no_padding()
                self._tokenizer = tokenizer
            return self._tokenizer

    def _offsets(self, text: str) -> list[tuple[int, int]]:
        if not text:
            return []
        return list(self.tokenizer().encode(text, add_special_tokens=False).offsets)

    def request_text(self, system: str, user: str) -> str:
        """The text the router embeds for a request: each segment cut to its share of the token budget."""
        system_offsets, user_offsets = self._offsets(system), self._offsets(user)
        system_take, user_take = embedding_token_budget(len(system_offsets), len(user_offsets))
        return join_embedding_text(_head(system, system_offsets, system_take), _head(user, user_offsets, user_take))

    def _embed_text(self, text: str) -> list[float]:
        key = hashlib.sha256(text.encode("utf-8", "surrogatepass")).digest()
        if self._cache_size > 0:
            with self._cache_lock:
                hit = self._cache.get(key)
                if hit is not None:
                    self._cache.move_to_end(key)
                    return list(hit)
        vector = tuple(float(value) for value in next(iter(self._load().embed([text]))))
        if self._cache_size > 0:
            with self._cache_lock:
                self._cache[key] = vector
                self._cache.move_to_end(key)
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
        return list(vector)

    def embed_many_sync(self, texts: list[str]) -> list[list[float]]:
        """Embeds a batch as given (no character cut, no LRU). For offline use."""
        return [[float(value) for value in vector] for vector in self._load().embed(texts)]

    def embed_sync(self, text: str) -> list[float]:
        return self._embed_text(text if self._max_chars is None else text[: self._max_chars])

    async def embed(self, text: str) -> list[float]:
        return await anyio.to_thread.run_sync(self.embed_sync, text)

    def embed_request_sync(self, system: str, user: str) -> list[float]:
        return self._embed_text(self.request_text(system, user))

    async def embed_request(self, system: str, user: str) -> list[float]:
        return await anyio.to_thread.run_sync(self.embed_request_sync, system, user)
