"""Router embedding input (feature version 2): token budget, cut at token boundaries, LRU, online/offline parity."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from modelmesh.cache.embeddings import FastEmbedEmbedder
from modelmesh.ml.features import (
    EMBEDDING_MAX_TOKENS,
    EMBEDDING_SPECIAL_TOKENS,
    EMBEDDING_SYSTEM_TOKENS,
    embedding_segments,
    embedding_token_budget,
)

CONTENT = EMBEDDING_MAX_TOKENS - EMBEDDING_SPECIAL_TOKENS
ROOT = Path(__file__).resolve().parents[2]


class WhitespaceTokenizer:
    """One token per whitespace-separated word, with character offsets like the real tokenizer's."""

    def encode(self, text: str, add_special_tokens: bool = False) -> Any:
        offsets, start = [], 0
        for word in text.split():
            begin = text.index(word, start)
            offsets.append((begin, begin + len(word)))
            start = begin + len(word)
        return SimpleNamespace(offsets=offsets)


class CountingModel:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> Any:
        self.calls.append(list(texts))
        return iter([[float(len(t)), 1.0] for t in texts])


def fake_embedder(**kwargs: Any) -> tuple[FastEmbedEmbedder, CountingModel]:
    embedder = FastEmbedEmbedder("test-model", **kwargs)
    model = CountingModel()
    embedder._model = model
    embedder._tokenizer = WhitespaceTokenizer()
    return embedder, model


def words(n: int, prefix: str) -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


@pytest.mark.parametrize(
    ("system", "user", "expected"),
    [
        (0, 10, (0, 10)),
        (0, 500, (0, CONTENT)),
        (10, 20, (10, 20)),
        (200, 50, (CONTENT - 50, 50)),
        (200, 300, (EMBEDDING_SYSTEM_TOKENS, CONTENT - EMBEDDING_SYSTEM_TOKENS)),
        (10, 300, (10, CONTENT - 10)),
        (300, 0, (CONTENT, 0)),
    ],
)
def test_token_budget_never_exceeds_the_cap_and_keeps_the_user_message(
    system: int, user: int, expected: tuple[int, int]
) -> None:
    got = embedding_token_budget(system, user)
    assert got == expected
    assert sum(got) <= CONTENT


def test_a_long_system_prompt_cannot_crowd_out_the_user_message() -> None:
    embedder, _ = fake_embedder()
    text = embedder.request_text(words(500, "s"), words(500, "u"))
    system_part, user_part = text.split("\n")
    assert system_part.split() == [f"s{i}" for i in range(EMBEDDING_SYSTEM_TOKENS)]
    assert user_part.split() == [f"u{i}" for i in range(CONTENT - EMBEDDING_SYSTEM_TOKENS)]


def test_short_texts_are_embedded_whole() -> None:
    embedder, _ = fake_embedder()
    assert embedder.request_text("", "  hello there  ") == "hello there"
    assert embedder.request_text("be brief", "hello") == "be brief\nhello"


def test_segments_are_system_prompt_and_final_user_message() -> None:
    messages = [
        {"role": "system", "content": "S"},
        {"role": "developer", "content": "D"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "A"},
        {"role": "user", "content": [{"type": "text", "text": "last"}]},
    ]
    assert embedding_segments(messages) == ("S\nD", "last")
    long = [{"role": "user", "content": "x" * 10_000}]
    assert len(embedding_segments(long)[1]) == 16 * EMBEDDING_MAX_TOKENS
    assert len(embedding_segments(long, max_chars=None)[1]) == 10_000


def test_lru_runs_the_model_once_per_text_and_evicts_the_oldest() -> None:
    embedder, model = fake_embedder(cache_size=2)
    first = embedder.embed_request_sync("", "alpha")
    assert embedder.embed_request_sync("", "alpha") == first
    assert len(model.calls) == 1
    embedder.embed_sync("beta")
    embedder.embed_sync("gamma")  # evicts "alpha"
    embedder.embed_request_sync("", "alpha")
    assert [c[0] for c in model.calls] == ["alpha", "beta", "gamma", "alpha"]


def test_lru_can_be_disabled_and_the_cache_path_keeps_its_character_cut() -> None:
    embedder, model = fake_embedder(cache_size=0, max_chars=4)
    embedder.embed_sync("abcdefgh")
    embedder.embed_sync("abcdefgh")
    assert model.calls == [["abcd"], ["abcd"]]


def test_threads_default_to_at_most_four() -> None:
    assert 1 <= FastEmbedEmbedder("m").threads <= 4
    assert FastEmbedEmbedder("m", threads=2).threads == 2


def _local_model_dir() -> str | None:
    path = os.environ.get("FASTEMBED_CACHE_PATH") or str(ROOT / ".cache" / "fastembed")
    return path if Path(path).is_dir() and any(Path(path).glob("models--*bge-small*")) else None


@pytest.mark.skipif(_local_model_dir() is None, reason="bge-small is not in the local fastembed cache")
def test_real_tokenizer_cut_fits_the_cap_and_offline_matches_online(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    cache_dir = _local_model_dir()
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", cache_dir or "")
    online = FastEmbedEmbedder("BAAI/bge-small-en-v1.5", cache_dir=cache_dir, threads=1)
    system = "You are a careful research assistant. Cite every claim. " * 30
    user = "Does the passage support the claim that tokenisation-heavy words, e.g. 'unbelievably', split? " * 40
    text = online.request_text(system, user)
    tokens = online.tokenizer().encode(text, add_special_tokens=True).ids
    assert len(tokens) <= EMBEDDING_MAX_TOKENS
    assert text.split("\n")[1].startswith("Does the passage")

    from modelmesh.ml.training.embedding import CachedEmbedder

    offline = CachedEmbedder(tmp_path / "e.npz")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    assert offline.request_texts([messages]) == [text]
