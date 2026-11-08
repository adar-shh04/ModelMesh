"""Cache correctness and tenant isolation against real Redis 8 (ADR-0006)."""

from __future__ import annotations

import math
from collections.abc import AsyncIterator
from typing import Any

import anyio
import pytest

from conftest import ROOMY_LIMITS, Api, assert_matches, build_api, sse_events
from doubles import chunks, make_model, reply
from modelmesh.cache.redis_cache import RedisResponseCache, lexical_guard
from modelmesh.core.errors import ErrorClass, ProviderError
from modelmesh.providers.types import NormalizedChunk, TokenUsage

pytestmark = [pytest.mark.anyio, pytest.mark.integration]

FP_A = "sha256:" + "a" * 64
FP_B = "sha256:" + "b" * 64


class AngleEmbedder:
    """Maps a text to a 384-d unit vector with the configured cosine similarity to a reference axis."""

    def __init__(self, similarities: dict[str, float]) -> None:
        self.similarities = similarities
        self.calls = 0

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        similarity = self.similarities.get(text, 0.0)
        vector = [0.0] * 384
        vector[0] = similarity
        vector[1] = math.sqrt(max(0.0, 1.0 - similarity * similarity))
        return vector


def status(body: dict[str, Any]) -> str:
    return body["modelmesh"]["cache"]["status"]


def deterministic(**extra: Any) -> dict[str, Any]:
    return {"temperature": 0, **extra}


@pytest.fixture
async def semantic(migrated_database: str) -> AsyncIterator[Api]:
    instance = await build_api(migrated_database, MODELMESH_SEMANTIC_CACHE_ENABLED=False)
    embedder = AngleEmbedder(
        {
            "How long is the warranty?": 1.0,
            "How long does the warranty last?": 0.985,
            "Tell me about the weather": 0.4,
            "Is the warranty valid abroad?": 1.0,
            "Is the warranty not valid abroad?": 0.999,
            "What changed in 2023?": 1.0,
            "What changed in 2024?": 0.999,
        }
    )
    instance.services.cache = RedisResponseCache(instance.services.redis, embedder, default_threshold=0.97)
    await instance.services.cache.prepare()
    yield instance
    await instance.client.aclose()
    await instance.services.aclose()


def semantic_args(text: str, *, fingerprint: str = FP_A, **extra: Any) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": text}],
        "temperature": 0,
        "cache": {"mode": "semantic", "context_fingerprint": fingerprint},
    } | extra


# ---------------------------------------------------------------- exact cache
async def test_exact_hit_reports_savings_and_no_routing(api: Api) -> None:
    api.only_models(make_model("m"), prices={"t/m": (1000, 1000)})
    api.scripted.script("m", reply("answer", usage=TokenUsage(100, 100)))
    first = await api.chat(model="t/m", **deterministic())
    assert status(first.json()) == "miss"
    second = await api.chat(model="t/m", **deterministic())
    body = second.json()
    assert_matches("ChatCompletionResponse", body)
    assert body["choices"][0]["message"]["content"] == "answer"
    cache = body["modelmesh"]["cache"]
    assert cache["status"] == "hit_exact"
    assert cache["original_request_id"] == first.headers["x-modelmesh-request-id"]
    assert cache["saved_estimated_usd"] == pytest.approx(0.2)
    assert body["modelmesh"]["routing"] is None
    assert body["modelmesh"]["cost"]["estimated_usd"] == 0
    assert body["usage"]["prompt_tokens"] == 100
    assert body["modelmesh"]["request_id"] == second.headers["x-modelmesh-request-id"]
    assert len(api.scripted.requests) == 1


async def test_nondeterministic_default_neither_reads_nor_writes(api: Api) -> None:
    for temperature_args in ({}, {"temperature": 0.7}):
        await api.services.redis.flushdb()
        first = await api.chat(**temperature_args)
        second = await api.chat(**temperature_args)
        assert status(first.json()) == "bypass"
        assert status(second.json()) == "bypass"
        assert not await api.services.redis.keys("mm:c:*")


async def test_seed_makes_a_request_cacheable(api: Api) -> None:
    await api.chat(temperature=0.7, seed=1)
    assert status((await api.chat(temperature=0.7, seed=1)).json()) == "hit_exact"


async def test_explicit_mode_opts_in_for_nondeterministic_requests(api: Api) -> None:
    await api.chat(temperature=0.9, cache={"mode": "exact"})
    assert status((await api.chat(temperature=0.9, cache={"mode": "exact"})).json()) == "hit_exact"


async def test_off_mode_bypasses(api: Api) -> None:
    await api.chat(cache={"mode": "off"}, temperature=0)
    assert status((await api.chat(cache={"mode": "off"}, temperature=0)).json()) == "bypass"
    assert not await api.services.redis.keys("mm:c:*")


async def test_key_with_caching_disabled_bypasses(api: Api) -> None:
    key = await api.new_key(scopes={"cache_enabled": False}, limits=ROOMY_LIMITS)
    headers = {"Authorization": f"Bearer {key}"}
    for _ in range(2):
        response = await api.chat(temperature=0, headers=headers)
        assert status(response.json()) == "bypass"


BASE: dict[str, Any] = {
    "model": "fake/echo-small",
    "messages": [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "first"},
        {"role": "user", "content": "q"},
    ],
    "temperature": 0,
    "cache": {"mode": "exact", "context_fingerprint": FP_A, "namespace": "ns1"},
    "metadata": {"task_type": "general"},
}
TOOLS = [{"type": "function", "function": {"name": "f"}}]


@pytest.mark.parametrize(
    "change",
    [
        {"cache": {"mode": "exact", "context_fingerprint": FP_B, "namespace": "ns1"}},
        {"cache": {"mode": "exact", "context_fingerprint": FP_A, "namespace": "ns2"}},
        {"cache": {"mode": "exact", "context_fingerprint": FP_A}},
        {"model": "fake/echo-medium"},
        {"temperature": 0.5},
        {"top_p": 0.5},
        {"max_tokens": 10},
        {"stop": ["x"]},
        {"seed": 3},
        {"presence_penalty": 0.1},
        {"frequency_penalty": 0.1},
        {"response_format": {"type": "json_object"}},
        {"tools": TOOLS},
        {"tools": TOOLS, "tool_choice": "required"},
        {"metadata": {"task_type": "final_response"}},
        {
            "messages": [
                {"role": "system", "content": "other"},
                {"role": "user", "content": "first"},
                {"role": "user", "content": "q"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "changed"},
                {"role": "user", "content": "q"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "first"},
                {"role": "user", "content": "q2"},
            ]
        },
    ],
)
async def test_every_partition_component_changes_the_key(api: Api, change: dict[str, Any]) -> None:
    first = await api.chat(**BASE)
    assert first.status_code == 200, first.text
    assert status((await api.chat(**BASE)).json()) == "hit_exact"
    changed = await api.chat(**(BASE | change))
    assert changed.status_code == 200, changed.text
    assert status(changed.json()) == "miss"


async def test_auto_routing_constraints_change_the_key(api: Api) -> None:
    base = BASE | {"model": "auto", "routing": {"policy": "cost_optimized"}}
    await api.chat(**base)
    assert status((await api.chat(**base)).json()) == "hit_exact"
    other = base | {"routing": {"policy": "quality_max"}}
    assert status((await api.chat(**other)).json()) == "miss"


async def test_exact_hit_never_crosses_tenants(api: Api) -> None:
    await api.chat(**BASE)
    other = await api.admin.create_tenant("other")
    key = await api.new_key(other["id"], limits=ROOMY_LIMITS)
    response = await api.chat(**BASE, headers={"Authorization": f"Bearer {key}"})
    assert status(response.json()) == "miss"


async def test_tightened_data_policy_blocks_an_older_hit(api: Api) -> None:
    api.only_models(
        make_model("trains", data_use="may_train", size_tier="small"),
        make_model("safe", data_use="no_training", size_tier="small"),
        prices={"t/trains": (0.1, 0.1), "t/safe": (5, 5)},
    )
    api.scripted.script("trains", reply("from trains"))
    args = {"model": "t/trains", "temperature": 0, "routing": {"fallback": "within_constraints"}}
    await api.chat(**args)
    assert status((await api.chat(**args)).json()) == "hit_exact"
    # Same partition, but the policy layer has removed this model, so the entry must not be served.
    key = await api.new_key(data_policy_ceiling="no_training_providers", limits=ROOMY_LIMITS)
    blocked = await api.chat(**args, headers={"Authorization": f"Bearer {key}"})
    assert blocked.status_code == 422
    assert "from trains" not in blocked.text


async def test_ttl_expires_and_is_capped_by_the_key(api: Api) -> None:
    key = await api.new_key(scopes={"cache_max_ttl_s": 1}, limits=ROOMY_LIMITS)
    headers = {"Authorization": f"Bearer {key}"}
    args = {"temperature": 0, "cache": {"mode": "exact", "ttl_s": 3600}}
    await api.chat(**args, headers=headers)
    stored = await api.services.redis.keys("mm:c:*")
    assert await api.services.redis.ttl(stored[0]) <= 1
    assert status((await api.chat(**args, headers=headers)).json()) == "hit_exact"
    await anyio.sleep(1.2)
    assert status((await api.chat(**args, headers=headers)).json()) == "miss"


async def test_filtered_responses_are_not_stored(api: Api) -> None:
    api.only_models(make_model("m"))
    api.scripted.script("m", reply("x", finish="content_filter"), reply("y"))
    await api.chat(model="t/m", temperature=0)
    assert status((await api.chat(model="t/m", temperature=0)).json()) == "miss"


async def test_failed_requests_are_not_stored(api: Api) -> None:
    api.only_models(make_model("m"))
    failure = ProviderError(ErrorClass.AUTH_ERROR, "x", http_status=401)
    api.scripted.script("m", failure)
    await api.chat(model="t/m", temperature=0)
    assert not await api.services.redis.keys("mm:c:*")


async def test_invalid_structured_output_is_not_cached(api: Api) -> None:
    api.only_models(make_model("m"))
    fmt = {"type": "json_object"}
    api.scripted.script("m", reply("nope"), reply("nope"))
    response = await api.chat(model="t/m", temperature=0, response_format=fmt)
    assert response.status_code == 502
    assert not await api.services.redis.keys("mm:c:*")


async def test_cache_hit_does_not_consume_budget(api: Api) -> None:
    api.only_models(make_model("m"), prices={"t/m": (1000, 1000)})
    key = await api.new_key(limits={"rpm": 1000, "tpm": 10_000_000, "concurrency": 100, "daily_budget_usd": 0.5})
    headers = {"Authorization": f"Bearer {key}"}
    api.scripted.script("m", reply(usage=TokenUsage(100, 100)))
    for _ in range(4):
        response = await api.chat(model="t/m", temperature=0, max_tokens=50, headers=headers)
        assert response.status_code == 200
    assert len(api.scripted.requests) == 1


# ---------------------------------------------------------------- streaming
async def test_complete_stream_is_stored_and_replayed(api: Api) -> None:
    args = {"temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    first = sse_events((await api.chat(**args)).text)
    assert first[-2]["modelmesh"]["cache"]["status"] == "miss"
    second = sse_events((await api.chat(**args)).text)
    assert second[-1] == "[DONE]"
    for event in second[:-1]:
        assert_matches("ChatCompletionChunk", event)
    assert second[-2]["modelmesh"]["cache"]["status"] == "hit_exact"
    assert second[-2]["modelmesh"]["routing"] is None
    text = "".join(e["choices"][0]["delta"].get("content", "") for e in second[:-1] if e["choices"])
    assert text == "echo: hello"


async def test_stream_cached_response_serves_a_non_stream_request(api: Api) -> None:
    await api.chat(temperature=0, stream=True)
    # stream is not part of the key: the answer is the same either way.
    response = await api.chat(temperature=0)
    assert status(response.json()) == "hit_exact"


async def test_interrupted_stream_is_never_stored(api: Api) -> None:
    api.only_models(make_model("m"))
    failure = ProviderError(ErrorClass.UPSTREAM_UNAVAILABLE, "x", http_status=503)
    api.scripted.script("m", [NormalizedChunk(content="partial"), failure])
    await api.chat(model="t/m", temperature=0, stream=True)
    assert not await api.services.redis.keys("mm:c:*")


async def test_stream_of_length_finish_is_stored(api: Api) -> None:
    api.only_models(make_model("m"))
    api.scripted.script("m", chunks("cut", finish="length", usage=TokenUsage(1, 1)))
    await api.chat(model="t/m", temperature=0, stream=True)
    assert await api.services.redis.keys("mm:c:*")


# ---------------------------------------------------------------- semantic cache
async def test_semantic_hit_above_threshold(semantic: Api) -> None:
    await semantic.chat(**semantic_args("How long is the warranty?"))
    response = await semantic.chat(**semantic_args("How long does the warranty last?"))
    cache = response.json()["modelmesh"]["cache"]
    assert cache["status"] == "hit_semantic"
    assert cache["similarity"] == pytest.approx(0.985, abs=1e-3)
    assert_matches("ChatCompletionResponse", response.json())


async def test_semantic_miss_below_threshold(semantic: Api) -> None:
    await semantic.chat(**semantic_args("How long is the warranty?"))
    response = await semantic.chat(**semantic_args("Tell me about the weather"))
    assert status(response.json()) == "miss"


@pytest.mark.parametrize(
    ("stored", "queried"),
    [
        ("Is the warranty valid abroad?", "Is the warranty not valid abroad?"),
        ("What changed in 2023?", "What changed in 2024?"),
    ],
)
async def test_lexical_guard_blocks_near_misses(semantic: Api, stored: str, queried: str) -> None:
    await semantic.chat(**semantic_args(stored))
    response = await semantic.chat(**semantic_args(queried))
    assert status(response.json()) == "miss"


async def test_semantic_requires_matching_fingerprint(semantic: Api) -> None:
    await semantic.chat(**semantic_args("How long is the warranty?"))
    response = await semantic.chat(**semantic_args("How long does the warranty last?", fingerprint=FP_B))
    assert status(response.json()) == "miss"


async def test_semantic_hit_never_crosses_tenants(semantic: Api) -> None:
    await semantic.chat(**semantic_args("How long is the warranty?"))
    other = await semantic.admin.create_tenant("other")
    key = await semantic.new_key(other["id"], limits=ROOMY_LIMITS)
    response = await semantic.chat(
        **semantic_args("How long does the warranty last?"), headers={"Authorization": f"Bearer {key}"}
    )
    assert status(response.json()) == "miss"


async def test_semantic_hit_never_crosses_namespaces(semantic: Api) -> None:
    first = semantic_args("How long is the warranty?")
    first["cache"] |= {"namespace": "ws1"}
    second = semantic_args("How long does the warranty last?")
    second["cache"] |= {"namespace": "ws2"}
    await semantic.chat(**first)
    assert status((await semantic.chat(**second)).json()) == "miss"


async def test_semantic_is_disabled_with_tools(semantic: Api) -> None:
    await semantic.chat(**semantic_args("How long is the warranty?", tools=TOOLS))
    response = await semantic.chat(**semantic_args("How long does the warranty last?", tools=TOOLS))
    assert status(response.json()) == "miss"
    assert not await semantic.services.redis.hexists((await semantic.services.redis.keys("mm:c:*"))[0], "embedding")


async def test_threshold_of_one_disables_a_task_type(migrated_database: str) -> None:
    api = await build_api(migrated_database, MODELMESH_SEMANTIC_CACHE_ENABLED=False)
    embedder = AngleEmbedder({"a b": 1.0, "a  b": 0.9999})
    api.services.cache = RedisResponseCache(api.services.redis, embedder, thresholds={"final_response": 1.0})
    await api.services.cache.prepare()
    try:
        meta = {"task_type": "final_response"}
        await api.chat(**semantic_args("a b", metadata=meta))
        assert status((await api.chat(**semantic_args("a  b", metadata=meta))).json()) == "miss"
    finally:
        await api.client.aclose()
        await api.services.aclose()


def test_lexical_guard_normalises_order_but_not_values() -> None:
    assert lexical_guard("from 2023 to 2024") == lexical_guard("to 2024 from 2023")
    assert lexical_guard("cost is 5") != lexical_guard("cost is 6")
    assert lexical_guard("allowed") != lexical_guard("not allowed")
    assert lexical_guard("isn't allowed") != lexical_guard("is allowed")
    assert lexical_guard('find "alpha"') != lexical_guard('find "beta"')


# ---------------------------------------------------------------- invalidation
async def seed_three(api: Api) -> None:
    await api.chat(
        temperature=0,
        messages=[{"role": "user", "content": "1"}],
        cache={"namespace": "a", "context_fingerprint": FP_A},
    )
    await api.chat(
        temperature=0,
        messages=[{"role": "user", "content": "2"}],
        cache={"namespace": "a", "context_fingerprint": FP_B},
    )
    await api.chat(
        temperature=0,
        messages=[{"role": "user", "content": "3"}],
        cache={"namespace": "b", "context_fingerprint": FP_A},
    )


async def test_invalidate_by_namespace(api: Api) -> None:
    await seed_three(api)
    response = await api.client.post("/v1/cache/invalidate", json={"namespace": "a"}, headers=api.headers)
    assert response.status_code == 200
    assert_matches("CacheInvalidateResponse", response.json())
    assert response.json() == {"deleted_entries": 2}
    assert len(await api.services.redis.keys("mm:c:*")) == 1


async def test_invalidate_by_fingerprint_and_both(api: Api) -> None:
    await seed_three(api)
    assert (
        await api.client.post("/v1/cache/invalidate", json={"context_fingerprint": FP_A}, headers=api.headers)
    ).json() == {"deleted_entries": 2}
    await seed_three(api)
    both = {"namespace": "a", "context_fingerprint": FP_B}
    assert (await api.client.post("/v1/cache/invalidate", json=both, headers=api.headers)).json() == {
        "deleted_entries": 1
    }


async def test_invalidate_is_tenant_scoped(api: Api) -> None:
    await seed_three(api)
    other = await api.admin.create_tenant("other")
    key = await api.new_key(other["id"], limits=ROOMY_LIMITS)
    response = await api.client.post(
        "/v1/cache/invalidate", json={"namespace": "a"}, headers={"Authorization": f"Bearer {key}"}
    )
    assert response.json() == {"deleted_entries": 0}
    assert len(await api.services.redis.keys("mm:c:*")) == 3


async def test_invalidate_requires_a_selector(api: Api) -> None:
    response = await api.client.post("/v1/cache/invalidate", json={}, headers=api.headers)
    assert response.status_code == 400
