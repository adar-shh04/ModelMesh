"""Records, usage aggregates, keys, limits and feedback (M3)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import anyio
import httpx
import pytest
from sqlalchemy import select, text

from conftest import ROOMY_LIMITS, Api, assert_matches, build_api, make_settings
from doubles import make_model, reply
from modelmesh.accounting.retention import purge_expired
from modelmesh.core.errors import ErrorClass, ProviderError
from modelmesh.persistence import models as db
from modelmesh.providers.types import TokenUsage

pytestmark = [pytest.mark.anyio, pytest.mark.integration]

ADMIN = {"Authorization": "Bearer test-admin-token"}


def client_for(api: Api, key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def message_body(**extra):
    return {"model": "fake/echo-small", "messages": [{"role": "user", "content": "hello"}]} | extra


def window() -> dict[str, str]:
    now = datetime.now(UTC)
    return {"from": (now - timedelta(minutes=5)).isoformat(), "to": (now + timedelta(minutes=5)).isoformat()}


# ---------------------------------------------------------------- records
async def test_success_produces_a_full_record(api: Api) -> None:
    api.only_models(make_model("m"), prices={"t/m": (1_000.0, 2_000.0)})
    api.scripted.script("m", reply("x", usage=TokenUsage(100, 50, cached_tokens=7, reasoning_tokens=3)))
    response = await api.chat(
        model="t/m", metadata={"task_type": "general", "correlation_id": "run-1", "application": "app"}
    )
    request_id = response.headers["x-modelmesh-request-id"]
    meta = response.json()["modelmesh"]
    assert meta["cost"]["estimated_usd"] == pytest.approx(0.2)
    assert meta["usage"]["cached_tokens"] == 7

    fetched = await api.client.get(f"/v1/requests/{request_id}", headers=api.headers)
    record = fetched.json()
    assert_matches("RequestRecord", record)
    assert record["status"] == "succeeded"
    assert record["metadata"]["correlation_id"] == "run-1"
    assert record["content"] is None
    assert record["modelmesh"]["usage"]["prompt_tokens"] == 100

    async with api.services.sessions() as session:
        usage = await session.get(db.UsageRecord, request_id)
        attempts = (
            (await session.execute(select(db.Attempt).where(db.Attempt.request_id == request_id))).scalars().all()
        )
    assert usage.usage_source == "provider_reported"
    assert usage.estimated_cost_usd == pytest.approx(0.2)
    assert usage.provider_reported_cost_usd is None
    assert [a.outcome for a in attempts] == ["success"]


@pytest.mark.parametrize(
    ("kwargs", "status", "stored_status"),
    [
        ({"n": 2}, 400, "rejected"),
        ({"model": "fake/none"}, 404, "rejected"),
        ({"model": "auto", "routing": {"min_quality": 0.5}}, 422, "rejected"),
    ],
)
async def test_rejected_requests_are_recorded(api: Api, kwargs, status, stored_status) -> None:
    response = await api.chat(**kwargs)
    assert response.status_code == status
    fetched = await api.client.get(f"/v1/requests/{response.headers['x-modelmesh-request-id']}", headers=api.headers)
    assert fetched.status_code == 200
    assert fetched.json()["status"] == stored_status
    assert fetched.json()["modelmesh"] is None
    assert fetched.json()["error"]["type"]
    assert_matches("RequestRecord", fetched.json())


async def test_upstream_failure_is_recorded_with_attempts(api: Api) -> None:
    api.only_models(make_model("m"))
    failure = ProviderError(ErrorClass.AUTH_ERROR, "no", http_status=401)
    api.scripted.script("m", failure)
    response = await api.chat(model="t/m")
    async with api.services.sessions() as session:
        record = await session.get(db.RequestRecord, response.headers["x-modelmesh-request-id"])
        attempts = (await session.execute(select(db.Attempt).where(db.Attempt.request_id == record.id))).scalars().all()
    assert (record.status, record.http_status, record.error_type) == ("failed", 502, "upstream_error")
    assert [a.outcome for a in attempts] == ["auth_error"]


async def test_estimated_usage_is_labelled_and_kept_separate(api: Api) -> None:
    api.only_models(make_model("reported"), make_model("guessed"), prices={"t/reported": (0, 0), "t/guessed": (0, 0)})
    api.scripted.script("reported", reply(usage=TokenUsage(100, 50, source="provider_reported")))
    from modelmesh.providers.types import NormalizedChatResponse

    api.scripted.script(
        "guessed", NormalizedChatResponse(content="abcd", tool_calls=None, finish_reason="stop", usage=None)
    )
    await api.chat(model="t/reported")
    await api.chat(model="t/guessed")
    report = (await api.client.get("/v1/usage", params=window() | {"group_by": "model"}, headers=api.headers)).json()
    assert_matches("UsageReport", report)
    by_model = {row["group"]: row for row in report["data"]}
    assert by_model["t/reported"]["prompt_tokens_provider_reported"] == 100
    assert by_model["t/reported"]["prompt_tokens_estimated"] == 0
    assert by_model["t/guessed"]["prompt_tokens_estimated"] > 0
    assert by_model["t/guessed"]["prompt_tokens_provider_reported"] == 0


async def test_usage_aggregates_match_hand_computed_values(api: Api) -> None:
    api.only_models(make_model("a"), make_model("b"), prices={"t/a": (1_000, 1_000), "t/b": (2_000, 2_000)})
    api.scripted.script("a", reply(usage=TokenUsage(100, 100)), reply(usage=TokenUsage(200, 0)))
    api.scripted.script("b", reply(usage=TokenUsage(50, 50), provider_reported_cost_usd=0.5))
    await api.chat(model="t/a", metadata={"task_type": "general"})
    await api.chat(model="t/a", metadata={"task_type": "general"})
    await api.chat(model="t/b", metadata={"task_type": "final_response"})
    await api.chat(n=2)  # rejected: counted as a failed request with no usage

    report = (await api.client.get("/v1/usage", params=window() | {"group_by": "model"}, headers=api.headers)).json()
    rows = {row["group"]: row for row in report["data"]}
    assert rows["t/a"]["requests"] == 2
    assert rows["t/a"]["succeeded"] == 2
    assert rows["t/a"]["prompt_tokens_provider_reported"] == 300
    assert rows["t/a"]["completion_tokens_provider_reported"] == 100
    assert rows["t/a"]["estimated_cost_usd"] == pytest.approx(0.4)
    assert rows["t/a"]["provider_reported_cost_usd"] is None
    assert rows["t/b"]["provider_reported_cost_usd"] == pytest.approx(0.5)
    assert rows["t/b"]["provider_reported_cost_coverage"] == 1.0
    assert report["totals"]["group"] is None
    assert report["totals"]["requests"] == 4
    assert report["totals"]["failed"] == 1
    assert report["totals"]["estimated_cost_usd"] == pytest.approx(0.6)


@pytest.mark.parametrize("group", ["provider", "task_type", "application", "day"])
async def test_usage_groupings(api: Api, group: str) -> None:
    await api.chat(metadata={"application": "a1", "task_type": "general"})
    report = (await api.client.get("/v1/usage", params=window() | {"group_by": group}, headers=api.headers)).json()
    assert_matches("UsageReport", report)
    assert report["data"]


async def test_usage_rejects_excessive_range(api: Api) -> None:
    now = datetime.now(UTC)
    params = {"from": (now - timedelta(days=93)).isoformat(), "to": now.isoformat()}
    response = await api.client.get("/v1/usage", params=params, headers=api.headers)
    assert response.status_code == 400


async def test_list_requests_by_correlation_id_paginates(api: Api) -> None:
    for _ in range(3):
        await api.chat(metadata={"correlation_id": "run-9"})
    await api.chat(metadata={"correlation_id": "other"})
    first = (
        await api.client.get("/v1/requests", params={"correlation_id": "run-9", "limit": 2}, headers=api.headers)
    ).json()
    assert_matches("RequestRecordList", first)
    assert len(first["data"]) == 2
    assert first["next_cursor"]
    second = (
        await api.client.get(
            "/v1/requests",
            params={"correlation_id": "run-9", "limit": 2, "cursor": first["next_cursor"]},
            headers=api.headers,
        )
    ).json()
    assert len(second["data"]) == 1
    assert second["next_cursor"] is None
    assert {r["id"] for r in first["data"]}.isdisjoint(r["id"] for r in second["data"])


# ---------------------------------------------------------------- content retention
async def test_content_is_not_retained_by_default(api: Api) -> None:
    response = await api.chat(messages=[{"role": "user", "content": "secret"}])
    async with api.services.sessions() as session:
        retained = await session.get(db.RequestContent, response.headers["x-modelmesh-request-id"])
    assert retained is None


async def test_content_retention_when_enabled(api: Api) -> None:
    key = await api.new_key(content_retention=True, limits=ROOMY_LIMITS)
    response = await api.client.post("/v1/chat/completions", json=message_body(), headers=client_for(api, key))
    record = (
        await api.client.get(f"/v1/requests/{response.headers['x-modelmesh-request-id']}", headers=client_for(api, key))
    ).json()
    assert_matches("RequestRecord", record)
    assert record["content"]["messages"][0]["content"] == "hello"
    assert record["content"]["response"]["content"] == "echo: hello"


async def test_expired_content_is_hidden_and_purged(api: Api) -> None:
    key = await api.new_key(content_retention=True, limits=ROOMY_LIMITS)
    response = await api.client.post("/v1/chat/completions", json=message_body(), headers=client_for(api, key))
    request_id = response.headers["x-modelmesh-request-id"]
    past = datetime.now(UTC) - timedelta(minutes=1)
    async with api.services.engine.begin() as connection:
        await connection.execute(text("UPDATE request_content SET expires_at = :t"), {"t": past})
    record = (await api.client.get(f"/v1/requests/{request_id}", headers=client_for(api, key))).json()
    assert record["content"] is None
    assert await purge_expired(api.services.sessions) == (1, 0)
    async with api.services.sessions() as session:
        assert await session.get(db.RequestContent, request_id) is None


async def test_old_execution_records_are_purged_after_90_days(api: Api) -> None:
    response = await api.chat()
    request_id = response.headers["x-modelmesh-request-id"]
    assert await purge_expired(api.services.sessions, now=datetime.now(UTC) + timedelta(days=89)) == (0, 0)
    assert await purge_expired(api.services.sessions, now=datetime.now(UTC) + timedelta(days=91)) == (0, 1)
    assert (await api.client.get(f"/v1/requests/{request_id}", headers=api.headers)).status_code == 404


async def test_tenant_deletion_removes_everything_including_cache(api: Api) -> None:
    other = await api.admin.create_tenant("doomed")
    key = await api.new_key(other["id"], limits=ROOMY_LIMITS)
    headers = client_for(api, key)
    await api.client.post("/v1/chat/completions", json=message_body(temperature=0), headers=headers)
    assert await api.services.redis.keys(f"mm:c:{other['id']}:*")
    response = await api.client.delete(f"/admin/v1/tenants/{other['id']}", headers=ADMIN)
    assert response.json()["cache_entries_deleted"] == 1
    assert not await api.services.redis.keys(f"mm:c:{other['id']}:*")
    assert (await api.client.get("/v1/models", headers=headers)).status_code == 401
    async with api.services.engine.connect() as connection:
        remaining = (
            await connection.execute(text("SELECT count(*) FROM requests WHERE tenant_id = :t"), {"t": other["id"]})
        ).scalar_one()
    assert remaining == 0


# ---------------------------------------------------------------- tenant isolation
async def test_other_tenants_cannot_read_requests_or_label_them(api: Api) -> None:
    response = await api.chat()
    request_id = response.headers["x-modelmesh-request-id"]
    other_tenant = await api.admin.create_tenant("other")
    other_key = await api.new_key(other_tenant["id"], limits=ROOMY_LIMITS)
    other = client_for(api, other_key)
    fetched = await api.client.get(f"/v1/requests/{request_id}", headers=other)
    assert fetched.status_code == 404
    assert fetched.json()["error"]["code"] == "request_not_found"
    label = {"request_id": request_id, "label_source": "automated_eval", "metric": "m", "score": 0.5, "evaluator": "e"}
    assert (await api.client.post("/v1/feedback", json=label, headers=other)).status_code == 404
    assert (await api.client.delete(f"/v1/feedback/{request_id}", headers=other)).status_code == 404
    listing = (await api.client.get("/v1/requests", headers=other)).json()
    assert listing["data"] == []
    report = (await api.client.get("/v1/usage", params=window(), headers=other)).json()
    assert report["totals"]["requests"] == 0


# ---------------------------------------------------------------- admin API and keys
async def test_admin_api_requires_the_token(api: Api) -> None:
    assert (await api.client.post("/admin/v1/tenants", json={"name": "x"})).status_code == 401
    assert (await api.client.post("/admin/v1/tenants", json={"name": "x"}, headers=api.headers)).status_code == 401


async def test_key_lifecycle_through_the_admin_api(api: Api) -> None:
    tenant = await api.client.post("/admin/v1/tenants", json={"name": "acme"}, headers=ADMIN)
    assert tenant.status_code == 201
    tenant_id = tenant.json()["id"]
    created = await api.client.post(f"/admin/v1/tenants/{tenant_id}/keys", json={"name": "k1"}, headers=ADMIN)
    assert created.status_code == 201
    assert created.json()["limits"]["rpm"] == 20
    assert created.json()["limits"]["tpm"] == 50_000
    assert created.json()["limits"]["daily_budget_usd"] == 0.25
    key = created.json()["key"]
    key_id = created.json()["id"]
    assert key.startswith("mm_live_")
    listed = await api.client.get(f"/admin/v1/tenants/{tenant_id}/keys", headers=ADMIN)
    assert key not in listed.text

    ok = await api.client.post("/v1/chat/completions", json=message_body(), headers=client_for(api, key))
    assert ok.status_code == 200

    rotated = await api.client.post(f"/admin/v1/keys/{key_id}/rotate", headers=ADMIN)
    new_key = rotated.json()["key"]
    assert (await api.client.get("/v1/models", headers=client_for(api, key))).status_code == 401
    assert (await api.client.get("/v1/models", headers=client_for(api, new_key))).status_code == 200

    await api.client.post(f"/admin/v1/keys/{key_id}/revoke", headers=ADMIN)
    started = asyncio.get_running_loop().time()
    revoked = await api.client.get("/v1/models", headers=client_for(api, new_key))
    assert asyncio.get_running_loop().time() - started < 1
    assert revoked.status_code == 401
    assert revoked.json()["error"]["code"] == "revoked_api_key"


async def test_key_is_never_stored_in_plaintext(api: Api) -> None:
    async with api.services.engine.connect() as connection:
        rows = (await connection.execute(text("SELECT * FROM api_keys"))).all()
    assert api.key not in repr(rows)


async def test_provider_disable_and_enable(api: Api) -> None:
    await api.client.put("/admin/v1/providers/fake", json={"enabled": False}, headers=ADMIN)
    disabled = await api.chat()
    assert disabled.status_code == 404
    status = (await api.client.get("/v1/providers", headers=api.headers)).json()
    assert next(p for p in status["data"] if p["id"] == "fake")["status"] == "disabled"
    await api.client.put("/admin/v1/providers/fake", json={"enabled": True}, headers=ADMIN)
    assert (await api.chat()).status_code == 200


# ---------------------------------------------------------------- limits
async def test_rpm_limit_holds_under_concurrency(api: Api) -> None:
    key = await api.new_key(limits={"rpm": 20, "tpm": 10_000_000, "concurrency": 1000, "daily_budget_usd": None})
    headers = client_for(api, key)
    results: list[httpx.Response] = []

    async def call() -> None:
        results.append(await api.client.post("/v1/chat/completions", json=message_body(), headers=headers))

    async with anyio.create_task_group() as group:
        for _ in range(50):
            group.start_soon(call)
    statuses = [r.status_code for r in results]
    assert statuses.count(200) == 20
    assert statuses.count(429) == 30
    rejected = next(r for r in results if r.status_code == 429)
    assert int(rejected.headers["retry-after"]) >= 1
    assert rejected.json()["error"]["code"] == "rpm"
    assert rejected.json()["error"]["details"]["scope"] == "key"
    assert_matches("ErrorResponse", rejected.json())


async def test_retry_after_is_accurate(api: Api) -> None:
    key = await api.new_key(limits={"rpm": 60, "tpm": 10_000_000, "concurrency": 1000, "daily_budget_usd": None})
    headers = client_for(api, key)
    while True:
        response = await api.client.post("/v1/chat/completions", json=message_body(), headers=headers)
        if response.status_code == 429:
            break
    wait = int(response.headers["retry-after"])
    assert wait == 1
    await anyio.sleep(wait + 0.05)
    assert (await api.client.post("/v1/chat/completions", json=message_body(), headers=headers)).status_code == 200


async def test_tpm_limit(api: Api) -> None:
    key = await api.new_key(limits={"rpm": 1000, "tpm": 1000, "concurrency": 1000, "daily_budget_usd": None})
    headers = client_for(api, key)
    small = message_body(max_tokens=100)
    for _ in range(2):  # Each reservation is reconciled down to actual usage, so both fit.
        assert (await api.client.post("/v1/chat/completions", json=small, headers=headers)).status_code == 200
    # Reservation about 850 tokens; the echo makes actual usage about 1500, which is reconciled upward.
    large = message_body(messages=[{"role": "user", "content": "x" * 3000}], max_tokens=100)
    assert (await api.client.post("/v1/chat/completions", json=large, headers=headers)).status_code == 200
    blocked = await api.client.post("/v1/chat/completions", json=small, headers=headers)
    assert blocked.status_code == 429
    assert blocked.json()["error"]["code"] == "tpm"


async def test_concurrency_limit_is_released(api: Api) -> None:
    api.only_models(make_model("slow"))
    api.scripted.delay_s = 0.2
    key = await api.new_key(limits={"rpm": 1000, "tpm": 10_000_000, "concurrency": 2, "daily_budget_usd": None})
    headers = client_for(api, key)
    body = message_body(model="t/slow")
    results: list[httpx.Response] = []

    async def call() -> None:
        results.append(await api.client.post("/v1/chat/completions", json=body, headers=headers))

    async with anyio.create_task_group() as group:
        for _ in range(5):
            group.start_soon(call)
    statuses = sorted(r.status_code for r in results)
    assert statuses == [200, 200, 429, 429, 429]
    assert next(r for r in results if r.status_code == 429).json()["error"]["code"] == "concurrency"
    assert (await api.client.post("/v1/chat/completions", json=body, headers=headers)).status_code == 200


async def test_concurrency_slot_released_after_streaming_failure(api: Api) -> None:
    key = await api.new_key(limits={"rpm": 1000, "tpm": 10_000_000, "concurrency": 1, "daily_budget_usd": None})
    headers = client_for(api, key)
    for _ in range(3):
        response = await api.client.post("/v1/chat/completions", json=message_body(stream=True), headers=headers)
        assert response.status_code == 200
        bad = await api.client.post(
            "/v1/chat/completions", json=message_body(model="fake/unavailable", stream=True), headers=headers
        )
        assert bad.status_code == 502


async def test_budget_upper_bound_precheck_and_exhaustion(api: Api) -> None:
    api.only_models(make_model("m"), prices={"t/m": (1_000.0, 1_000.0)})
    key = await api.new_key(limits={"rpm": 1000, "tpm": 10_000_000, "concurrency": 100, "daily_budget_usd": 0.1})
    headers = client_for(api, key)
    too_big = message_body(model="t/m", max_tokens=1000)  # upper bound 1000 * $1000/M = $1 > $0.1
    response = await api.client.post("/v1/chat/completions", json=too_big, headers=headers)
    assert response.status_code == 402
    assert response.json()["error"]["code"] == "daily_budget"
    assert api.scripted.requests == []

    api.scripted.script("m", reply(usage=TokenUsage(100, 100)))
    spend = message_body(model="t/m", max_tokens=90)
    first = await api.client.post("/v1/chat/completions", json=spend, headers=headers)
    assert first.status_code == 200
    assert first.json()["modelmesh"]["cost"]["estimated_usd"] == pytest.approx(0.2)
    exhausted = await api.client.post("/v1/chat/completions", json=spend, headers=headers)
    assert exhausted.status_code == 402
    details = exhausted.json()["error"]["details"]
    assert details["scope"] == "key"
    assert details["period"] == "day"
    assert details["resets_at"]
    assert_matches("ErrorResponse", exhausted.json())


async def test_budget_counters_rebuild_from_postgres(api: Api) -> None:
    api.only_models(make_model("m"), prices={"t/m": (1_000.0, 1_000.0)})
    key = await api.new_key(limits={"rpm": 1000, "tpm": 10_000_000, "concurrency": 100, "daily_budget_usd": 0.1})
    headers = client_for(api, key)
    api.scripted.script("m", reply(usage=TokenUsage(100, 100)))
    assert (
        await api.client.post("/v1/chat/completions", json=message_body(model="t/m", max_tokens=90), headers=headers)
    ).status_code == 200
    for stored in await api.services.redis.keys("mm:bud:*"):
        await api.services.redis.delete(stored)
    rebuilt = await api.client.post(
        "/v1/chat/completions", json=message_body(model="t/m", max_tokens=90), headers=headers
    )
    assert rebuilt.status_code == 402


async def test_redis_outage_fails_closed(migrated_database: str) -> None:
    api = await build_api(migrated_database)
    await api.services.redis.aclose()
    broken = make_settings(MODELMESH_DATABASE_URL=migrated_database, MODELMESH_REDIS_URL="redis://127.0.0.1:1/0")
    from redis.asyncio import Redis

    from modelmesh.auth.keys import KeyAuthenticator
    from modelmesh.auth.limits import BudgetTracker, RateLimiter

    dead = Redis.from_url(broken.redis_url, decode_responses=True, socket_connect_timeout=0.2)
    services = api.services
    services.redis = dead
    services.authenticator = KeyAuthenticator(services.sessions, dead, broken.api_key_pepper)
    services.limiter = RateLimiter(dead)
    services.budgets = BudgetTracker(dead, services.records.spend_since)
    try:
        response = await api.chat()
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "dependency_unavailable"
        health = await api.client.get("/v1/health")
        assert health.status_code == 503
        assert health.json()["status"] == "down"
    finally:
        await api.client.aclose()
        await services.aclose()


# ---------------------------------------------------------------- feedback
LABEL = {"label_source": "automated_eval", "metric": "faithfulness", "score": 0.86, "evaluator": "eval/1"}


async def requested(api: Api, **kwargs) -> str:
    return (await api.chat(**kwargs)).headers["x-modelmesh-request-id"]


async def test_feedback_idempotency(api: Api) -> None:
    request_id = await requested(api)
    body = LABEL | {"request_id": request_id, "metric_version": "0.2", "accepted": True}
    first = await api.client.post("/v1/feedback", json=body, headers=api.headers)
    assert first.status_code == 201
    assert_matches("FeedbackResponse", first.json())
    assert first.json()["status"] == "stored"
    assert first.json()["training_eligible"] is True
    replay = await api.client.post("/v1/feedback", json=body, headers=api.headers)
    assert replay.status_code == 200
    assert replay.json()["status"] == "replayed"
    assert replay.json()["id"] == first.json()["id"]
    conflict = await api.client.post("/v1/feedback", json=body | {"score": 0.1}, headers=api.headers)
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "feedback_conflict"
    new_version = await api.client.post(
        "/v1/feedback", json=body | {"metric_version": "0.3", "score": 0.1}, headers=api.headers
    )
    assert new_version.status_code == 201


@pytest.mark.parametrize(
    ("change", "task_type"),
    [
        ({"label_source": "user_signal"}, None),
        ({"attribution": "workflow"}, None),
        ({}, "evaluation_judge"),
    ],
)
async def test_training_eligibility(api: Api, change, task_type) -> None:
    kwargs = {"metadata": {"task_type": task_type}} if task_type else {}
    request_id = await requested(api, **kwargs)
    response = await api.client.post(
        "/v1/feedback", json=LABEL | {"request_id": request_id} | change, headers=api.headers
    )
    assert response.json()["training_eligible"] is False


async def test_feedback_validation_and_unknown_request(api: Api) -> None:
    request_id = await requested(api)
    assert (
        await api.client.post(
            "/v1/feedback", json=LABEL | {"request_id": request_id, "score": 1.5}, headers=api.headers
        )
    ).status_code == 400
    missing = await api.client.post(
        "/v1/feedback", json=LABEL | {"request_id": "mmr_01JC3ZR2A4B6C8D0E2F4G6H8J0"}, headers=api.headers
    )
    assert missing.status_code == 404


async def test_feedback_batch_per_item_results(api: Api) -> None:
    request_id = await requested(api)
    items = [
        LABEL | {"request_id": request_id},
        LABEL | {"request_id": request_id},
        LABEL | {"request_id": request_id, "score": 0.2},
        LABEL | {"request_id": "mmr_01JC3ZR2A4B6C8D0E2F4G6H8J0"},
    ]
    response = await api.client.post("/v1/feedback/batch", json={"items": items}, headers=api.headers)
    assert response.status_code == 200
    assert_matches("FeedbackBatchResponse", response.json())
    assert [r["status"] for r in response.json()["results"]] == ["stored", "replayed", "conflict", "not_found"]
    assert [r["index"] for r in response.json()["results"]] == [0, 1, 2, 3]


async def test_feedback_batch_size_limit(api: Api) -> None:
    request_id = await requested(api)
    items = [LABEL | {"request_id": request_id}] * 501
    response = await api.client.post("/v1/feedback/batch", json={"items": items}, headers=api.headers)
    assert response.status_code == 400


async def test_feedback_delete(api: Api) -> None:
    request_id = await requested(api)
    await api.client.post("/v1/feedback", json=LABEL | {"request_id": request_id}, headers=api.headers)
    await api.client.post(
        "/v1/feedback", json=LABEL | {"request_id": request_id, "metric": "other"}, headers=api.headers
    )
    record = (await api.client.get(f"/v1/requests/{request_id}", headers=api.headers)).json()
    assert record["feedback_count"] == 2
    narrowed = await api.client.delete(f"/v1/feedback/{request_id}", params={"metric": "other"}, headers=api.headers)
    assert narrowed.json() == {"deleted": 1}
    remaining = await api.client.delete(f"/v1/feedback/{request_id}", headers=api.headers)
    assert remaining.json() == {"deleted": 1}
    assert_matches("FeedbackDeleteResponse", remaining.json())
