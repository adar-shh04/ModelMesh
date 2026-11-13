"""M7 dashboard: server-rendered pages over the same service-layer queries as the API.

Acceptance (implementation plan, M7): pages render against seeded data; every number equals the API's figure from the
same query; the admin token is required; prompt content is shown only when retention is on. Also: no provider call.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from typing import Any

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import select

from conftest import ROOMY_LIMITS, Api
from doubles import make_model, reply
from modelmesh.accounting.queries import GROUP_COLUMNS
from modelmesh.api.dashboard import SESSION_COOKIE, USAGE_FIELDS
from modelmesh.core.errors import ErrorClass, ProviderError
from modelmesh.persistence import models as db
from modelmesh.providers.types import TokenUsage

pytestmark = [pytest.mark.anyio, pytest.mark.integration]

ADMIN = {"Authorization": "Bearer test-admin-token"}
PAGES = ["/dashboard/usage", "/dashboard/reliability", "/dashboard/requests", "/dashboard/keys", "/dashboard/routing",
         "/dashboard/benchmarks"]  # fmt: skip


class Tables(HTMLParser):
    """Collects `data-field` cells by table id and row `data-group`: {table: {group: {field: (value, text)}}}."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: dict[str, dict[str, dict[str, tuple[str, str]]]] = {}
        self._table: str | None = None
        self._group: str | None = None
        self._cell: tuple[str, str] | None = None
        self._text: list[str] = []
        self.inputs: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {name: value or "" for name, value in attrs}
        if tag == "table":
            self._table = attributes.get("id", "")
            self.tables.setdefault(self._table, {})
        elif tag == "tr" and "data-group" in attributes:
            self._group = attributes["data-group"]
        elif tag == "td" and "data-field" in attributes:
            self._cell = (attributes["data-field"], attributes.get("data-value", ""))
            self._text = []
        elif tag == "input" and "name" in attributes:
            self.inputs.setdefault(attributes["name"], attributes.get("value", ""))

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self._cell is not None and self._table is not None and self._group is not None:
            field, value = self._cell
            self.tables[self._table].setdefault(self._group, {})[field] = (value, "".join(self._text).strip())
            self._cell = None
        elif tag == "tr":
            self._group = None
        elif tag == "table":
            self._table = None


def parse(html: str) -> Tables:
    parser = Tables()
    parser.feed(html)
    return parser


def assert_same(page_value: str, api_value: Any) -> None:
    """A page's `data-value` must be exactly the API's value: same number, same null."""
    if api_value is None:
        assert page_value == ""
    elif isinstance(api_value, bool):
        assert page_value == ("true" if api_value else "false")
    elif isinstance(api_value, int | float):
        assert float(page_value) == float(api_value)
    else:
        assert page_value == str(api_value)


def window() -> dict[str, str]:
    now = datetime.now(UTC)
    return {"from": (now - timedelta(minutes=5)).isoformat(), "to": (now + timedelta(minutes=5)).isoformat()}


async def seed(api: Api) -> list[str]:
    """Successes on two models with different prices, an upstream failure, and a rejected request."""
    api.only_models(make_model("a"), make_model("b"), prices={"t/a": (1_000, 2_000), "t/b": (3_000, 500)})
    api.scripted.script("a", reply("one", usage=TokenUsage(100, 40)), reply("two", usage=TokenUsage(60, 10)))
    api.scripted.script("b", reply("three", usage=TokenUsage(30, 70)))
    api.scripted.script("b", ProviderError(ErrorClass.AUTH_ERROR, "upstream", http_status=401))
    ids = []
    for model, task in (("t/a", "general"), ("t/a", "query_classification"), ("t/b", "general"), ("t/b", "general")):
        metadata = {"task_type": task, "application": "dash", "correlation_id": "c1"}
        response = await api.chat(model=model, metadata=metadata)
        ids.append(response.headers["x-modelmesh-request-id"])
    rejected = await api.chat(n=2)
    assert rejected.status_code == 400
    ids.append(rejected.headers["x-modelmesh-request-id"])
    return ids


# ---------------------------------------------------------------- access
async def test_every_page_requires_the_admin_token(api: Api) -> None:
    for path in PAGES:
        for headers in ({}, {"Authorization": "Bearer wrong"}, api.headers):
            response = await api.client.get(path, params={"tenant_id": api.tenant_id}, headers=headers)
            assert response.status_code == 401, (path, headers)
            assert "Sign in" in response.text
            assert "<table" not in response.text
    api.client.cookies.set(SESSION_COOKIE, "forged", domain="mesh.test", path="/dashboard")
    forged = await api.client.post("/dashboard/keys", content=f"tenant_id={api.tenant_id}&name=x")
    assert forged.status_code == 401
    async with api.services.sessions() as session:
        keys = (await session.execute(select(db.ApiKey).where(db.ApiKey.tenant_id == api.tenant_id))).scalars().all()
    assert len(keys) == 1


async def test_login_sets_a_cookie_derived_from_the_token(api: Api) -> None:
    bad = await api.client.post("/dashboard/login", content="token=nope")
    assert bad.status_code == 401
    assert SESSION_COOKIE not in bad.cookies

    good = await api.client.post("/dashboard/login", content="token=test-admin-token")
    assert good.status_code == 303
    header = good.headers["set-cookie"]
    assert "HttpOnly" in header
    assert "samesite=strict" in header.lower()
    assert "test-admin-token" not in header
    assert api.client.cookies.get(SESSION_COOKIE)  # kept by the client, sent back on /dashboard paths
    page = await api.client.get("/dashboard/usage")
    assert page.status_code == 200
    assert page.headers["cache-control"] == "no-store"
    assert "script-src" not in page.headers["content-security-policy"]  # no scripts are allowed at all

    csrf = parse(page.text).inputs["csrf"]
    assert (await api.client.post("/dashboard/logout", content="csrf=bad")).status_code == 403
    out = await api.client.post("/dashboard/logout", content=f"csrf={csrf}")
    assert out.status_code == 303
    assert not api.client.cookies.get(SESSION_COOKIE)
    assert (await api.client.get("/dashboard/usage")).status_code == 401


async def test_dashboard_stays_closed_if_the_admin_token_is_empty(api: Api) -> None:
    # Start-up already refuses an empty admin token; the dashboard must not open even if that check is bypassed.
    api.services.settings.admin_token = SecretStr("")
    for headers in ({}, {"Authorization": "Bearer "}, {"Authorization": "Bearer"}):
        response = await api.client.get("/dashboard/usage", headers=headers)
        assert response.status_code == 503
        assert "MODELMESH_ADMIN_TOKEN is not set" in response.text
    login = await api.client.post("/dashboard/login", content="token=")
    assert login.status_code == 503


# ---------------------------------------------------------------- rendering
async def test_pages_render_against_seeded_data(api: Api) -> None:
    ids = await seed(api)
    await api.admin.create_tenant("<script>alert(1)</script>")
    for path in PAGES:
        response = await api.client.get(path, params={"tenant_id": api.tenant_id}, headers=ADMIN)
        assert response.status_code == 200, path
        assert response.headers["content-type"].startswith("text/html")
        assert "<script>alert(1)</script>" not in response.text  # escaped
        assert "<script" not in response.text  # no JavaScript anywhere

    usage = parse((await api.client.get("/dashboard/usage", params=window(), headers=ADMIN)).text)
    assert set(usage.tables["usage"]) == {"t/a", "t/b", "unknown", "__total__"}

    listing = await api.client.get("/dashboard/requests", params={"correlation_id": "c1"}, headers=ADMIN)
    for request_id in ids[:4]:
        assert request_id in listing.text
    assert ids[4] not in listing.text  # the rejected request carried no correlation id

    async with api.services.sessions() as session:
        trace_id = (await session.get(db.RequestRecord, ids[0])).trace_id
    detail = await api.client.get(f"/dashboard/requests/{ids[0]}", headers=ADMIN)
    assert detail.status_code == 200
    assert f"http://127.0.0.1:16686/trace/{trace_id}" in detail.text
    failed = await api.client.get(f"/dashboard/requests/{ids[3]}", headers=ADMIN)
    assert "auth_error" in failed.text  # the attempt outcome

    routing = parse((await api.client.get("/dashboard/routing", headers=ADMIN)).text)
    assert set(routing.tables["catalog"]) == {"t/a", "t/b"}
    assert routing.tables["catalog"]["t/a"]["status"][1] == "available"
    assert routing.tables["tasks"]["query_classification"]["calibrated"][1] == "no"

    missing = await api.client.get("/dashboard/requests/mmr_01ARZ3NDEKTSV4RRFFQ69G5FAV", headers=ADMIN)
    assert missing.status_code == 404
    bad = await api.client.get("/dashboard/usage", params={"from": "yesterday"}, headers=ADMIN)
    assert bad.status_code == 400


async def test_request_from_another_tenant_is_not_shown(api: Api) -> None:
    ids = await seed(api)
    other = await api.admin.create_tenant("other")
    response = await api.client.get(f"/dashboard/requests/{ids[0]}", params={"tenant_id": other["id"]}, headers=ADMIN)
    assert response.status_code == 404


# ---------------------------------------------------------------- numbers equal the API
@pytest.mark.parametrize("group_by", list(GROUP_COLUMNS))
async def test_usage_page_numbers_equal_the_usage_api(api: Api, group_by: str) -> None:
    await seed(api)
    params = window() | {"group_by": group_by}
    report = (await api.client.get("/v1/usage", params=params, headers=api.headers)).json()
    page = await api.client.get("/dashboard/usage", params=params | {"tenant_id": api.tenant_id}, headers=ADMIN)
    table = parse(page.text).tables["usage"]

    expected = {row["group"]: row for row in report["data"]} | {"__total__": report["totals"]}
    assert set(table) == set(expected)
    assert report["totals"]["requests"] == 5
    for group, row in expected.items():
        assert set(table[group]) == {field for field, _, _ in USAGE_FIELDS}
        for field, _, kind in USAGE_FIELDS:
            value, text = table[group][field]
            assert_same(value, row[field])
            if kind == "int" and row[field] is not None:
                assert text == str(row[field])


async def test_reliability_page_numbers_equal_the_api(api: Api) -> None:
    await seed(api)
    params = window() | {"tenant_id": api.tenant_id}
    page = parse((await api.client.get("/dashboard/reliability", params=params, headers=ADMIN)).text)
    providers = (await api.client.get("/v1/providers", headers=api.headers)).json()["data"]
    assert set(page.tables["providers"]) == {row["id"] for row in providers}
    for row in providers:
        for field, (value, _) in page.tables["providers"][row["id"]].items():
            assert_same(value, row[field])

    report = (await api.client.get("/v1/usage", params=window() | {"group_by": "model"}, headers=api.headers)).json()
    by_model = {row["group"]: row for row in report["data"]}
    assert set(page.tables["models"]) == set(by_model)
    for group, cells in page.tables["models"].items():
        for field, (value, _) in cells.items():
            assert_same(value, by_model[group][field])

    models = {row["id"]: row for row in (await api.client.get("/v1/models", headers=api.headers)).json()["data"]}
    for model_id, cells in page.tables["breakers"].items():
        assert cells["status"][0] == models[model_id]["modelmesh"]["status"]
        assert cells["circuit_state"][0] == api.services.breakers.for_model(model_id).state


# ---------------------------------------------------------------- content
async def test_prompt_content_is_shown_only_when_retention_is_on(api: Api) -> None:
    retained_key = await api.new_key(content_retention=True, limits=ROOMY_LIMITS)
    body = {"model": "fake/echo-small", "messages": [{"role": "user", "content": "kept-prompt-7f3"}]}
    kept = await api.client.post("/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {retained_key}"})
    plain = await api.chat(messages=[{"role": "user", "content": "private-prompt-9c1"}])
    kept_id, plain_id = kept.headers["x-modelmesh-request-id"], plain.headers["x-modelmesh-request-id"]

    plain_page = await api.client.get(f"/dashboard/requests/{plain_id}", headers=ADMIN)
    assert plain_page.status_code == 200
    assert "private-prompt-9c1" not in plain_page.text
    assert 'id="no-content"' in plain_page.text

    kept_page = await api.client.get(f"/dashboard/requests/{kept_id}", headers=ADMIN)
    assert "kept-prompt-7f3" in kept_page.text
    assert 'id="content-messages"' in kept_page.text

    listing = await api.client.get("/dashboard/requests", headers=ADMIN)
    assert kept_id in listing.text
    assert plain_id in listing.text
    assert "kept-prompt-7f3" not in listing.text  # the list never shows content
    assert "private-prompt-9c1" not in listing.text


# ---------------------------------------------------------------- keys
async def test_key_management_creates_rotates_and_revokes(api: Api) -> None:
    keys_page = await api.client.get("/dashboard/keys", headers=ADMIN)
    csrf = parse(keys_page.text).inputs["csrf"]

    refused = await api.client.post("/dashboard/keys", content=f"tenant_id={api.tenant_id}&name=x", headers=ADMIN)
    assert refused.status_code == 403

    created = await api.client.post(
        "/dashboard/keys", content=f"csrf={csrf}&tenant_id={api.tenant_id}&name=dash&env=test", headers=ADMIN
    )
    assert created.status_code == 200
    assert created.headers["cache-control"] == "no-store"
    secret = created.text.split('id="issued-secret">', 1)[1].split("<", 1)[0]
    auth = {"Authorization": f"Bearer {secret}"}
    assert (await api.client.get("/v1/models", headers=auth)).status_code == 200
    listed = await api.admin.list_keys(api.tenant_id)
    new_key = next(key for key in listed if key["name"] == "dash")
    assert new_key["limits"]["rpm"] == api.services.settings.default_key_rpm  # D1 defaults for new keys
    again = await api.client.get("/dashboard/keys", headers=ADMIN)
    assert secret not in again.text  # shown once only

    rotated = await api.client.post(
        f"/dashboard/keys/{new_key['id']}/rotate", content=f"csrf={csrf}&tenant_id={api.tenant_id}", headers=ADMIN
    )
    rotated_secret = rotated.text.split('id="issued-secret">', 1)[1].split("<", 1)[0]
    assert rotated_secret != secret
    assert (await api.client.get("/v1/models", headers=auth)).status_code == 401
    auth = {"Authorization": f"Bearer {rotated_secret}"}
    assert (await api.client.get("/v1/models", headers=auth)).status_code == 200

    revoked = await api.client.post(
        f"/dashboard/keys/{new_key['id']}/revoke", content=f"csrf={csrf}&tenant_id={api.tenant_id}", headers=ADMIN
    )
    assert revoked.status_code == 303
    assert (await api.client.get("/v1/models", headers=auth)).status_code == 401
    page = parse((await api.client.get("/dashboard/keys", headers=ADMIN)).text)
    assert page.tables["keys"][new_key["id"]]["status"][1] == "revoked"

    other = await api.admin.create_tenant("other")
    cross = await api.client.post(
        f"/dashboard/keys/{api.key_id}/revoke", content=f"csrf={csrf}&tenant_id={other['id']}", headers=ADMIN
    )
    assert cross.status_code == 400
    assert (await api.client.get("/v1/models", headers=api.headers)).status_code == 200


# ---------------------------------------------------------------- no provider calls
async def test_rendering_never_calls_a_provider(api: Api, monkeypatch: pytest.MonkeyPatch) -> None:
    ids = await seed(api)
    calls_before = len(api.scripted.requests)

    async def refuse(self: Any, request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"network call to {request.url.host} while rendering the dashboard")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", refuse)
    for path in [*PAGES, f"/dashboard/requests/{ids[0]}"]:
        response = await api.client.get(path, params={"tenant_id": api.tenant_id}, headers=ADMIN)
        assert response.status_code == 200, path
    assert len(api.scripted.requests) == calls_before
