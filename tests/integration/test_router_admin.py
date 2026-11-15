"""M6 against real Postgres: register, shadow (logs only, traffic unchanged) and gated promotion with approval."""

from __future__ import annotations

import logging
from pathlib import Path

import anyio
import pytest

from conftest import Api
from router_fixtures import write_artifact

pytestmark = [pytest.mark.anyio, pytest.mark.integration]

ADMIN = {"Authorization": "Bearer test-admin-token"}
AUTO = {"model": "auto", "routing": {"policy": "cost_optimized"}}


class StubEmbedder:
    async def embed(self, text: str) -> list[float]:
        return [0.0, 1.0]


async def register(api: Api, tmp_path: Path, version: str = "test-router", **metrics: object) -> None:
    path = write_artifact(tmp_path / f"{version}.json", version)
    body = {"version": version, "artifact_uri": str(path), "metrics": {"preliminary": True, **metrics}}
    response = await api.client.post("/admin/v1/router-models", json=body, headers=ADMIN)
    assert response.status_code == 201, response.text
    assert response.json()["status"] == "candidate"


async def test_router_admin_requires_the_admin_token(api: Api) -> None:
    assert (await api.client.get("/admin/v1/router-models")).status_code == 401
    assert (await api.client.get("/admin/v1/router-models", headers=api.headers)).status_code == 401


async def test_shadow_mode_logs_decisions_without_changing_traffic(
    api: Api, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    api.services.router_runtime.embedder = StubEmbedder()
    before = await api.chat(**AUTO)
    assert before.status_code == 200, before.text
    await register(api, tmp_path)
    shadow = await api.client.post("/admin/v1/router-models/test-router/shadow", headers=ADMIN)
    assert shadow.status_code == 200 and shadow.json()["status"] == "shadow"
    assert api.services.router_runtime.shadow is not None
    with caplog.at_level(logging.INFO, logger="modelmesh.pipeline"):
        after = await api.chat(**AUTO)
    assert after.status_code == 200, after.text
    assert after.json()["model"] == before.json()["model"]
    assert after.json()["modelmesh"]["routing"]["mode"] == before.json()["modelmesh"]["routing"]["mode"]
    assert after.json()["modelmesh"]["routing"]["router_version"] is None  # no learned router is serving
    for _ in range(50):  # the shadow decision is computed after the response, off the request path
        if any(getattr(r, "routing_shadow", None) for r in caplog.records):
            break
        await anyio.sleep(0.02)
    records = [r.routing_shadow for r in caplog.records if getattr(r, "routing_shadow", None)]  # type: ignore[attr-defined]
    assert records and records[0]["router_version"] == "test-router"
    assert records[0]["live_choice"] == after.json()["model"]


async def test_promotion_needs_approval_and_a_passing_non_preliminary_report(api: Api, tmp_path: Path) -> None:
    await register(api, tmp_path)
    direct = await api.client.post(
        "/admin/v1/router-models/test-router/promote", json={"approved_by": "po", "confirm": True}, headers=ADMIN
    )
    assert direct.status_code == 409 and direct.json()["error"]["code"] == "invalid_transition"
    await api.client.post("/admin/v1/router-models/test-router/shadow", headers=ADMIN)
    unapproved = await api.client.post("/admin/v1/router-models/test-router/promote", json={}, headers=ADMIN)
    assert unapproved.status_code == 400
    blocked = await api.client.post(
        "/admin/v1/router-models/test-router/promote", json={"approved_by": "po", "confirm": True}, headers=ADMIN
    )
    assert blocked.status_code == 409
    assert "report_is_preliminary" in blocked.json()["error"]["details"]["blockers"]
    assert api.services.router_runtime.active is None

    await register(api, tmp_path, "passing", preliminary=False, criterion_passed=True, ece_calibrated=0.03)
    await api.client.post("/admin/v1/router-models/passing/shadow", headers=ADMIN)
    promoted = await api.client.post(
        "/admin/v1/router-models/passing/promote", json={"approved_by": "po", "confirm": True}, headers=ADMIN
    )
    assert promoted.status_code == 200, promoted.text
    assert promoted.json()["status"] == "active" and promoted.json()["approved_by"] == "po"
    assert api.services.router_runtime.active is not None
    listed = (await api.client.get("/admin/v1/router-models", headers=ADMIN)).json()["data"]
    assert {row["version"]: row["status"] for row in listed} == {"test-router": "candidate", "passing": "active"}
    retired = await api.client.post("/admin/v1/router-models/passing/retire", headers=ADMIN)
    assert retired.json()["status"] == "retired"
    assert api.services.router_runtime.active is None
