"""Router artifact lifecycle (ADR-0004): candidate -> shadow -> active -> retired, with human approval.

Nothing promotes automatically. ``active`` needs an explicit admin call naming the approver, a shadow period,
and offline metrics that meet the gate below. P0 reports are preliminary and can never promote.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from modelmesh.core.errors import ApiError
from modelmesh.ml.artifacts import ArtifactError, RouterArtifact, artifact_sha256
from modelmesh.persistence.models import RouterModel

MAX_PROMOTABLE_ECE = 0.05
TRANSITIONS: dict[str, set[str]] = {
    "candidate": {"shadow", "retired"},
    "shadow": {"active", "candidate", "retired"},
    "active": {"retired"},
    "retired": set(),
}


def promotion_blockers(metrics: Mapping[str, Any]) -> list[str]:
    """Reasons the offline metrics do not allow promotion to active. Empty means the gate is met."""
    blockers = []
    if metrics.get("preliminary", True) is not False:
        blockers.append("report_is_preliminary")
    if metrics.get("criterion_passed") is not True:
        blockers.append("adr_0004_success_criterion_not_met")
    ece = metrics.get("ece_calibrated")
    if not isinstance(ece, int | float) or ece > MAX_PROMOTABLE_ECE:
        blockers.append(f"ece_above_{MAX_PROMOTABLE_ECE}")
    return blockers


def check_transition(current: str, target: str) -> None:
    allowed = TRANSITIONS.get(current, set())
    if target not in allowed:
        raise ApiError(
            "conflict",
            f"A router model cannot move from {current} to {target}.",
            code="invalid_transition",
            details={"from": current, "to": target, "allowed": sorted(allowed)},
        )


def as_dict(row: RouterModel) -> dict[str, Any]:
    return {
        "version": row.version,
        "algorithm": row.algorithm,
        "artifact_uri": row.artifact_uri,
        "artifact_sha256": row.artifact_sha256,
        "dataset_manifest_sha256": row.dataset_manifest_sha256,
        "metrics": row.metrics,
        "mlflow_run_id": row.mlflow_run_id,
        "status": row.status,
        "approved_by": row.approved_by,
        "approved_at": row.approved_at.isoformat() if row.approved_at else None,
        "promotion_blockers": promotion_blockers(row.metrics),
    }


def load_artifact(path: str, sha256: str | None = None) -> RouterArtifact:
    try:
        return RouterArtifact.load(Path(path), expected_sha256=sha256)
    except (OSError, ValueError, KeyError) as exc:
        raise ApiError(
            "invalid_request", f"The router artifact could not be loaded: {exc}", code="invalid_artifact"
        ) from exc


class RouterModelService:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def register(
        self, version: str, artifact_uri: str, metrics: dict[str, Any], mlflow_run_id: str | None
    ) -> dict[str, Any]:
        artifact = load_artifact(artifact_uri)
        if artifact.version != version:
            raise ApiError("invalid_request", "The version does not match the artifact.", code="version_mismatch")
        async with self._sessions() as session, session.begin():
            if await session.get(RouterModel, version) is not None:
                raise ApiError("conflict", "This router version is already registered.", code="already_registered")
            row = RouterModel(
                version=version,
                algorithm=artifact.algorithm,
                artifact_uri=artifact_uri,
                artifact_sha256=artifact_sha256(Path(artifact_uri)),
                dataset_manifest_sha256=str(artifact.document.get("dataset_manifest_sha256", "")),
                metrics=metrics,
                mlflow_run_id=mlflow_run_id,
                status="candidate",
            )
            session.add(row)
            return as_dict(row)

    async def list_models(self) -> list[dict[str, Any]]:
        async with self._sessions() as session:
            rows = (await session.execute(select(RouterModel).order_by(RouterModel.created_at))).scalars()
            return [as_dict(row) for row in rows]

    async def transition(
        self, version: str, target: str, *, approved_by: str | None = None
    ) -> tuple[dict[str, Any], RouterArtifact | None]:
        """Moves one version. A new shadow or active version demotes the previous holder of that status.

        Returns the row and, when the target is served (shadow or active), the verified artifact to install.
        """
        async with self._sessions() as session, session.begin():
            row = await session.get(RouterModel, version, with_for_update=True)
            if row is None:
                raise ApiError("not_found", "No such router version.", code="router_model_not_found")
            check_transition(row.status, target)
            if target == "active":
                blockers = promotion_blockers(row.metrics)
                if not approved_by:
                    blockers.append("approval_missing")
                if blockers:
                    raise ApiError(
                        "conflict",
                        "Promotion to active is blocked.",
                        code="promotion_blocked",
                        details={"blockers": blockers},
                    )
            artifact = None
            if target in ("active", "shadow"):
                artifact = load_artifact(row.artifact_uri, row.artifact_sha256)  # still the registered file
                others = await session.execute(select(RouterModel).where(RouterModel.status == target))
                for other in others.scalars():
                    if other.version != version:
                        other.status = "retired" if target == "active" else "candidate"
                        other.updated_at = datetime.now(UTC)
            row.status = target
            row.updated_at = datetime.now(UTC)
            if target == "active":
                row.approved_by, row.approved_at = approved_by, datetime.now(UTC)
            return as_dict(row), artifact

    async def runtime_artifacts(self) -> dict[str, RouterArtifact]:
        """The artifacts to serve at start-up, by status (active, shadow). A changed file is skipped."""
        loaded: dict[str, RouterArtifact] = {}
        async with self._sessions() as session:
            rows = await session.execute(select(RouterModel).where(RouterModel.status.in_(("active", "shadow"))))
            for row in rows.scalars():
                try:
                    loaded[row.status] = RouterArtifact.load(
                        Path(row.artifact_uri), expected_sha256=row.artifact_sha256
                    )
                except (OSError, ArtifactError, ValueError, KeyError):
                    continue
        return loaded
