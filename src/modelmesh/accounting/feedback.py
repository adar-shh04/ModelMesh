"""Quality labels from clients. They are stored as candidate training data only (contract section 5)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from modelmesh.contract.schemas import FeedbackRequest
from modelmesh.core.errors import ApiError
from modelmesh.core.ids import new_feedback_id
from modelmesh.persistence import models as db


def _is_training_eligible(item: FeedbackRequest, request_task_type: str | None) -> bool:
    return (
        item.label_source != "user_signal"
        and item.attribution == "direct"
        and (request_task_type or item.task_type) != "evaluation_judge"
    )


def _response(row: db.Feedback, status: str) -> dict[str, Any]:
    return {
        "id": row.id,
        "request_id": row.request_id,
        "status": status,
        "training_eligible": row.training_eligible,
        "created_at": row.created_at,
    }


def _not_found() -> ApiError:
    return ApiError("not_found", "Unknown request id.", code="request_not_found", param="request_id")


class FeedbackService:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def submit(self, tenant_id: str, item: FeedbackRequest) -> dict[str, Any]:
        """Store a label. Returns status `stored` or `replayed`; raises ApiError otherwise."""
        version = item.metric_version or ""
        async with self._sessions() as session:
            task_type = (
                await session.execute(
                    select(db.RequestRecord.task_type).where(
                        db.RequestRecord.id == item.request_id, db.RequestRecord.tenant_id == tenant_id
                    )
                )
            ).first()
            if task_type is None:
                raise _not_found()
            existing = await self._find(session, item, version)
            if existing is not None:
                return self._replay_or_conflict(existing, item)
            row = db.Feedback(
                id=new_feedback_id(),
                request_id=item.request_id,
                tenant_id=tenant_id,
                label_source=item.label_source,
                metric=item.metric,
                metric_version=version,
                score=item.score,
                higher_is_better=item.higher_is_better,
                accepted=item.accepted,
                attribution=item.attribution,
                evaluator=item.evaluator,
                eval_run_id=item.eval_run_id,
                dataset_version=item.dataset_version,
                labelled_at=item.labelled_at,
                task_type=item.task_type,
                notes=item.notes,
                training_eligible=_is_training_eligible(item, task_type[0]),
            )
            session.add(row)
            try:
                await session.commit()
            except IntegrityError:  # Two identical submissions raced; the loser is treated as a replay.
                await session.rollback()
                existing = await self._find(session, item, version)
                if existing is None:
                    raise
                return self._replay_or_conflict(existing, item)
            await session.refresh(row)
            return _response(row, "stored")

    @staticmethod
    async def _find(session: AsyncSession, item: FeedbackRequest, version: str) -> db.Feedback | None:
        statement = select(db.Feedback).where(
            db.Feedback.request_id == item.request_id,
            db.Feedback.metric == item.metric,
            db.Feedback.metric_version == version,
            db.Feedback.evaluator == item.evaluator,
        )
        return (await session.execute(statement)).scalar_one_or_none()

    @staticmethod
    def _replay_or_conflict(existing: db.Feedback, item: FeedbackRequest) -> dict[str, Any]:
        if existing.score != item.score or existing.accepted != item.accepted:
            raise ApiError(
                "conflict",
                "Feedback for this request, metric and evaluator already exists with a different value.",
                code="feedback_conflict",
            )
        return _response(existing, "replayed")

    async def delete(
        self,
        tenant_id: str,
        request_id: str,
        *,
        metric: str | None,
        evaluator: str | None,
        metric_version: str | None,
    ) -> int:
        async with self._sessions.begin() as session:
            owned = await session.execute(
                select(db.RequestRecord.id).where(
                    db.RequestRecord.id == request_id, db.RequestRecord.tenant_id == tenant_id
                )
            )
            if owned.first() is None:
                raise _not_found()
            statement = delete(db.Feedback).where(
                db.Feedback.request_id == request_id, db.Feedback.tenant_id == tenant_id
            )
            if metric is not None:
                statement = statement.where(db.Feedback.metric == metric)
            if evaluator is not None:
                statement = statement.where(db.Feedback.evaluator == evaluator)
            if metric_version is not None:
                statement = statement.where(db.Feedback.metric_version == metric_version)
            result = await session.execute(statement)
            return int(result.rowcount or 0)  # type: ignore[attr-defined]
