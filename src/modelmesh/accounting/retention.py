"""Periodic deletion of expired content and old execution records (ADR-0010)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import anyio
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from modelmesh.persistence import models as db

logger = logging.getLogger("modelmesh.retention")

METADATA_RETENTION_DAYS = 90
SWEEP_INTERVAL_S = 3600.0


async def purge_expired(
    sessions: async_sessionmaker[AsyncSession],
    *,
    now: datetime | None = None,
    metadata_days: int = METADATA_RETENTION_DAYS,
) -> tuple[int, int]:
    """Delete expired prompt and response content and old request records. Returns the two counts."""
    current = now or datetime.now(UTC)
    async with sessions.begin() as session:
        content = await session.execute(delete(db.RequestContent).where(db.RequestContent.expires_at <= current))
        records = await session.execute(
            delete(db.RequestRecord).where(db.RequestRecord.created_at < current - timedelta(days=metadata_days))
        )
    return int(content.rowcount or 0), int(records.rowcount or 0)  # type: ignore[attr-defined]


async def retention_loop(sessions: async_sessionmaker[AsyncSession]) -> None:
    while True:
        try:
            await purge_expired(sessions)
        except Exception:  # Never let housekeeping take the service down.
            logger.exception("retention sweep failed")
        await anyio.sleep(SWEEP_INTERVAL_S)
