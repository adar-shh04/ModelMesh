"""Async engine and session factory."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine


def make_engine(url: str) -> AsyncEngine:
    return create_async_engine(
        url, pool_size=10, max_overflow=10, pool_pre_ping=True, connect_args={"connect_timeout": 5}
    )


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)
