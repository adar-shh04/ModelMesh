"""Alembic environment. Runs with psycopg's synchronous mode, using the same URL as the application."""

from __future__ import annotations

import os

from alembic import context
from sqlalchemy import create_engine, pool

from modelmesh.core.settings import Settings
from modelmesh.persistence.models import Base

target_metadata = Base.metadata


def database_url() -> str:
    return os.environ.get("MODELMESH_DATABASE_URL") or Settings().database_url


def run_migrations_offline() -> None:
    context.configure(url=database_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(database_url(), poolclass=pool.NullPool, connect_args={"connect_timeout": 10})
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
