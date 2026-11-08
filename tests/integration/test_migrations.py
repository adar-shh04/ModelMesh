"""Alembic round trip against real Postgres (ADR-0010)."""

from __future__ import annotations

import psycopg
import pytest
from alembic import command
from alembic.util.exc import CommandError

from conftest import PG_ADMIN_DSN, alembic_config

pytestmark = pytest.mark.integration

DB = "modelmesh_migration_test"
URL = PG_ADMIN_DSN.replace("postgresql://", "postgresql+psycopg://").rsplit("/", 1)[0] + f"/{DB}"


def recreate() -> None:
    with psycopg.connect(PG_ADMIN_DSN, autocommit=True, connect_timeout=5) as connection:
        connection.execute(f'DROP DATABASE IF EXISTS "{DB}" WITH (FORCE)')
        connection.execute(f'CREATE DATABASE "{DB}"')


def test_upgrade_downgrade_upgrade_and_no_drift() -> None:
    recreate()
    config = alembic_config(URL)
    command.upgrade(config, "head")
    command.downgrade(config, "base")
    command.upgrade(config, "head")
    command.check(config)  # raises if models and migrations differ


def test_check_detects_drift_when_not_upgraded() -> None:
    recreate()
    with pytest.raises(CommandError):
        command.check(alembic_config(URL))
