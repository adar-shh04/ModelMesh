#!/usr/bin/env bash
# Runs the same gates as CI. Postgres and Redis must be up: docker compose up -d postgres redis
set -euo pipefail
uv run ruff check src tests migrations
uv run ruff format --check src tests migrations
uv run mypy
uv run lint-imports
uv run pytest tests "$@"
