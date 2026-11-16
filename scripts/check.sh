#!/usr/bin/env bash
# Runs the same gates as CI. Postgres and the test Redis must be up: docker compose --profile test up -d postgres redis-test
set -euo pipefail
uv run ruff check src tests migrations loadtests
uv run ruff format --check src tests migrations loadtests
uv run mypy
uv run lint-imports
uv run pytest tests "$@"
