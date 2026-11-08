# ADR-0002: Python version, tooling, FastAPI and Pydantic v2

- Status: Accepted (architecture phase)
- Date: 2026-11-08
- Spec refs: §2.5 (stack), §5.2 (type-checked contracts, CI, reproducibility)

## Context

The machine has only CPython 3.14.3 and no uv (see ENVIRONMENT.md). The spec's stack is Python, FastAPI, scikit-learn, Redis, PostgreSQL, MLflow, OpenTelemetry, pytest, HTTPX and Locust. A PyPI wheel survey on 2026-11-08 found Windows wheels for both cp313 and cp314 for every planned dependency: pydantic-core 2.50, SQLAlchemy 2.1, psycopg-binary 3.3, numpy 2.5, scikit-learn 1.9, LightGBM 4.7 (ships a `py3-none-win_amd64` wheel), onnxruntime 1.31, gevent 26.9, mypy 2.4, and others. Some projects (gevent, httpx, mlflow, pyright) still don't list 3.14 in their classifiers, so they have probably been tested less on it.

## Options considered

| Option | Pros | Cons |
|---|---|---|
| System 3.14.3 + pip/venv | Nothing to install. `uuid.uuid7()` is in the stdlib | Newest interpreter, so a stray transitive dependency without a cp314 wheel would need MSVC to build. No lockfile tool |
| **3.13 managed by uv** (chosen) | Every dependency has had cp313 wheels for more than a year. uv gives a cross-platform lockfile, fast installs, and pins the interpreter without touching the system | Requires installing uv (one small binary) |
| 3.12 via uv | Most conservative | Gains nothing over 3.13: numpy, scipy and xgboost already need ≥3.12 |
| Poetry / pip-tools | Familiar | Slower. Poetry doesn't manage Python versions |

## Decision

- **Python 3.13** (`requires-python = ">=3.13,<3.15"`), pinned with `uv python pin 3.13`. CI also runs the unit tests on 3.14 as a non-blocking job, to prepare an upgrade.
- **uv** for environments and the lockfile (`uv.lock` is committed). Dependency groups: `dev` (lint, type checking, tests), `ml` (offline training and benchmarking), `load` (Locust). The runtime image installs only the default group.
- **FastAPI + Pydantic v2.** Contract models in `api/schemas` are the single source of truth for validation. A CI test compares the OpenAPI that FastAPI generates against `docs/contract/openapi-v1.yaml` and fails on any difference in paths, schemas or required fields. FastAPI is the spec's choice. It is async-native (needed for streaming and concurrent upstream calls) and its OpenAPI generation supports contract conformance tests.
- **Uvicorn** as the ASGI server. On Linux, uvloop and httptools come with `uvicorn[standard]`. uvloop has no Windows wheels and is skipped there automatically.
- **Ruff** for lint and format, replacing flake8, isort and black with one tool.
- **mypy `--strict`** with the `pydantic.mypy` plugin as the CI type gate. Pyright is not added as a second gate: two checkers disagree on edge cases and double the maintenance. Editors may use Pylance informally.
- **pytest** with the **anyio** plugin, which ships with anyio (already a Starlette/httpx dependency), instead of adding pytest-asyncio. **respx** mocks upstream HTTPX calls in adapter tests. **import-linter** enforces ADR-0001.
- **No** structlog, tenacity or pybreaker. Stdlib `logging` with a roughly 40-line JSON formatter (trace-id injection through contextvars) is enough. Retry and circuit breaking are tied to the fallback and deadline logic (ADR-0008), so a small hand-written implementation is clearer than adapting a generic library. aiobreaker is unmaintained (last release 2021).

### Runtime dependencies (justified)

| Package | Why |
|---|---|
| fastapi, uvicorn | API layer (spec) |
| pydantic, pydantic-settings | Contract models. Typed env configuration (spec §5.2, centralized config) |
| httpx | Upstream provider calls with per-phase timeouts and streaming. Used directly instead of provider SDKs (ADR-0003) |
| sqlalchemy[asyncio], psycopg[binary], alembic | Postgres ORM/Core, one driver for sync migrations and async app use, and migrations (ADR-0010) |
| redis | Rate limits, quotas, exact and semantic cache (ADR-0006, ADR-0010) |
| jsonschema | Validating structured outputs against the caller's schema (ADR-0007) |
| fastembed (onnxruntime) | CPU embeddings (`BAAI/bge-small-en-v1.5`, MIT, 384-d) for router features and the semantic cache, without torch |
| numpy | Feature vectors, kNN, calibration tables |
| lightgbm | Inference for the GBDT router, only if it wins in ADR-0004. Otherwise it moves to the `ml` group |
| opentelemetry-api/-sdk/-exporter-otlp-proto-http, -instrumentation-fastapi/-httpx | Traces and metrics (ADR-0009). The HTTP exporter avoids grpcio |
| python-ulid | Sortable `mmr_` request ids on 3.13 (no stdlib uuid7 before 3.14) |
| jinja2 | Minimal dashboard templates (M7) |

`ml` group: scikit-learn, pandas, pyarrow, datasets (HF), mlflow. `load` group: locust.

## Consequences

- The developer must install uv once (`py -m pip install --user uv`). The system Python stays untouched.
- The Docker image uses `python:3.13-slim` and `uv sync --frozen --no-dev`.
- Hand-written retry and breaker code needs thorough unit tests (ADR-0008).

## Verification plan

- M1: `uv sync` on Windows and inside the Linux image both succeed from the same `uv.lock`.
- M1: CI runs `ruff check`, `ruff format --check`, `mypy --strict src`, `lint-imports`, and `pytest -m "not live"`.
- M1: an OpenAPI conformance test (generated spec compared with `openapi-v1.yaml`) is green.
