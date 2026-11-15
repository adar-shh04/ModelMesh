# Phase 2 verification notes (M1 to M4, M5 harness)

Recorded 2026-11-09 from commands actually run on the development machine (Windows 11, Docker Desktop capped at 4 GB).
"Not run" means exactly that. No live provider was called, because no API keys exist and Ollama is not installed.

## How to reproduce

```bash
uv sync --group ml
docker compose up -d postgres redis
./scripts/check.sh
```

Latest gate results are in the table at the end of this file.

## M1: skeleton, one adapter, unified API

| Acceptance test | Result |
|---|---|
| FastAPI routes equal `openapi-v1.yaml`; Pydantic models match its properties and required fields | Passed (`tests/contract`) |
| Redocly or Spectral lint of the contract | **Not run.** Downloading the linter through `npx` was refused by the sandbox's permission check |
| Official `openai` SDK against the fake provider: non-stream, stream with `include_usage` true and false, tools, in-stream error, unknown field gives 400 | Passed |
| Ollama adapter fixtures | Passed |
| Ollama adapter against a real Ollama | **Not run** (not installed) |
| Span tree, no content attributes, `traceparent` continuation, `trace_id` in body | Passed (in-memory exporter) |
| Trace visible in Jaeger | Passed: 9 spans for one `traceparent` through the Compose stack |
| `docker compose up`: health `ok` | Passed in 15 s with images built and the volume present. A clone on a machine with no images was not timed |
| Reachable from another container through `host.docker.internal:8100` | Passed |
| Stack memory | API about 100 MB idle and 326 MB with the embedding model loaded; Postgres about 75 MB; Redis about 15 MB; Jaeger about 15 MB |
| Import boundaries; runtime does not load sklearn, mlflow or datasets | Passed |

## M2: all adapters, reliability, rules-based routing

| Acceptance test | Result |
|---|---|
| Per-provider fixtures for every ADR-0003 case | Passed |
| Live smoke per configured model | **Not run**: 24 live tests skipped (no keys, no Ollama). Every UNVERIFIED cell in ADR-0003 stays unverified |
| Breaker state machine, backoff bounds, taxonomy table, `Retry-After`, deadline exhaustion | Passed |
| Fault injection: fallback never violates `data_policy`, cost or capability; explicit model gets no fallback | Passed |
| Each policy on a synthetic catalog; `no_eligible_model` details | Passed |
| Uncalibrated task types under every policy (contract section 2.1.1) | Passed |
| Structured output: native, best-effort, emulated, repair, 502, buffered stream | Passed |
| Measured Ollama throughput on this GPU | **Not done** (not installed) |

## M3: accounting, keys, quotas, feedback

| Acceptance test | Result |
|---|---|
| Every request, including rejected ones, produces a record | Passed |
| `/v1/usage` matches a hand-computed fixture; estimated and reported figures stay separate | Passed |
| Revocation effective within 1 s; plaintext key never stored or logged; cross-tenant ids return 404 | Passed |
| 50 concurrent tasks admit exactly the RPM limit; `Retry-After` accurate; budget 402 on upper bound; Redis down gives 503 | Passed |
| Feedback 201/200/409, batch, delete, `training_eligible` rules | Passed |
| Alembic upgrade, downgrade, upgrade; empty autogenerate diff, against real Postgres | Passed |

## M4: exact and semantic caching

| Acceptance test | Result |
|---|---|
| Every partition component changes the key; no cross-tenant or cross-namespace hit | Passed |
| Lexical-guard negatives, changed fingerprint, tightened policy, interrupted streams | Passed |
| Real Redis 8: index, filtered KNN, TTL, tenant-scoped purge | Passed |
| Semantic hit in the running container with the baked-in model | Passed |
| Exact lookup p95 under 5 ms; semantic under 20 ms | Passed: 0.93 ms and 13.82 ms (CPU, embedding warm, local Redis) |
| Threshold calibration | See below |

**Calibration.** `python -m modelmesh.ml.cache_calibration` on hand-written pairs found 0.96 as the lowest threshold with no false hits on 42 validation negatives. On held-out pairs it gave 0 false hits out of 43 (95% interval 0 to 8.2%) and 14 hits out of 15 (70% to 99%). That sample cannot demonstrate a false-hit rate at the 0.5% target, and it says nothing about particular task types. The configured threshold therefore stays at the more conservative 0.97 for all task types. Semantic caching remains opt-in per request.

## M5: benchmark harness (no pilot)

Built: dataset loaders, prompt templates, manifests, automatic metrics, the judge with family exclusion, a resumable runner over the public API, and a review page for human confirmation of labels (DECISIONS D3).

The pilot has **not** been run. It needs API keys, and then the product owner's OpenRouter decision (D2).

| Check | Result |
|---|---|
| Automatic metrics against reference values; judge canary; family exclusion; resumable runner; review page | Passed |
| Harness run end to end against the Compose stack with the fake provider (run, judge, review-sample, report) | Passed; no real provider was called |
| Loaders against real downloads | 13 of 14 sources produced items. **`musique_suff` is not verified**: downloading the MuSiQue `default` config stalled for over 5 minutes, twice |
| `prepare --profile pilot` as a whole | **Not completed**, because of the stalled `musique_suff` download |

## M7: minimal dashboard

Server-rendered Jinja2 pages under `/dashboard` (`src/modelmesh/api/dashboard.py`, templates in `src/modelmesh/api/templates/`). No JavaScript and no build step; the pages send a CSP with no script source. Pages: usage and cost, latency and reliability (provider and model breakers), request explorer with Jaeger links (`MODELMESH_JAEGER_UI_URL`), key management (create, rotate, revoke), routing policy, and a benchmark page that says it needs M6.

Tests: `tests/integration/test_dashboard.py`, 14 tests against real Postgres and Redis, recorded 2026-11-14.

| Acceptance test | Result |
|---|---|
| Pages render against seeded data | Passed: every page returns 200 with seeded successes, an upstream failure and a rejected request; tenant names are escaped |
| Every number comes from the same service-layer query as `/v1/usage`, asserted equal | Passed: the usage page uses `AccountingQueries.usage`, and each cell's value equals the `/v1/usage` figure for all five `group_by` values. The reliability page equals `/v1/providers` (now one shared `provider_entries` function), `/v1/usage?group_by=model` and `/v1/models` status |
| Admin token required | Passed: no token, a wrong token and a tenant API key all get 401 on every page; a forged cookie cannot create a key; with an empty admin token the dashboard answers 503 |
| No prompt content unless retention is on | Passed: a request from a key without retention shows no content; one from a retention key shows it on its detail page only; the request list never shows content |
| Never calls a provider | Passed: rendering every page with outbound HTTP transports patched to fail, and with no new scripted-adapter calls |
| Key management | Passed: create shows the secret once (`no-store`) with the D1 default limits; rotate kills the old secret; revoke takes effect; form posts need the CSRF value; a key cannot be revoked through another tenant |

Not verified:
- **A browser.** The tests drive the ASGI app in-process. Nobody has looked at the pages in a browser, including the layout at phone width and the dark theme.
- Benchmark comparison: not built, because M6 has not produced a report.
- The login form has no attempt limit, and the session cookie is derived from the admin token, so it lasts until the token changes. Both are acceptable for a local operator page and are not suitable for an exposed deployment.

**Compose stack.** On 2026-11-14 the `api` image was rebuilt with jinja2 and only the `api` container was recreated; it came back healthy. Against the pilot data, every page returned 200 with the admin token and 401 without it. On the usage page grouped by model, all 140 cells (9 models and the total) equalled `GET /v1/usage` for the benchmark key over 2026-11-10 to 2026-11-14 (2,923 requests). A request detail page showed its Jaeger link and no content (the benchmark key has retention off).

**Test isolation.** The integration fixtures `FLUSHDB` the Redis at `TEST_REDIS_URL`, which defaults to `127.0.0.1:56379/0`, the same database the Compose gateway uses for quota counters, budgets, rate limits and the cache. While the pilot gateway was running, the gates were run with `TEST_REDIS_URL=redis://127.0.0.1:56390/0` against a throwaway `redis:8-alpine` container, so that the gateway's state was not wiped.

## M6: router training and evaluation (preliminary, P0 data)

Recorded 2026-11-15. Offline only: no provider was called. Full report: `M6_P0_PRELIMINARY_REPORT.md`, generated from `M6_P0_results.json`.

| Acceptance test | Result |
|---|---|
| Synthetic data: the evaluation code recovers a known Oracle and hand-computed AIQ | Passed (`tests/unit/test_router_evaluation.py`; also a kNN router recovers the Oracle on separable synthetic data in `test_router_training.py`) |
| No group or near-duplicate leakage between splits (asserted) | Passed in unit tests and asserted in the P0 run: 0 groups cross splits, highest cross-split cosine 0.9490 (threshold 0.95) |
| Test manifest frozen before model selection; test touched once | Partly. The P0 split manifest is written by the run that selects models, from a fixed seed, before selection; it is committed as `data/manifests/p0_router_split.json`. The test set was evaluated twice by identical code (a smoke run, then the reported run). P1 must freeze its own manifest first |
| Learned vs B0a and B0b on held-out-query and workload-shift splits, paired CIs; ADR-0004 criterion reported | Done. **The criterion does not pass on P0**: learned minus B0b on held-out queries 0.017 [-0.023, 0.070]; on leave-one-dataset-out -0.037 [-0.096, 0.014] |
| Router p95 overhead under 10 ms excluding embedding, under 25 ms including it | **Half met.** 1.06 ms excluding embedding; 411.9 ms including embedding (fastembed, one thread, i5-11400H) |
| Shadow mode logs decisions without changing traffic; promotion needs the admin approval call | Unit-tested (shadow decisions never change the live estimates; promotion gate and transitions). The integration test against Postgres was **not run** (Docker stopped) |
| LightGBM and logistic-regression runtime predictions equal the libraries' own | Passed (`test_router_training.py`) |

Not verified: every Postgres/Redis integration test (Docker stopped), including `test_router_admin.py` and migration `0002_router_models`; the shadow path through a running gateway; the MLflow UI.

## Latest gate results

2026-11-15, after M6, with Docker Desktop stopped:

| Gate | Result |
|---|---|
| ruff check and format | Clean |
| mypy --strict | Clean (90 source files) |
| import-linter | 4 contracts kept |
| pytest without `tests/integration` | 359 passed, 24 skipped (all 24 are live-provider tests, which did **not** run) |
| pytest `tests/integration` | **Not run**: Postgres and the test Redis were unreachable (203 tests errored or failed on connection; 1 pure test passed) |

The last full run with real Postgres and Redis was 2026-11-14: 531 passed, 24 skipped, 0 failed.
