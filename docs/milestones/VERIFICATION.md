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

## Latest gate results

| Gate | Result |
|---|---|
| ruff check and format | Clean (94 files) |
| mypy --strict | Clean (76 source files) |
| import-linter | 3 contracts kept |
| pytest (real Postgres and Redis) | 502 passed, 24 skipped, 0 failed. All 24 skips are live-provider tests, which did **not** run |
