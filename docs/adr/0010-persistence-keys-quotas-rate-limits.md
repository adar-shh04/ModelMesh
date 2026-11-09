# ADR-0010: Persistence, Redis usage, API keys, quotas and rate limits

- Status: Accepted (architecture phase)
- Date: 2026-11-08
- Spec refs: §2.3 F/H, §2.5 (Authentication, Accounting, Persistence), §5.2 (migrations), §5.3 (retention, deletion), §6 ("API-key management and rate limits work")
- Resolves contract open question 4: content retention defaults.

## Context

ModelMesh needs durable records of tenants, keys, requests, attempts, usage, feedback, benchmark observations and router versions. It also needs low-latency counters for rate limits and quotas. The spec assigns Postgres to persistent data and Redis to ephemeral state, and asks us not to duplicate technologies.

## Options considered

- **Migrations:** Alembic (SQLAlchemy-native, autogenerate, offline SQL) vs plain SQL files (dbmate, yoyo). **Alembic** fits because SQLAlchemy is already in use.
- **Driver:** asyncpg (fastest, async only) vs **psycopg 3** (sync and async, one driver for the app, Alembic and scripts). psycopg 3 is chosen. The speed gap doesn't matter here: about 3 queries per request, mostly batched.
- **Rate-limit algorithm:** fixed window (bursty at window edges) vs sliding log (memory-heavy) vs **token bucket / GCRA in a Redis Lua script** (atomic, O(1)). GCRA is chosen.
- **API-key hashing:** bcrypt/argon2 (slow by design, which matters for low-entropy passwords) vs **HMAC-SHA-256 with a server pepper**. Keys are 256-bit random, so slow hashing adds nothing except per-request latency. HMAC is chosen.

## Decision

### API keys

- Format: `mm_<env>_<43-char base64url of 32 random bytes>`, with env `live` or `test`.
- Stored as `HMAC-SHA256(pepper, key)` plus a display prefix (first 12 characters). The plaintext key is shown once at creation.
- The pepper is `MODELMESH_API_KEY_PEPPER` (env).
- Revocation is immediate. The verify cache (Redis, 60 s) is deleted on revoke.
- **Per-key settings:**
  - `allowed_providers`/`allowed_models`
  - a **ceiling `data_policy`** (requests can tighten it, never loosen it)
  - RPM, TPM, max concurrent requests
  - daily and monthly budget in estimated USD
  - `cache_enabled`, `cache_max_ttl_s`
  - `content_retention`
- **Admin API** (`/admin/v1/…`): tenants, keys, provider enablement, routing policy defaults, router promotion. It is authenticated with `MODELMESH_ADMIN_TOKEN` for v1. It is **not** part of the client inference contract and is specified separately in M3.

### Rate limits and quotas

**Defaults for newly created keys (DECISIONS D1): 20 RPM, 50K TPM, $0.25/day estimated cost.** Keys can have explicit overrides. The benchmark harness, load tests and integration tests use explicitly configured keys, and those overrides are recorded with each run.

- **RPM:** GCRA in Redis, keyed by key id.
- **TPM:** the estimated tokens (input + `max_tokens`) are reserved before the call and reconciled with actual usage afterwards.
- **Concurrency:** a Redis counter with TTL safety.
- Going over RPM, TPM or concurrency returns **429 `rate_limited`** with `Retry-After` and `X-RateLimit-*` headers.
- **Budget:** daily and monthly estimated-USD spend, kept as Redis counters mirrored from Postgres usage. Before routing, the check uses the request's **upper-bound** estimate. Going over returns **402 `budget_exceeded`**.
- Redis is not the source of truth. Counters are rebuilt from Postgres if lost, and when Redis is down the API **fails closed** (503) for limited keys.

### Postgres schema (outline)

Postgres 17. The already-pulled `postgres:17-alpine` image is reused because C: is short on space (ENVIRONMENT.md). Supported until 2029.

| Table | Key columns |
|---|---|
| `tenants` | id, name, status, created_at |
| `api_keys` | id (`key_…`), tenant_id, hmac, prefix, name, scopes jsonb, limits jsonb, data_policy_ceiling, content_retention, status, created_at, revoked_at, last_used_at |
| `models` | id (`provider/model`), provider, family, context_window, max_output_tokens, capabilities jsonb, data_use, free_tier, status, catalog_version |
| `pricing` | model_id, pricing_version, input_per_mtok_usd, output_per_mtok_usd, effective_from |
| `requests` | id (`mmr_…`), tenant_id, api_key_id, client_request_id, trace_id, correlation_id, application, task_type, end_user_hash, stream, status (`succeeded`/`failed`/`stream_interrupted`/`rejected`), http_status, error_type, error_code, model_id, provider, routing jsonb, cache_status, fallback_used, structured_output jsonb, latency_total_ms, latency_provider_ms, ttft_ms, created_at, completed_at. Indexed on (tenant_id, created_at), correlation_id, trace_id |
| `attempts` | request_id, seq, provider, model_id, outcome, http_status, retryable, latency_ms, possibly_billed, started_at |
| `usage_records` | request_id, prompt_tokens, completion_tokens, cached_tokens, reasoning_tokens, usage_source, estimated_cost_usd, provider_reported_cost_usd, pricing_version |
| `request_content` | request_id, messages jsonb, response jsonb, expires_at. **Only when the key's `content_retention` is enabled** |
| `feedback` | id, request_id, tenant_id, label_source, metric, score, accepted (nullable), evaluator, task_type, notes, created_at. UNIQUE(request_id, metric, evaluator) |
| `bench_items` | id, dataset, source_item_id, task_type, split, content_sha256, manifest_version |
| `bench_observations` | id, bench_item_id, model_id, request_id, run_id |
| `quality_labels` | id, bench_observation_id or request_id, metric, score, accepted, label_source (`auto`/`judge`/`human`), judge_model, rubric_version, created_at |
| `router_models` | version, algorithm, artifact_uri, artifact_sha256, dataset_manifest_sha256, metrics jsonb, mlflow_run_id, status (`candidate`/`shadow`/`active`/`retired`), approved_by, approved_at |

### Retention defaults (answers open question 4)

- **Prompt and response content retention: off by default**, per key. When enabled, the default TTL is 7 days, deleted by a periodic job.
- Execution metadata (no content) is kept 90 days, configurable.
- Feedback and quality labels are kept until the tenant is deleted.
- Benchmark content lives only in local files (ADR-0005), never in request content tables.
- Tenant deletion cascades in Postgres and purges `mm:*:{tenant}:*` in Redis.

### Redis keyspace

Redis 8, AOF off, `maxmemory` 256 MB with `allkeys-lru` (only for cache keys; see below).

- `mm:rl:{key_id}:*`: rate-limit and concurrency state
- `mm:bud:{key_id}:{day|month}`: budget counters
- `mm:quota:{provider}:{model}:*`: upstream quota tracker (ADR-0008)
- `mm:c:{tenant}:…`: cache entries and the vector index (ADR-0006)
- `mm:kv:{hmac_prefix}`: key-verification cache

Eviction must never drop limit counters before cache entries. Limits and cache therefore live in **separate logical DBs**: cache in DB 1 with LRU, limits in DB 0 with `noeviction`. If the Query Engine requires DB 0 for indexes, which it historically did, the roles swap: cache in DB 0, limits in DB 1. This is verified in M4.

## Consequences

- Plain, well-understood components. Every operation that matters for correctness is atomic, through Lua scripts or Postgres transactions.
- Request rows are written once at completion (plus one row on rejection), so writes stay small. Partitioning is deferred until volume justifies it.

## Verification plan

- Alembic: `upgrade head` → `downgrade base` → `upgrade head` round-trip in CI against Postgres. An autogenerate diff must be empty.
- Rate-limit tests with real Redis: the burst allowance, `Retry-After` correctness, concurrent requests from 50 tasks not exceeding the RPM limit, and fail-closed behaviour when Redis is stopped.
- Key tests: revoked keys are rejected within 1 s, a key is never stored or logged in plaintext (canary), and a cross-tenant `/v1/requests/{id}` returns 404, not 403, so existence is not leaked.
- The budget test covers the upper-bound pre-check and the reconciliation drift reported in usage.

## Amendment 2026-11-08 (contract 1.0-rc2, CR-13)

- The `feedback` table adds the columns `metric_version` (default `''`), `higher_is_better`, `attribution` (`direct`/`workflow`), `eval_run_id`, `dataset_version`, `labelled_at` and `training_eligible`.
- The unique key becomes (request_id, metric, metric_version, evaluator).
- `DELETE /v1/feedback/{request_id}` hard-deletes rows.
- Dataset builds record label ids, so deleted labels are excluded from the next rebuild.
- The `requests.status` enum gains `cancelled`.

## Amendment 2026-11-09 (implementation)

- The model catalog and pricing are served from versioned YAML (`config/`). The `models` and `pricing` tables in the outline were not created, and each usage record stores its `pricing_version`.
- `bench_*` tables are replaced by local run files (ADR-0005 amendment). `router_models` arrives with M6.
- RPM and TPM use GCRA with a burst equal to the limit: a key can spend a full minute's allowance at once and it then refills smoothly, so a 60-second window can see up to about twice the limit at the very start.
- A TPM reservation larger than the remaining allowance is rejected, even if the call would turn out to use fewer tokens.
- Postgres and Redis are published on loopback (ports 55432 and 56379) by default instead of behind an `expose-db` profile (ADR-0011), so local tests can reach them.
