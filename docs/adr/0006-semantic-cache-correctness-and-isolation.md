# ADR-0006: Cache correctness and tenant isolation

- Status: Accepted (architecture phase)
- Date: 2026-11-08
- Spec refs: §2.3 E, §2.4 step 4, §5.3 (cross-workspace cache leakage), §8 item 6

## Context

Two requests that look alike don't necessarily deserve the same answer. A cached answer must never cross a tenant boundary, never come from a different model configuration, and never be served as current when it was grounded in an older version of a document (Mezvuat's `context_fingerprint`). The cache also stores prompt and response content, which makes it a retention concern.

## Options considered

| Topic | Options | Choice |
|---|---|---|
| Store | Redis (spec); Postgres + pgvector; in-process | **Redis 8**, which has the Query Engine (vector index, TAG filters) built in. Redis Stack is deprecated. Built-in modules in the official image are UNVERIFIED, so M4 starts with a smoke test. Fallback if they are absent: FLAT search in-process over a small per-partition candidate set read from Redis |
| Client library | redis-py directly; RedisVL `SemanticCache` | **redis-py directly**. RedisVL's cache semantics (threshold over the whole prompt) don't match the partitioning rules below, and wrapping it costs more than about 100 lines of FT.SEARCH code |
| Embeddings | provider embedding API; sentence-transformers (torch); fastembed ONNX | **fastembed `BAAI/bge-small-en-v1.5`** on CPU, shared with router features. No external call, no data-policy issue, no torch |
| Semantic scope | whole conversation; final user turn only | **final user turn only**. Everything before it must match exactly |

## Decision

### Keys

- `tenant_id` always comes from the authenticated API key, **never from the request body**.
- **Partition key** P = SHA-256 of the canonical JSON of:
  - `tenant_id`, `cache.namespace` (optional caller partition, e.g. a hashed workspace id)
  - the **resolution target**: the concrete model id, or `auto` + policy + canonicalised hard constraints
  - the system prompt hash and the hash of `messages[:-1]`
  - generation parameters: temperature, top_p, max_tokens, stop, seed, presence/frequency penalty
  - `response_format` (including the full schema), `tools`, `tool_choice`
  - `context_fingerprint`
  - `task_type`
  - the cache-format version
- **Exact key** = `mm:c:{tenant_id}:x:` + SHA-256(P ‖ canonical(messages[-1])).
- **Semantic entry:**
  - The vector of the final user message is stored with TAG fields `tenant` and `partition = P`.
  - A lookup is a KNN query **pre-filtered by `@tenant` and `@partition`**. A hit needs cosine ≥ θ(task_type) **and** a lexical guard: the multisets of numbers, negation tokens ("not", "no", "never", "without", …) and quoted strings must be identical.
  - This stops "is X allowed" from matching "is X not allowed" and "2023" from matching "2024".

### Eligibility rules

These resolve the draft's §2.3. The `cache` block itself is unchanged apart from the additions listed in the contract changelog.

- `cache.mode`: `off` | `exact` (default) | `semantic`.
- When `temperature > 0` and no `seed` is given, the default `exact` mode is downgraded to **bypass**. Explicitly setting `cache.mode` opts in, as in the draft.
- Semantic mode **requires** `context_fingerprint` (400 otherwise). It is disabled when `tools` are present, and θ defaults to **0.97** for every task_type until M4 calibration. Final-response-style tasks may stay effectively disabled (θ = 1.0) if calibration shows false hits.
- On read, the cached entry's provider and model must **pass the current request's eligibility filter**: `data_policy`, allowed/denied lists, and the key's current permissions. A permission tightened after caching therefore never serves old content.
- Only `finish_reason ∈ {stop, length}` responses that passed structured-output validation are stored. Errors, content-filtered and partial (broken) streams are never stored.
- Streaming requests may be served from cache by replaying the stored response as chunks (`modelmesh.cache.status = hit_*`). A successful stream is stored after completion.
- TTL: `cache.ttl_s` default 3600, capped by the tenant's `cache_max_ttl_s` (default 86,400).
- On a hit:
  - top-level `usage` = the original usage, for client compatibility
  - `modelmesh.cost.estimated_usd = 0`
  - `modelmesh.cache.saved_estimated_usd` = the original estimate
  - `modelmesh.cache.original_request_id` is set
  - `routing` is `null` (no routing happened)

### Invalidation and deletion

- Mezvuat's correctness doesn't depend on explicit invalidation: new document versions produce a new `context_fingerprint`, so old entries simply stop matching and expire.
- **Additive endpoint `POST /v1/cache/invalidate`** with `{namespace?, context_fingerprint?}` purges the tenant's matching entries, for example when a workspace is deleted (spec §5.3). The purge is tenant-scoped by construction: it deletes by scanning the `mm:c:{tenant}:…` keyspace plus an `FT.SEARCH` on the tags.
- Tenant deletion purges `mm:c:{tenant_id}:*`.

### Calibration of θ (M4)

- Build near-miss pairs from benchmark prompts. Positives are paraphrases. Negatives are hard cases: negation flips, number and date changes, entity swaps.
- Choose θ per task_type so that the **false-hit rate is ≤ 0.5%** on validation pairs, then report the hit rate and the false-hit rate with CIs on test pairs.
- If no θ < 1.0 meets the target for a task type, semantic caching stays off for it.

## Consequences

- Semantic hits will be rare, because the partitioning is strict. That trade is deliberate: correctness over hit rate. Exact caching delivers most of the savings for repeated pipeline calls.
- The cache holds content, so it is covered by the tenant's retention settings and the purge endpoint. `cache.mode: off` is available per request, and a tenant can disable caching entirely.

## Verification plan

- Property-based and unit tests:
  - Changing any partition component changes the key.
  - Tenant A can never read B's entry, including with an identical namespace or fingerprint, and including through semantic KNN.
  - Lexical-guard negatives never hit.
  - A tightened `data_policy` blocks an old hit.
  - Streams that fail mid-way are not stored.
- Integration test against real Redis 8: FT index creation, filtered KNN, TTL expiry, and the purge endpoint.
- M4 report: θ calibration (false-hit and hit rates per task_type, with CIs), and cache latency overhead p95 (target under 5 ms exact, under 20 ms semantic including embedding).

## Amendment 2026-11-09 (M4)

- `redis:8-alpine` was checked: Redis 8.10 with the `search`, `timeseries` and `bf` modules loaded, and `FT.CREATE` with a `VECTOR` field succeeds. The in-process fallback was not needed.
- Redis 8 is offered under RSALv2, SSPLv1 or AGPLv3 (`LICENSE.txt` in the Redis repository, checked 2026-11-09). It runs unmodified as a separate server, under RSALv2. The product owner has not yet reviewed that reading of D5, so the store stays behind `ResponseCache`.
- Limits and cache share Redis DB 0. Eviction is not configured on the local stack, so the separation by logical DB in ADR-0010 is deferred; `maxmemory` is not set either.
