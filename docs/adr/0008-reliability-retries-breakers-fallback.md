# ADR-0008: Reliability (retries, backoff, circuit breakers, quotas, fallback)

- Status: Accepted (architecture phase)
- Date: 2026-11-08
- Spec refs: §2.3 D, §2.4 step 7, §4.3, §6 ("Fallback, retry and circuit-breaker behaviour tested")
- Resolves contract open question 2 (`max_latency_ms` semantics).

## Context

Free-tier providers fail often and in different ways:
- per-minute vs daily quota 429s (Gemini `rate_limit_exceeded` vs `quota_exceeded`)
- Groq 498 at flex capacity
- OpenRouter errors inside HTTP 200 responses, and 503 when no upstream matches the routing requirements
- local Ollama being down or still loading a model

Retries must separate transient from permanent errors. Fallback must never break the caller's constraints (spec §2.3 D).

## Options considered

- **Libraries (tenacity, pybreaker)** vs **a small hand-written module.** Retry, deadline, quota and fallback decisions are tightly coupled: one deadline budget across attempts, attempt records for the response, and breaker updates. Generic libraries would need adapting at every step, and pybreaker's shared storage is synchronous. **Hand-written** wins, at about 300 lines with thorough tests.
- **Breaker state per process vs in Redis.** v1 runs one API container, possibly with several workers. The breaker exists to stop hammering a failing provider, and per-process state does that well enough. **Per process** for v1. **Quota tracking must be shared**, because a daily quota is a global fact, so it lives in **Redis**.

## Decision

### Normalised error taxonomy

Each adapter maps upstream errors (ADR-0003) to these classes.

| Class | Examples | Retry same model | Fallback to another model | Breaker |
|---|---|---|---|---|
| `auth_error` | 401/403 from provider | never | yes; provider marked `misconfigured` and an alert raised | opens provider |
| `invalid_request` | 400 caused by our translation | never | no; this is a ModelMesh bug: 500 `internal_error`, logged | no |
| `incompatible` | 400 due to an unsupported param or schema the catalog said was fine | never | yes; catalog-drift metric | no |
| `context_length_exceeded` | 400/413 context | never | yes, to a larger-context eligible model | no |
| `content_filtered` | safety/recitation block, 403 moderation | never | no; returned to caller as `upstream_error`/`content_filtered` | no |
| `rate_limited` | 429 per-minute, Groq 498 | yes, if `Retry-After` ≤ remaining deadline | yes, otherwise | no |
| `quota_exhausted` | Gemini 429 `quota_exceeded`, OpenRouter daily free limit, 402 credits | never | yes; model marked unavailable until reset | no (quota state instead) |
| `timeout` | connect/read timeout, 408, 504 | once | yes | counts |
| `upstream_unavailable` | 500/502/503, connection refused, empty 200 | once | yes | counts |
| `schema_validation_failed` | ADR-0007 | one repair | yes | no |

### Retry and backoff

- At most **1 retry per model** for transient classes, so there are at most 2 attempts per model.
- Backoff is exponential with **full jitter**: `sleep = random(0, min(cap, base·2^n))`, base 250 ms, cap 2 s.
- `Retry-After` is honoured when it is shorter than the remaining deadline. Otherwise the model is skipped.

### Circuit breaker

- Kept per (provider, model) and per provider (for connection-level errors).
- Sliding window of the last 20 calls. It **opens** at ≥ 50% failures with at least 5 calls, or on 3 consecutive timeouts.
- Open for 30 s, then **half-open** with one probe request.
- Open models are skipped and recorded as `circuit_open` attempts. Their state is exposed in `/v1/providers` (`circuit_state`).

### Quota tracker

Shared through Redis.

- Each (provider, model) has known limits from config (RPM/RPD/TPM/TPD). These are updated from response headers (Groq `x-ratelimit-*`, where requests = RPD and tokens = TPM) and from `quota_exhausted` errors, whose reset is midnight Pacific for Gemini and midnight UTC for OpenRouter.
- The router treats a model with too little remaining quota as **ineligible**, before any call, so the free tier is never hammered.

### Fallback within constraints

- Fallback candidates are **only** the router's ranked eligible list (ADR-0004). That list was already filtered by every hard constraint (`allowed`/`denied` lists, `required_capabilities`, `data_policy`, `min_quality`, latency, cost).
- Before each attempt, the cost check is **re-run** against `max_cost_usd − cost_possibly_incurred`. Failed attempts that may have been billed are counted conservatively: timeouts after the request body was sent, and streams that broke mid-way. The `accounting` note records "possibly billed".
- **No hard constraint is ever relaxed.** If the list runs out, the response is 502 `upstream_error`, or 503 `service_unavailable` / `upstream_capacity` when every candidate failed for rate or quota reasons. The `details` field lists each attempt's outcome.
- `routing.fallback`:
  - `within_constraints` is the default for `model: "auto"`.
  - **The default for an explicit model is `none`** (retries on that model only). This is a clarification of the draft: a caller who named a model shouldn't silently get a different one.
  - Explicitly setting `within_constraints` with an explicit model allows fallback to router-ranked alternatives that meet the same constraints. `fallback_used: true` is then reported.

### Deadlines (answers open question 2)

- `routing.max_latency_ms` is a **routing budget**. A model is eligible only if its predicted p90 end-to-end latency is ≤ `max_latency_ms`.
- **Additive field `routing.timeout_ms`** is the **hard wall-clock deadline** for the whole request: all attempts, retries and fallback.
  - Default when `max_latency_ms` is set: `min(2 × max_latency_ms, server_max)`. Otherwise the server default.
  - Server default 60 s and server max 300 s, both configurable.
- For streams, the deadline applies to the time until the first token. After that, an **idle timeout** (default 30 s with no upstream bytes) ends the stream with an in-stream error.
- When the deadline is reached, the response is 504 `upstream_timeout`, with the attempts in `details`.

### Health

- `/v1/health` reports `degraded` when any enabled provider's circuit is open, or when Postgres or Redis latency goes over its threshold. It reports `down` when the database or Redis is unreachable.
- Ollama's health check also reports whether the model is loaded.

## Consequences

- The behaviour is predictable and fully reported in `modelmesh.attempts`.
- Per-process breakers mean each worker learns failures separately, which is acceptable at this scale. Moving them to Redis is a later ADR if replicas are added.
- "Possibly billed" accounting is conservative and may over-estimate cost after timeouts. That is labelled.

## Verification plan

- Unit tests:
  - the breaker state machine: closed→open→half-open→closed and half-open→open
  - jitter bounds, using an injected RNG and clock
  - each taxonomy class against its retry and fallback decision
  - `Retry-After` handling
  - deadline exhaustion
- Fault-injection integration tests through the `fake` provider and respx:
  - a Gemini quota error marks the model unavailable until reset
  - fallback skips a model that violates `data_policy`, even if it is the only healthy one, giving a 502 rather than a violation
  - a cost-ceiling re-check during fallback
  - an explicit model with no fallback
- M8: a chaos run with Locust, injecting 30% upstream failures. Reports the success rate, p95 latency and breaker transitions.
