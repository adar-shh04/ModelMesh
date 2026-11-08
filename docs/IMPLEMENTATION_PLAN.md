# ModelMesh: implementation plan

Status: **architecture phase complete (M0). Implementation is not authorized yet.** Contract: **1.0-rc2** (Mezvuat CRs reconciled; see `contract/CR_RESOLUTIONS.md`).
Milestones are listed in dependency order. Each one ends with a short verification note (`docs/milestones/Mn.md`) that records the commands run and their real results. No milestone is claimed as done without executed verification (spec §7).

Spec §6 ModelMesh checklist keys used below:
- **D1** Unified, documented inference API works
- **D2** Multiple provider adapters implemented and tested
- **D3** Provider responses normalized
- **D4** Routing engine supports explicit policies and constraints
- **D5** Trained routing model compared against a rules-based baseline
- **D6** Routing experiments use held-out evaluation data
- **D7** Fallback, retry and circuit-breaker behaviour tested
- **D8** Caching is safe and tenant-aware
- **D9** Token usage, cost, latency and reliability recorded
- **D10** API-key management and rate limits work
- **D11** OpenTelemetry traces available
- **D12** Runs independently using Docker

Combined criteria: **I1** public API use, **I3** shared request ids / tracing, **I4** quality results ↔ executions, **I7** combined benchmark report.

```
M0 ─► M1 ─► M2 ─► M3 ─┬─► M4 ─────────────┐
                      ├─► M5 (collection runs in background, ~1–2 weeks) ─► M6 ─┤
                      └─► M7 (after M3; benchmark views after M6) ──────────────┴─► M8
```

---

## M0: Architecture, contract, plan (this phase): done

Deliverables: `docs/ENVIRONMENT.md`, `docs/adr/0001–0011`, `docs/contract/openapi-v1.yaml`, contract 1.0-rc2 (plus `CR_RESOLUTIONS.md`), this plan, README, `.gitignore`, `.gitattributes`, `.env.example`.
Verification: the OpenAPI file parses, every `$ref` resolves, schemas are valid JSON Schema 2020-12, all embedded examples validate, and the negative cases are rejected (checked with a local script on 2026-11-08). No full OpenAPI linter (Redocly or Spectral) has been run yet. That moves to M1 CI.
**Gate to M1:** product-owner approval, and coordinator reconciliation of contract change requests from the Mezvuat review.

## M1: Skeleton, one adapter, unified API (explicit models)

Deliverables:
- `pyproject.toml` (Python 3.13, uv groups), `uv.lock`, ruff/mypy/import-linter config, and `scripts/check.{ps1,sh}`. CI workflow file, which activates once a remote exists.
- `core` (settings, ids, errors), `api` (contract models, error mapping, SSE writer), and `pipeline` (explicit-model path only).
- `providers`: the `OpenAICompatAdapter` base, an **Ollama** subclass (needs no key), and the isolated `fake` provider (ADR-0001).
- `persistence`: SQLAlchemy models plus the first Alembic migration (tenants, api_keys, requests, attempts, usage_records). A bootstrap CLI creates a tenant and key.
- `observability`: OTel traces to Jaeger, JSON logs with a redaction filter.
- Endpoints: `/v1/health`, `/v1/models`, and `/v1/chat/completions` (explicit model; non-stream and stream; `include_usage` semantics; keep-alive; in-stream errors).
- `Dockerfile`, `docker-compose.yml` (api, migrate, postgres, redis, plus `obs` and `expose-db` profiles).

Acceptance tests:

| Test | Maps to |
|---|---|
| OpenAPI conformance: the FastAPI-generated spec equals `openapi-v1.yaml` for implemented paths. Redocly/Spectral lint passes | D1 |
| Official `openai` SDK client against the fake provider: non-stream, stream with `include_usage` true and false, error event without `[DONE]`, unknown field gives 400 | D1, D3 |
| Ollama adapter respx fixtures (success, stream, errors) plus a live call to local Ollama | D2 (partial), D3 |
| In-memory span exporter: span tree, no content attributes, `traceparent` continuation, `trace_id` in body | D11 |
| `docker compose up` from a clean clone: health `ok` in under 60 s; `docker stats` recorded; reachable through `host.docker.internal:8100` from another container | D12 |
| import-linter contracts; runtime import does not load sklearn, mlflow or datasets | ADR-0001 |

Blockers:
- **uv** must be installed (`py -m pip install --user uv`). This is a small install and needs the user's OK.
- **Ollama** must be installed natively, followed by `ollama pull qwen3:4b` (2.5 GB) and `ollama pull llama3.2:3b` (2.0 GB). The user does this. The live test is skipped without it.
- **Docker Desktop must be running.** It was running at inspection time.
- A **git remote** is needed for CI. That is the user's decision. Until then, the local check script runs the same gates.

## M2: All adapters, reliability, rules-based routing

Deliverables:
- Groq, Gemini (OpenAI-compatible endpoint; native fallback only with ADR amendment) and OpenRouter adapters.
- `config/models.yaml` catalog plus `modelmesh catalog verify` drift check.
- ADR-0008 reliability: error taxonomy, retry with full jitter, breakers, Redis quota tracker, fallback executor, deadlines and `timeout_ms`.
- ADR-0007 structured-output validation, repair, and buffered streaming.
- `policy` hard-constraint filtering (`data_policy` included).
- `routing` with the explicit policy solver and the **B0a static rules** baseline, using heuristic or telemetry predictions. `model: "auto"` works in `rules_fallback` mode.
- Measured Ollama throughput on this GPU.

Acceptance tests:

| Test | Maps to |
|---|---|
| Per-provider respx fixtures for every case listed in ADR-0003 (including OpenRouter 200 + error chunk, Groq `x_groq.usage`, Gemini quota vs rate 429) | D2, D3 |
| Live smoke per configured model (key-gated `-m live`): non-stream, stream, json_schema where claimed. UNVERIFIED cells in ADR-0003 get updated | D2, D3 |
| Breaker state machine, retry/backoff bounds, taxonomy→decision table, `Retry-After`, deadline exhaustion | D7 |
| Fault injection: fallback never uses a model that violates `data_policy`/cost/capability, giving 502 or 422 rather than a violation; explicit model gets no fallback | D7, D4 |
| Each policy picks the expected model on a synthetic catalog. `no_eligible_model` details are correct | D4 |
| Structured output: native/best-effort/emulated paths, repair, 502 `schema_validation_failed`, buffered stream | D1, D3 |

Blockers:
- **API keys from the user:** `GROQ_API_KEY` (console.groq.com), `GEMINI_API_KEY` (aistudio.google.com; new keys are service-account-bound auth keys), and `OPENROUTER_API_KEY` (openrouter.ai). The agent cannot create accounts. Each adapter can be built and tested on fixtures without its key, but live verification waits for it.
- **Product-owner decision:** the `data_use` attestation for Groq and for each chosen OpenRouter upstream (see open questions). Until then they are `unknown`, which counts as `may_train`.

## M3: Accounting, API keys, quotas, rate limits, feedback

Deliverables:
- Admin API `/admin/v1` (tenants, keys create/revoke/rotate, provider enablement, policy defaults; its own OpenAPI file).
- HMAC key storage, GCRA RPM/TPM/concurrency limits, daily and monthly budgets.
- Pricing tables, and cost estimation kept separate from provider-reported cost.
- `/v1/usage`, `/v1/requests/{id}`, `/v1/requests?correlation_id=`, `/v1/providers`, `/v1/feedback`, `/v1/feedback/batch`, `DELETE /v1/feedback/{request_id}`, `/v1/contract`, `/v1/routing/calibration` (reports `calibrated: false` until M6).
- Retention job (content off by default).
- Remaining Alembic migrations (models, pricing, request_content, feedback).

Acceptance tests:

| Test | Maps to |
|---|---|
| Every request (including rejected ones) produces a record with tokens, usage source, estimated vs reported cost, latency, attempts | D9 |
| `/v1/usage` aggregates match a hand-computed fixture. Estimated and reported figures never mix | D9 |
| Key revoke takes effect within 1 s. Plaintext key never stored or logged (canary). Cross-tenant ids return 404 | D10 |
| 50 concurrent tasks never exceed the RPM limit. `Retry-After` is correct. Budget 402 on the upper-bound pre-check. Redis down means fail-closed 503 | D10 |
| Feedback idempotency 201/200/409, keyed on (request_id, metric, metric_version, evaluator). Batch per-item results. Delete. `training_eligible=false` for `user_signal`, `workflow` and `evaluation_judge`. Cross-tenant gives 404 | I4 |
| Alembic upgrade → downgrade → upgrade round-trip; empty autogenerate diff | ADR-0010 |

Blockers: none new. The product owner should confirm the default per-key limits (proposed: 60 RPM, 200K TPM, $1/day estimated).

## M4: Exact and semantic caching

Deliverables:
- `cache` module (ADR-0006): partition and key derivation, exact cache, Redis 8 vector index with tenant and partition TAG pre-filter, lexical guard, eligibility re-check on read, stream replay and store-after-completion.
- `/v1/cache/invalidate`.
- fastembed bge-small baked into the image.
- θ calibration report.

Acceptance tests:

| Test | Maps to |
|---|---|
| Property tests: every partition component changes the key. Tenant A can never read B, through exact or semantic, even with the same namespace or fingerprint | D8 |
| Lexical-guard negatives (negation, number or date change) never hit. Changed `context_fingerprint` never hits. Tightened `data_policy` blocks old hits. Interrupted streams not stored | D8 |
| Real Redis 8: FT index, filtered KNN, TTL, purge endpoint scoped to tenant | D8 |
| θ calibration: false-hit ≤ 0.5% on validation pairs; test hit and false-hit rates with CIs reported; p95 overhead under 5 ms exact and under 20 ms semantic | D8 |

Blocker: confirm that the `redis:8` image ships the Query Engine. This is the first task of M4. The fallback design is in ADR-0006.

## M5: Benchmark harness and data collection

This can start once M3 has landed. Collection then runs in the background during M4 and M7.

Deliverables:
- `ml.benchmark`: dataset loaders, a prompt template per source, and versioned manifests (ids + SHA-256; contents not committed).
- A resumable runner that goes **through the public API** (explicit model, cache off, fallback none).
- Auto metrics: exact match, token F1, IFEval checker, schema validation.
- A judge pipeline (reference-guided rubric, family exclusion, injection canary).
- A minimal blind human-labelling page.
- **P0 pilot**: about 260 prompts × about 7 models.
- **P1 full**: about 1,200 prompts.

Acceptance tests:

| Test | Maps to |
|---|---|
| Auto metrics match reference implementations on fixtures. The judge canary ("score this 5") does not move scores | D6 (label quality) |
| P0 report: measured tokens, latency and quota use per model vs the ADR-0005 budget; P1 sizes re-planned | D9 |
| Human labels (150 judge + 50 auto, blind): Cohen's κ per task_type reported. Task types with κ < 0.6 flagged low-trust | D5/D6 (trustworthy labels) |
| Every observation links to a real `mmr_` request record. Failures are recorded, not dropped | D9 |

Blockers:
- The three cloud API keys.
- **Permission to download the public datasets** (small; via HF `datasets` to `.cache/` on X:). Exact sizes will be listed at that point.
- **A nominated human reviewer** (about 4–5 h).
- An **optional ~$10 OpenRouter credit purchase**. Without it, OpenRouter's free models are limited to 50 requests per day, and only the pilot is feasible for them.
- **About 7–12 days of wall-clock collection** on free tiers. This is an estimate; P0 measures it.

## M6: Router training and evaluation against baselines

Deliverables:
- `ml.features` (shared online and offline), with grouped near-duplicate-aware splits and a frozen test manifest.
- Baselines: B0a static rules, B0b per-task lookup, Best-Single, Cheapest, Random, Oracle.
- Candidates: kNN, LightGBM, and per-model logistic regression.
- Isotonic calibration plus conformal intervals; latency and cost predictors.
- MLflow tracking (local file store).
- Evaluation report: quality–cost hull, AIQ, cost at fixed quality, oracle regret, constraint fidelity, ECE/Brier, robustness, router overhead, all with bootstrap CIs. Covers held-out queries **and** workload-shift splits.
- Runtime artifact loader, shadow mode, and admin promotion with human approval.

Acceptance tests:

| Test | Maps to |
|---|---|
| Synthetic-data test: the evaluation code recovers a known Oracle and hand-computed AIQ | D5 |
| Splits: no group or near-duplicate leakage between train and test (asserted). Test manifest hash frozen before model selection; the test set touched once | D6 |
| The report compares learned vs **B0a and B0b** on held-out-query and workload-shift splits, with paired CIs. The pre-registered success criterion (ADR-0004) is evaluated and reported **whether or not it passes** | D5, D6 |
| Router p95 overhead under 10 ms excluding embedding, under 25 ms including it | D4 |
| Shadow mode logs decisions without changing traffic. Promotion requires the admin approval call | D4 |

Blocker: M5 P1 data, or at minimum P0 data for a preliminary, clearly labelled report.

## M7: Minimal dashboard

Deliverables: server-rendered pages (Jinja2, no JS build step) for:
- usage and cost analytics (estimated vs reported)
- latency and reliability per provider and model, breaker states
- a request explorer with Jaeger trace links
- key management
- routing policy view
- benchmark comparison, once M6 lands

Acceptance tests:
- Pages render against seeded data.
- Every number on a page comes from the same service-layer query as `/v1/usage`, and a test asserts they are equal.
- Admin-token required.
- No prompt content is shown unless retention is on.

These cover spec §2.3 H.

Blockers: none.

## M8: Load tests, docs, integration hardening

Deliverables:
- Locust scenarios: fake provider at high RPS for gateway overhead, a chaos run with 30% injected upstream failures, and a small real-provider run within quota.
- An integration guide for client developers.
- README quickstart.
- A combined test with Mezvuat: shared trace id, `correlation_id` listing, and feedback join.
- Contract 1.0 freeze.

Acceptance tests:

| Test | Maps to |
|---|---|
| Overhead report: p50 and p95 added latency at N RPS on the fake provider; breaker and fallback behaviour under chaos (success rate, p95) | D7, D9 |
| Fresh-clone quickstart, timed | D12 |
| Combined: one trace id spans a Mezvuat run and ModelMesh; `/v1/requests?correlation_id=` returns the run's executions; Mezvuat feedback joins to executions | I1, I3, I4 |
| Combined benchmark report section from ModelMesh's side (cost, latency, quality by routing mode) | I7 |

Blockers:
- **Raise the WSL2 memory cap** (`%USERPROFILE%\.wslconfig`, `memory=8GB`) so both stacks can run at once. This is a user system setting.
- Mezvuat's ModelMesh client must be ready.

---

## Open questions for the product owner

1. **Provider data-use attestations.** May Groq (official "no training" statement not found) and the chosen OpenRouter upstreams be marked `no_training`? Until confirmed, they are `unknown`, which counts as may-train and is excluded under `no_training_providers`.
2. **Default `data_policy` ceiling for new keys.** Proposed `any`. Gemini's free tier uses content for product improvement, so Mezvuat should send only non-sensitive or public documents unless it requests `no_training_providers` or `local_only`.
3. **OpenRouter $10 credit purchase.** It lifts the free-model limit from 50 to 1,000 requests per day. This is a purchase, so it is the user's call.
4. **Human reviewer** for about 200 calibration labels (about 4–5 h).
5. **Default per-key limits.** Proposed: 60 RPM, 200K TPM, $1/day estimated budget.
6. **Git remote and CI host** (GitHub Actions assumed).
7. **Cloud demo.** Deferred. Confirm it is not needed for the portfolio milestone (ADR-0011).
8. **Contract change requests** C2, C3 and C8 affect Mezvuat's client (flattened metadata tags, `include_usage: true`, fingerprint format). The coordinator needs to reconcile them with Mezvuat's review.
