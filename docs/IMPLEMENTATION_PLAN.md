# ModelMesh: implementation plan

Status: **M1 to M4, the M5 harness and the M7 dashboard (without the benchmark view) are implemented. The M5 P0 pilot is partly collected (`milestones/P0_PILOT_REPORT.md`); P1 waits on the product owner (D2).** Results and what was not verified: `docs/milestones/VERIFICATION.md`. Contract: **1.0, frozen** (Mezvuat CRs in `contract/CR_RESOLUTIONS.md`). Product-owner decisions: `DECISIONS.md`, which overrides this plan where they differ.
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

Deliverables: `docs/ENVIRONMENT.md`, `docs/adr/0001–0011`, `docs/contract/openapi-v1.yaml`, contract 1.0 (plus `CR_RESOLUTIONS.md`), this plan, README, `.gitignore`, `.gitattributes`, `.env.example`.
Verification: the OpenAPI file parses, every `$ref` resolves, schemas are valid JSON Schema 2020-12, all embedded examples validate, and the negative cases are rejected (checked with a local script on 2026-11-08). No full OpenAPI linter (Redocly or Spectral) has been run yet. That moves to M1 CI.
**Gate to M1:** an explicit go-ahead to start implementation. The contract is frozen and the Mezvuat change requests are reconciled.

## M1: Skeleton, one adapter, unified API (explicit models)

Deliverables:
- `pyproject.toml` (Python 3.13, uv groups), `uv.lock`, ruff/mypy/import-linter config, and `scripts/check.{ps1,sh}`. CI workflow file, which activates once a remote exists.
- `core` (settings, ids, errors), `api` (contract models, error mapping, SSE writer), and `pipeline` (explicit-model path only).
- `providers`: the `OpenAICompatAdapter` base, an **Ollama** subclass (needs no key), and the isolated `fake` provider (ADR-0001).
- `persistence`: SQLAlchemy models plus the first Alembic migration (tenants, api_keys, requests, attempts, usage_records). A bootstrap CLI creates a tenant and key.
- `observability`: OTel traces to Jaeger, JSON logs with a redaction filter.
- Endpoints: `/v1/health`, `/v1/models`, and `/v1/chat/completions` (explicit model; non-stream and stream; `include_usage` semantics; keep-alive; in-stream errors).
- `Dockerfile`, `docker-compose.yml` (api, migrate, postgres, redis, plus `obs` and `expose-db` profiles).
- MIT `LICENSE` (D6) and a CI check that every dependency's licence is compatible, with no AGPL dependencies (D5).

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
- **uv** and Python 3.13 must be installed (`py -m pip install --user uv`). Approved (D9).
- **Ollama** must be installed natively, followed by `ollama pull qwen3:4b` (2.5 GB) and `ollama pull llama3.2:3b` (2.0 GB). The user does this. The live test is skipped without it, but Ollama is a required path for `local_only` requests (D4), so M2 can't be signed off without a passing live test.
- **Docker Desktop must be running.** It was running at inspection time.
- CI runs on GitHub Actions (D7). Commits stay local until the product owner says to push. Until then, the local check script runs the same gates.

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
| Uncalibrated task types (contract §2.1.1): every policy succeeds without `min_quality` and reports the right warning; an explicit `min_quality` returns 422; `local_only` with only small models still routes | D4 |
| Structured output: native/best-effort/emulated paths, repair, 502 `schema_validation_failed`, buffered stream | D1, D3 |

Blockers:
- **API keys from the user:** `GROQ_API_KEY` (console.groq.com), `GEMINI_API_KEY` (aistudio.google.com; new keys are service-account-bound auth keys), and `OPENROUTER_API_KEY` (openrouter.ai). The agent cannot create accounts. Each adapter can be built and tested on fixtures without its key, but live verification waits for it.
- Gemini free tier, Groq and OpenRouter stay classified as `may_train` or `unknown` until reviewed (D4). Clients choose per request through `routing.data_policy`, which is never relaxed, including during fallback.

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

Blockers: none new. Default limits for new keys are 20 RPM, 50K TPM and $0.25/day estimated (D1). Benchmark, load and integration runs use explicitly configured keys, and record those overrides with each run.

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
- **Stop after P0** and report the yield at free limits and what the OpenRouter credit would change (D2). The product owner decides before P1 continues.
- **P1 full**: about 1,200 prompts, after that decision.

Acceptance tests:

| Test | Maps to |
|---|---|
| Auto metrics match reference implementations on fixtures. The judge canary ("score this 5") does not move scores | D6 (label quality) |
| P0 report: measured tokens, latency and quota use per model vs the ADR-0005 budget; P1 sizes re-planned | D9 |
| LLM pre-labels confirmed or corrected by the product owner (D3), 150 judge + 50 auto, with the pre-label hidden on a random subset; agreement with and without it reported. Cohen's κ per task_type reported. Task types with κ < 0.6 flagged low-trust | D5/D6 (trustworthy labels) |
| Every observation links to a real `mmr_` request record. Failures are recorded, not dropped | D9 |

Blockers:
- The three cloud API keys.
- Dataset downloads are approved, cached on X: (D9).
- The product owner's time to confirm labels (about 4–5 h, D3).
- The **~$10 OpenRouter credit is deferred** (D2). Without it, OpenRouter's free models are limited to 50 requests per day, so the pilot runs at free limits and the product owner decides afterwards. An agent never makes this purchase.
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

## Open questions for the product owner: resolved

All were answered in `DECISIONS.md` (2026-11-08).

| # | Question | Resolution |
|---|---|---|
| 1 | Provider data-use attestations | **Resolved (D4).** Gemini free tier, Groq and OpenRouter stay `may_train` or `unknown` until reviewed |
| 2 | `data_policy` for new keys | **Resolved (D4).** Clients choose per request: Mezvuat sends `any` for verified public-corpus workspaces and `local_only` for private ones. Never relaxed |
| 3 | OpenRouter $10 credit | **Resolved as deferred (D2).** Stop after the M5 pilot, report the yield, and the product owner decides then |
| 4 | Human reviewer | **Resolved (D3).** LLM pre-labels, the product owner confirms or corrects |
| 5 | Default per-key limits | **Resolved (D1).** 20 RPM, 50K TPM, $0.25/day estimated |
| 6 | Git remote and CI host | **Resolved (D7).** GitHub with GitHub Actions; push only on the product owner's say-so |
| 7 | Cloud demo | **Resolved (D8).** Not in this phase; local Docker Compose only |
| 8 | Contract change requests | **Resolved.** Reconciled in `contract/CR_RESOLUTIONS.md`; contract frozen at 1.0 |
