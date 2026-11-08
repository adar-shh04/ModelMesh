# ADR-0001: Repository layout and module boundaries

- Status: Accepted (architecture phase)
- Date: 2026-11-08
- Spec refs: §2.5 (layers), §2.4 (lifecycle), §4.2 (ownership), §5.1 (independent repos), §7

## Context

ModelMesh is one deployable service with an offline ML workflow beside it: benchmarking, router training and evaluation. The spec's §2.5 lists 11 layers. The spec forbids microservices without a workload that justifies them, and forbids any dependency on Mezvuat. We need boundaries that can be enforced, so the production path (request → response) cannot accidentally import heavy training code, and so provider quirks cannot leak out of the adapters.

## Options considered

1. **Modular monolith, one Python package, with import rules enforced in CI** (chosen).
2. Several packages in one repo (a uv workspace: `modelmesh-gateway`, `modelmesh-ml`, `modelmesh-bench`). Stronger isolation, but more packaging overhead than a one-developer project needs right now.
3. Separate services (gateway, router service, accounting worker). Rejected: no workload justifies it, it costs extra memory under the 4 GB Docker cap (ENVIRONMENT.md), and it puts network hops on the routing path.

## Decision

A src-layout single package with these modules. Each module maps to one row of spec §2.5.

```
modelmesh/                      repo root
├── src/modelmesh/
│   ├── core/            shared kernel: settings (pydantic-settings), ids (mmr_ ULIDs), error taxonomy, clock
│   ├── api/             §2.5 API gateway: FastAPI app factory, routers per endpoint, contract models
│   │                    (Pydantic v2, mirror openapi-v1.yaml), SSE writer, error→HTTP mapping
│   ├── auth/            §2.5 Authentication: API-key verification, tenant scoping, rate limits & quotas
│   ├── policy/          §2.5 Policy engine: hard-constraint evaluation, data_policy, budgets → eligible set
│   ├── routing/         §2.5 Router: policy objectives, rules baseline, learned-router runtime (loads a
│   │                    versioned artifact), decision objects. No training code.
│   ├── ml/              §2.5 ML engine
│   │   ├── features/    feature extraction, imported by BOTH routing (online) and training (offline)
│   │   ├── artifacts/   artifact schema + loader (model file + feature spec + calibration table)
│   │   ├── training/    offline only: dataset builds, splits, kNN/LightGBM/LR training, calibration
│   │   ├── evaluation/  offline only: AIQ, cost-at-quality, regret, bootstrap CIs, reports
│   │   └── benchmark/   offline only: dataset loaders, runner (calls the gateway's public API), judges
│   ├── providers/       §2.5 Provider adapters: base protocol, OpenAI-compatible base, gemini, groq,
│   │                    openrouter, ollama, model catalog loader, SSE parser
│   ├── cache/           §2.5 Cache: key derivation, exact cache, semantic cache (Redis vector index)
│   ├── reliability/     §2.5 Reliability: retry/backoff, circuit breakers, provider-quota tracker, fallback executor
│   ├── accounting/      §2.5 Accounting: pricing tables, cost estimation, usage records
│   ├── observability/   §2.5 Observability: OTel setup, span helpers, metrics, JSON logging, redaction
│   ├── persistence/     §2.5 Persistence: SQLAlchemy models, repositories, Redis client, Alembic env
│   ├── pipeline/        request lifecycle (spec §2.4): composes auth→policy→cache→routing→execute→account
│   └── dashboard/       minimal server-rendered dashboard (M7); reads through service layer, not raw SQL
├── migrations/          Alembic versions
├── config/              non-secret, versioned data: model catalog, pricing, routing policy defaults
├── tests/               unit/, integration/, contract/ (OpenAPI conformance), live/ (key-gated), load/
├── docs/                spec/, contract/, adr/, ENVIRONMENT.md, IMPLEMENTATION_PLAN.md
├── data/                gitignored: downloaded datasets, benchmark outputs (only manifests committed)
└── artifacts/           gitignored: trained router artifacts (only the manifest committed)
```

**Dependency rules.** These are enforced with `import-linter` contracts in CI.

- `api` → `pipeline` → {`auth`, `policy`, `routing`, `cache`, `reliability`, `providers`, `accounting`, `observability`} → {`persistence`, `core`}.
- `providers` depends only on `core` and `observability`. Adapters never touch persistence or routing.
- `routing` may import `ml.features` and `ml.artifacts` only. **Online code never imports `ml.training`, `ml.evaluation` or `ml.benchmark`.** This keeps scikit-learn, MLflow and `datasets` out of the runtime image.
- `ml.benchmark` talks to ModelMesh **only over its public HTTP API**, the same way any client would. Benchmark data collection therefore goes through the production path (routing metadata, attempts, usage, traces) and validates the contract.
- Nothing imports `api` except the ASGI entrypoint. Nothing in this repo imports Mezvuat code, and there is no Mezvuat-specific branch in the code. "mezvuat" appears only as an example value of `metadata.application`.

**Test doubles.** A deterministic `fake` provider (used for load tests and CI) lives in `providers/fake.py`. It is registered only when `MODELMESH_ENABLE_FAKE_PROVIDER=true`, and the settings model refuses to enable it when `MODELMESH_ENV=production`. This satisfies the mandate that mocks be clearly isolated.

## Consequences

- One image and one process to run. Module boundaries are as strict as CI makes them, so the import-linter contracts are the real boundary.
- `ml.features` sits on the hot path and in training. That avoids train/serve skew, but its latency budget (p95 under 10 ms for the router as a whole) is checked by a benchmark test.
- If a separate worker is ever needed (for example asynchronous accounting writes), it can be split out of `pipeline` and `accounting` without crossing boundaries.

## Verification plan

- M1: `lint-imports` passes in CI. A deliberately violating import in a test fixture makes it fail.
- M1: a test asserts that `import modelmesh.api.app` does not load `sklearn`, `mlflow` or `datasets` (checked via `sys.modules`).
- M6: a router-latency micro-benchmark (p95 under 10 ms excluding embedding, and under 25 ms including bge-small on CPU), recorded in the M6 report.
