# Architecture Decision Records

Template: **Context → Options considered → Decision → Consequences → Verification plan**, plus a header with status, date and spec references.
An ADR changes only through a new "Supersedes ADR-NNNN" record or a dated amendment section. Facts marked UNVERIFIED must be re-checked when the corresponding milestone starts.

| ADR | Title | Spec §8 item | Resolves contract open question |
|---|---|---|---|
| [0001](0001-repository-layout-and-module-boundaries.md) | Repository layout and module boundaries | n/a | n/a |
| [0002](0002-python-tooling-and-web-framework.md) | Python 3.13, uv, ruff, mypy, pytest, FastAPI + Pydantic v2 | n/a | n/a |
| [0003](0003-provider-adapter-interface.md) | Provider adapter interface; Gemini, Groq, OpenRouter, Ollama | n/a | n/a |
| [0004](0004-learned-routing-algorithm-and-baseline.md) | Learned routing algorithm, baselines, metrics, splits | 4 | Q1 (quality scale) |
| [0005](0005-benchmark-dataset-and-quality-labels.md) | Benchmark dataset, quality labels, free-tier budget | 5 | n/a |
| [0006](0006-semantic-cache-correctness-and-isolation.md) | Cache correctness and tenant isolation | 6 | n/a |
| [0007](0007-streaming-and-structured-output.md) | Streaming and structured-output compatibility | 7 | Q3 (structured output emulation) |
| [0008](0008-reliability-retries-breakers-fallback.md) | Retries, backoff, circuit breakers, quotas, fallback | n/a | Q2 (latency budget vs timeout) |
| [0009](0009-observability-and-trace-correlation.md) | OpenTelemetry, Langfuse decision, trace correlation | 8 | n/a |
| [0010](0010-persistence-keys-quotas-rate-limits.md) | Postgres/Alembic, Redis, API keys, quotas, rate limits | n/a | Q4 (retention default) |
| [0011](0011-deployment.md) | Docker Compose locally; cloud deferred | 10 | n/a |

Spec §8 items 1–3 and 9 (embedding/reranking models, Qdrant, BM25, document parsing) belong to Mezvuat and are out of scope here.
