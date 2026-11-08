# ModelMesh

*One interface. Multiple models. Intelligent inference.*

ModelMesh is a self-hostable, provider-independent LLM inference gateway. Applications call one OpenAI-compatible API, and ModelMesh handles provider authentication, request and response translation, retries and fallback, caching, usage accounting and tracing. Its defining feature is a **learned router** that picks a model per request based on *measured* quality, cost, latency and reliability, under explicit, caller-supplied constraints. That router is evaluated honestly against a rules-based baseline on held-out data.

Initial providers: Google Gemini (AI Studio), Groq, OpenRouter, and local models via Ollama.
ModelMesh is independent of any one client application. Mezvuat AI is its first consumer, but integration happens only through the public API.

## Status: architecture phase

No application code exists yet. This repository currently contains:

- `docs/spec/PRODUCT_SPEC.md`: product specification (source of truth)
- `docs/ENVIRONMENT.md`: verified development-machine facts and their implications
- `docs/adr/`: architecture decision records 0001–0011
- `docs/contract/inference-api-v1.md` and `openapi-v1.yaml`: the public API contract (1.0-rc2; Mezvuat change requests resolved in `CR_RESOLUTIONS.md`)
- `docs/IMPLEMENTATION_PLAN.md`: milestones, acceptance tests and blockers

Implementation starts at milestone M1 once the product owner authorizes it.

## Configuration

Settings come from environment variables. `.env.example` lists the variable names. Copy it to `.env`, which is gitignored, and fill in values locally. Never commit secrets.
