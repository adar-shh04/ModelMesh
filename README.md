# ModelMesh

*One interface. Multiple models. Intelligent inference.*

ModelMesh is a self-hostable, provider-independent LLM inference gateway. Applications call one OpenAI-compatible API, and ModelMesh handles provider authentication, request and response translation, retries and fallback, caching, usage accounting and tracing. Routing is driven by explicit, caller-supplied policies and constraints. A learned router, trained on measured quality, cost, latency and reliability, is the planned next stage and will be compared against the rules baseline on held-out data.

Providers: Google Gemini (AI Studio), Groq, OpenRouter, and local models through Ollama.
ModelMesh is independent of any one client application. Mezvuat AI is its first consumer, but integration happens only through the public API.

## Status

| Milestone | State |
|---|---|
| M0 Architecture and contract (frozen at 1.0) | Done |
| M1 Skeleton, adapters, unified API | Done |
| M2 All adapters, reliability, rules-based routing | Done; live-provider tests not run (no API keys yet) |
| M3 Accounting, API keys, limits, feedback | Done |
| M4 Exact and semantic caching | Done |
| M5 Benchmark harness | Harness built. P0 pilot partly collected (`docs/milestones/P0_PILOT_REPORT.md`) |
| M6 Learned router | Not started |
| M7 Dashboard | Done, except the benchmark view, which needs M6 |
| M8 Load tests | Not started |

No router has been trained yet. Until one is, `routing.mode` is `rules_fallback` and every task type reports `calibrated: false`. Verification notes for each milestone are in `docs/milestones/`.

## Quickstart

Docker Compose is the only requirement.

```bash
cp .env.example .env
# Set at least MODELMESH_API_KEY_PEPPER and MODELMESH_ADMIN_TOKEN to long random strings.
# Add provider keys, or set MODELMESH_ENABLE_FAKE_PROVIDER=true to try the API with a deterministic fake provider.
docker compose up -d
curl http://localhost:8100/v1/health
```

Create a tenant and an API key with the admin API, using your `MODELMESH_ADMIN_TOKEN`:

```bash
curl -X POST localhost:8100/admin/v1/tenants -H "Authorization: Bearer $ADMIN" -H 'content-type: application/json' -d '{"name":"demo"}'
curl -X POST localhost:8100/admin/v1/tenants/<tenant id>/keys -H "Authorization: Bearer $ADMIN" -H 'content-type: application/json' -d '{"name":"demo-key"}'
```

Then use any OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(api_key="mm_live_...", base_url="http://localhost:8100/v1")
client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={"routing": {"policy": "cost_optimized"}},
)
```

New keys default to 20 requests per minute, 50,000 tokens per minute and $0.25 of estimated cost per day. Pass `limits` when you create a key to change them.

The contract is `docs/contract/inference-api-v1.md` and `docs/contract/openapi-v1.yaml`. The operator API is `docs/contract/admin-api.yaml`.

The operator dashboard is at `http://localhost:8100/dashboard`. Sign in with `MODELMESH_ADMIN_TOKEN`. It shows usage and cost, latency and reliability, requests with trace links, keys and the routing policy, and it never calls a provider. Keep it on a trusted network: the sign-in form has no attempt limit.

For Ollama, run it on the host and set `OLLAMA_BASE_URL=http://host.docker.internal:11434`. Set `OLLAMA_CONTEXT_LENGTH` on the Ollama server to size the context window.

## Development

```bash
uv sync --group ml
docker compose up -d postgres redis
./scripts/check.sh    # or scripts/check.ps1: ruff, mypy, import-linter and pytest
```

Live-provider tests are marked `live` and are skipped without credentials. A skipped test is not a pass.

## Licences

ModelMesh is MIT-licensed (see `LICENSE`). No dependency combined into ModelMesh is AGPL-licensed, and a test enforces this for the Python runtime dependencies.

ModelMesh depends on a **Redis 8** server as a separate process. Redis does not ship as part of ModelMesh. Redis 8 is offered under your choice of RSALv2, SSPLv1 or AGPLv3 (checked 2026-11-09 against `LICENSE.txt` in the Redis repository). This project uses it unmodified, under **RSALv2**, reached over the network. If you self-host ModelMesh you choose your own Redis licence and are responsible for complying with it. The cache sits behind an interface (`modelmesh.cache.base.ResponseCache`) so it can be replaced.
