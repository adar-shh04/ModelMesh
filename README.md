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
| M5 Benchmark harness | Harness built. P0 pilot partly collected (`docs/milestones/P0_PILOT_REPORT.md`). P1 full run in progress on free tiers |
| M6 Learned router | Preliminary, on P0 pilot data (`docs/milestones/M6_P0_PRELIMINARY_REPORT.md`). Not promoted |
| M7 Dashboard | Done, except the benchmark view, which needs M6 |
| M8 Load tests, docs | Integration guide, quickstart and Locust scenarios written (`loadtests/`). Load tests and the timed fresh-clone test not yet run |

A preliminary learned router was trained on the P0 pilot. It did not pass the ADR-0004 criterion and is not promoted, so `routing.mode` is `rules_fallback` and every task type reports `calibrated: false`. Verification notes for each milestone are in `docs/milestones/`.

## Quickstart

You need Docker with Compose v2 and `git`. The commands below are for a POSIX shell (Linux, macOS, WSL or Git Bash on Windows).

**1. Clone and configure.**

```bash
git clone https://github.com/adar-shh04/ModelMesh.git modelmesh && cd modelmesh
cp .env.example .env
```

Edit `.env` and set the two required secrets to long random strings, for example the output of `openssl rand -hex 32`:

```bash
MODELMESH_API_KEY_PEPPER=<random>
MODELMESH_ADMIN_TOKEN=<random>
```

Then either add at least one provider key (`GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`, or `OLLAMA_BASE_URL`), or set `MODELMESH_ENABLE_FAKE_PROVIDER=true` to try the API with a deterministic fake provider and no external calls. Leave every other variable empty to use its default. If port 8100 is taken, set `MODELMESH_API_HOST_PORT`.

**2. Start the stack.** The first start builds the image. That downloads the locked Python dependencies and the embedding model, so it needs network access and takes a few minutes. Later starts reuse the image.

```bash
docker compose up -d
docker compose ps                      # wait until api is "healthy"
curl http://localhost:8100/v1/health
```

`migrate` runs the database migrations once and exits. `api`, `postgres` and `redis` keep running.

**3. Create a tenant and an API key** with the admin API. The key's secret is shown once, in the `key` field.

```bash
export ADMIN=<your MODELMESH_ADMIN_TOKEN>
curl -s -X POST localhost:8100/admin/v1/tenants -H "Authorization: Bearer $ADMIN" \
  -H 'content-type: application/json' -d '{"name":"demo"}'
# -> {"id": "<tenant id>", ...}
curl -s -X POST localhost:8100/admin/v1/tenants/<tenant id>/keys -H "Authorization: Bearer $ADMIN" \
  -H 'content-type: application/json' -d '{"name":"demo-key"}'
# -> {"id": "...", "key": "mm_live_...", "prefix": "...", "limits": {...}}
export MM_KEY=<the "key" value>
```

You can also create keys in the dashboard at `http://localhost:8100/dashboard`. Sign in with `MODELMESH_ADMIN_TOKEN`.

**4. Send a request.**

```bash
curl -s localhost:8100/v1/chat/completions -H "Authorization: Bearer $MM_KEY" -H 'content-type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Hello"}],"routing":{"policy":"cost_optimized"}}'
```

With only the fake provider, `auto` chooses among the fake models. The reply is `echo: Hello`, and the `modelmesh` object shows the routing decision, the attempts, cost and latency. Any OpenAI SDK works too:

```python
from openai import OpenAI

client = OpenAI(api_key="mm_live_...", base_url="http://localhost:8100/v1")
response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={"routing": {"policy": "cost_optimized"}, "metadata": {"task_type": "general"}},
)
print(response.choices[0].message.content, response.model_extra["modelmesh"]["model"])
```

**5. Stop.** `docker compose down` stops the stack and keeps the database. `docker compose down -v` also deletes it.

New keys default to 20 requests per minute, 50,000 tokens per minute and $0.25 of estimated cost per day. Pass `limits` when you create a key to change them.

**Next steps:**
- Client developers: start with **`docs/INTEGRATION_GUIDE.md`**. It covers error handling and fallback, `data_policy`, cache namespaces and feedback.
- The contract is `docs/contract/inference-api-v1.md` and `docs/contract/openapi-v1.yaml`.
- The operator API is `docs/contract/admin-api.yaml`.

**Dashboard.** The operator dashboard at `/dashboard` shows:
- usage and cost
- latency and reliability
- requests, with trace links
- keys and the routing policy

It never calls a provider. Keep it on a trusted network: the sign-in form has no attempt limit.

**Ollama.** Run Ollama on the host and set `OLLAMA_BASE_URL=http://host.docker.internal:11434`. Set `OLLAMA_CONTEXT_LENGTH` on the Ollama server to size the context window.

**Tracing.** `docker compose --profile obs up -d jaeger` starts Jaeger at `http://localhost:16686`. Set `OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4318` in `.env` to send traces to it.

## Development

```bash
uv sync --group ml
docker compose --profile test up -d postgres redis-test   # tests use their own Redis on 56390, never the gateway's
./scripts/check.sh    # or scripts/check.ps1: ruff, mypy, import-linter and pytest
```

Live-provider tests are marked `live` and are skipped without credentials. A skipped test is not a pass.

Load tests (Locust) live in `loadtests/`. Run them only against a separate instance; `loadtests/README.md` explains how.

## Licences

ModelMesh is MIT-licensed (see `LICENSE`). No dependency combined into ModelMesh is AGPL-licensed, and a test enforces this for the Python runtime dependencies.

ModelMesh depends on a **Redis 8** server as a separate process. Redis does not ship as part of ModelMesh. Redis 8 is offered under your choice of RSALv2, SSPLv1 or AGPLv3 (checked 2026-11-14 against `LICENSE.txt` in the Redis repository, for the 8.10.3 server in the Compose stack; see ADR-0006). This project uses it unmodified, under **RSALv2**, reached over the network. If you self-host ModelMesh you choose your own Redis licence and are responsible for complying with it. The cache sits behind an interface (`modelmesh.cache.base.ResponseCache`) so it can be replaced.
