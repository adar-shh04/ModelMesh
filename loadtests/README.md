# M8 load tests

Three Locust scenarios from `docs/IMPLEMENTATION_PLAN.md` (M8). Results go in `docs/milestones/M8_LOAD_REPORT.md`.

| File | What it measures | Target gateway configuration |
|---|---|---|
| `fake_overhead.py` | Gateway overhead (p50, p95) at a fixed request rate, fake provider, cache off | `MODELMESH_ENABLE_FAKE_PROVIDER=true`, delay 0, failure rate 0 |
| `chaos.py` | Success rate, p95, fallback use and breaker states with 30% of upstream calls failing | `MODELMESH_ENABLE_FAKE_PROVIDER=true`, `MODELMESH_FAKE_PROVIDER_FAILURE_RATE=0.3` |
| `real_provider.py` | A short run against real providers, capped at 20 requests by default | Provider keys; **free-tier quotas are shared with anything else using the same accounts** |

Support files:
- `mmload.py`: request bodies, response parsing and the per-request recorder (no Locust import; unit-tested in `tests/unit/test_loadtests.py`).
- `mmload_locust.py`: the shared Locust user and the `/v1/providers` poller.
- `setup_key.py`: creates a tenant and key on the target and writes the key to `.cache/load_key`. The key is never printed.
- `summarize.py`: turns a run's JSON Lines output into the Markdown tables the report uses.
- `compose.load.yml`: overrides for a second Compose stack.
- `serve.py`: runs the API in-process (option B).

## Rules

- **Never point a load test at a gateway that is doing other work**, such as a running benchmark. Its latency numbers would be skewed and its quotas, rate limits and budgets used up. Always run against a separate instance.
- There is no default base URL. Locust's `--host` is the target, and it is required.
- Keys for load tests carry explicit limits, as `docs/DECISIONS.md` (D1) requires. `setup_key.py --profile fake` sets limits far above the load, so ModelMesh's own limiter never shapes the result. `--profile real` sets the D1 defaults explicitly. Record the limits it prints with the run.
- The load generator shares the machine with the gateway in a local run. Report the machine, and say what else was running.

## 1. Start a separate instance

Pick one of these two options.

### Option A: a second Compose stack (closest to a deployment)

Create `.env.load` in the repository root. It is gitignored by the `.env.*` rule. Use new random secrets, leave every provider key empty for the fake scenarios, and choose ports that nothing else uses:

```bash
MODELMESH_ENV_FILE=.env.load
MODELMESH_API_KEY_PEPPER=<random>
MODELMESH_ADMIN_TOKEN=<random>
MODELMESH_API_HOST_PORT=8200
MODELMESH_PG_HOST_PORT=55433
MODELMESH_REDIS_HOST_PORT=56381
```

Then build the image under its own tag and start the stack under its own project name:

```bash
docker build -t modelmesh-api:load .
docker compose -p mmload --env-file .env.load -f docker-compose.yml -f loadtests/compose.load.yml up -d --no-build
curl http://127.0.0.1:8200/v1/health
```

- **Failure rate.** For the chaos run, set `MODELMESH_FAKE_PROVIDER_FAILURE_RATE=0.3` in `.env.load`, then recreate only this project's API: `docker compose -p mmload ... up -d --no-build api`.
- **Removal.** `docker compose -p mmload ... down -v` removes this stack and its volume only.
- **Memory.** The stack needs about 1.2 GB more inside Docker's VM. With Docker capped at 4 GB, check `docker stats` before starting it next to another stack.

### Option B: the API in-process on the host

This is faster to set up. It uses a separate Postgres database and a separate Redis. Don't use the test Redis (`redis-test`, port 56390) while the test suite runs, because the suite flushes it.

```bash
export MODELMESH_DATABASE_URL=postgresql+psycopg://modelmesh:modelmesh@127.0.0.1:55432/modelmesh_load
export MODELMESH_REDIS_URL=redis://127.0.0.1:56390/0
export MODELMESH_API_KEY_PEPPER=<random> MODELMESH_ADMIN_TOKEN=<random>
export MODELMESH_ENABLE_FAKE_PROVIDER=true MODELMESH_FAKE_PROVIDER_FAILURE_RATE=0
# Create the database once (psql or any client), then:
uv run alembic upgrade head
uv run python loadtests/serve.py --port 8200   # uvicorn on a selector event loop, which psycopg needs on Windows
```

With option B, the gateway's environment is the shell's, so make sure no provider keys are exported for the fake scenarios.

## 2. Create a key

```bash
MODELMESH_ADMIN_TOKEN=<the instance's token> uv run python loadtests/setup_key.py --base-url http://127.0.0.1:8200 --profile fake
```

## 3. Run

```bash
mkdir -p data/load

# Overhead: 50 users x 10 req/s = 500 req/s target for 5 minutes
MM_LOAD_OUT=data/load/fake_overhead.jsonl uv run --group load locust -f loadtests/fake_overhead.py \
  --host http://127.0.0.1:8200 --headless -u 50 -r 10 -t 5m --csv data/load/fake_overhead

# Chaos (gateway restarted with MODELMESH_FAKE_PROVIDER_FAILURE_RATE=0.3)
MM_LOAD_OUT=data/load/chaos.jsonl uv run --group load locust -f loadtests/chaos.py \
  --host http://127.0.0.1:8200 --headless -u 20 -r 5 -t 5m --csv data/load/chaos

# Real providers: a gateway with provider keys, a key from --profile real, at most 20 requests
MM_LOAD_OUT=data/load/real_provider.jsonl MM_LOAD_REAL_PROVIDERS=groq uv run --group load locust \
  -f loadtests/real_provider.py --host http://127.0.0.1:8200 --headless -u 1 -r 1 -t 10m

uv run python loadtests/summarize.py data/load/fake_overhead.jsonl data/load/chaos.jsonl
```

`data/` is gitignored. Copy the summaries into `docs/milestones/M8_LOAD_REPORT.md`, together with:
- the machine
- what else was running
- the gateway's configuration and the key's limits
- the exact command

To find the request rate at which overhead starts to grow, repeat the overhead run at several user counts. Keep `MM_LOAD_RPS_PER_USER` fixed.

## What the numbers mean

- **Client p50 and p95** are measured by the Locust process from sending the request to reading the whole body. They include the loopback network and Locust's own cost.
- **Overhead p50 and p95** come from the gateway's `modelmesh.latency_ms.overhead` on each successful response, which is total time minus provider time. With the fake provider at zero delay, this is the gateway's own cost: authentication, limits, routing, execution and accounting.
- **Chaos.** A request fails only when every eligible attempt fails, or when the breakers have opened on every candidate. Record the success rate, the p95, the share of successful requests that needed fallback, the attempts per request, and the breaker states seen in the `/v1/providers` polls.
