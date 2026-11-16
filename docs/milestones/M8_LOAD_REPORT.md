# M8 load-test report

> **Status: NOT YET RUN.** The scenarios are written (`loadtests/`). None has been run as a load test. Every result table below is empty on purpose.
>
> They were not run because the P1 benchmark is using the only gateway and the machine (started 2026-11-16, expected to last about two weeks). A load test on the same machine would skew P1's latency data, and the real-provider scenario would spend the same free-tier quotas.

How to run the scenarios against a separate instance: `loadtests/README.md`.

## Acceptance tests (IMPLEMENTATION_PLAN.md, M8)

| Test | Scenario | Result |
|---|---|---|
| Overhead report: p50 and p95 added latency at N RPS on the fake provider | `loadtests/fake_overhead.py` | **Not yet run** |
| Breaker and fallback behaviour under chaos (success rate, p95) | `loadtests/chaos.py`, gateway with `MODELMESH_FAKE_PROVIDER_FAILURE_RATE=0.3` | **Not yet run** |
| Small real-provider run within quota | `loadtests/real_provider.py`, at most 20 requests | **Not yet run** |
| Fresh-clone quickstart, timed | README quickstart on a clean machine or a clean Docker state | **Not yet run**: it needs a stack built and started from nothing, which cannot happen while the P1 gateway must stay up |
| Combined test with Mezvuat (shared trace id, `correlation_id` listing, feedback join) | Not written yet | **Not yet run**: needs Mezvuat's client and both stacks running at once (WSL2 memory cap) |

## Scenario 1: gateway overhead, fake provider

**Configuration to record:**
- the gateway (option A or B in `loadtests/README.md`), with `MODELMESH_FAKE_PROVIDER_DELAY_MS=0` and `MODELMESH_FAKE_PROVIDER_FAILURE_RATE=0`
- the key limits printed by `setup_key.py --profile fake`
- users, `MM_LOAD_RPS_PER_USER` and the duration
- the machine, and what else was running

| Target RPS | Achieved RPS | Success rate | Client p50 / p95 (ms) | Gateway overhead p50 / p95 (ms) |
|---|---|---|---|---|
| not yet run | | | | |

## Scenario 2: chaos, 30% upstream failures

| Achieved RPS | Success rate | Client p95 (ms) | Successful requests using fallback | Attempts per request | Breaker states seen |
|---|---|---|---|---|---|
| not yet run | | | | | |

**Expected behaviour to check:**
- A failed attempt falls back to the next eligible fake model.
- A request fails only when every attempt fails.
- Breakers open and close as ADR-0008 specifies. The `/v1/providers` polls are in the run's output file.

## Scenario 3: real providers within quota

| Provider(s) | Requests | Success rate | Client p50 / p95 (ms) | Gateway overhead p50 / p95 (ms) |
|---|---|---|---|---|
| not yet run | | | | |

## Harness smoke check (not a result)

Two short smoke runs (10 s in total) checked that the Locust files load, authenticate, send valid requests, parse responses and write output that `summarize.py` reads. It was not a measurement. Its latencies were taken on a machine running the P1 benchmark, and they are not reported.

### Smoke run (2026-11-16)

**Target.** The API ran in-process through `loadtests/serve.py` on port 8201:
- the fake provider only, with no provider keys in its environment and no `.env` read
- the test database (`modelmesh_test`) and the test Redis (port 56390)

It made no provider calls and sent no traffic to port 8100. The key came from `setup_key.py --profile fake`.

**Runs.**
- `fake_overhead.py`: 5 s, 2 users at 5 req/s each, stream share 0.3. 20 requests, all successful, streamed and non-streamed. The output file parsed with `summarize.py`.
- `chaos.py`: 5 s, against the same API restarted with `MODELMESH_FAKE_PROVIDER_FAILURE_RATE=0.3`. 30 requests, all successful.
  - Attempt outcomes recorded: 30 `success`, 18 `upstream_unavailable`, 1 `circuit_open`. 12 requests needed more than one attempt.
  - The `/v1/providers` poller recorded 4 snapshots.
- `real_provider.py`: only loaded (`locust --list`), not run.

**What the smoke run did not show.** It cannot show anything about performance:
- The runs were seconds long, on a machine loaded by the P1 benchmark.
- The load generator and the gateway shared the machine.

**Fixed during the smoke run.** `uvicorn` on Windows picks the Proactor event loop, which psycopg's async driver rejects: the database health check was `down`. `loadtests/serve.py` starts uvicorn on a selector loop instead. The Compose image runs on Linux and is not affected.
