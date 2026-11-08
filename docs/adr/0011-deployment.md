# ADR-0011: Deployment (local Docker Compose; cloud deferred)

- Status: Accepted (architecture phase)
- Date: 2026-11-08
- Spec refs: §2.6 (no Kubernetes control plane), §5.1 (independent deployability), §5.2 (Docker-based dev, CI), §6 ("Runs independently using Docker"), §8 item 10

## Context

The product must run on its own with Docker, and alongside Mezvuat without depending on it. On this machine the Docker VM is capped at 4 GB, C: is 94% full, other projects' containers hold ports 5432 and 6380, Ollama will run natively on Windows, and there is no git remote or CI yet (ENVIRONMENT.md). Nothing in the spec requires cloud deployment.

## Options considered

1. **Docker Compose locally, cloud deferred** (chosen).
2. A single small cloud VM running the same compose stack. This is the natural next step, but it costs money and adds secret management, TLS and egress. That choice belongs to the product owner.
3. Kubernetes or managed container platforms. These are out of scope (spec §2.6). There is no workload that needs them.

## Decision

`docker-compose.yml` with project name `modelmesh`. Every host port is configurable through env.

| Service | Image | Host port (default) | Memory limit | Profile |
|---|---|---|---|---|
| `api` | built from `Dockerfile` (`python:3.13-slim`, multi-stage, `uv sync --frozen --no-dev`, non-root, bge-small ONNX baked in at build time) | `8100` | 768 MB | default |
| `migrate` | same image, runs `alembic upgrade head` once | n/a | n/a | default |
| `postgres` | `postgres:17-alpine` | not published (`55432` with the `expose-db` profile) | 384 MB | default |
| `redis` | `redis:8-alpine` (the alpine variant is UNVERIFIED for the Query Engine; use the plain `redis:8` image if the M4 smoke test fails) | not published (`56379` with `expose-db`) | 320 MB | default |
| `jaeger` | `jaegertracing/jaeger:2.x` (exact tag pinned in M1) | `16686` UI, OTLP internal only | 256 MB | `obs` |

- **Ollama is not in compose.** It runs natively, and the API reaches it at `http://host.docker.internal:11434` (`OLLAMA_BASE_URL`). A compose `extra_hosts: host-gateway` entry handles this on non-Desktop engines.
- **Coexisting with Mezvuat.** Mezvuat's containers call ModelMesh at `http://host.docker.internal:8100/v1`, the same public endpoint any client uses. There is no shared Docker network, database or volume. An optional external network can be documented later, but it is never required.
- Secrets come from `.env`, which is gitignored and has no defaults for secrets. The container fails fast if a required secret is missing for an enabled provider.
- The default stack budget is about 1.7 GB of limits, and expected real use is under 1.2 GB. Measured in M1 with `docker stats`.
- **CI:** GitHub Actions (lint, type check, unit tests, integration tests with Postgres and Redis service containers, Alembic round-trip, OpenAPI conformance, image build). Hosting is GitHub (DECISIONS D7), and pushes happen only when the product owner says so. Until then, the same checks run through a `make`-style script (`scripts/check.ps1` and `scripts/check.sh`).
- **Release:** the image is tagged with the version and git SHA. The contract version is served in `X-ModelMesh-Contract`.

**Cloud: not in this phase (DECISIONS D8).** If the product owner later wants a public demo, the cheapest credible option is the same compose stack on a single VM behind Caddy (automatic TLS). Ollama would be replaced by cloud providers or a small CPU model there. This would be recorded as a new ADR with cost estimates. No cloud-specific code is written now.

## Consequences

- `docker compose up` gives a working gateway with no host dependencies beyond Docker and the optional native Ollama.
- The local stack is the only deployment target that is tested. Cloud readiness is limited to 12-factor config and a stateless API container.

## Verification plan

- M1: `docker compose up -d` from a clean clone. `/v1/health` returns `ok` within 60 s, and `docker stats` memory is recorded in the M1 notes.
- M1: from inside a throwaway container on a separate network, `curl http://host.docker.internal:8100/v1/health` succeeds. This simulates how Mezvuat reaches ModelMesh.
- M2: the `api` container reaches native Ollama through `host.docker.internal`.
- M8: a fresh-machine walkthrough of README quickstart, timed and documented.

## Amendment 2026-11-08 (docs/DECISIONS.md D5, D6)

- The `grafana/otel-lgtm` profile is removed because its components are AGPL (D5).
- **To confirm in M1:** Redis 8 is distributed under a choice of RSALv2, SSPLv1 or AGPLv3 (from memory; not re-verified). ModelMesh would run it unmodified as a separate service under RSALv2, not AGPL. If the product owner reads D5 as excluding it, ADR-0006 needs another vector store.
- The repository licence is MIT (D6). M1 adds `LICENSE` and a dependency licence check.
