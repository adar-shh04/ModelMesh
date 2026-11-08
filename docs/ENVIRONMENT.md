# ModelMesh: development environment inspection

Inspected: 2026-11-08, about 02:40 local time, using read-only commands only. Nothing was installed or changed.
Re-run these checks before M1 starts. Several facts below (Docker state, ports) depend on what else is running on the machine.

## 1. Verified facts

| Area | Finding | How verified |
|---|---|---|
| OS | Windows 11 Home Single Language, build 26300 | `Win32_OperatingSystem` |
| CPU | Intel Core i5-11400H, 6 cores / 12 threads | `Win32_Processor` |
| RAM | 15.7 GB | `Win32_ComputerSystem` |
| GPU | NVIDIA RTX 3050 Ti Laptop, **4096 MiB VRAM**, driver 610.60, CUDA UMD 13.3, WDDM | `nvidia-smi` |
| Disk | **C: 456 GB, 94% used (about 30 GB free)**. X: 47 GB, 23% used (36 GB free) | `df -h` |
| Python | **Only CPython 3.14.3** (`C:\Python314`). The `py` launcher lists only 3.14. pip 25.3. PyYAML and jsonschema are installed globally (not ours, left untouched) | `py -0p`, `python --version` |
| uv | **Not installed** (not on PATH, not a pip module) | `uv --version`, `py -m uv` |
| Node | v24.13.0, npm 11.6.2 | `node --version` |
| Git | 2.37.1.windows.1. Global `core.autocrlf=true` | `git --version`, `git config` |
| Repo | `X:\projects\modelmesh` on branch `main`, no commits. It contained only `docs/spec/PRODUCT_SPEC.md` and `docs/contract/inference-api-v1.md` | `git status` |
| Docker | CLI 29.8.1, Compose v5.5.1. **Daemon was running at inspection time** (server 29.8.1, context `desktop-linux`). The coordinator's note said the daemon was not running; that is no longer true, but the user may stop it again | `docker info` |
| Docker VM | WSL2 backend. **`.wslconfig` caps the VM at `memory=4GB` (`MemTotal` reports 3.82 GiB)** with `swap=2GB`. The `nvidia` container runtime is registered | `docker info`, `%USERPROFILE%\.wslconfig` |
| Docker disk | Images 19 GB and **build cache 34 GB (18 GB reclaimable)**, on C: | `docker system df` |
| Port conflicts | Another project's containers are running: `careeros-postgres-1` publishes **5432** and `careeros-redis-1` publishes **6380**. Ports 6379, 8000, 8080, 8100, 11434, 4317, 4318 and 16686 were free | `docker ps`, `netstat -ano` |
| Cached images | `postgres:17-alpine` (424 MB) and `redis:7-alpine` are already pulled. `redis:8` and Jaeger are not | `docker images` |
| Ollama | **Not installed** (no binary, no `%LOCALAPPDATA%\Programs\Ollama`, no `~/.ollama`) | filesystem check |
| Postgres / Redis CLIs | Not installed on the host. Containers will provide them | `psql`, `redis-cli` |
| Provider API keys | None available (as expected). None were requested or handled | n/a |

## 2. Implications and decisions

1. **Python version.** The system 3.14.3 would work: a PyPI wheel survey (2026-11-08) found cp314 `win_amd64` wheels for every library we plan to use, including LightGBM, onnxruntime, psycopg-binary, and gevent for Locust. We still pin **3.13 through uv** for maturity and a wider range of older pinned wheels; see ADR-0002. This needs uv, which is a single small binary. Proposed install, not run: `py -m pip install --user uv`, then `uv python install 3.13`. Nothing heavy is installed globally. Every dependency lives in `.venv` inside the repo.
2. **The 4 GB Docker VM is the binding constraint.**
   - The ModelMesh stack (API, Postgres, Redis, Jaeger) is budgeted at under 1.2 GB.
   - **Self-hosted Langfuse (Postgres + ClickHouse + Redis + MinIO; ClickHouse alone needs about 8 GiB) cannot run here.** This is one input to ADR-0009.
   - Running ModelMesh and Mezvuat's stack together will probably exceed 4 GB. **The user should raise `memory=` in `%USERPROFILE%\.wslconfig` (suggest 8 GB) before combined integration (M8).** That is a system setting, so the user must change it, not the agent.
3. **Ollama runs natively on Windows, not in Docker.** This uses the GPU directly (driver 610.60 is newer than the documented minimum of 551.61) and keeps model memory outside the 4 GB VM cap. Containers reach it at `http://host.docker.internal:11434`.
   - With 4 GB VRAM, 3–4B models (`qwen3:4b` at 2.5 GB, `llama3.2:3b` at 2.0 GB) are realistic.
   - Use `num_ctx` of about 4–8K and `OLLAMA_KV_CACHE_TYPE=q8_0`.
   - 7–8B models will split between CPU and GPU and run slowly.
   - Throughput on this GPU is **unmeasured**. It will be measured during M2 and not assumed.
4. **Ports.** ModelMesh must not use 5432 or 6380, and must never stop the other project's containers.
   - Postgres and Redis are not published to the host by default. When they are, they use env-configurable host ports (defaults 55432 and 56379).
   - API on 8100. Jaeger UI on 16686, under the optional `obs` profile.
   - Compose project name is `modelmesh`, so volumes and networks are namespaced.
5. **Disk.** C: is 94% full and Docker stores its data there.
   - Images stay small: `python:3.13-slim` base, no torch in the runtime image, and the ONNX embedding model (about 130 MB) instead of sentence-transformers/torch.
   - We reuse the already-pulled `postgres:17-alpine`.
   - Hugging Face datasets and model caches go to `X:` through `HF_HOME=<repo>/.cache/huggingface` (gitignored).
   - The user may want to run `docker builder prune`, which would reclaim about 18 GB. That is the user's call; the agent will not run it.
6. **Line endings.** Because global `core.autocrlf=true`, shell scripts and Dockerfiles copied into Linux images could get CRLF. A `.gitattributes` file forcing LF for `*.sh`, `Dockerfile` and `*.py` is included in this phase.
7. **GPU in Docker.** The `nvidia` runtime is available. It is not needed in v1, since the runtime embedding model runs on CPU and Ollama runs on the host.

## 3. Not verified / to verify at implementation time

- That the official `redis:8` image loads the Query Engine (FT.CREATE / vector search) by default. ADR-0006 depends on this, and M4 starts with a smoke test.
- Actual container memory use of each service under the 4 GB cap. M1 measures it with `docker stats`.
- Ollama tokens/s for the candidate models on this GPU (M2).
- Free-tier rate limits for each provider account. They are shown per account in each provider's console and differ from third-party figures; see ADR-0003 and ADR-0005.
- Network egress to provider endpoints. It cannot be tested without keys.
