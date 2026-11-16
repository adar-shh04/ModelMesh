"""M8 scenario 3: a small real-provider run, bounded so that it stays inside free-tier quotas.

This calls real providers through the target gateway. It sends at most ``MM_LOAD_REAL_MAX_REQUESTS``
requests in total (default 20) at ``MM_LOAD_REAL_RPS`` requests per second per user (default 0.1, i.e. one
every 10 s), then stops the run. Run it with one user. Check each provider's current free-tier limits
before raising either number, and never run it while a benchmark is using the same provider accounts.

    uv run --group load locust -f loadtests/real_provider.py --host http://127.0.0.1:8200 \
        --headless -u 1 -r 1 -t 10m --csv data/load/real_provider

Environment: ``MM_LOAD_REAL_MAX_REQUESTS``, ``MM_LOAD_REAL_RPS``, ``MM_LOAD_REAL_PROVIDERS`` (comma-separated
``routing.allowed_providers``, default ``groq``), ``MM_LOAD_POLICY`` (default cost_optimized),
``MM_LOAD_DATA_POLICY`` (default ``any``; ``local_only`` restricts the run to Ollama).
"""

from __future__ import annotations

import os
import threading
from typing import Any

from locust import constant_throughput, task
from mmload import TASK_TYPES, chat_body, env_list
from mmload_locust import GatewayUser, install

MAX_REQUESTS = int(os.environ.get("MM_LOAD_REAL_MAX_REQUESTS", "20"))
RPS_PER_USER = float(os.environ.get("MM_LOAD_REAL_RPS", "0.1"))
PROVIDERS = env_list("MM_LOAD_REAL_PROVIDERS", "groq")
POLICY = os.environ.get("MM_LOAD_POLICY", "cost_optimized")
DATA_POLICY = os.environ.get("MM_LOAD_DATA_POLICY", "any")

_sent = 0
_lock = threading.Lock()


def meta() -> dict[str, Any]:
    return {
        "max_requests": MAX_REQUESTS,
        "rps_per_user": RPS_PER_USER,
        "providers": PROVIDERS,
        "policy": POLICY,
        "data_policy": DATA_POLICY,
        "sent": _sent,
    }


install("real_provider", meta)


class RealProviderUser(GatewayUser):
    scenario = "real_provider"
    wait_time = constant_throughput(RPS_PER_USER)

    def on_start(self) -> None:
        from locust.runners import WorkerRunner

        if isinstance(self.environment.runner, WorkerRunner):
            raise SystemExit("real_provider.py caps requests per process; run it without workers")
        super().on_start()

    @task
    def routed_chat(self) -> None:
        global _sent
        with _lock:
            if _sent >= MAX_REQUESTS:
                if self.environment.runner is not None:
                    self.environment.runner.quit()
                return
            task_type = TASK_TYPES[_sent % len(TASK_TYPES)]
            _sent += 1
        body = chat_body(task_type, policy=POLICY, allowed_providers=PROVIDERS, max_tokens=128, run_id=self.run_id)
        body["routing"]["data_policy"] = DATA_POLICY
        self.chat(body, f"chat:{task_type}")
