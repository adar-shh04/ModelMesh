"""M8 scenario 2: chaos. 30% of upstream calls fail; measure fallback, breakers, success rate and p95.

The target gateway must run with ``MODELMESH_ENABLE_FAKE_PROVIDER=true`` and
``MODELMESH_FAKE_PROVIDER_FAILURE_RATE=0.3`` (every fake model then answers 503 on about 30% of calls).
Requests route with ``model: "auto"`` and the default ``fallback: "within_constraints"``, so a failed attempt
falls back to the next eligible fake model. ``/v1/providers`` is polled every ``MM_LOAD_PROVIDER_POLL_S``
seconds (default 5) to record circuit-breaker states during the run.

    uv run --group load locust -f loadtests/chaos.py --host http://127.0.0.1:8200 \
        --headless -u 20 -r 5 -t 5m --csv data/load/chaos

Environment: as ``fake_overhead.py`` (``MM_LOAD_RPS_PER_USER`` default 5 here), plus
``MM_LOAD_PROVIDER_POLL_S``. The injected rate itself is server configuration; record it with the run
(``MM_LOAD_INJECTED_FAILURE_RATE``, default 0.3, is copied into the output's metadata, nothing more).
"""

from __future__ import annotations

import os
import random
from typing import Any

from locust import constant_throughput, task
from mmload import TASK_TYPES, chat_body, env_list
from mmload_locust import GatewayUser, install

RPS_PER_USER = float(os.environ.get("MM_LOAD_RPS_PER_USER", "5"))
MODELS = env_list("MM_LOAD_MODELS", "fake/echo-small,fake/echo-medium,fake/echo-large")
POLICY = os.environ.get("MM_LOAD_POLICY", "cost_optimized")
POLL_S = float(os.environ.get("MM_LOAD_PROVIDER_POLL_S", "5"))
INJECTED = float(os.environ.get("MM_LOAD_INJECTED_FAILURE_RATE", "0.3"))


def meta() -> dict[str, Any]:
    return {
        "rps_per_user": RPS_PER_USER,
        "models": MODELS,
        "policy": POLICY,
        "injected_failure_rate_declared": INJECTED,
        "provider_poll_s": POLL_S,
    }


install("chaos", meta, poll_providers_s=POLL_S)


class ChaosUser(GatewayUser):
    scenario = "chaos"
    wait_time = constant_throughput(RPS_PER_USER)

    @task
    def routed_chat(self) -> None:
        task_type = random.choice(TASK_TYPES)
        body = chat_body(task_type, policy=POLICY, allowed_models=MODELS, run_id=self.run_id)
        self.chat(body, f"chat:{task_type}")
