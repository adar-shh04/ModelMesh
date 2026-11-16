"""M8 scenario 1: gateway overhead at high RPS against the fake provider.

The target gateway must run with ``MODELMESH_ENABLE_FAKE_PROVIDER=true`` and, for a pure overhead figure,
``MODELMESH_FAKE_PROVIDER_DELAY_MS=0`` and ``MODELMESH_FAKE_PROVIDER_FAILURE_RATE=0``. Requests route with
``model: "auto"`` among the fake echo models, with the cache off, so each request runs authentication, rate
limits, routing, the executor and accounting. The gateway's own ``latency_ms.overhead`` is recorded per request
next to the client-observed latency.

    uv run --group load locust -f loadtests/fake_overhead.py --host http://127.0.0.1:8200 \
        --headless -u 50 -r 10 -t 5m --csv data/load/fake_overhead

Environment: ``MM_LOAD_RPS_PER_USER`` (default 10; total target RPS is users x this), ``MM_LOAD_MODELS``
(comma-separated, default the three fake echo models), ``MM_LOAD_STREAM_SHARE`` (default 0.1),
``MM_LOAD_POLICY`` (default cost_optimized).
"""

from __future__ import annotations

import os
import random
from typing import Any

from locust import constant_throughput, task
from mmload import TASK_TYPES, chat_body, env_list
from mmload_locust import GatewayUser, install

RPS_PER_USER = float(os.environ.get("MM_LOAD_RPS_PER_USER", "10"))
MODELS = env_list("MM_LOAD_MODELS", "fake/echo-small,fake/echo-medium,fake/echo-large")
STREAM_SHARE = float(os.environ.get("MM_LOAD_STREAM_SHARE", "0.1"))
POLICY = os.environ.get("MM_LOAD_POLICY", "cost_optimized")


def meta() -> dict[str, Any]:
    return {"rps_per_user": RPS_PER_USER, "models": MODELS, "stream_share": STREAM_SHARE, "policy": POLICY}


install("fake_overhead", meta)


class FakeOverheadUser(GatewayUser):
    scenario = "fake_overhead"
    wait_time = constant_throughput(RPS_PER_USER)

    @task
    def routed_chat(self) -> None:
        task_type = random.choice(TASK_TYPES)
        stream = task_type != "citation_verification" and random.random() < STREAM_SHARE
        body = chat_body(task_type, policy=POLICY, allowed_models=MODELS, stream=stream, run_id=self.run_id)
        self.chat(body, f"{'stream' if stream else 'chat'}:{task_type}")
