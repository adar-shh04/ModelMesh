"""Locust glue shared by the M8 scenarios: a gateway user, the recorder hooks and a ``/v1/providers`` poller."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

import gevent
import requests
from locust import HttpUser, events
from locust.env import Environment
from locust.runners import WorkerRunner
from mmload import Recorder, api_key, headers, output_path, sample_from_response

RECORDER = Recorder()


def install(scenario: str, meta: Callable[[], dict[str, Any]], *, poll_providers_s: float = 0.0) -> None:
    """Writes the recorder to ``MM_LOAD_OUT`` when the test stops; optionally polls breaker states meanwhile."""
    poller: list[Any] = []
    started: list[float] = []

    @events.test_start.add_listener
    def _start(environment: Environment, **_: Any) -> None:
        started.append(time.time())
        if poll_providers_s > 0 and not isinstance(environment.runner, WorkerRunner):
            poller.append(gevent.spawn(_poll, environment.host or "", poll_providers_s))

    @events.test_stop.add_listener
    def _stop(environment: Environment, **_: Any) -> None:
        for greenlet in poller:
            greenlet.kill()
        path = output_path(scenario)
        times = {"started_at": started[0] if started else None, "stopped_at": time.time()}
        RECORDER.write(path, {"scenario": scenario, "host": environment.host, **times, **meta()})
        print(f"[mmload] wrote {len(RECORDER.samples)} samples to {path}")


def _poll(host: str, every_s: float) -> None:
    key = api_key()
    while True:
        try:
            response = requests.get(f"{host.rstrip('/')}/v1/providers", headers=headers(key), timeout=5)
            RECORDER.add_providers(response.json() if response.ok else {"status": response.status_code})
        except (requests.RequestException, ValueError) as exc:
            RECORDER.add_providers({"error": type(exc).__name__})
        gevent.sleep(every_s)


class GatewayUser(HttpUser):
    abstract = True
    scenario = "base"

    def on_start(self) -> None:
        self.key = api_key()
        self.run_id = os.environ.get("MM_LOAD_RUN_ID", f"load-{self.scenario}")

    def chat(self, body: dict[str, Any], kind: str) -> None:
        stream = bool(body.get("stream"))
        started = time.perf_counter()
        with self.client.post(
            "/v1/chat/completions", json=body, headers=headers(self.key), name=kind, stream=stream, catch_response=True
        ) as response:
            text = response.text
            client_ms = (time.perf_counter() - started) * 1000
            parsed: Any = None
            if not stream or response.status_code != 200:
                try:
                    parsed = response.json()
                except ValueError:
                    parsed = None
            sample = sample_from_response(
                self.scenario,
                kind,
                response.status_code,
                client_ms,
                parsed,
                stream_text=text if stream and response.status_code == 200 else None,
            )
            RECORDER.add(sample)
            if sample.ok:
                response.success()
            else:
                response.failure(f"{response.status_code} {sample.error_type or 'incomplete'}")
