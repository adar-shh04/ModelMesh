"""Circuit breakers (ADR-0008). State is per process for v1."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from typing import Literal

CircuitState = Literal["closed", "open", "half_open"]


class CircuitBreaker:
    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        window: int = 20,
        min_calls: int = 5,
        failure_threshold: float = 0.5,
        open_seconds: float = 30.0,
        consecutive_timeouts: int = 3,
        probe_expiry_seconds: float = 330.0,
    ) -> None:
        self._clock = clock
        self._outcomes: deque[bool] = deque(maxlen=window)
        self._min_calls = min_calls
        self._threshold = failure_threshold
        self._open_seconds = open_seconds
        self._consecutive_timeouts_limit = consecutive_timeouts
        self._consecutive_timeouts = 0
        self._opened_at: float | None = None
        self._probe_in_flight = False
        self._probe_started: float | None = None
        # No request outlives the 300 s maximum deadline, so a probe still in flight after this is abandoned (a code
        # path that took the probe and never reported a verdict). Without the expiry such a leak would wedge the
        # breaker in half-open until the process restarts.
        self._probe_expiry_seconds = probe_expiry_seconds

    @property
    def state(self) -> CircuitState:
        if self._opened_at is None:
            return "closed"
        if self._clock() - self._opened_at >= self._open_seconds:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        state = self.state
        if state == "closed":
            return True
        if state != "half_open":
            return False
        now = self._clock()
        abandoned = self._probe_started is not None and now - self._probe_started >= self._probe_expiry_seconds
        if not self._probe_in_flight or abandoned:
            self._probe_in_flight = True
            self._probe_started = now
            return True
        return False

    def release_probe(self) -> None:
        """A probe that ended without a verdict (e.g. a client cancel) must not wedge the breaker."""
        self._probe_in_flight = False
        self._probe_started = None

    def record_success(self) -> None:
        self._outcomes.append(True)
        self._consecutive_timeouts = 0
        if self._opened_at is not None:
            self._opened_at = None
            self.release_probe()
            self._outcomes.clear()

    def record_failure(self, *, timeout: bool = False) -> None:
        self._outcomes.append(False)
        self._consecutive_timeouts = self._consecutive_timeouts + 1 if timeout else 0
        if self._opened_at is not None:
            self.trip()
            return
        failures = sum(not ok for ok in self._outcomes)
        tripped_rate = len(self._outcomes) >= self._min_calls and failures / len(self._outcomes) >= self._threshold
        if tripped_rate or self._consecutive_timeouts >= self._consecutive_timeouts_limit:
            self.trip()

    def trip(self) -> None:
        self._opened_at = self._clock()
        self.release_probe()


class BreakerRegistry:
    """One breaker per model and one per provider (for connection-level failures)."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._model: dict[str, CircuitBreaker] = {}
        self._provider: dict[str, CircuitBreaker] = {}
        self.misconfigured: set[str] = set()

    def for_model(self, model_id: str) -> CircuitBreaker:
        return self._model.setdefault(model_id, CircuitBreaker(clock=self._clock))

    def for_provider(self, provider: str) -> CircuitBreaker:
        return self._provider.setdefault(provider, CircuitBreaker(clock=self._clock))

    def provider_state(self, provider: str, model_ids: list[str]) -> CircuitState:
        states = [self.for_provider(provider).state, *(self.for_model(model).state for model in model_ids)]
        if "open" in states:
            return "open"
        if "half_open" in states:
            return "half_open"
        return "closed"
