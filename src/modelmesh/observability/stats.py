"""In-process rolling statistics per model. They feed /v1/providers and the latency and reliability predictors."""

from __future__ import annotations

import statistics
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass

_MAX_SAMPLES = 2000
_RETENTION_S = 3600.0


@dataclass(frozen=True)
class Sample:
    at: float
    ok: bool
    latency_ms: int
    input_tokens: int
    output_tokens: int


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


class ProviderStats:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._samples: dict[tuple[str, str], deque[Sample]] = defaultdict(lambda: deque(maxlen=_MAX_SAMPLES))

    def record(
        self,
        provider: str,
        model: str,
        *,
        ok: bool,
        latency_ms: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        self._samples[(provider, model)].append(Sample(self._clock(), ok, latency_ms, input_tokens, output_tokens))

    def _window(self, provider: str | None, model: str | None, window_s: float) -> list[Sample]:
        cutoff = self._clock() - min(window_s, _RETENTION_S)
        return [
            sample
            for (sample_provider, sample_model), samples in self._samples.items()
            if (provider is None or provider == sample_provider) and (model is None or model == sample_model)
            for sample in samples
            if sample.at >= cutoff
        ]

    def error_rate(self, provider: str, window_s: float = 300.0) -> float | None:
        samples = self._window(provider, None, window_s)
        return sum(not s.ok for s in samples) / len(samples) if samples else None

    def latency_percentiles(self, provider: str, window_s: float = 300.0) -> tuple[int | None, int | None]:
        values = [float(s.latency_ms) for s in self._window(provider, None, window_s) if s.ok]
        if not values:
            return None, None
        return int(percentile(values, 0.5)), int(percentile(values, 0.95))

    def failure_probability(self, model: str, window_s: float = 900.0) -> float | None:
        samples = self._window(None, model, window_s)
        return sum(not s.ok for s in samples) / len(samples) if samples else None

    def latency_prediction(
        self, model: str, output_tokens: int, *, min_samples: int = 20, window_s: float = 3600.0
    ) -> tuple[int, int] | None:
        """Predicted p50 and p90 from a least-squares fit of latency on output tokens, or None if sparse."""
        samples = [s for s in self._window(None, model, window_s) if s.ok]
        if len(samples) < min_samples:
            return None
        xs = [float(s.output_tokens) for s in samples]
        ys = [float(s.latency_ms) for s in samples]
        mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
        variance = sum((x - mean_x) ** 2 for x in xs)
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / variance if variance else 0.0
        slope = max(slope, 0.0)
        intercept = mean_y - slope * mean_x
        residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys, strict=True)]
        p50 = max(1.0, intercept + slope * output_tokens)
        return int(p50), int(p50 + max(0.0, percentile(residuals, 0.9)))
