"""Pre-execution estimates of tokens, cost, latency, quality and failure for each candidate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from modelmesh.observability.stats import ProviderStats
from modelmesh.providers.catalog import CatalogModel, PricingTable

DEFAULT_OUTPUT_UPPER_BOUND = 4096
TOKENS_PER_MESSAGE = 4

# Median completion lengths assumed until telemetry says otherwise.
PREDICTED_OUTPUT_TOKENS: dict[str, int] = {
    "query_classification": 20,
    "query_rewriting": 40,
    "research_planning": 200,
    "evidence_assessment": 60,
    "evidence_synthesis": 300,
    "citation_verification": 80,
    "final_response": 400,
    "evaluation_judge": 150,
    "general": 250,
}


@dataclass(frozen=True)
class QualityEstimate:
    probability: float
    interval: tuple[float, float] | None


class QualityPredictor(Protocol):
    """Calibrated quality predictions. The learned router (M6) implements this."""

    router_version: str | None

    def is_task_calibrated(self, task_type: str) -> bool: ...

    def predict(self, model_id: str, task_type: str, features: dict[str, Any]) -> QualityEstimate | None: ...


class NoCalibration:
    """Until a promoted router exists there is no calibrated evidence for any task type."""

    router_version: str | None = None

    def is_task_calibrated(self, task_type: str) -> bool:
        return False

    def predict(self, model_id: str, task_type: str, features: dict[str, Any]) -> QualityEstimate | None:
        return None


def message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    return ""


class TokenEstimator:
    """chars * ratio + per-message overhead. The ratio per family is corrected from provider-reported usage."""

    def __init__(self, initial_ratio: float = 0.25, smoothing: float = 0.05) -> None:
        self._initial_ratio = initial_ratio
        self._smoothing = smoothing
        self._ratios: dict[str, float] = {}

    def estimate_text(self, text: str, family: str | None = None) -> int:
        return int(len(text) * self._ratios.get(family or "", self._initial_ratio)) + 1

    def estimate_messages(self, messages: list[dict[str, Any]], family: str | None = None) -> int:
        return sum(self.estimate_text(message_text(m), family) + TOKENS_PER_MESSAGE for m in messages)

    def observe(self, family: str, chars: int, reported_tokens: int) -> None:
        if chars <= 0 or reported_tokens <= 0:
            return
        current = self._ratios.get(family, self._initial_ratio)
        self._ratios[family] = current + self._smoothing * (reported_tokens / chars - current)


@dataclass(frozen=True)
class CandidateEstimate:
    model: CatalogModel
    input_tokens: int
    predicted_output_tokens: int
    upper_bound_output_tokens: int
    predicted_cost_usd: float
    upper_bound_cost_usd: float
    latency_p50_ms: int
    latency_p90_ms: int
    latency_prior_used: bool
    quality: float | None
    quality_interval: tuple[float, float] | None
    failure_probability: float


class Estimator:
    def __init__(
        self, pricing: PricingTable, stats: ProviderStats, tokens: TokenEstimator, quality: QualityPredictor
    ) -> None:
        self.pricing = pricing
        self.stats = stats
        self.tokens = tokens
        self.quality = quality

    def estimate(
        self,
        model: CatalogModel,
        *,
        messages: list[dict[str, Any]],
        max_tokens: int | None,
        task_type: str,
        features: dict[str, Any] | None = None,
    ) -> CandidateEstimate:
        input_tokens = self.tokens.estimate_messages(messages, model.family)
        upper_output = min(max_tokens or DEFAULT_OUTPUT_UPPER_BOUND, model.max_output_tokens)
        predicted_output = min(PREDICTED_OUTPUT_TOKENS.get(task_type, 250), upper_output)
        telemetry = self.stats.latency_prediction(model.id, predicted_output)
        if telemetry is None:
            p50 = model.latency_ttft_ms + int(1000 * predicted_output / model.latency_tokens_per_s)
            p90 = int(p50 * 1.5)
        else:
            p50, p90 = telemetry
        quality = self.quality.predict(model.id, task_type, features or {})
        return CandidateEstimate(
            model=model,
            input_tokens=input_tokens,
            predicted_output_tokens=predicted_output,
            upper_bound_output_tokens=upper_output,
            predicted_cost_usd=self.pricing.cost(model.id, input_tokens, predicted_output),
            upper_bound_cost_usd=self.pricing.cost(model.id, input_tokens, upper_output),
            latency_p50_ms=p50,
            latency_p90_ms=p90,
            latency_prior_used=telemetry is None,
            quality=quality.probability if quality else None,
            quality_interval=quality.interval if quality else None,
            failure_probability=self.stats.failure_probability(model.id) or 0.0,
        )
