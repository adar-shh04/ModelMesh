"""Explicit routing policies and the rules baseline B0a (ADR-0004, contract section 2.1.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from modelmesh.providers.catalog import TIER_ORDER, SizeTier
from modelmesh.routing.estimates import CandidateEstimate

DEFAULT_MIN_QUALITY = 0.6
BALANCED_DELTA = 0.05
ESCALATION_INPUT_TOKENS = 8000
ESCALATION_SCHEMA_PROPERTIES = 10

KNOWN_TASK_TYPES = (
    "query_classification",
    "query_rewriting",
    "research_planning",
    "evidence_assessment",
    "evidence_synthesis",
    "citation_verification",
    "final_response",
    "evaluation_judge",
    "general",
)

TASK_TIER: dict[str, SizeTier] = {
    "query_classification": "small",
    "query_rewriting": "small",
    "evidence_assessment": "small",
    "citation_verification": "medium",
    "general": "medium",
    "research_planning": "large",
    "evidence_synthesis": "large",
    "final_response": "large",
    "evaluation_judge": "large",
}

Mode = Literal["learned", "rules_fallback"]


@dataclass
class SolveInput:
    policy: str
    task_type: str
    calibrated: bool
    explicit_min_quality: float | None
    max_latency_ms: int | None
    input_tokens: int
    schema_properties: int = 0


@dataclass
class SolveResult:
    ranked: list[CandidateEstimate]
    mode: Mode
    reason: str
    warnings: list[str] = field(default_factory=list)


def _cost_key(e: CandidateEstimate) -> tuple[float, float, int]:
    return (e.predicted_cost_usd, -(e.quality if e.quality is not None else -1.0), e.latency_p50_ms)


def rules_tier(task_type: str, input_tokens: int, schema_properties: int) -> SizeTier:
    tier = TASK_TIER.get(task_type, "medium")
    if input_tokens > ESCALATION_INPUT_TOKENS or schema_properties > ESCALATION_SCHEMA_PROPERTIES:
        tier = TIER_ORDER[min(TIER_ORDER.index(tier) + 1, len(TIER_ORDER) - 1)]
    return tier


def _tier_preference(target: SizeTier) -> list[SizeTier]:
    """The target tier first, then the nearest tiers, preferring larger ones."""
    index = TIER_ORDER.index(target)
    return sorted(TIER_ORDER, key=lambda tier: (abs(TIER_ORDER.index(tier) - index), TIER_ORDER.index(tier) < index))


def rank_by_rules(candidates: list[CandidateEstimate], target: SizeTier) -> list[CandidateEstimate]:
    order = {tier: position for position, tier in enumerate(_tier_preference(target))}
    return sorted(candidates, key=lambda e: (order[e.model.size_tier], e.predicted_cost_usd, e.latency_p50_ms))


def _largest_first(candidates: list[CandidateEstimate]) -> list[CandidateEstimate]:
    return sorted(
        candidates,
        key=lambda e: (-TIER_ORDER.index(e.model.size_tier), e.predicted_cost_usd, e.latency_p50_ms),
    )


def _partition_by_evidence(
    candidates: list[CandidateEstimate],
) -> tuple[list[CandidateEstimate], list[CandidateEstimate]]:
    return [e for e in candidates if e.quality is not None], [e for e in candidates if e.quality is None]


def _quality(e: CandidateEstimate) -> float:
    assert e.quality is not None
    return e.quality


def _solve_cost(req: SolveInput, candidates: list[CandidateEstimate]) -> SolveResult:
    if not req.calibrated:
        return SolveResult(
            sorted(candidates, key=_cost_key),
            "rules_fallback",
            "lowest estimated cost; task type is not calibrated so no quality floor applies",
            ["default_min_quality_skipped:uncalibrated"],
        )
    if req.explicit_min_quality is not None:
        return SolveResult(sorted(candidates, key=_cost_key), "learned", "lowest estimated cost meeting min_quality")
    meeting = [e for e in candidates if e.quality is not None and e.quality >= DEFAULT_MIN_QUALITY]
    if not meeting:
        return SolveResult(
            sorted(candidates, key=_cost_key),
            "learned",
            "lowest estimated cost; no eligible model met the default quality floor",
            [f"default_min_quality_unmet:{DEFAULT_MIN_QUALITY}"],
        )
    others = [e for e in candidates if e not in meeting]
    return SolveResult(
        sorted(meeting, key=_cost_key) + sorted(others, key=_cost_key),
        "learned",
        "lowest estimated cost meeting the default quality floor",
        [f"default_min_quality:{DEFAULT_MIN_QUALITY}"],
    )


def _solve_balanced(req: SolveInput, candidates: list[CandidateEstimate]) -> SolveResult:
    if not req.calibrated:
        target = rules_tier(req.task_type, req.input_tokens, req.schema_properties)
        return SolveResult(
            rank_by_rules(candidates, target),
            "rules_fallback",
            f"rules policy: {target} tier, cheapest eligible model",
            [f"quality_uncalibrated:{req.task_type}"],
        )
    with_evidence, without = _partition_by_evidence(candidates)
    if not with_evidence:
        target = rules_tier(req.task_type, req.input_tokens, req.schema_properties)
        return SolveResult(rank_by_rules(without, target), "rules_fallback", f"rules policy: {target} tier", [])
    best = max(_quality(e) for e in with_evidence)
    within = [e for e in with_evidence if _quality(e) >= best - BALANCED_DELTA]
    rest = sorted([e for e in with_evidence if e not in within], key=lambda e: (-_quality(e), e.predicted_cost_usd))
    return SolveResult(
        sorted(within, key=_cost_key) + rest + sorted(without, key=_cost_key),
        "learned",
        f"cheapest eligible model within {BALANCED_DELTA} of best predicted quality",
    )


def _solve_quality_max(req: SolveInput, candidates: list[CandidateEstimate]) -> SolveResult:
    if not req.calibrated:
        return SolveResult(
            _largest_first(candidates),
            "rules_fallback",
            "rules policy: largest eligible tier, cheapest within it",
            [f"quality_uncalibrated:{req.task_type}"],
        )
    with_evidence, without = _partition_by_evidence(candidates)
    return SolveResult(
        sorted(with_evidence, key=lambda e: (-_quality(e), e.predicted_cost_usd)) + sorted(without, key=_cost_key),
        "learned" if with_evidence else "rules_fallback",
        "highest predicted quality",
    )


def solve(req: SolveInput, candidates: list[CandidateEstimate]) -> SolveResult:
    if req.policy == "cost_optimized":
        result = _solve_cost(req, candidates)
    elif req.policy == "quality_balanced":
        result = _solve_balanced(req, candidates)
    elif req.policy == "quality_max":
        result = _solve_quality_max(req, candidates)
    elif req.policy == "latency_optimized":
        result = SolveResult(
            sorted(candidates, key=lambda e: (e.latency_p50_ms, e.predicted_cost_usd)),
            "learned" if req.calibrated else "rules_fallback",
            "lowest predicted latency",
        )
    else:
        result = SolveResult(
            sorted(candidates, key=lambda e: (e.failure_probability, e.predicted_cost_usd)),
            "learned" if req.calibrated else "rules_fallback",
            "lowest predicted failure probability, tie-broken on cost",
        )
    if req.max_latency_ms is not None and any(e.latency_prior_used for e in candidates):
        result.warnings.append("latency_prior_used")
    return result
