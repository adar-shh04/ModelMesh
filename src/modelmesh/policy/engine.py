"""Hard-constraint evaluation: which candidates may serve this request (spec section 2.4 step 3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from modelmesh.core.errors import ApiError
from modelmesh.providers.catalog import CatalogModel
from modelmesh.providers.types import StructuredMode
from modelmesh.routing.estimates import CandidateEstimate

DATA_POLICY_RANK = {"any": 0, "no_training_providers": 1, "local_only": 2}


@dataclass(frozen=True)
class CapabilityNeeds:
    response_format_kind: str | None = None
    schema_portable: bool = False
    has_tools: bool = False
    tool_choice_forced: bool = False
    stream: bool = False
    required: frozenset[str] = frozenset()


@dataclass(frozen=True)
class StructuredPlan:
    mode: StructuredMode | None
    buffered: bool


@dataclass
class Constraints:
    data_policy: str = "any"
    max_cost_usd: float | None = None
    max_latency_ms: int | None = None
    min_quality: float | None = None
    allowed_providers: set[str] | None = None
    denied_providers: set[str] = field(default_factory=set)
    allowed_models: set[str] | None = None
    budget_remaining_usd: float | None = None
    quota_unavailable: set[str] = field(default_factory=set)
    needs: CapabilityNeeds = field(default_factory=CapabilityNeeds)


def effective_data_policy(requested: str | None, ceiling: str) -> str:
    if requested is None:
        return ceiling
    if DATA_POLICY_RANK[requested] < DATA_POLICY_RANK[ceiling]:
        raise ApiError(
            "permission_denied",
            "The requested data_policy is looser than this key allows.",
            code="data_policy_exceeds_key",
            param="routing.data_policy",
        )
    return requested


def plan_structured(model: CatalogModel, needs: CapabilityNeeds) -> StructuredPlan | None:
    """How this model would serve the requested format, or None if it cannot."""
    kind = needs.response_format_kind
    required = needs.required
    if kind not in ("json_object", "json_schema"):
        return StructuredPlan(None, False)
    mode: StructuredMode
    if kind == "json_object":
        if "json_object" in required and not model.json_object:
            return None
        return StructuredPlan("native_best_effort" if model.json_object else "emulated", needs.stream)
    if model.schema_mode is None or (needs.has_tools and not model.schema_with_tools):
        mode = "emulated"
    elif model.schema_mode == "strict" and needs.schema_portable:
        mode = "native_strict"
    else:
        mode = "native_best_effort"
    if "json_schema" in required and mode == "emulated":
        return None
    if "json_schema_strict" in required and mode != "native_strict":
        return None
    buffered = needs.stream and not (mode == "native_strict" and model.schema_stream)
    if buffered and "streaming" in required:
        return None
    return StructuredPlan(mode, buffered)


def satisfies_capabilities(model: CatalogModel, needs: CapabilityNeeds) -> bool:
    if needs.has_tools and not model.tools:
        return False
    if needs.tool_choice_forced and not model.tool_choice:
        return False
    if "tools" in needs.required and not model.tools:
        return False
    if "streaming" in needs.required and not model.streaming:
        return False
    if "json_object" in needs.required and not model.json_object:
        return False
    for token in needs.required:
        if token.startswith("long_context:") and model.context_window < int(token.split(":", 1)[1]):
            return False
    return plan_structured(model, needs) is not None


def _violations(estimate: CandidateEstimate, c: Constraints) -> list[str]:
    model = estimate.model
    found: list[str] = []
    if c.allowed_providers is not None and model.provider not in c.allowed_providers:
        found.append("allowed_providers")
    if model.provider in c.denied_providers:
        found.append("denied_providers")
    if c.allowed_models is not None and model.id not in c.allowed_models:
        found.append("allowed_models")
    if c.data_policy == "local_only" and not model.local:
        found.append("data_policy")
    if c.data_policy == "no_training_providers" and model.data_use != "no_training":
        found.append("data_policy")
    if not satisfies_capabilities(model, c.needs):
        found.append("required_capabilities")
    if estimate.input_tokens > model.context_window:
        found.append("context_window")
    if c.max_cost_usd is not None and estimate.upper_bound_cost_usd > c.max_cost_usd:
        found.append("max_cost_usd")
    if c.max_latency_ms is not None and estimate.latency_p90_ms > c.max_latency_ms:
        found.append("max_latency_ms")
    if c.min_quality is not None and (estimate.quality is None or estimate.quality < c.min_quality):
        found.append("min_quality")
    if c.budget_remaining_usd is not None and estimate.upper_bound_cost_usd > c.budget_remaining_usd:
        found.append("budget")
    if model.id in c.quota_unavailable:
        found.append("quota")
    return found


def _best_available(constraint: str, estimates: list[CandidateEstimate]) -> float | None:
    if not estimates:
        return None
    if constraint == "max_cost_usd":
        return min(e.upper_bound_cost_usd for e in estimates)
    if constraint == "max_latency_ms":
        return float(min(e.latency_p90_ms for e in estimates))
    if constraint == "min_quality":
        values = [e.quality for e in estimates if e.quality is not None]
        return max(values) if values else None
    return None


def filter_candidates(estimates: list[CandidateEstimate], c: Constraints) -> list[CandidateEstimate]:
    """Return the eligible candidates, or raise the error the contract defines for an empty set."""
    results = [(e, _violations(e, c)) for e in estimates]
    eligible = [e for e, violations in results if not violations]
    if eligible:
        return eligible
    all_violations = [set(v) for _, v in results]
    if all_violations and all(v == {"quota"} for v in all_violations):
        raise ApiError(
            "service_unavailable",
            "Every eligible upstream is currently rate- or quota-limited.",
            code="upstream_capacity",
            details={"attempts": []},
        )
    if all_violations and all(v == {"budget"} for v in all_violations):
        raise ApiError(
            "budget_exceeded",
            "This request would exceed the key's budget.",
            code="daily_budget",
            details={"scope": "key", "period": "day"},
        )
    if all_violations and all(v == {"context_window"} for v in all_violations):
        raise ApiError(
            "request_too_large",
            "The estimated input exceeds every eligible context window.",
            code="context_too_large",
        )

    counts: dict[str, int] = {}
    otherwise_eligible: dict[str, list[CandidateEstimate]] = {}
    for estimate, violations in results:
        for constraint in violations:
            counts[constraint] = counts.get(constraint, 0) + 1
            if len(violations) == 1:
                otherwise_eligible.setdefault(constraint, []).append(estimate)
    details: dict[str, Any] = {
        "candidates_considered": len(estimates),
        "violated_constraints": [
            {
                "constraint": constraint,
                "excluded_candidates": count,
                "best_available": _best_available(constraint, otherwise_eligible.get(constraint, [])),
            }
            for constraint, count in sorted(counts.items())
        ],
    }
    raise ApiError(
        "no_eligible_model",
        "No candidate model satisfies all hard constraints.",
        code="constraints_unsatisfiable",
        details=details,
    )
