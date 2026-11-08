"""Read models for the catalog and calibration endpoints."""

from __future__ import annotations

import time
from typing import Any

from modelmesh.auth.keys import AuthContext
from modelmesh.pipeline.services import Services

_ACCEPTANCE_CRITERIA = {
    "query_classification": "Exact label match",
    "query_rewriting": "Reference-guided rubric score of 4/5 or higher",
    "research_planning": "Reference-guided rubric score of 4/5 or higher",
    "evidence_assessment": "Sufficiency label correct",
    "evidence_synthesis": "Token F1 of 0.5 or higher against gold and a faithfulness judge score of 4/5 or higher",
    "citation_verification": "JSON valid against the requested schema and support label correct",
    "final_response": "Per source: programmatic, exact match, accuracy or judge score of 4/5 or higher",
    "general": "Per source: programmatic, exact match, accuracy or judge score of 4/5 or higher",
    "evaluation_judge": "Not calibrated in v1",
}


def acceptance_criteria(task_type: str) -> str:
    return _ACCEPTANCE_CRITERIA[task_type]


async def model_entries(svc: Services, auth: AuthContext) -> list[dict[str, Any]]:
    created = int(time.time())
    models = svc.registry.enabled_models()
    if auth.allowed_providers is not None:
        models = [m for m in models if m.provider in auth.allowed_providers]
    if auth.allowed_models is not None:
        models = [m for m in models if m.id in auth.allowed_models]
    blocked = await svc.quota.unavailable(models)
    entries: list[dict[str, Any]] = [
        {
            "id": "auto",
            "object": "model",
            "created": created,
            "owned_by": "modelmesh",
            "modelmesh": {
                "kind": "router",
                "provider": "modelmesh",
                "family": None,
                "context_window": None,
                "max_output_tokens": None,
                "capabilities": ["json_object", "json_schema", "tools", "streaming"],
                "pricing": None,
                "data_use": None,
                "status": "available" if models else "unavailable",
            },
        }
    ]
    for model in models:
        price = svc.pricing.get(model.id)
        state = svc.breakers.for_model(model.id).state
        status = "quota_exhausted" if model.id in blocked else ("available" if state == "closed" else "degraded")
        entries.append(
            {
                "id": model.id,
                "object": "model",
                "created": created,
                "owned_by": model.provider,
                "modelmesh": {
                    "kind": "model",
                    "provider": model.provider,
                    "family": model.family,
                    "context_window": model.context_window,
                    "max_output_tokens": model.max_output_tokens,
                    "capabilities": model.capability_tokens(),
                    "pricing": {
                        "input_per_mtok_usd": price.input_per_mtok_usd,
                        "output_per_mtok_usd": price.output_per_mtok_usd,
                        "pricing_version": svc.pricing.version,
                        "free_tier": model.free_tier,
                    },
                    "data_use": model.data_use,
                    "status": status,
                },
            }
        )
    return entries
