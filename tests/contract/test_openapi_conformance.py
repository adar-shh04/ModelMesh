"""The implemented API and its models must agree with the published OpenAPI contract."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from conftest import CONTRACT, make_settings
from modelmesh.api.routes import router
from modelmesh.contract import schemas as contract_models

SCHEMAS = CONTRACT["components"]["schemas"]
# Schemas that are JSON-Schema aliases or unions rather than objects with a Pydantic counterpart.
NON_MODELS = {"RequestId", "ModelId", "TaskType", "ChatMessage", "MessageContent", "ResponseFormat", "ToolChoice"}
# Names that differ between the contract and the implementation.
RENAMES = {"ChatCompletionChunk": None, "EmbeddingResponse": None}
MODEL_SCHEMAS = sorted(name for name in SCHEMAS if name not in NON_MODELS and RENAMES.get(name, name) is not None)


def contract_operations() -> set[tuple[str, str]]:
    return {(method.upper(), path) for path, item in CONTRACT["paths"].items() for method in item}


def test_every_contract_operation_is_routed_and_no_extra_ones() -> None:
    implemented = {(method, route.path) for route in router.routes for method in route.methods}
    assert implemented == contract_operations()


@pytest.mark.parametrize("name", MODEL_SCHEMAS)
def test_model_matches_contract_properties_and_required(name: str) -> None:
    model: type[BaseModel] = getattr(contract_models, name)
    generated = model.model_json_schema(by_alias=True)
    contract = SCHEMAS[name]
    assert set(generated.get("properties", {})) == set(contract.get("properties", {}))
    # A property the contract requires must be required by the model, except where the model supplies a
    # value the contract permits the client to omit.
    assert set(contract.get("required", [])) <= set(generated.get("required", [])) | set(
        key for key, value in generated.get("properties", {}).items() if "default" in value
    )


def test_input_models_reject_unknown_fields() -> None:
    for name in ("ChatCompletionRequest", "RoutingOptions", "CacheOptions", "FeedbackRequest"):
        assert getattr(contract_models, name).model_config.get("extra") == "forbid"


def test_fake_provider_refused_in_production() -> None:
    with pytest.raises(ValueError, match="production"):
        make_settings(MODELMESH_ENV="production")
