"""Contract models. These mirror `docs/contract/openapi-v1.yaml`; tests/contract enforces that they agree.

Request models forbid unknown fields. Response models are built with every required field set explicitly and
serialised with `exclude_unset=True`, so optional fields are omitted rather than sent as null.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

REQUEST_ID_PATTERN = r"^mmr_[0-9A-HJKMNP-TV-Z]{26}$"
RequestId = Annotated[str, Field(pattern=REQUEST_ID_PATTERN)]
ModelId = Annotated[str, Field(min_length=1, max_length=200, pattern=r"^(auto|[a-z0-9-]+/.+)$")]
TaskType = Annotated[str, Field(max_length=64)]
Fingerprint = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
Namespace = Annotated[str, Field(max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")]
CorrelationId = Annotated[str, Field(max_length=128, pattern=r"^[A-Za-z0-9_:.-]+$")]
ShortName = Annotated[str, Field(max_length=64)]

_METADATA_KEY = re.compile(r"^[A-Za-z0-9_.:-]+$")


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Output(BaseModel):
    model_config = ConfigDict(extra="ignore")

    def to_json(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_unset=True, by_alias=True)


# ---------------------------------------------------------------- messages
class TextContentPart(_Input):
    type: Literal["text"]
    text: str


MessageContent = str | Annotated[list[TextContentPart], Field(min_length=1)]


class SystemMessage(_Input):
    role: Literal["system", "developer"]
    content: MessageContent
    name: ShortName | None = None


class UserMessage(_Input):
    role: Literal["user"]
    content: MessageContent
    name: ShortName | None = None


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"]
    function: ToolCallFunction


class AssistantMessageInput(_Input):
    role: Literal["assistant"]
    content: MessageContent | None = None
    name: ShortName | None = None
    tool_calls: list[ToolCall] | None = None
    refusal: str | None = None


class ToolMessage(_Input):
    role: Literal["tool"]
    content: MessageContent
    tool_call_id: Annotated[str, Field(max_length=128)]


ChatMessage = Annotated[
    SystemMessage | UserMessage | AssistantMessageInput | ToolMessage, Field(discriminator="role")
]


# ---------------------------------------------------------------- request options
class StreamOptions(_Input):
    include_usage: bool = False


class ResponseFormatText(_Input):
    type: Literal["text"]


class ResponseFormatJsonObject(_Input):
    type: Literal["json_object"]


class JsonSchemaSpec(_Input):
    # `schema` shadows a BaseModel attribute, hence the alias.
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
    description: str | None = None
    schema_: dict[str, Any] = Field(alias="schema")
    strict: bool = False


class ResponseFormatJsonSchema(_Input):
    type: Literal["json_schema"]
    json_schema: JsonSchemaSpec


ResponseFormat = Annotated[
    ResponseFormatText | ResponseFormatJsonObject | ResponseFormatJsonSchema, Field(discriminator="type")
]


class ToolFunction(_Input):
    name: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None


class ToolDefinition(_Input):
    type: Literal["function"]
    function: ToolFunction


class ToolChoiceFunctionName(_Input):
    name: str


class ToolChoiceFunction(_Input):
    type: Literal["function"]
    function: ToolChoiceFunctionName


ToolChoice = Literal["none", "auto", "required"] | ToolChoiceFunction


class RequestMetadata(BaseModel):
    """A string-to-string map with four reserved keys."""

    model_config = ConfigDict(extra="allow")

    application: ShortName | None = None
    task_type: TaskType | None = None
    correlation_id: CorrelationId | None = None
    end_user_id: Annotated[str, Field(max_length=128)] | None = None

    @model_validator(mode="before")
    @classmethod
    def _check_map(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            raise ValueError("metadata must be an object")
        if len(value) > 16:
            raise ValueError("metadata may have at most 16 keys")
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 64 or not _METADATA_KEY.match(key):
                raise ValueError("invalid metadata key")
            if not isinstance(item, str) or len(item) > 512:
                raise ValueError("metadata values must be strings of at most 512 characters")
        return value

    def as_map(self) -> dict[str, str]:
        return {k: v for k, v in self.model_dump(exclude_none=True).items() if isinstance(v, str)}


Capability = Annotated[
    str,
    Field(
        pattern=r"^(json_object|json_schema|json_schema_strict|tools|streaming|long_context:[1-9][0-9]{0,8})$"
    ),
]
PolicyName = Literal[
    "cost_optimized", "latency_optimized", "quality_balanced", "quality_max", "reliability_first"
]
DataPolicy = Literal["any", "no_training_providers", "local_only"]


class RoutingOptions(_Input):
    policy: PolicyName = "quality_balanced"
    max_cost_usd: Annotated[float, Field(ge=0)] | None = None
    max_latency_ms: Annotated[int, Field(ge=1)] | None = None
    timeout_ms: Annotated[int, Field(ge=1, le=300_000)] | None = None
    min_quality: Annotated[float, Field(ge=0, le=1)] | None = None
    allowed_providers: list[Annotated[str, Field(max_length=32)]] | None = None
    denied_providers: list[Annotated[str, Field(max_length=32)]] | None = None
    allowed_models: list[ModelId] | None = None
    required_capabilities: list[Capability] | None = None
    data_policy: DataPolicy | None = None
    fallback: Literal["within_constraints", "none"] | None = None


class CacheOptions(_Input):
    mode: Literal["off", "exact", "semantic"] | None = None
    ttl_s: Annotated[int, Field(ge=1, le=604_800)] | None = None
    context_fingerprint: Fingerprint | None = None
    namespace: Namespace | None = None


class ChatCompletionRequest(_Input):
    model: ModelId
    messages: Annotated[list[ChatMessage], Field(min_length=1, max_length=512)]
    temperature: Annotated[float, Field(ge=0, le=2)] | None = None
    top_p: Annotated[float, Field(ge=0, le=1)] | None = None
    max_tokens: Annotated[int, Field(ge=1)] | None = None
    max_completion_tokens: Annotated[int, Field(ge=1)] | None = None
    stop: (
        Annotated[str, Field(max_length=256)]
        | Annotated[list[Annotated[str, Field(max_length=256)]], Field(max_length=4)]
        | None
    ) = None
    seed: int | None = None
    n: Literal[1] | None = None
    logprobs: Literal[False] | None = None
    presence_penalty: Annotated[float, Field(ge=-2, le=2)] | None = None
    frequency_penalty: Annotated[float, Field(ge=-2, le=2)] | None = None
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] | None = None
    user: Annotated[str, Field(max_length=256)] | None = None
    store: bool | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    response_format: ResponseFormat | None = None
    tools: Annotated[list[ToolDefinition], Field(max_length=128)] | None = None
    tool_choice: ToolChoice | None = None
    parallel_tool_calls: bool | None = None
    metadata: RequestMetadata | None = None
    routing: RoutingOptions | None = None
    cache: CacheOptions | None = None


# ---------------------------------------------------------------- chat response
class PromptTokensDetails(_Output):
    cached_tokens: int | None = None


class CompletionTokensDetails(_Output):
    reasoning_tokens: int | None = None


class Usage(_Output):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: PromptTokensDetails | None = None
    completion_tokens_details: CompletionTokensDetails | None = None


class AssistantMessage(_Output):
    role: Literal["assistant"]
    content: str | None
    tool_calls: list[ToolCall] | None = None
    refusal: str | None = None


class Choice(_Output):
    index: Literal[0]
    message: AssistantMessage
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"]
    logprobs: None = None


class RoutingDecision(_Output):
    mode: Literal["learned", "rules_fallback", "explicit_model"]
    policy: PolicyName | None
    router_version: str | None
    task_type: str
    candidates_considered: int
    candidates_eligible: int
    predicted_quality: float | None
    predicted_quality_interval: list[float] | None
    predicted_latency_ms: int | None
    predicted_latency_p90_ms: int | None
    estimated_cost_usd: float | None
    reason: str
    warnings: list[str]


AttemptOutcome = Literal[
    "success",
    "rate_limited",
    "quota_exhausted",
    "upstream_timeout",
    "upstream_unavailable",
    "upstream_error",
    "incompatible",
    "context_length_exceeded",
    "content_filtered",
    "schema_validation_failed",
    "auth_error",
    "circuit_open",
    "stream_interrupted",
    "client_cancelled",
]


class Attempt(_Output):
    provider: str
    model: str
    outcome: AttemptOutcome
    latency_ms: int | None
    http_status: int | None
    retryable: bool
    possibly_billed: bool


class CacheInfo(_Output):
    status: Literal["hit_exact", "hit_semantic", "miss", "bypass"]
    similarity: float | None = None
    original_request_id: RequestId | None = None
    saved_estimated_usd: float | None = None


class ModelMeshUsage(_Output):
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    source: Literal["provider_reported", "estimated"]


class CostInfo(_Output):
    estimated_usd: float
    provider_reported_usd: float | None
    failed_attempts_estimated_usd: float
    pricing_version: str


class LatencyInfo(_Output):
    total: int
    provider: int | None
    overhead: int
    ttft: int | None


class StructuredOutputInfo(_Output):
    mode: Literal["native_strict", "native_best_effort", "emulated"]
    validated: Literal[True]
    repair_attempts: int


class StreamingInfo(_Output):
    mode: Literal["native", "buffered"]


class ModelMeshMetadata(_Output):
    request_id: RequestId
    client_request_id: str
    trace_id: Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
    provider: str
    model: str
    routing: RoutingDecision | None
    attempts: list[Attempt]
    fallback_used: bool
    cache: CacheInfo
    usage: ModelMeshUsage
    cost: CostInfo
    latency_ms: LatencyInfo
    structured_output: StructuredOutputInfo | None
    streaming: StreamingInfo | None


class ChatCompletionResponse(_Output):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    system_fingerprint: str | None = None
    choices: Annotated[list[Choice], Field(min_length=1, max_length=1)]
    usage: Usage
    modelmesh: ModelMeshMetadata


# ---------------------------------------------------------------- errors
class ViolatedConstraint(_Output):
    constraint: str
    excluded_candidates: int
    best_available: float | None


class ValidationErrorItem(_Output):
    path: str
    keyword: str


class ErrorDetails(_Output):
    candidates_considered: int | None = None
    violated_constraints: list[ViolatedConstraint] | None = None
    attempts: list[Attempt] | None = None
    scope: Literal["key", "tenant"] | None = None
    period: Literal["minute", "day", "month"] | None = None
    resets_at: datetime | None = None
    validation_errors: list[ValidationErrorItem] | None = None


class ErrorObject(_Output):
    type: str
    code: str | None
    message: str
    param: str | None
    request_id: RequestId
    client_request_id: str | None
    retryable: bool
    details: ErrorDetails


class ErrorResponse(_Output):
    error: ErrorObject
    modelmesh: ModelMeshMetadata | None = None


# ---------------------------------------------------------------- embeddings
class EmbeddingRequest(_Input):
    model: Annotated[str, Field(min_length=1, max_length=200)]
    input: (
        Annotated[str, Field(min_length=1)]
        | Annotated[list[Annotated[str, Field(min_length=1)]], Field(min_length=1, max_length=256)]
    )
    encoding_format: Literal["float"] = "float"
    dimensions: Annotated[int, Field(ge=1)] | None = None
    user: Annotated[str, Field(max_length=256)] | None = None
    metadata: RequestMetadata | None = None


# ---------------------------------------------------------------- catalog
class Pricing(_Output):
    input_per_mtok_usd: float
    output_per_mtok_usd: float
    pricing_version: str
    free_tier: bool


class ModelInfo(_Output):
    kind: Literal["model", "router"]
    provider: str
    family: str | None
    context_window: int | None
    max_output_tokens: int | None
    capabilities: list[str]
    pricing: Pricing | None
    data_use: Literal["no_training", "may_train", "unknown"] | None
    status: Literal["available", "degraded", "unavailable", "quota_exhausted", "disabled"]


class Model(_Output):
    id: str
    object: Literal["model"]
    created: int
    owned_by: str
    modelmesh: ModelInfo


class ModelList(_Output):
    object: Literal["list"]
    data: list[Model]


# ---------------------------------------------------------------- operations
class ComponentStatus(_Output):
    status: Literal["ok", "degraded", "down"]


class ProviderCounts(_Output):
    ok: int
    degraded: int
    down: int


class HealthComponents(_Output):
    postgres: ComponentStatus
    redis: ComponentStatus
    providers: ProviderCounts


class Health(_Output):
    status: Literal["ok", "degraded", "down"]
    contract_version: str
    components: HealthComponents


class ContractInfo(_Output):
    contract_version: str
    contract_release: str
    endpoints: list[str]
    extensions: list[str]
    task_types: list[str]
    capabilities: list[str]
    routing_policies: list[str]
    feedback_metrics: list[str]


class ProviderStatus(_Output):
    id: str
    status: Literal["ok", "degraded", "down", "disabled", "misconfigured"]
    circuit_state: Literal["closed", "open", "half_open"]
    error_rate_5m: float | None
    p50_latency_ms: int | None
    p95_latency_ms: int | None
    models_available: int


class ProviderStatusList(_Output):
    data: list[ProviderStatus]


class CalibrationModel(_Output):
    model: str
    calibrated: bool
    calibration_samples: int


class CalibrationTaskType(_Output):
    task_type: str
    calibrated: bool
    acceptance_criterion: str
    acceptance_criterion_version: str | None = None
    benchmark_manifest: str | None
    calibration_samples: int
    ece: float | None
    brier: float | None
    judge_human_kappa: float | None = None
    models: list[CalibrationModel]


class RoutingCalibration(_Output):
    router_version: str | None
    generated_at: datetime
    task_types: list[CalibrationTaskType]


# ---------------------------------------------------------------- accounting
class UsageRow(_Output):
    group: str | None
    requests: int
    succeeded: int
    failed: int
    cache_hits: int
    fallback_events: int
    prompt_tokens_provider_reported: int
    completion_tokens_provider_reported: int
    prompt_tokens_estimated: int
    completion_tokens_estimated: int
    estimated_cost_usd: float
    provider_reported_cost_usd: float | None
    provider_reported_cost_coverage: float
    latency_p50_ms: int | None
    latency_p95_ms: int | None


class UsageReport(_Output):
    model_config = ConfigDict(populate_by_name=True)

    from_: datetime = Field(alias="from")
    to: datetime
    group_by: Literal["model", "provider", "task_type", "application", "day"]
    data: list[UsageRow]
    totals: UsageRow


class RetainedContent(_Output):
    messages: list[dict[str, Any]]
    response: dict[str, Any] | None
    expires_at: datetime


class RequestRecord(_Output):
    id: RequestId
    client_request_id: str
    created_at: datetime
    completed_at: datetime | None
    status: Literal["succeeded", "failed", "stream_interrupted", "cancelled", "rejected", "in_progress"]
    http_status: int | None
    api_key_id: str
    endpoint: Literal["chat.completions", "embeddings"]
    stream: bool
    metadata: dict[str, str]
    modelmesh: dict[str, Any] | None
    error: dict[str, Any] | None
    feedback_count: int
    content: RetainedContent | None


class RequestRecordList(_Output):
    data: list[RequestRecord]
    next_cursor: str | None


# ---------------------------------------------------------------- feedback
class FeedbackRequest(_Input):
    request_id: RequestId
    label_source: Literal["automated_eval", "human_review", "user_signal"]
    metric: Annotated[str, Field(pattern=r"^[a-z0-9_.:-]{1,64}$")]
    metric_version: Annotated[str, Field(max_length=64, pattern=r"^[A-Za-z0-9_.:+-]+$")] | None = None
    score: Annotated[float, Field(ge=0, le=1)]
    higher_is_better: bool = True
    accepted: bool | None = None
    attribution: Literal["direct", "workflow"] = "direct"
    evaluator: Annotated[str, Field(min_length=1, max_length=128)]
    eval_run_id: Annotated[str, Field(max_length=128)] | None = None
    dataset_version: Annotated[str, Field(max_length=128)] | None = None
    labelled_at: datetime | None = None
    task_type: TaskType | None = None
    notes: Annotated[str, Field(max_length=2000)] | None = None


class FeedbackResponse(_Output):
    id: Annotated[str, Field(pattern=r"^fb_[0-9A-HJKMNP-TV-Z]{26}$")]
    request_id: RequestId
    status: Literal["stored", "replayed"]
    training_eligible: bool
    created_at: datetime


class FeedbackBatchRequest(_Input):
    items: Annotated[list[FeedbackRequest], Field(min_length=1, max_length=500)]


class FeedbackBatchItem(_Output):
    index: int
    status: Literal["stored", "replayed", "conflict", "not_found", "invalid"]
    feedback: FeedbackResponse | None = None
    error: ErrorObject | None = None


class FeedbackBatchResponse(_Output):
    results: list[FeedbackBatchItem]


class FeedbackDeleteResponse(_Output):
    deleted: int


# ---------------------------------------------------------------- cache
class CacheInvalidateRequest(_Input):
    namespace: Namespace | None = None
    context_fingerprint: Fingerprint | None = None

    @model_validator(mode="after")
    def _at_least_one(self) -> CacheInvalidateRequest:
        if self.namespace is None and self.context_fingerprint is None:
            raise ValueError("at least one of namespace or context_fingerprint is required")
        return self


class CacheInvalidateResponse(_Output):
    deleted_entries: int
