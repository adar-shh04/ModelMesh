# ModelMesh Inference API: Contract v1 (1.0-rc2)

Status: **release candidate 2**. This release incorporates the dispositions of Mezvuat's change requests CR-01 to CR-14 (see `CR_RESOLUTIONS.md`).

- **Owner:** ModelMesh.
- **Consumers:** Mezvuat AI and any OpenAI-compatible client.
- **Normative schemas:** `docs/contract/openapi-v1.yaml` (OpenAPI 3.1). This document defines semantics. If the two disagree, the YAML wins and the disagreement is a bug.
- **Consumer-side copy:** `mezvuat-ai/docs/contract/inference-api-v1.md`. This is a read-only mirror: Mezvuat raises change requests and never edits it.
- **Rationale:** `docs/adr/` (cited as ADR-NNNN).

## 0. Principles

- **OpenAI-compatible core.** A stock OpenAI SDK pointed at `base_url=<modelmesh>/v1` must work with `model` set to a concrete id or to `"auto"`.
  - ModelMesh extensions (`routing`, `cache`) are optional top-level fields. SDK users send them through `extra_body`.
  - `metadata` has OpenAI's own shape: a string→string map (§2).
- **Strict validation and negotiation.** The server rejects unknown fields with **400 `invalid_request`** (code `unknown_parameter`). This applies to unknown top-level fields and to unknown fields inside `routing` or `cache`.
  - To avoid sending fields an older server does not know, clients read **`GET /v1/contract`** at start-up. It lists the supported extension fields, task types, capabilities, policies and feedback metrics. Clients then send only those.
  - **Hard parameters** are honoured exactly, or the model is ineligible. They are never silently dropped. They are: `response_format`, `tools`, `tool_choice`, `stop`, `n`, `logprobs`.
  - **Soft parameters** are forwarded where the provider supports them. If one is dropped, the response reports it in `routing.warnings` as `param_dropped:<name>`. They are: `presence_penalty`, `frequency_penalty`, `seed`, `top_p`, `reasoning_effort`, `user`.
- **Versioning.** The path prefix is `/v1`. Changes within v1 are additive and optional only. Responses carry `X-ModelMesh-Contract: 1.0`. Breaking changes go to `/v2`.
- **Units.**
  - Money: USD as JSON numbers (`*_usd`).
  - Durations: milliseconds (`*_ms`), or seconds when the name ends in `_s`.
  - Tokens: integers.
  - Timestamps: RFC 3339 UTC, except the OpenAI-compatible `created` field, which is in Unix seconds.
- **Honesty fields.** Every predicted or estimated value is labelled as such (`predicted_*`, `estimated_*`, `source`). Estimated cost is not billing data. Provider-reported cost is reported separately.

## 1. Authentication and correlation

| Header | Direction | Required | Meaning |
|---|---|---|---|
| `Authorization: Bearer mm_<env>_<key>` | req | yes (except `/v1/health`, `/v1/contract`) | Tenant-scoped ModelMesh API key. Provider keys never leave ModelMesh. |
| `X-Request-ID` | req/resp | no | Client-generated id (UUID or ULID). Echoed back, and generated if absent. Stored as `client_request_id`. **It is not an idempotency key**: send a new one per attempt. ModelMesh never deduplicates inference requests. |
| `traceparent` / `tracestate` | req | no | W3C Trace Context. ModelMesh's server span becomes a **child of the caller's span**, so both share one trace id. ModelMesh uses a parent-based sampler: if the caller's sampled flag is 0, ModelMesh records no spans but still returns `trace_id`. These headers are not forwarded to providers. |
| `X-ModelMesh-Request-ID` | resp | always | ModelMesh execution id (`mmr_<ULID>`), **assigned on receipt** before authentication. Primary key for `/v1/requests/{id}` and `/v1/feedback`. |
| `X-ModelMesh-Contract` | resp | always | Contract version served (`1.0`). |
| `Retry-After` | resp | **always on 429**; on 503 when known | Integer seconds. |
| `X-RateLimit-{Limit,Remaining,Reset}-Requests` | resp | for rate-limited keys | Back-off guidance. |

**Correlation keys a client should store** (ADR-0009):
- `X-Request-ID`
- the `mmr_` id
- `modelmesh.trace_id`
- `metadata.correlation_id`

**What happens to `metadata`.** It is stored with the execution record and returned by `GET /v1/requests`. **Only `task_type` influences routing and the cache key.** Every other key is used only for storage and analytics.

## 2. `POST /v1/chat/completions`

### Request

**Standard OpenAI fields:**
- `model`: `"auto"` or `<provider>/<provider-model-id>`, split on the first `/`. Example: `groq/openai/gpt-oss-20b`.
- `messages`: roles `system`, `developer`, `user`, `assistant`, `tool`. Content is a string or an array of text parts. Image and audio parts return 400 in v1.
- `temperature`, `top_p`
- `max_tokens` and `max_completion_tokens`: if both are sent, they must be equal.
- `stop`, `seed`, `presence_penalty`, `frequency_penalty`, `reasoning_effort`
- `stream`, `stream_options.include_usage`
- `response_format`: `text`, `json_object` or `json_schema`
- `tools`, `tool_choice`, `parallel_tool_calls`
- `n`: must be 1.
- `logprobs`: unsupported.
- `user`, `store`: accepted and ignored.

Extensions (all optional):

```jsonc
{
  "routing": {
    "policy": "quality_balanced",        // §2.1; default "quality_balanced"; used only when model == "auto"
    "max_cost_usd": 0.02,                // ceiling on ESTIMATED cost (upper bound via max_tokens), incl. possibly-billed failed attempts
    "max_latency_ms": 10000,             // routing budget: eligible iff PREDICTED p90 latency <= this
    "timeout_ms": 20000,                 // hard deadline for the whole request; default min(2*max_latency_ms, server max) or 60000
    "min_quality": 0.7,                  // calibrated P(acceptable) for task_type (§2.4); only for calibrated task types
    "allowed_providers": ["groq", "gemini"],
    "denied_providers": [],
    "allowed_models": [],
    "required_capabilities": ["json_schema"],  // json_object | json_schema | json_schema_strict | tools | streaming | long_context:<n>
    "data_policy": "any",                // "any" | "no_training_providers" | "local_only"; may only tighten the key's ceiling
    "fallback": "within_constraints"     // default for auto; default for explicit models is "none"
  },
  "cache": {
    "mode": "exact",                     // "off" | "exact" (default) | "semantic"
    "ttl_s": 3600,                       // capped by the key's max TTL
    "context_fingerprint": "sha256:<64 lowercase hex>", // opaque; REQUIRED for semantic; part of every cache key
    "namespace": "ws_7c1e"               // optional partition WITHIN the tenant (e.g. hashed workspace id)
  },
  "metadata": {                          // OpenAI-shaped: <= 16 string keys, string values <= 512 chars
    "application": "mezvuat",            // reserved
    "task_type": "evidence_synthesis",   // reserved, §2.2
    "correlation_id": "run_01J…",        // reserved: <= 128 chars, [A-Za-z0-9_:.-]
    "end_user_id": "hash:…",             // reserved: <= 128 chars, opaque, pre-hashed; analytics only; NOT part of the cache key
    "workflow_node": "synthesize"        // any other key is a free-form tag
  }
}
```

**Constraint semantics.** Hard constraints filter the candidate models, for explicit models as well as `auto`. The hard constraints are:
- `max_cost_usd`
- `max_latency_ms`
- `min_quality`
- the provider and model allow/deny lists
- `required_capabilities`
- `data_policy`

If no candidate survives, the response is **422 `no_eligible_model`**. Its `details.violated_constraints` lists, for each constraint, `{constraint, excluded_candidates, best_available}`.

ModelMesh never relaxes a hard constraint silently, including during fallback. Specific cases:
- When `min_quality` is set, a model with no calibrated evidence for that task_type is ineligible (§2.4).
- `data_use` is attested by the operator **per configured provider account and tier**. For example, a Gemini free-tier account is `may_train` and a paid account is `no_training`. `no_training_providers` excludes `may_train` and `unknown`.
- A `data_policy` looser than the key's ceiling returns 403.

### 2.1 Routing policies (ADR-0004)

| Policy | Objective | Constraints honoured |
|---|---|---|
| `cost_optimized` | Minimum estimated cost | `min_quality` (policy default 0.6 if unset, reported in warnings), `max_latency_ms`, `max_cost_usd` |
| `latency_optimized` | Minimum predicted p50 latency | `min_quality`, `max_cost_usd` |
| `quality_balanced` | **Cheapest model whose predicted quality is within δ = 0.05 of the best eligible prediction** | All |
| `quality_max` | Maximum predicted quality | All that are set |
| `reliability_first` | Minimum predicted failure probability, tie-broken on cost | All |

The router returns a ranked list of eligible models. That list is also the fallback order.

If no learned router is active, or the learned router has low confidence for a request, ModelMesh uses the documented rules policy and reports `routing.mode = "rules_fallback"`. Low confidence means the request is out of distribution, or its task_type or model was not seen in training.

### 2.2 `task_type` (v1)

| Value | Use |
|---|---|
| `query_classification` | Low-latency labelling and routing of user queries |
| `query_rewriting` | Reformulating a query |
| `research_planning` | Decomposing a question into sub-questions or steps |
| `evidence_assessment` | Short structured judgment of whether the evidence is sufficient |
| `evidence_synthesis` | Answering from supplied evidence |
| `citation_verification` | Structured check of whether a cited passage supports a claim |
| `final_response` | User-facing answer generation |
| `evaluation_judge` | LLM-as-judge for offline evaluation |
| `general` | Default |

- New values may be added within v1. `GET /v1/contract` lists the values the server supports.
- An unknown value is treated as `general` and reported in `routing.warnings` as `unknown_task_type:<value>`.
- **`evaluation_judge`** requests are routed and accounted for normally. However, they are **excluded from router quality-training data**: feedback on them is `training_eligible: false`, and they are not mixed into labels for other task types. Their latency and reliability telemetry is still used.
- **Callers should pin an explicit judge model** (with `fallback: "none"`) for reproducible evaluations, and choose a family that differs from the model being judged. If routing chooses the judge, the evaluator changes whenever the router does, and evaluations stop being comparable.

### 2.3 Cache correctness (ADR-0006)

**Tenant isolation.** The tenant always comes from the API key, never from the request body. Entries never cross tenants.

**`namespace`** scopes entries *within* a tenant.
- It is part of both exact and semantic keys, so a hit never crosses namespaces.
- An absent namespace is its own partition, distinct from every named namespace.
- The same namespace string used by two tenants shares nothing.

**`context_fingerprint`** is **opaque and caller-computed**. It must match `^sha256:[0-9a-f]{64}$`. ModelMesh never interprets it.
- It is part of every cache key.
- It is **required for semantic mode**. Without it, the response is 400 `missing_context_fingerprint`.
- Non-normative example: a RAG client might hash its workspace, prompt version, and the sorted `(chunk id, chunk version hash)` pairs. Then an answer grounded in an older document version is never reused for newer evidence.
- `namespace` and `context_fingerprint` are independent. Both are part of the key, and using both is allowed.

**Partition.** Every cache key is built from:
- the tenant and `namespace`
- the resolved target: the concrete model, or `auto` plus the policy and the canonical hard constraints
- a hash of the system prompt and of every message except the last
- the generation parameters, the full `response_format` schema, and `tools` and `tool_choice`
- `context_fingerprint`, `task_type`, and the cache-format version

The **exact key** is the partition plus the final message.

**A semantic hit** requires all of:
- the same partition
- cosine similarity of the final user message at or above the threshold for the task_type
- identical numbers, negations and quoted strings

Semantic mode is disabled when `tools` are present.

**Reading.** On read, the cached entry's model must still pass the current request's eligibility checks, such as `data_policy` and the provider and model restrictions.

**Default mode for non-deterministic requests.** When `cache.mode` is omitted and the request has `temperature > 0` with no `seed`, the cache is **neither read nor written**, and the response reports `cache.status: "bypass"`. Setting `cache.mode` explicitly opts in.

**Writing.** Only complete, validated responses are stored, meaning those whose `finish_reason` is `stop` or `length`. Errors, cancelled streams and interrupted streams are never stored.

**On a hit:**
- `routing` is `null`.
- `cache.original_request_id` and `saved_estimated_usd` are set.
- `cost.estimated_usd` is 0.
- Top-level `usage` is the original request's usage.

**Purge:** `POST /v1/cache/invalidate` (§4).

### 2.4 Quality scale and calibration (ADR-0004, ADR-0005)

`min_quality` and `predicted_quality` are **calibrated probabilities that the response passes the task_type's acceptance criterion**. The criteria for v1:

| task_type | Acceptance criterion (v1) | Label source |
|---|---|---|
| `query_classification` | Exact label match | Automatic |
| `query_rewriting` | Reference-guided rubric score of 4/5 or higher (same intent as the gold rewrite, self-contained) | LLM judge, calibrated against human labels |
| `research_planning` | Reference-guided rubric score of 4/5 or higher (covers the gold reasoning hops) | LLM judge, calibrated against human labels |
| `evidence_assessment` | Sufficiency label correct (proxy: answerable vs unanswerable items, e.g. MuSiQue-Full) | Automatic |
| `evidence_synthesis` | Token F1 of 0.5 or higher against gold **and** a faithfulness judge score of 4/5 or higher | Automatic + judge |
| `citation_verification` | JSON valid against the schema **and** support label correct | Automatic |
| `final_response`, `general` | Per source: programmatic (IFEval), exact match (GSM8K), accuracy (MMLU-Pro), or judge score of 4/5 or higher (MT-Bench) | Mixed |
| `evaluation_judge` | **Not calibrated in v1.** With `min_quality` set, the response is always 422 | n/a |

Reading `min_quality: 0.7`: among requests like this one, at least about 70% of the chosen model's responses are expected to be acceptable. `predicted_quality_interval` is a 90% interval. Both are `null` when there is no calibrated evidence.

**Machine-readable evidence: `GET /v1/routing/calibration`.** For each task_type it returns:
- `calibrated`, the criterion and its version
- the benchmark manifest hash and the number of calibration samples
- ECE and Brier score
- Cohen's κ between judge and human labels, for judge-labelled tasks
- per-model coverage

Clients should send `min_quality` only for task types where `calibrated` is `true`. Until router training (M6) has produced results, every task type reports `calibrated: false`.

### 2.5 Structured output (ADR-0007)

**Validation always happens.** ModelMesh validates every `json_schema` response against JSON Schema 2020-12, and checks that every `json_object` response parses as a JSON object.
- On failure it makes one repair attempt, then falls back to the next eligible model.
- If everything fails, the response is **502 `upstream_error`, code `schema_validation_failed`**, `retryable: true`. `details.validation_errors` is a truncated list of `{path, keyword}` entries with no values or content.
- **A 200 response is always valid against the requested format.**

**Enforcement modes** are reported in `modelmesh.structured_output.mode`:
- `native_strict`: constrained decoding
- `native_best_effort`: the provider honours the schema but ignores some keywords
- `emulated`: the schema is given in the prompt and the output is validated

**Requiring a mode:**
- No requirement: any mode is allowed, and the output is still validated.
- `required_capabilities: ["json_schema"]`: native modes only.
- `required_capabilities: ["json_schema_strict"]`: `native_strict` only.

**Portable schema subset.** A schema inside this subset is eligible for `native_strict` on every native-capable model. This will be verified by live tests in M2.
- The root is an object.
- Every property is listed in `required`.
- Every object has `additionalProperties: false`.
- Only these keywords are used: `type` (nullable written as `["T","null"]`), `properties`, `required`, `items`, `enum`, `const`, `description`, `minItems`, `maxItems`.
- Nesting depth is at most 5.
- None of these are used: `$ref`, `oneOf`, `anyOf`, `allOf`, `pattern`, `format`, `minLength`, `maxLength`, `uniqueItems`.

Schemas outside the subset are still accepted and validated, but they may qualify only for `native_best_effort` or `emulated`.

**Streaming with structured output.**
- `stream: true` with `json_schema` is **allowed**. If the chosen model cannot stream structured output, ModelMesh uses **buffered streaming**: it calls upstream without streaming, validates the output, and replays it as chunks with `modelmesh.streaming.mode = "buffered"`.
- If the caller also lists `streaming` in `required_capabilities`, ModelMesh requires native structured streaming. Buffered mode is then excluded, and if no model qualifies the response is 422.

### Response (non-streaming)

The response is a standard OpenAI `chat.completion` object:
- `model` is the concrete model that was used.
- `choices` has exactly one entry.
- `usage` includes `prompt_tokens_details.cached_tokens` and `completion_tokens_details.reasoning_tokens` when they are known.

It also carries a `modelmesh` object:

```jsonc
"modelmesh": {
  "request_id": "mmr_01J…",
  "client_request_id": "…",              // echo of X-Request-ID
  "trace_id": "4bf92f35…",               // W3C trace id (32 hex)
  "provider": "groq",
  "model": "groq/openai/gpt-oss-120b",
  "routing": {                           // null on cache hits
    "mode": "learned",                   // "learned" | "rules_fallback" | "explicit_model"
    "policy": "quality_balanced",        // null for explicit models
    "router_version": "router-2026-12-01-lgbm",
    "task_type": "evidence_synthesis",   // resolved
    "candidates_considered": 7,
    "candidates_eligible": 3,
    "predicted_quality": 0.82,
    "predicted_quality_interval": [0.74, 0.89],
    "predicted_latency_ms": 1400,        // p50
    "predicted_latency_p90_ms": 2600,
    "estimated_cost_usd": 0.00071,       // pre-execution upper bound
    "reason": "cheapest eligible model within 0.05 of best predicted quality",
    "warnings": []
  },
  "attempts": [
    {"provider": "gemini", "model": "gemini/gemini-3.6-flash", "outcome": "upstream_timeout", "latency_ms": 10000, "http_status": null, "retryable": true, "possibly_billed": true},
    {"provider": "groq", "model": "groq/openai/gpt-oss-120b", "outcome": "success", "latency_ms": 1320, "http_status": 200, "retryable": false, "possibly_billed": true}
  ],
  "fallback_used": true,
  "cache": {"status": "miss", "similarity": null, "original_request_id": null, "saved_estimated_usd": null},  // hit_exact | hit_semantic | miss | bypass
  "usage": {"prompt_tokens": 2310, "completion_tokens": 412, "cached_tokens": null, "reasoning_tokens": 96, "source": "provider_reported"},
  "cost": {"estimated_usd": 0.00059, "provider_reported_usd": null, "failed_attempts_estimated_usd": 0.00012, "pricing_version": "2026-11-01"},
  "latency_ms": {"total": 11450, "provider": 1320, "overhead": 130, "ttft": null},
  "structured_output": {"mode": "native_strict", "validated": true, "repair_attempts": 0},  // null if text
  "streaming": null                       // {"mode": "native" | "buffered"} for streams
}
```

### Streaming

With `stream: true` the response is `text/event-stream` carrying OpenAI `chat.completion.chunk` events, terminated by `data: [DONE]`.

**Commit point.** The HTTP status and headers are sent once the first upstream content is ready. The headers are `X-ModelMesh-Request-ID`, `X-Request-ID` and `X-ModelMesh-Contract`, so a stream that later breaks can always be correlated. Any failure *before* the commit point is an ordinary JSON error with its real HTTP status, exactly as for a non-streaming request. Such failures include all fallback attempts failing (502, 503 or 504), and 4xx rejections. `routing.timeout_ms` bounds the wait for the commit point. No keep-alive is sent before it.

**After the commit point:**
- Comment lines (`: keep-alive`, sent every 15 s while waiting) may appear, and clients must ignore them.
- A 30 s upstream idle timeout applies.

**Usage and metadata.**
- `stream_options.include_usage` **defaults to false**, as in OpenAI. When it is true, the last chunk before `[DONE]` has `choices: []`, `usage`, and the full `modelmesh` object.
- When it is false, the chunk carrying `finish_reason` also carries `modelmesh`.
- Mezvuat's client should set `include_usage: true`.

**`finish_reason`** is normalized to `stop`, `length`, `tool_calls` or `content_filter`. Provider-specific values are mapped onto these. Failures are never expressed as a `finish_reason`.

**Mid-stream failure.** If an upstream fails after the commit point, ModelMesh emits one event, `data: {"error": {…}, "modelmesh": {…}}`. The `error` object uses the §3 schema with code `stream_interrupted`. The `modelmesh` object carries the attempts and the usage so far, with `usage.source`. The stream then **closes without `[DONE]`**, so a missing `[DONE]` always means the response is incomplete. The record's status becomes `stream_interrupted`.

**Client disconnect** cancels the upstream request. Usage is still recorded: estimated if the provider reported none, and flagged `possibly_billed`. The record's status becomes `cancelled`.

## 3. Errors

All non-2xx bodies and in-stream error events use this shape:

```json
{"error": {"type": "no_eligible_model", "code": "constraints_unsatisfiable", "message": "…", "param": null,
           "request_id": "mmr_…", "client_request_id": "…", "retryable": false,
           "details": {"candidates_considered": 7, "violated_constraints": [{"constraint": "max_cost_usd", "excluded_candidates": 2, "best_available": 0.031}]}}}
```

- `request_id` is always present. Unauthenticated requests get an id, but no record is stored for them.
- `client_request_id` echoes `X-Request-ID`. It is `null` if the client did not send one.
- `details` depends on the error type:
  - `no_eligible_model`: `candidates_considered`, `violated_constraints[]`
  - `upstream_error`, `upstream_timeout`, and `service_unavailable` with code `upstream_capacity`: `attempts[]`, the same shape as `modelmesh.attempts`
  - `budget_exceeded` and `rate_limited`: `{scope: key|tenant, period: minute|day|month, resets_at}`
  - `schema_validation_failed`: `validation_errors[]`
  - Fields that don't apply are omitted. `details` never contains content.

| HTTP | `type` | Retryable | Typical `code`s |
|---|---|---|---|
| 400 | `invalid_request` | no | unknown_parameter, invalid_value, missing_context_fingerprint, unsupported_content_part, max_tokens_mismatch |
| 401 | `authentication_error` | no | invalid_api_key, revoked_api_key |
| 402 | `budget_exceeded` | no | daily_budget, monthly_budget, request_quota |
| 403 | `permission_denied` | no | model_not_allowed, provider_not_allowed, data_policy_exceeds_key |
| 404 | `not_found` | no | model_not_found, request_not_found (also returned for another tenant's ids) |
| 409 | `conflict` | no | feedback_conflict |
| 413 | `request_too_large` | no | body_too_large, context_too_large |
| 422 | `no_eligible_model` | no | constraints_unsatisfiable |
| 429 | `rate_limited` | yes (`Retry-After`, always present) | rpm, tpm, concurrency: **ModelMesh's own per-key limits only** |
| 500 | `internal_error` | yes | n/a |
| 501 | `not_implemented` | no | embeddings_disabled |
| 502 | `upstream_error` | yes | all_attempts_failed, schema_validation_failed, content_filtered |
| 503 | `service_unavailable` | yes | upstream_capacity (every eligible upstream is rate- or quota-limited; `Retry-After` when known), dependency_unavailable |
| 504 | `upstream_timeout` | yes | deadline_exceeded |
| in-stream | `upstream_error` | no (partial output already sent) | stream_interrupted |

### 3.1 Client fallback guidance (normative)

| Signal | Meaning | What the client should do |
|---|---|---|
| Network error or timeout reaching ModelMesh; **500, 502, 503, 504** | **ModelMesh cannot serve this right now** | Retry within its own budget, then it **may** use its direct-provider fallback. That fallback must enforce the same data policy itself. |
| **429** `rate_limited` | The caller has exceeded its **own** ModelMesh limits. ModelMesh is healthy. | **Back off** for `Retry-After` seconds, within the caller's deadline. Do **not** treat it as unavailability. If `Retry-After` exceeds the remaining deadline, fail the step, or degrade within the caller's own policy. Falling back to a direct provider here would bypass limits set by the operator. That is a client policy decision and is not recommended. |
| 400, 401, 402, 403, 404, 409, 413, 422, 501 | The request is wrong, not allowed, or unsatisfiable | **Never fall back.** Doing so would bypass policy, or repeat a broken request. |
| In-stream `stream_interrupted` | Partial output was delivered | Discard the partial output. Retrying the whole request is allowed. |

Upstream quota exhaustion is deliberately reported as **503 `upstream_capacity`**, never 429, so that it triggers the first row.

## 4. Other endpoints

- **`GET /v1/contract`** (unauthenticated): returns `{contract_version, contract_release, endpoints[], extensions[], task_types[], capabilities[], routing_policies[], feedback_metrics[]}` for start-up negotiation.
- **`GET /v1/models`**: OpenAI list format, filtered to the models the key may use.
  - Includes a pseudo-model `auto` (`modelmesh.kind = "router"`).
  - Each item adds `modelmesh: {kind, provider, family, context_window, max_output_tokens, capabilities[], pricing: {input_per_mtok_usd, output_per_mtok_usd, pricing_version, free_tier}, data_use, status}`.
  - `data_use` (`no_training` | `may_train` | `unknown`) is attested per configured provider account and tier.
- **`GET /v1/routing/calibration`**: per-task_type calibration evidence (§2.4).
- **`GET /v1/health`** (unauthenticated):
  - Body: `{status: ok|degraded|down, contract_version, components: {postgres, redis, providers: {ok, degraded, down}}}`.
  - HTTP 200 when `ok` or `degraded`, 503 when `down`.
- **`GET /v1/providers`**: per provider, `{id, status, circuit_state, error_rate_5m, p50_latency_ms, p95_latency_ms, models_available}`.
- **`GET /v1/usage?from=&to=&group_by=model|provider|task_type|application|day[&api_key_id=]`**:
  - Aggregates scoped to the tenant.
  - Provider-reported and estimated tokens and costs are in separate columns, with `provider_reported_cost_coverage`.
  - Maximum range: 92 days.
- **`GET /v1/requests/{mmr_id}`**: the execution record.
  - Contents: status (`succeeded`, `failed`, `stream_interrupted`, `cancelled`, `rejected`, `in_progress`), timestamps, metadata, the `modelmesh` object (or `null` for rejected requests), error, and `feedback_count`.
  - Prompt and response content is included only if the key has content retention enabled. Retention is **off by default**, with a 7-day TTL when on.
- **`GET /v1/requests?correlation_id=&from=&to=&limit=&cursor=`**: the tenant's records, newest first. No content is included unless retention is enabled.
- **`POST /v1/cache/invalidate`**: body `{namespace?, context_fingerprint?}`, at least one required. Purges the tenant's matching entries and returns `{deleted_entries}`.
- **`POST /v1/embeddings`**: OpenAI-compatible, optional in v1. Concrete models only, with no routing and no cache. Returns 501 when disabled.

## 5. Feedback (quality-label interface)

### `POST /v1/feedback`

```jsonc
{
  "request_id": "mmr_01J…",
  "label_source": "automated_eval",      // "automated_eval" | "human_review" | "user_signal"
  "metric": "faithfulness",              // ^[a-z0-9_.:-]{1,64}$; vocabulary below
  "metric_version": "0.2.1",             // optional; part of the idempotency key
  "score": 0.86,                         // 0..1
  "higher_is_better": true,              // optional, default true
  "accepted": true,                      // optional binary verdict under the client's criterion (preferred for training)
  "attribution": "direct",               // optional: "direct" (default) | "workflow" (end-to-end score shared by several requests)
  "evaluator": "mezvuat-eval/ragas-0.x+citation-v1",
  "eval_run_id": "mlflow:3f2a…",         // optional provenance
  "dataset_version": "eval-2026-11",     // optional
  "labelled_at": "2026-11-08T12:00:00Z", // optional
  "task_type": "evidence_synthesis",
  "notes": null                          // never personal data, prompt content or document content
}
```

**Responses.**
- The idempotency key is (`request_id`, `metric`, `metric_version`, `evaluator`).
- **201** when stored. **200** with `status: replayed` for an identical replay. **409** when the same key arrives with a different `score` or `accepted`.
- **400** for an invalid body, such as a score outside 0..1. Validation errors are 400 throughout this contract.
- **404** when the request id is unknown or belongs to another tenant. The response does not reveal which.
- The response body is `{id: "fb_…", request_id, status, training_eligible, created_at}`.

**Batch: `POST /v1/feedback/batch`.**
- Body `{items: [...]}`, with 1 to 500 items.
- The response is 200 with `results[]`, one entry per item: `{index, status: stored|replayed|conflict|not_found|invalid, feedback?, error?}`.
- Items are processed independently, not atomically.

**Delete: `DELETE /v1/feedback/{request_id}[?metric=&evaluator=&metric_version=]`.**
- Hard-deletes the tenant's matching labels and returns `{deleted}`.
- Deleted labels are excluded from every future dataset build. Trained artifacts are not retrained automatically.

**Metric vocabulary.** It is additive. Other values that match the pattern are accepted.
- RAGAS: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`
- Client custom: `citation_validity`, `citation_support_rate`, `temporal_accuracy`, `schema_valid`, `task_success`
- User signal: `user_thumbs`

**Training use.**
- Feedback is stored as **candidate** training data.
- Using it for router retraining requires an explicit, versioned dataset build and an offline evaluation. Nothing changes production routing automatically.
- `training_eligible` is false when any of these hold:
  - `label_source = user_signal`
  - `attribution = workflow`
  - the labelled request's task_type is `evaluation_judge`

## 6. Resolved questions (from DRAFT 0.1 §6)

1. **Calibration scale:** §2.4, plus `GET /v1/routing/calibration` (ADR-0004 and ADR-0005).
2. **`max_latency_ms`:** a routing budget on predicted p90 latency. `routing.timeout_ms` is the hard deadline (ADR-0008).
3. **Structured output:** emulation with validation is allowed. `json_schema` requires a native mode and `json_schema_strict` requires `native_strict`. Invalid output is never returned (ADR-0007).
4. **Retention:** prompt and response content retention is off by default, with a 7-day TTL when enabled (ADR-0010).

## 7. Changelog

### 1.0-rc2 (2026-11-08): Mezvuat change requests reconciled

Full per-CR table: `CR_RESOLUTIONS.md`.

| # | Change | Source | Client impact |
|---|---|---|---|
| R1 | New task types `evidence_assessment` and `evaluation_judge`. Judge requests are excluded from router training data, and pinning an explicit judge model is recommended | CR-01 | Mezvuat may send the new values. **Check `GET /v1/contract` first** |
| R2 | New capability `json_schema_strict`. The portable schema subset is documented. `details.validation_errors` added | CR-02 | Opt-in |
| R3 | **Stream commit point.** Failures before the first content are now real HTTP errors, not in-stream events. There is no keep-alive before the commit point. In-stream error events carry `modelmesh`. Client disconnect behaviour is defined (status `cancelled`). `finish_reason` is normalized | CR-03 | Handle 5xx on stream requests as for non-streaming ones. This **changes rc1**, which sent keep-alives before the first token |
| R4 | `context_fingerprint` is defined as opaque. The Mezvuat formula is now a non-normative example. Namespace scoping is spelled out | CR-04 | None (rc1 already had the format and `namespace`) |
| R5 | When `cache` is omitted and the request is non-deterministic, there is **no read and no write** | CR-05 | None |
| R6 | `ErrorObject.request_id` is **always present** (non-nullable). `client_request_id` added. `details` gets a typed schema, and **`violated_constraints` changes from an rc1 map to a list** of `{constraint, excluded_candidates, best_available}`. `Retry-After` is required on 429. Client fallback guidance (§3.1) added. `X-Request-ID` is documented as not an idempotency key | CR-06 | Parse the list form |
| R7 | `correlation_id` pattern `[A-Za-z0-9_:.-]`. How metadata is used is now stated. `traceparent` child-span semantics and sampling note added | CR-08 | Keep correlation ids within the pattern |
| R8 | `data_use` is defined per provider account and tier | CR-10 | None |
| R9 | Per-task acceptance criteria table and `GET /v1/routing/calibration` | CR-11 | Send `min_quality` only when `calibrated: true` |
| R10 | `GET /v1/contract` for negotiation. Unknown fields are still 400 (not 422) | CR-12 | Read at start-up |
| R11 | Feedback gains `metric_version` (now in the idempotency key), `higher_is_better`, `attribution`, `eval_run_id`, `dataset_version`, `labelled_at`. Also added: `POST /v1/feedback/batch` (up to 500), `DELETE /v1/feedback/{request_id}`, a replay status, and the metric vocabulary | CR-13 | Optional fields |

Rejected sub-items, explained in `CR_RESOLUTIONS.md`:
- `[DONE]` after an in-stream error (CR-03.2)
- `finish_reason: "error"` (CR-03.4)
- 429 as a direct-provider fallback trigger (CR-06.1)
- 422 for an invalid feedback score (CR-13.3)

### 1.0-rc1 (2026-11-08): changes from DRAFT 0.1

Breaking or behavioural changes, relative to the draft:

| # | Change | Reason |
|---|---|---|
| C1 | Schema and unknown-field errors are **400** `invalid_request`. 422 is reserved for `no_eligible_model` | The draft contradicted itself (§0 said 422, §3 said 400). OpenAI uses 400 for invalid parameters |
| C2 | `metadata` is an **OpenAI-shaped string→string map** with at most 16 keys. The nested `tags` object is replaced by free-form extra keys | OpenAI defines `metadata` as a string map. A nested object breaks typed OpenAI SDKs and compatible tooling. **Mezvuat impact: flatten `tags`** |
| C3 | `stream_options.include_usage` defaults to **false**. When false, `modelmesh` rides on the `finish_reason` chunk | The usage chunk has `choices: []`, which crashes naive SDK code (`chunk.choices[0]`). **Mezvuat impact: set `include_usage: true`** |
| C4 | `quality_balanced` means the cheapest model within δ = 0.05 of the best predicted quality. The draft had "max Q − λ·cost" | λ was undefined, and its units are awkward (quality per USD at about 1e-4 USD per call). δ is explicit and auditable |
| C5 | Explicit models default to `fallback: "none"`. Hard constraints are enforced on explicit models too | A caller who named a model should not silently get another. Constraints are policy |
| C6 | Upstream capacity exhaustion returns **503** `upstream_capacity`. 429 is used only for ModelMesh's own limits | Separates "you are over your limit" from "ModelMesh can't serve right now", and triggers Mezvuat's 503 fallback |
| C7 | Stream errors close **without `[DONE]`**. Keep-alive comments are specified | A truncated stream can never look like a complete one |
| C8 | `cache.context_fingerprint` format is `sha256:<64 lowercase hex>`. `semantic` without it returns 400 | Makes the field verifiable. **Mezvuat impact: emit this exact format** |

Additions:

| # | Addition | Reason |
|---|---|---|
| A1 | `routing.timeout_ms` | Open question 2 (ADR-0008) |
| A2 | `cache.namespace` and `POST /v1/cache/invalidate` | Spec §2.3 E and §5.3 |
| A3 | `modelmesh.trace_id` and `GET /v1/requests?correlation_id=` | Spec §6 combined criteria (ADR-0009) |
| A4 | Prediction and estimation honesty fields: routing p90 and estimated cost; attempt `http_status` and `possibly_billed`; cache similarity, origin and savings; cached and reasoning tokens; `provider_reported_usd` and failed-attempt cost; `ttft`; `routing: null` on cache hits | Spec §2.3 F |
| A5 | `modelmesh.structured_output` and `modelmesh.streaming` | Open question 3 (ADR-0007) |
| A6 | `error.param`; the error types `not_found`, `conflict`, `request_too_large`, `internal_error`, `not_implemented`; a `code` vocabulary | Endpoints listed in the draft, and OpenAI compatibility |
| A7 | Feedback `accepted` field, a response schema, and 201/200/409 semantics | Router quality scale (ADR-0004) |
| A8 | `/v1/models`: the `auto` entry, plus `kind`, `family`, `max_output_tokens`, `data_use`, `pricing.free_tier` and `status` | Discoverability and `data_policy` transparency |
| A9 | `X-RateLimit-*` headers. `/v1/health` returns 503 when down | Client back-off and load balancers |
| A10 | Explicit parameter rules: `n` = 1; no `logprobs`; text-only content; `max_tokens` consistency; hard vs soft parameters | Makes "never silently dropped" concrete (ADR-0003) |

Editorial: example model ids moved to `groq/openai/gpt-oss-*`. Groq shut down its Llama models for free and developer tiers on 2026-08-16.

### DRAFT 0.1 (coordinator)

Initial draft.
