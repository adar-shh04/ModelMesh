# ModelMesh integration guide for client developers

This guide covers how to call ModelMesh from an application: what to send, how to handle each kind of failure, when you may fall back to a provider directly, how to keep data inside your policy, how to partition the cache and how to send quality feedback.

It explains the contract; it does not replace it. Where they differ, **`docs/contract/inference-api-v1.md` (semantics) and `docs/contract/openapi-v1.yaml` (schemas) win**. Section numbers below (§) refer to that contract.

Contract version: **1.0, frozen**. Changes within `/v1` are additive and optional. Every response carries `X-ModelMesh-Contract: 1.0`.

## 1. What you need

- **A base URL.** For example `http://localhost:8100/v1` for the local Compose stack.
- **An API key** (`mm_live_…` or `mm_test_…`), created by the operator with the admin API or the dashboard.
  - Each key belongs to one tenant. Each key has rate limits, a daily estimated-cost budget and a **data-policy ceiling**.
  - New keys default to 20 requests per minute, 50,000 tokens per minute and $0.25 of estimated cost per day (`docs/DECISIONS.md` D1). Ask the operator for explicit limits if you need more.
- **Any OpenAI-compatible client.** ModelMesh's extensions are optional top-level fields. With the OpenAI SDKs, send them through `extra_body`.

Provider keys never leave ModelMesh. Your application holds only its ModelMesh key.

## 2. Start-up: negotiate, then send only what the server knows

The server rejects unknown fields with **400 `invalid_request` / `unknown_parameter`**. This includes unknown fields inside `routing` and `cache`.

At start-up, read `GET /v1/contract`. It needs no authentication. Then:
- Check that `contract_version` starts with `1.`.
- Send only the extension fields listed in `extensions`.
- Send only the task types in `task_types`. An unknown task type is not an error: it is treated as `general` and reported as a warning.
- Send only the policies in `routing_policies` and the capabilities in `capabilities`.

Also read `GET /v1/routing/calibration` (authenticated). Send `routing.min_quality` **only** for task types where `calibrated` is `true` (see §4).

## 3. A request

```python
from openai import OpenAI

client = OpenAI(api_key=MODELMESH_KEY, base_url="http://localhost:8100/v1", max_retries=0)  # retries are yours, see §5

response = client.chat.completions.create(
    model="auto",                       # or a concrete id such as "groq/openai/gpt-oss-120b"
    messages=[{"role": "system", "content": "Answer from the evidence only."},
              {"role": "user", "content": "…"}],
    max_tokens=400,
    extra_headers={"X-Request-ID": new_uuid()},           # new for every attempt; it is not an idempotency key
    extra_body={
        "routing": {"policy": "cost_optimized", "data_policy": "any", "timeout_ms": 20000},
        "cache": {"mode": "exact", "namespace": "ws_" + sha256(workspace_id)[:16]},
        "metadata": {"application": "myapp", "task_type": "evidence_synthesis", "correlation_id": run_id},
    },
)
mm = response.model_extra["modelmesh"]   # request_id, trace_id, routing, attempts, cache, usage, cost, latency_ms
```

**Rules that matter:**
- **`model: "auto"` lets ModelMesh choose** within your constraints and fall back to the next eligible model if an attempt fails.
  - A concrete model id is honoured exactly. It gets **no fallback** unless you set `routing.fallback`.
  - `routing.policy` applies only to `auto`.
- **`metadata.task_type`** is the only metadata key that affects routing and the cache key. Set it on every request; it defaults to `general`.
  - `correlation_id` (at most 128 characters, `[A-Za-z0-9_:.-]`) groups the requests of one run or workflow.
  - `end_user_id` must already be hashed. It is used for analytics only.
  - Other keys are free-form tags. Never put prompt content or personal data in metadata.
- **Correlation.** Store these four values with your own records:
  - `X-Request-ID`
  - the `mmr_…` id (`modelmesh.request_id`, also in the `X-ModelMesh-Request-ID` header)
  - `modelmesh.trace_id`
  - `metadata.correlation_id`
- **Tracing.** Send `traceparent` and ModelMesh's span becomes a child of yours, so both share one trace id.
- **Hard constraints filter models and are never relaxed**, including during fallback:
  - `max_cost_usd`, `max_latency_ms` and `min_quality`
  - the provider and model allow and deny lists
  - `required_capabilities` and `data_policy`

  If nothing survives them, you get **422 `no_eligible_model`**. `details.violated_constraints` says which constraint excluded how many models, and the best value available.
- **`routing.timeout_ms`** is the hard deadline for the whole request, fallbacks included. `max_latency_ms` is different: it is a routing filter on *predicted* p90 latency.
- **Structured output** (`response_format` `json_object` or `json_schema`) is always validated. A 200 is always valid against your schema.
  - If every attempt fails validation, you get 502 with code `schema_validation_failed` (retryable).
  - To guarantee constrained decoding, require `json_schema` (any native mode) or `json_schema_strict` (`native_strict` only) in `required_capabilities`.
- **Streaming.** Set `stream_options.include_usage: true`. The full `modelmesh` object then arrives in the last chunk.
  - Ignore SSE comment lines (`: keep-alive`).
  - A stream that ends **without `data: [DONE]`** is incomplete. If it ended on an error event with code `stream_interrupted`, discard the partial output.

## 4. Routing, quality and what `routing.mode` tells you

- **`routing.mode`** is one of:
  - `learned`: quality estimates came from a promoted learned router.
  - `rules_fallback`: rules decided, because no learned router is active or it had low confidence for this request.
  - `explicit_model`: you named the model.
- **Today every task type is `calibrated: false`, and the mode is `rules_fallback`.** The P0 learned router was not promoted (`docs/milestones/M6_P0_PRELIMINARY_REPORT.md`).
  - Requests without `min_quality` work normally.
  - With `min_quality` set on an uncalibrated task type, a request returns **422**, because no model has calibrated evidence (§2.1.1).
- **The `cost_optimized` default floor (0.6)** applies only to calibrated task types and is never a hard constraint. Its effect is reported in `routing.warnings`, for example `default_min_quality_skipped:uncalibrated`.
- **Warnings.** `routing.warnings` also reports:
  - dropped soft parameters (`param_dropped:<name>`)
  - an unknown task type (`unknown_task_type:<value>`)
  - the use of a latency prior (`latency_prior_used`)

  Log them. They are not errors.

## 5. Error handling and fallback rules

Every error body has the same shape: `{"error": {type, code, message, param, request_id, client_request_id, retryable, details}}`. Errors never contain prompt content. Act on `type` and the HTTP status (§3, §3.1):

| You get | Meaning | Do this |
|---|---|---|
| Network error, timeout, **500, 502, 503, 504** | ModelMesh cannot serve this right now. ModelMesh has already tried every eligible model | Retry with backoff within your own deadline, with a **new `X-Request-ID`** each time. After that, you **may** use a direct-provider fallback, which must enforce the same `data_policy` itself (§6) |
| **429 `rate_limited`** | You exceeded **your own key's** limits. ModelMesh is healthy | Wait `Retry-After` seconds (always present) if that fits your deadline, otherwise fail the step. **Do not** treat it as an outage and do not fall back to a provider: that would bypass the operator's limits |
| **503 `service_unavailable` / `upstream_capacity`** | Every eligible upstream is rate- or quota-limited | Treat as unavailable (first row). `Retry-After` is present when known |
| 400, 401, 402, 403, 404, 409, 413, 422, 501 | The request is invalid, not allowed, over budget or unsatisfiable | **Never retry unchanged and never fall back.** Fix the request, or surface the error |
| In-stream `stream_interrupted` | Partial output was sent, then the upstream failed | Discard the partial output. Retrying the whole request is allowed |

Notes:
- **`402 budget_exceeded`** means the key's estimated-cost budget for the day or month is used up. `details.resets_at` says when it resets.
- **There is no idempotency key, and ModelMesh never deduplicates inference requests.**
  - A retry is a new request. It is recorded and, if it reaches a provider, possibly billed.
  - `modelmesh.attempts[].possibly_billed` marks upstream attempts that may have cost money even though they failed.
  - Budget your retries accordingly.
- **Fallback inside ModelMesh** is controlled by `routing.fallback`:
  - `within_constraints`, the default for `auto`, tries the next model in the ranked list.
  - `none`, the default for a concrete model, does not.
  - `modelmesh.fallback_used` and `modelmesh.attempts` show what happened.
- **Pin the model for evaluation judges.** For `evaluation_judge` requests, use a concrete model with `fallback: "none"`, from a different family than the model being judged. A routed judge changes whenever the router does (§2.2).

A minimal retry policy that follows these rules:

```python
RETRYABLE = {500, 502, 503, 504}

def call_with_policy(send, deadline_s: float):
    """`send()` makes one request with a fresh X-Request-ID and returns (status, body, headers)."""
    delay = 0.5
    while True:
        status, body, headers = send()
        if status < 400:
            return body
        remaining = deadline_s - now()
        if status == 429:
            wait = float(headers["Retry-After"])
            if wait > remaining:
                raise StepFailed(body)          # degrade within your own policy; do not bypass the gateway
            sleep(wait)
            continue
        if status in RETRYABLE and remaining > delay:
            sleep(delay)
            delay = min(delay * 2, 8)
            continue
        if status in RETRYABLE:
            return direct_provider_fallback_or_fail(body)   # only if it enforces the same data_policy
        raise RequestRejected(body)                          # 4xx: never fall back
```

## 6. `data_policy`: keeping data where it is allowed to go

`routing.data_policy` restricts which provider accounts may see the request (§2, D4):

| Value | Eligible models |
|---|---|
| `any` | All configured models |
| `no_training_providers` | Only accounts the operator has attested as `no_training`. `may_train` and `unknown` are excluded |
| `local_only` | Only local models (Ollama on the operator's machine) |

- **The key has a ceiling.** If you omit `data_policy`, the key's ceiling applies. If you send a looser one than the ceiling, you get **403 `data_policy_exceeds_key`**.
- **It is never relaxed.** This holds during fallback, and when reading a cache entry: a cached answer from a model your policy excludes is not served.
- **If no eligible model exists, you get 422.** For example, `local_only` when Ollama is not configured, or every local model's breaker is open. Do not retry that by loosening the policy.
- **Data use is attested per provider account and tier**, and is visible as `data_use` in `GET /v1/models`. The free tiers of Gemini, Groq and OpenRouter are classified `may_train` or `unknown` until reviewed (D4). So today, `no_training_providers` leaves only models whose account has been attested `no_training`, which may be none.
- **Your own direct-provider fallback must enforce the same policy.** A `local_only` request must never reach a hosted provider through your fallback path.
- **Set the policy per request from the data's sensitivity**, not per application. Mezvuat, for example, sends `any` for verified public-corpus workspaces and `local_only` for private ones.

## 7. Cache namespaces and fingerprints

The cache is keyed per tenant. The tenant comes from the API key, so entries never cross tenants (§2.3).

- **`cache.mode`:**
  - `exact` is the default.
  - `semantic` also matches a similar final user message: same partition, cosine at or above the task type's threshold, identical numbers, negations and quoted strings.
  - `off` neither reads nor writes.
  - Semantic mode is disabled when `tools` are present.
- **Non-deterministic requests are not cached by default.** If you omit `cache.mode` and send `temperature > 0` without a `seed`, the cache is bypassed (`cache.status: "bypass"`). Setting `cache.mode` explicitly opts in.
- **`cache.namespace`** partitions entries *within* the tenant.
  - Use one per isolation boundary, for example a hashed workspace id (`"ws_" + sha256(workspace_id)[:16]`). Never use the raw id if it is sensitive.
  - No namespace is its own partition.
- **`cache.context_fingerprint`** (`sha256:<64 lowercase hex>`) is opaque to ModelMesh. It is **required for semantic mode**; without it you get 400 `missing_context_fingerprint`.
  - Hash everything the answer depends on that is not in the messages: for a RAG client, the workspace, the prompt version and the sorted `(chunk id, chunk version hash)` pairs.
  - Then an answer grounded in old evidence is never reused for new evidence.
- **Both namespace and fingerprint are part of every key**, exact and semantic. The rest of the key covers:
  - the model or the routing target (policy and hard constraints)
  - the system prompt and earlier messages
  - generation parameters, `response_format` and tools
  - `task_type`
- **On a hit:**
  - `modelmesh.routing` is `null`.
  - `cache.status` is `hit_exact` or `hit_semantic`.
  - `cost.estimated_usd` is 0, and `cache.original_request_id` points at the request that produced the answer.
- **Invalidate** with `POST /v1/cache/invalidate {"namespace": …, "context_fingerprint": …}`. Send at least one of the two. Call it when a workspace's documents change or a workspace is deleted.
- **Only complete answers are stored**, those with `finish_reason` `stop` or `length`. Errors and interrupted streams never are.

## 8. Feedback: telling ModelMesh how good an answer was

Labels feed the learned router's training data, after an explicit, versioned dataset build. Nothing changes routing automatically (§5).

```http
POST /v1/feedback
{"request_id": "mmr_…", "label_source": "automated_eval", "metric": "faithfulness", "metric_version": "1.0",
 "score": 0.86, "accepted": true, "evaluator": "myapp-eval/ragas-0.x", "task_type": "evidence_synthesis"}
```

- **`request_id`** is the `mmr_…` id of the execution being labelled. An unknown id, or another tenant's, gives 404.
- **The idempotency key is `(request_id, metric, metric_version, evaluator)`.**
  - Replaying the same label returns 200 `replayed`.
  - Changing `score` or `accepted` under the same key returns **409**. To revise a label, bump `metric_version` or delete the label first.
- **`accepted`** is your binary verdict under your own acceptance criterion. Send it when you have one: it is the preferred training signal.
- **`attribution: "workflow"`** marks an end-to-end score shared by several requests. Such labels are stored but are `training_eligible: false`.
  - `label_source: "user_signal"` (for example thumbs) is also not training-eligible.
  - So is any label on an `evaluation_judge` request.
- **Batch** up to 500 labels with `POST /v1/feedback/batch`. Items are processed independently, and each gets its own status.
- **Delete** with `DELETE /v1/feedback/{request_id}`, optionally filtered by `metric`, `evaluator` and `metric_version`. Deleted labels are left out of every future dataset build.
- **`notes` must never contain personal data, prompt content or document content.**

To join your evaluation results to executions, list a run's requests with `GET /v1/requests?correlation_id=<run id>`. It returns the tenant's records, newest first, without content unless the key has retention enabled. Send one feedback item per request id.

## 9. Observability for clients

- **`GET /v1/requests/{mmr_id}`** returns one execution record: status, attempts, routing, usage, cost and error. Prompt and response content is included only if the key has content retention enabled. Retention is off by default.
- **`GET /v1/usage?from=&to=&group_by=model|provider|task_type|application|day`** returns usage and cost for your tenant. Estimated and provider-reported figures are in separate columns. Estimated cost is not billing data.
- **`GET /v1/providers`** shows provider health and circuit-breaker state. **`GET /v1/health`** needs no authentication.

## 10. Checklist

- [ ] Read `GET /v1/contract` at start-up. Send only the fields, task types, policies and capabilities it lists.
- [ ] Set `metadata.task_type` and `metadata.correlation_id` on every request.
- [ ] Send a new `X-Request-ID` per attempt, and propagate `traceparent`.
- [ ] Store the `mmr_` id, `trace_id`, `X-Request-ID` and `correlation_id` with your records.
- [ ] Retry only 5xx and network errors, with backoff, within a deadline. Honour `Retry-After` on 429. Never fall back on 4xx or 429.
- [ ] Make any direct-provider fallback enforce the same `data_policy`.
- [ ] Choose `data_policy` per request from the data's sensitivity. Never loosen it to get an answer.
- [ ] Send `min_quality` only for task types that `GET /v1/routing/calibration` reports as calibrated.
- [ ] Use a hashed per-workspace `cache.namespace`, and a `context_fingerprint` over your evidence versions for semantic mode. Invalidate when the evidence changes.
- [ ] For streams, set `include_usage: true` and treat a stream without `[DONE]` as incomplete.
- [ ] Send feedback with a stable `(metric, metric_version, evaluator)`. Use `accepted` when you have a verdict and `attribution: "workflow"` for end-to-end scores.
