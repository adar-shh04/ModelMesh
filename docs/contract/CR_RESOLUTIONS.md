# Resolution of Mezvuat change requests against Contract v1

- **Input:** `mezvuat-ai/docs/contract/CHANGE_REQUESTS.md`. This file was read, not modified. The CRs were raised against DRAFT 0.1.
- **Output:** contract **1.0-rc2** (`inference-api-v1.md` and `openapi-v1.yaml`).
- **Resolved by:** the ModelMesh agent (contract owner), 2026-11-08. The coordinator reconciles any objections.

## Disposition counts

| Disposition | Count | CRs |
|---|---|---|
| Accepted | 2 | CR-01, CR-08 |
| Accepted with modification | 7 | CR-02, CR-03, CR-06, CR-10, CR-11, CR-12, CR-13 |
| Already in rc1 | 5 | CR-04, CR-05, CR-07, CR-09, CR-14 |
| Rejected (whole CR) | 0 | n/a |

Four sub-items inside accepted CRs were rejected: CR-03.2, CR-03.4, CR-06.1 and CR-13.3. Details are below.

## Per-CR resolution

| CR | Disposition | Rationale | Contract section |
|---|---|---|---|
| CR-01 task types | **accepted** | Adds `evidence_assessment` and `evaluation_judge`. Judge requests are routed normally but **excluded from router quality-training data** (`training_eligible: false`) because their outputs are evaluator artefacts, not task outcomes. Pinning an explicit judge model is recommended so evaluations stay reproducible as the router changes. Proxy benchmark for `evidence_assessment`: answerable vs unanswerable items (ADR-0005 amendment) | md §2.2, §2.4, §5; yaml `TaskType`, `FeedbackResponse` |
| CR-02 structured output | **accepted with modification** | rc1 already validates *every* structured response, not only when the capability is required. It also reports `structured_output` and fails with 502 `upstream_error`/`schema_validation_failed`, so Mezvuat's error name maps to rc1's existing code. Changes: add the `json_schema_strict` capability (native_strict only), while `json_schema` keeps its rc1 meaning (native, no emulation). With no capability required, emulation plus validation is allowed. This is a superset of Mezvuat's two levels. Mode name stays `native_best_effort`, not `native`. Added `details.validation_errors` (path + keyword only) and the **portable schema subset** | md §2.5; yaml `RoutingOptions.required_capabilities`, `ResponseFormat`, `ErrorDetails` |
| CR-03 streaming | **accepted with modification** | (1) Headers before the first event: accepted. rc2 defines a **commit point** at the first upstream content, so failures before it become real HTTP errors (a change from rc1's early keep-alives). (2) Error envelope plus a final `modelmesh` object: accepted. **`[DONE]` after an error: rejected.** `[DONE]` means normal completion, and naive clients that read until `[DONE]` would treat truncated output as complete. (3) Disconnect cancels upstream, usage still recorded: accepted (status `cancelled`). (4) Normalized `finish_reason`: accepted as `stop`/`length`/`tool_calls`/`content_filter`. **`error` as a finish_reason: rejected.** It is not an OpenAI value and would break typed SDKs, and failures are signalled by the error event. (5) Streaming with `json_schema`: allowed through buffered mode. Callers who want 422 instead add `streaming` to `required_capabilities` | md §2.5, Streaming; yaml info.description, `ErrorResponse.modelmesh`, `Attempt.outcome`, `RequestRecord.status` |
| CR-04 fingerprint + namespace | **already in rc1** (wording clarified) | rc1 already had the `^sha256:[0-9a-f]{64}$` pattern, `cache.namespace` in exact and semantic keys, and 400 `missing_context_fingerprint`. rc2 makes the fingerprint explicitly **opaque**, demotes the Mezvuat formula to a non-normative example, and states that **a namespace scopes only within a tenant**: the tenant prefix comes from the API key, an absent namespace is its own partition, and identical namespace strings in two tenants share nothing. Fingerprint and namespace are independent and can be used together | md §2.3; yaml `CacheOptions` |
| CR-05 non-deterministic default | **already in rc1** (clarified) | rc1 already downgraded the default to bypass. rc2 states it as **no read and no write**, with `cache.status: bypass` | md §2.3 |
| CR-06 errors / fallback | **accepted with modification** | (1) **429 as a fallback trigger: rejected.** A 429 means the caller exceeded its own ModelMesh limits, which ModelMesh as a whole is not unavailable. Falling back would bypass limits set by the operator. Clients should back off for `Retry-After`. Upstream exhaustion is already 503 `upstream_capacity`, which is in the fallback set. `Retry-After` is now **required** on every 429. The normative fallback table is §3.1. (2) Typed `details`: accepted. `violated_constraints` becomes a list of `{constraint, excluded_candidates, best_available}`, plus `attempts`, `scope`/`period`/`resets_at`. (3) `client_request_id` in errors: accepted. (4) `mmr_` id always assigned on receipt: accepted, and `request_id` is now non-nullable. (5) `X-Request-ID` documented as not an idempotency key. No `Idempotency-Key` in v1, because inference responses are not safe to dedupe; it can be added later as an additive change. (6) No fallback on 4xx: confirmed, with 404/409/413/501 added to the list | md §1, §3, §3.1; yaml `ErrorObject`, `ErrorDetails`, `RateLimited`, headers |
| CR-07 timeout | **already in rc1** | `routing.timeout_ms` with a default of min(2 × `max_latency_ms`, server max), or 60 s when no budget is given. Returns 504 `upstream_timeout` | md §2 |
| CR-08 correlation | **accepted** | `correlation_id` pattern `[A-Za-z0-9_:.-]`, at most 128 characters. `end_user_id` at most 128. Tags follow OpenAI metadata limits (16 keys including reserved ones, values at most 512, which is looser than the requested 256, so compatible). Metadata is stored and returned, and only `task_type` affects routing or the cache. `traceparent` makes ModelMesh spans children of the caller's span (same trace id), and a sampled flag of 0 means no ModelMesh spans | md §1, §2; yaml `RequestMetadata`, list params |
| CR-09 list by correlation id | **already in rc1** | `GET /v1/requests?correlation_id=&limit=&cursor=`. Tenant-scoped, and no content unless retention is enabled | md §4 |
| CR-10 data policy per tier | **accepted with modification** | The rc1 field name `data_use` (`no_training`/`may_train`/`unknown`) is kept instead of `data_training: bool`, to avoid two fields with the same meaning. It is now defined **per configured provider account and tier**, and `unknown` is ineligible under `no_training_providers`, as requested | md §2, §4; yaml `Model.modelmesh.data_use` |
| CR-11 `min_quality` calibration | **accepted with modification** | Published as a dedicated `GET /v1/routing/calibration` (per task_type: calibrated flag, criterion and version, benchmark manifest hash, sample count, ECE, Brier, judge-human κ, per-model coverage) rather than inside `/v1/models`, because calibration is per (task, model) and per router version. md §2.4 now has the per-task acceptance-criteria table. **Still missing until M6:** actual calibration numbers. Every task reports `calibrated: false` until then, so Mezvuat's plan not to send `min_quality` is correct | md §2.4, §4; yaml `/v1/routing/calibration`, `RoutingCalibration` |
| CR-12 negotiation | **accepted with modification** | Added an unauthenticated `GET /v1/contract` (version, release, endpoints, extension field paths, task types, capabilities, policies, metrics) instead of overloading `/v1/health`. The strict rejection stays, as **400** (rc1 already changed it from 422) | md §0, §4; yaml `/v1/contract`, `ContractInfo` |
| CR-13 feedback | **accepted with modification** | (1) All six optional fields are accepted. `metric_version` joins the **idempotency key**, so re-scoring with a new metric version doesn't return 409. Only `attribution: direct` labels are training-eligible. (2) Batch: accepted, but as a separate `POST /v1/feedback/batch` (1–500 items, per-item results) rather than overloading the single endpoint, which keeps both schemas strict. (3) 201/200/404: accepted. **422 for an invalid score: rejected**, because validation errors are 400 throughout the contract (rc1 C1). (4) Vocabulary: Mezvuat's names adopted (they are the producer), plus `task_success`. (5) `DELETE /v1/feedback/{request_id}`: accepted. Deleted labels are excluded from future dataset builds, and trained artifacts are not retrained automatically. (6) `notes` must not contain content: confirmed | md §5; yaml `/v1/feedback`, `/v1/feedback/batch`, `/v1/feedback/{request_id}`, `Feedback*` |
| CR-14 open-question positions | **already in rc1** | Consistent with rc1's resolutions of all four questions. The Q3 nuance (`json_schema_strict`) is handled under CR-02 | md §6 |

## Client-code impact summary (Mezvuat)

1. Use **flat** metadata keys, not `metadata.tags.*`. This came from rc1 C2. Mezvuat's interim `metadata.tags.task_type_intended` becomes unnecessary because CR-01 is accepted.
2. Set `stream_options.include_usage: true` (rc1 C3).
3. Emit `context_fingerprint` as `sha256:<64 lowercase hex>`, and `cache.namespace` matching `[A-Za-z0-9._:-]{1,128}`.
4. `correlation_id` must match `[A-Za-z0-9_:.-]{1,128}`.
5. Parse `details.violated_constraints` as a **list** (rc2 R6).
6. On **streaming** requests, failures before the first content now arrive as **HTTP 5xx JSON**, not as in-stream events. Treat them like non-streaming errors.
7. On **429**, back off for `Retry-After`. Fall back to a direct provider only on network errors or 5xx (§3.1).
8. Call `GET /v1/contract` at start-up and send only the extensions it lists.
9. Send `min_quality` only once `GET /v1/routing/calibration` reports `calibrated: true` for the task type.
10. For judge calls, pin an explicit model with `fallback: "none"`.

## Items for the product owner

- **`data_policy` vs the current provider pool.** Mezvuat will send `no_training_providers` by default for document content. Under the current attestations, that leaves **only local Ollama models eligible**:
  - Gemini free tier is `may_train`.
  - Groq and OpenRouter upstreams are `unknown`, because no official no-training statement was verified.

  Before integration, decide one of the following: attest Groq's (and selected OpenRouter endpoints') data use after reviewing their terms; use a paid Gemini tier; or accept local-only for document-bearing calls.
- **Rejected 429 fallback (CR-06.1).** If the product owner prefers that Mezvuat be allowed to fall back on 429, that is a client-side policy. No contract change would be needed. The contract recommends against it.
