# ADR-0003: Provider adapter interface and initial providers

- Status: Accepted (architecture phase). Provider facts must be re-verified when M1/M2 starts, because catalogs change monthly.
- Date: 2026-11-08
- Spec refs: §2.3 A, B; §2.4 steps 6–8; §5.3

## Context

ModelMesh must hide authentication, request formats, rate limits and usage reporting from clients. The initial targets are Gemini (Google AI Studio), Groq, OpenRouter and Ollama. All four now offer an OpenAI-compatible Chat Completions surface, and each has quirks that affect correctness. Provider documentation was reviewed on **2026-11-08**. The sources were read through a summarising fetcher, and items marked UNVERIFIED could not be confirmed on an official page.

### Provider facts (as of 2026-11-08)

| | Gemini (AI Studio) | Groq | OpenRouter | Ollama (local) |
|---|---|---|---|---|
| Auth | Native: `x-goog-api-key`. Compat layer: `Authorization: Bearer`. New keys are service-account-bound "auth keys" since 2026-05-28 | `Authorization: Bearer` | `Authorization: Bearer`. Optional `HTTP-Referer`, `X-OpenRouter-Title` | none (any dummy bearer) |
| OpenAI-compatible base | `https://generativelanguage.googleapis.com/v1beta/openai/`, labelled **beta** | `https://api.groq.com/openai/v1`. Returns 400 for `logprobs`, `logit_bias`, `top_logprobs`, `messages[].name`, `n>1` | `https://openrouter.ai/api/v1` (native shape). Unsupported params may be **silently ignored** unless `provider.require_parameters: true` | `http://localhost:11434/v1`. No `tool_choice`, `logprobs`, `logit_bias`, `n` |
| Usage in stream | compat `stream_options.include_usage`: UNVERIFIED | `include_usage`. Usage may arrive in `x_groq.usage` (UNVERIFIED) | **always** sent as a final chunk before `[DONE]` (repeats `finish_reason`) | `include_usage` supported |
| Mid-stream error | SSE error event (documented for the Interactions API; compat behaviour UNVERIFIED) | UNVERIFIED | HTTP 200, then a chunk with top-level `error` and `finish_reason:"error"`. Also `: OPENROUTER PROCESSING` keep-alive comments | UNVERIFIED |
| Cost reported | no | no (timing fields `queue_time`, `prompt_time`, …) | **yes**: `usage.cost`, plus `GET /api/v1/generation?id=` | n/a (free, local) |
| json_schema | Native `responseJsonSchema`. Silently ignores `minLength`/`maxLength`/`minProperties`/`maxProperties`/`uniqueItems`/`contains`. Compat path documented only through SDK `parse` | strict (constrained decoding) on `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `qwen/qwen3.8-27b`, `meta-models/muse-spark-open`. `json_object` on all models. **Not combinable with streaming or tools** | per upstream endpoint; filter `supported_parameters=structured_outputs` | `format` = JSON Schema (native). `response_format` on `/v1` (partially verified) |
| Tools | yes. Tools + structured output only on Gemini 3 (preview) | yes (not with structured output) | per endpoint (`supported_parameters=tools`) | yes, model-dependent. No `tool_choice` |
| Free-tier limits | Shown only in the AI Studio console: **UNVERIFIED**. Third-party sources say about 10 RPM and 500–1,500 RPD per model. Per project, resets midnight Pacific | Official page shows the Developer plan: gpt-oss-20b/120b at 30 RPM, 1K RPD, 8K TPM, **200K TPD**. Free-plan numbers UNVERIFIED | `:free` models: 20 RPM. **50 req/day** if under 10 credits purchased, **1,000/day** with 10 or more | hardware-bound |
| Rate-limit signals | 429 `rate_limit_exceeded` (per-minute) vs 429 `quota_exceeded` (daily). Headers undocumented | `x-ratelimit-limit/remaining-requests` (**RPD**), `…-tokens` (**TPM**), `…-reset-*`, `retry-after` on 429 | `Retry-After` on 429. `GET /api/v1/key` shows `free_model_daily_requests` | none |
| Data use | **Free tier: content used to improve Google products, human review possible.** Paid: not used for training. EEA/UK/CH users: paid only | No retention by default. Up to 30 days abuse logs. ZDR toggle. "No training": official statement UNVERIFIED | Varies by upstream. `provider.data_collection:"deny"`, `provider.zdr:true`, account-level training toggles (separate for free and paid) | local, none |
| Models endpoint | `inputTokenLimit`, `outputTokenLimit`, `thinking`. No pricing | `context_window`, `active`. No pricing | `context_length`, **`pricing`**, `supported_parameters`, `expiration_date` | names (details via `/api/show`) |

**Catalog churn observed during this review:**
- Groq shut down `llama-3.1-8b-instant` and `llama-3.3-70b-versatile` for free and developer tiers on **2026-08-16** (replacements are `openai/gpt-oss-20b` and `openai/gpt-oss-120b`).
- Gemini 2.5 Flash and Pro are deprecated, with automatic re-routing on 2026-12-01.
- `ai.google.dev/gemini-api/docs/*` now redirects to `aistudio.google.com/docs/*`.

Sources (accessed 2026-11-08):
- Gemini: https://aistudio.google.com/docs/openai, …/docs/api-key, …/docs/structured-output, …/docs/models, …/docs/rate-limits, …/docs/api-errors, …/docs/terms, …/docs/pricing, https://aistudio.google.com/api-reference/models
- Groq: https://console.groq.com/docs/openai, …/docs/structured-outputs, …/docs/models, …/docs/deprecations, …/docs/rate-limits, …/docs/errors, …/docs/your-data
- OpenRouter: https://openrouter.ai/docs/api-reference/authentication, …/api-reference/streaming, …/use-cases/usage-accounting, …/api-reference/get-a-generation, …/features/structured-outputs, …/guides/overview/models, …/api-reference/limits, …/api-reference/errors, …/features/provider-routing, …/guides/privacy/provider-logging
- Ollama: https://docs.ollama.com/api/openai-compatibility, …/api/chat, …/capabilities/structured-outputs, …/capabilities/tool-calling, …/faq, …/windows

## Options considered

1. **Official provider SDKs** (`google-genai`, `groq`, `openai` for OpenRouter and Ollama). This gives four dependency trees, each with its own retry logic (which we must disable to own retries) and its own error types. Provider extras (`x_groq`, OpenRouter's HTTP-200 error chunks, keep-alive comments) end up behind SDK abstractions.
2. **One generic OpenAI-compatible adapter with per-provider "quirk" subclasses, over raw httpx** (chosen).
3. A third-party abstraction such as LiteLLM. It solves "many providers", but adapters are the core of this product (spec §2.3 B). Delegating them would hide error classification, usage extraction and streaming behaviour, which we must own and test. It is also a large dependency.

## Decision

### Adapter protocol (sketch)

The real signatures will be written in M1.

```python
class ProviderAdapter(Protocol):
    provider_id: str                                   # "gemini" | "groq" | "openrouter" | "ollama"
    async def chat(self, req: NormalizedChatRequest, *, deadline: Deadline) -> NormalizedChatResponse
    def stream(self, req: NormalizedChatRequest, *, deadline: Deadline) -> AsyncIterator[NormalizedChunk]
    async def list_models(self) -> list[ProviderModelInfo]          # used to validate the catalog, not trusted blindly
    async def health(self) -> ProviderHealth
    def classify_error(self, exc_or_response) -> NormalizedError     # ADR-0008 taxonomy
    def supports(self, model: CatalogModel, requirement: CapabilityRequirement) -> bool
```

- `NormalizedChatRequest` is ModelMesh's internal, provider-neutral form. The adapter translates it into the provider payload. Parameter handling is driven by the catalog:
  - **Hard parameters** (`response_format`, `tools`, `tool_choice`, `stop`, `n`, `logprobs`): the request is either honoured or the model is ineligible. These are never silently dropped.
  - **Soft parameters** (`presence_penalty`, `frequency_penalty`, `seed`, `top_p`, `reasoning_effort`, `user`): forwarded where supported. When dropped, this is reported in `modelmesh.routing.warnings`.
- `NormalizedChunk` and `NormalizedChatResponse` carry content, tool calls, finish reason and usage, with `usage_source` set to `provider_reported` or `estimated`, plus `provider_reported_cost_usd` (OpenRouter) and raw provider timing fields.
- **Base class `OpenAICompatAdapter`** implements the HTTP call, SSE parsing (skips `:` comment lines, handles `data: [DONE]`, detects `{"error":…}` chunks even after HTTP 200), timeouts and error classification. Subclasses override:
  - **groq**: reject or strip unsupported params. Read usage from `usage` or `x_groq.usage`. Parse `x-ratelimit-*` headers (requests = RPD, tokens = TPM) into the quota tracker (ADR-0008). Map 498 (flex capacity) to retryable.
  - **openrouter**: set `provider.require_parameters: true` whenever a hard parameter is present, so unsupported params are not silently ignored. Map `routing.data_policy` to `provider.data_collection:"deny"` / `zdr`. Treat an empty 200 response as an upstream failure. Use `usage.cost` as `provider_reported_cost_usd`. Optional attribution headers come from config.
  - **gemini**: compat base URL. Distinguish 429 `rate_limit_exceeded` from `quota_exceeded` (mark the model unavailable until the Pacific-midnight reset). If contract tests show the compat layer fails a needed capability (stream usage or json_schema fidelity), a native `generateContent` adapter replaces this subclass for that capability. That decision will be recorded in an ADR amendment with test evidence.
  - **ollama**: no auth. Drop `tool_choice:"auto"`, and make the model ineligible when `tool_choice` is `required` or names a function. Pass `num_ctx` from catalog config. Also call `/api/ps` for health.
- **Model catalog** (`config/models.yaml`, versioned, operator-maintained) is the authority on context window, capabilities (per mode, e.g. `json_schema` vs `json_schema+stream`), family, `data_use` (`no_training` | `may_train` | `unknown`, operator-attested), free-tier flag and status. Provider `/models` endpoints are used to *detect drift*: an unknown or deactivated id marks the model `unavailable` and raises a log/metric alarm. They are never used to auto-enable models.
- **Model ids**: `<provider>/<provider-model-id>`, split on the first `/`. For example `groq/openai/gpt-oss-20b` and `openrouter/meta-llama/…:free`.
- **Credentials** come from env vars only (`GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`). They are held in `SecretStr` and never logged, traced or returned. An adapter without a key is registered as `disabled`, and its models are absent from `/v1/models`.

### Initial candidate pool

To be confirmed when M2 starts.

| Provider | Models (tentative) | Role |
|---|---|---|
| gemini | the current Flash-Lite (`gemini-3.1-flash-lite` or `gemini-3.5-flash-lite`) and current Flash (`gemini-3.6-flash`) | capable, cheap. Free tier trains on data |
| groq | `openai/gpt-oss-20b`, `openai/gpt-oss-120b` | fast. Native strict json_schema (non-streaming) |
| openrouter | 1–2 `:free` models from a different family (chosen at M2 by `supported_parameters`) | family diversity, provider-reported cost |
| ollama | `qwen3:4b`, `llama3.2:3b` | local, `data_use: no_training`, satisfies `local_only` |

## Consequences

- One HTTP/SSE implementation is tested once. Provider quirks are isolated in small subclasses (spec §2.3 B: "provider-specific behaviour stays inside adapters").
- We carry the burden of tracking API changes. This is mitigated by catalog drift detection and live smoke tests.
- Gemini's free tier is incompatible with `data_policy: no_training_providers`, and so is any provider whose `data_use` is `unknown`. Some routing requests will therefore have small eligible sets. That is correct behaviour, not a bug.

## Verification plan

- **Unit/contract tests (no keys):** respx-recorded, sanitised fixtures per provider. Covered cases:
  - success
  - streaming with and without usage
  - OpenRouter 200 + error chunk
  - keep-alive comments
  - 400/401/403/404/408/413/429 (with and without Retry-After)/498/500/502/503
  - malformed JSON and an empty body
  - Each case is asserted against the normalized output and the error taxonomy.
- **Live smoke tests** (`pytest -m live`, skipped without keys): one non-stream and one stream request per configured model, plus a json_schema request where the catalog claims support. Recorded token usage must be present or explicitly `estimated`. These tests also re-verify every UNVERIFIED cell above, and results are written back into this ADR.
- **Drift check:** `modelmesh catalog verify` compares `config/models.yaml` with provider `/models` responses and exits non-zero on mismatch.
