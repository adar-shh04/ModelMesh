# ADR-0007: Streaming and structured-output compatibility across providers

- Status: Accepted (architecture phase)
- Date: 2026-11-08
- Spec refs: §2.3 A/B, §3.3 ("citation verification: reliable structured output"), §8 item 7
- Resolves contract open question 3 (structured output on providers without native `json_schema`).

## Context

Providers differ in several ways (ADR-0003, 2026-11-08):

- **Groq** supports strict `json_schema` only on some models, and **not with streaming or tools**.
- **Gemini** ignores several JSON-Schema keywords (`minLength`, `maxLength`, `uniqueItems`, …) and allows tools combined with structured output only on Gemini 3 preview.
- **OpenRouter** support depends on the upstream endpoint, and it may silently ignore parameters unless `require_parameters` is set.
- **Ollama** constrains output to a schema through `format`, has no `tool_choice`, and model quality varies.

Stream usage reporting and mid-stream error formats differ too. Clients (Mezvuat first) need structured output they can rely on, and a stock OpenAI SDK must keep working.

## Options considered

For providers without native schema support:
1. **Mark them ineligible.** This is safe, but it shrinks the pool a lot, for example to only some Groq models when streaming.
2. **Emulate everywhere:** prompt plus validate plus repair. This gives a large pool, but quality and latency vary, and a caller who needs a guarantee can't ask for one.
3. **Tiered (chosen):** native modes are preferred and can be *required*. Emulation is allowed only when the caller does not require native support. **Every** structured response is validated by ModelMesh, whatever the mode.

For streaming combined with a required schema on a model that can't stream structured output:
- (a) mark the model ineligible
- (b) **buffered streaming (chosen as a fallback mode):** call upstream without streaming, validate, then emit the result as SSE chunks. This is labelled honestly.

## Decision

### Capability vocabulary

Used in the catalog and in `routing.required_capabilities`:
- `json_object`
- `json_schema` (native schema-constrained or strict decoding)
- `tools`
- `streaming`
- `long_context:<n_tokens>`

Combinations are checked by the router, not just single flags. The catalog lists support per mode, for example `json_schema: {stream: false, with_tools: false}` for Groq.

### Structured-output modes

Reported in `modelmesh.structured_output`.

| Mode | When | Guarantee |
|---|---|---|
| `native_strict` | provider enforces the schema (constrained decoding) | schema-valid by construction. Still re-validated |
| `native_best_effort` | provider accepts a schema without strict enforcement (e.g. Gemini's ignored keywords, OpenRouter non-strict) | validated by ModelMesh |
| `emulated` | schema injected into the system prompt, `json_object` mode if available | validated by ModelMesh. Allowed only if `json_schema` is **not** in `required_capabilities` |

Rules:
- `required_capabilities: ["json_schema"]` means **native modes only**. Emulation-only models are filtered out. If none remain, the response is `422 no_eligible_model`.
- **Validation:** every `json_schema` response is parsed and validated with `jsonschema` (Draft 2020-12) against the caller's schema. Every `json_object` response must parse to a JSON object.
- **On validation failure:** one **repair attempt** on the same model (the validation error is appended as a user turn). It is recorded as an attempt with outcome `schema_validation_failed`. After that, fall back to the next eligible model (ADR-0008). If all fail, the response is `502 upstream_error` with code `schema_validation_failed`.
- **A successful response is always schema-valid.** ModelMesh never returns a response that is known to be invalid.

### Tools

- Tool calls are normalised to OpenAI `tool_calls` format, with parallel tool calls when the provider supports them.
- `tool_choice` set to `required` or naming a function makes Ollama ineligible. `tool_choice: "auto"` is dropped for Ollama, which is equivalent since it is Ollama's default.

### Streaming

Wire format: `text/event-stream`, OpenAI `chat.completion.chunk` objects, `data: [DONE]` terminator.

- **Usage chunk.** **Changed from the draft:** `stream_options.include_usage` defaults to **false**, which is OpenAI's semantics.
  - Reason: when it is true, the final chunk has `choices: []`. Naive OpenAI-SDK client code (`chunk.choices[0]`) crashes on it, which breaks the "stock SDK must work" principle.
  - When `include_usage` is true, the last chunk before `[DONE]` carries `usage` plus the full `modelmesh` object.
  - When it is false, the chunk carrying `finish_reason` also carries the `modelmesh` object. SDKs ignore unknown fields. Usage is then also available through `GET /v1/requests/{id}`.
  - Mezvuat's client should set `include_usage: true`.
- **Upstream usage.** ModelMesh always asks upstream for stream usage where supported: Groq `include_usage` (reading `usage` or `x_groq.usage`), OpenRouter always includes it, Ollama `include_usage`, Gemini to be verified. When upstream gives none, tokens are estimated and `usage.source = "estimated"`.
- **Upstream comments** (`: OPENROUTER PROCESSING`) are consumed, never forwarded.
- **Keep-alive.** ModelMesh emits SSE comment lines (`: keep-alive`) every 15 s while waiting, including before the first token during retries and fallback. SSE parsers ignore comments.
- **Fallback window.** Fallback is possible **only before the first content chunk is sent**, which is unchanged from the draft. Upstream output is held back until the first content token arrives, so a failed first attempt is invisible to the client.
- **Mid-stream failure.** After the first chunk, an upstream failure produces one event, `data: {"error": {…ErrorObject…}}`, and the stream **closes without `[DONE]`**. A missing `[DONE]` therefore means the response is incomplete. The openai-python SDK raises `APIError` on such an event. The request record shows `status = "stream_interrupted"`, and partial outputs are never cached.
- **Buffered streaming.** Used for `stream: true` combined with structured output where the chosen model can't stream structured output, or where validation must happen before delivery. ModelMesh makes a non-streamed upstream call, validates, then emits the content as chunks. `modelmesh.streaming.mode = "buffered"` (otherwise `"native"`), and time-to-first-token reflects the full generation. The router's latency predictor accounts for this mode.

### Soft parameters

When a soft parameter (ADR-0003) can't be honoured by the chosen model, it is listed in `modelmesh.routing.warnings` as `param_dropped:<name>`. Hard parameters are never dropped.

## Consequences

- Clients get one guarantee: if ModelMesh returns 200 for a structured request, the content validates against the schema. Native enforcement is opt-in and strict.
- Repair attempts and buffered streaming add latency, and both are visible in the metadata.
- `jsonschema` becomes a runtime dependency (ADR-0002).

## Verification plan

- Unit tests per mode, using respx fixtures:
  - native-strict success
  - best-effort output that violates `maxLength` (caught, repaired)
  - emulated invalid then valid
  - all invalid, ending in 502 `schema_validation_failed`
  - required `json_schema` with no native candidate, ending in 422
- Streaming tests:
  - chunk sequence with and without `include_usage`
  - keep-alive comments
  - fallback before the first token, which the client never sees
  - mid-stream error event without `[DONE]`
  - buffered mode
  - an OpenRouter 200 + error chunk being normalised
- A compatibility test using the **official `openai` Python SDK** as the client (dev dependency, tests only): non-stream, stream, `include_usage` true/false, `response_format` json_schema, tools.
- Live tests (key-gated) re-verify the provider support matrix and update the catalog.

## Amendment 2026-11-08 (contract 1.0-rc2, CR-02 and CR-03)

- **Stream commit point (supersedes "keep-alive ... including before the first token").**
  - HTTP status and headers are sent when the first upstream content is ready.
  - Failures before that point, including when every fallback attempt fails, are returned as normal JSON errors with their real status (502/503/504). That matches non-streaming behaviour and the client fallback rules.
  - Keep-alive comments are sent only after the commit point. `timeout_ms` bounds the wait.
- **In-stream error event.** The event is `{"error": …, "modelmesh": …}` (attempts and partial usage), with code `stream_interrupted`, and still **no `[DONE]`**.
- **Client disconnect.** It cancels upstream. Usage is still recorded (estimated, `possibly_billed`), and the status is `cancelled`.
- **`finish_reason`.** Normalized to `stop`/`length`/`tool_calls`/`content_filter`. There is no `error` value.
- **New capability `json_schema_strict`.** It requires `native_strict` only. `json_schema` keeps its meaning (native modes).
- **`details.validation_errors`.** On `schema_validation_failed`, this carries up to 10 `{path, keyword}` entries and no values.
- **Portable schema subset.** Documented in contract §2.5. M2 live tests verify it against each native-capable model, and the subset is adjusted from that evidence.
- **Opting out of buffered mode.** `streaming` in `required_capabilities` together with `stream` + `json_schema` excludes buffered mode, so the request gets 422 if no model can stream structured output natively.
