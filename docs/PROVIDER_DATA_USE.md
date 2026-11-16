# Provider data use (D4 review)

Checked **2026-11-17** by the implementation agent, using web fetches of the published pages listed below. This is a summary for the product owner, **not legal advice and not a change**. D4 says providers stay `may_train` or `unknown` until the owner has reviewed them, so `config/models.yaml` is **unchanged**. The proposed classifications at the end take effect only if the owner approves them.

The runtime `data_use` values are `no_training`, `may_train` and `unknown` (`modelmesh.providers.catalog.DataUse`). A request with `data_policy: no_training_providers` can only use `no_training` models, and `local_only` can only use Ollama.

## Summary

| Provider and tier (catalog models) | Training on prompts and outputs | Retention and logging | Current `data_use` | Proposed |
|---|---|---|---|---|
| Gemini API, **unpaid** tier (`gemini/*`) | **Yes.** Used to improve Google products and ML technologies; human reviewers may read them | Not stated for unpaid use. Grounding with Search keeps data 30 days. Terms say not to submit sensitive, confidential or personal data | `may_train` | keep `may_train` (confirmed) |
| Groq, free tier (`groq/openai/*`) | **No**, unless the customer permits it (Services Agreement §4.2). No free-tier exception found | No retention by default. Reliability and abuse-monitoring logs up to 30 days. Zero Data Retention is available to all customers | `unknown` | `no_training`, if the owner accepts §4.2 for a free account. Optionally enable ZDR |
| OpenRouter (the router itself) | Only if "use of inputs/outputs" is opted in (off by default) | Prompts not stored by default. Request metadata (tokens, latency) always stored. Prompts may be sampled for anonymous categorisation | n/a: the upstream provider decides | n/a |
| OpenRouter `liquid/lfm-3-1.7b:free`, upstream **Liquid** (fp8, one endpoint) | **Likely yes.** The model description begins "Prompts and outputs may be retained and used..." (truncated at the source). Search snippets of the model page say retained and used to train Liquid's models | Not verified | `unknown` | `may_train` |
| OpenRouter `nvidia/nemotron-3.5-super-vl:free`, upstream **NVIDIA** (one endpoint) | **Likely yes**, under the NVIDIA API Trial Terms. Other NVIDIA `:free` pages say prompts and outputs are logged to improve NVIDIA's models; this model's page was not readable | Trial terms reportedly session-only, except security and abuse logging and product improvement (not verified) | `unknown` | `may_train` |
| Ollama, local (`ollama/*`) | No. Prompts never leave the machine | Ollama's policy: local prompts and responses are not collected or transmitted; the app sends limited metadata (version, request counts) | `no_training` | keep |

## Details and sources

### Gemini API, unpaid tier

Source: [Gemini API Additional Terms of Service](https://aistudio.google.com/docs/terms) (redirected from `ai.google.dev/gemini-api/terms`), "Effective March 23, 2026". Checked 2026-11-17.

- **Unpaid Services:** Google uses submitted content and responses "to provide, improve, and develop Google products and services".
  - Human reviewers may read, annotate and process input and output, after it is disconnected from the account, API key and project.
  - The terms say: "Do not submit sensitive, confidential, or personal information to the Unpaid Services."
- **Paid Services:** prompts and responses are not used to improve products. They are logged for a limited period, only for abuse detection and legal requirements.
- **Region:** in the EEA, Switzerland and the UK, the paid data-use terms apply to all use, unpaid quota included. ModelMesh's account is not in that region as far as this review knows, so the owner should confirm.
- **For ModelMesh:** this confirms `may_train`, and the ADR-0005 rule that only public benchmark data goes to Gemini's free tier. P1's judge step sends public prompts and model outputs only.

### Groq, free tier

Sources, checked 2026-11-17:
- [GroqCloud Services Agreement](https://console.groq.com/docs/legal/services-agreement), "Last Modified: June 22, 2026"
- [Your Data in GroqCloud](https://console.groq.com/docs/your-data), undated
- [Groq Privacy Policy](https://groq.com/privacy-policy), effective November 12, 2025

What they say:
- **§4.2:** Groq "is not permitted to use Inputs or Outputs for training or fine-tuning" models, unless the customer explicitly permits or instructs it. Groq does not access, use, store or retain Inputs or Outputs except as necessary to provide the service, comply with law, keep it reliable and check compliance with the acceptable-use policy.
- **§8.2:** customer training data is used only to provide the services.
- **No free-tier exception found.** §5.1 allows services "designated as fee-free" without changing the data terms.
- **Beta Services (§3.6)** are outside the DPA and BAA. The agreement does not say that changes training use. Whether the two gpt-oss models are designated beta was not checked.
- **"Your Data":** "By default, Groq does not retain customer data for inference requests." Reliability and abuse logs, which can include inputs and outputs, are "retained for up to 30 days". Every customer can turn on Zero Data Retention in Data Controls. Data is stored in Google Cloud in the US.
- **Privacy Policy:** it excludes customer content from its scope and defers to the Services Agreement and DPA.

### OpenRouter

Sources, checked 2026-11-17, none dated:
- [Data collection](https://openrouter.ai/docs/guides/privacy/data-collection)
- [Provider logging](https://openrouter.ai/docs/features/privacy-and-logging)
- [Provider selection](https://openrouter.ai/docs/guides/routing/provider-selection)

What they say:
- **OpenRouter's own storage.** It does not store prompts or completions by default. The exceptions are opt-in logging, opt-in "use of inputs/outputs" (which earns a 1% discount), CSAM review, and Safety Retention endpoints.
  - Metadata such as token counts and latency is always stored.
  - A small sample of prompts may be categorised anonymously for rankings.
- **Upstream providers.** Each endpoint carries the provider's own policy. There are separate training settings for free and paid models.
  - The account setting, or `provider.data_collection: "deny"` per request, excludes providers that "store user data non-transiently and may train on it".
  - `provider.zdr: true` restricts routing to zero-data-retention endpoints.
- **Endpoints API.** It lists **one** endpoint for each `:free` model: provider `Nvidia` (quantization unknown) and provider `Liquid` (fp8), with no structured data-policy fields in the public response.
  - [nemotron endpoints](https://openrouter.ai/api/v1/models/nvidia/nemotron-3.5-super-vl:free/endpoints)
  - [lfm endpoints](https://openrouter.ai/api/v1/models/liquid/lfm-3-1.7b:free/endpoints)
- **Liquid.** The lfm model description in that API response, and in the cached `/v1/models` list of 2026-11-09, is cut off at the source: "Prompts and outputs may be retained and used...". Search snippets of the model page say the data may be used to train Liquid's models. The full sentence could not be read.
- **NVIDIA.** The nemotron-3.5-super-vl description has no data-use text in the API. Web-search snippets of other NVIDIA `:free` pages on OpenRouter say the free endpoint logs prompts and outputs to improve NVIDIA's models, and that use is governed by the [NVIDIA API Trial Terms of Service](https://assets.ngc.nvidia.com/products/api-catalog/legal/NVIDIA%20API%20Trial%20Terms%20of%20Service.pdf). That PDF was fetched (created 2025-10-07, per its metadata), but its text could not be extracted here.
  - Secondary sources quote its §3.3(iv) as allowing NVIDIA to use prompts and outputs to improve its products.
  - Secondary sources quote its §2.3 as session-only storage, with exceptions.
  - **Not verified.**

### Ollama, local

Source: [Ollama Privacy Policy](https://ollama.com/privacy), "Last updated: March 2026". Checked 2026-11-17.

- Local runs: Ollama does not collect, store or transmit prompts or responses ("We don't see your prompts or data when you run locally").
- The app sends limited metadata: app version, request counts and download metadata.
- Ollama's cloud models are a separate service. ModelMesh does not use them.
- ModelMesh calls the local Ollama server only, so `no_training` holds. That depends on the deployment never pointing `ollama/*` at a remote or cloud host.

## Proposed changes (need the owner's approval; not applied)

1. **Groq: `unknown` → `no_training`.** Basis: Services Agreement §4.2, with no free-tier exception found. Before approving, the owner should check:
   - whether the free account is bound by that agreement (signed up through the console), and whether the gpt-oss models are designated **Beta Services** (§3.6);
   - whether to turn on **Zero Data Retention** in Groq's Data Controls. That removes the 30-day reliability and abuse logs, but disables features that need retention. The benchmark does not use batch or fine-tuning.

   Effect: Mezvuat's `no_training_providers` requests would become eligible for both Groq models, not only Ollama. The Groq quota is shared, so this also moves Mezvuat load onto Groq.
2. **OpenRouter `liquid/lfm-3-1.7b:free` and `nvidia/nemotron-3.5-super-vl:free`: `unknown` → `may_train`.** The evidence points to training use, and `may_train` is the stricter class, so this changes no request that is allowed today. Under `no_training_providers`, `unknown` and `may_train` are both excluded already.
3. **Gemini free tier and Ollama:** no change. The terms confirm the current values.
4. **Optional, OpenRouter account setting:** turn **off** "allow providers that may train" for **free** models. That would block both `:free` models (one endpoint each). It is a trade-off, not a fix: it would end the P1 OpenRouter lane.

## What the owner must check (not verifiable here)

- **The NVIDIA API Trial Terms PDF**, §§2.2–2.4 and 3.3. The text could not be extracted, so the training and retention reading above is from secondary sources.
- **The full Liquid data-use sentence** on the lfm model page in a browser, and Liquid's own terms. OpenRouter's public pages and API cut it off.
- **OpenRouter's account privacy settings** for this account: whether free-model training is allowed, and whether logging or "use of inputs/outputs" is on. They must be off for prompts not to be stored by OpenRouter.
- **Groq:** the beta designation of the models, and the ZDR setting, as in proposed change 1.
- **Gemini:** the account's billing region, because the EEA, Switzerland and UK rule changes the terms.
- These pages change. Every row above is as of 2026-11-17. Re-check before relying on them, and record the date in `config/models.yaml` when the classification changes.
