# ADR-0005: Benchmark dataset, quality labels and free-tier budget

- Status: Accepted (architecture phase). Item counts are re-planned after the pilot (P0) measures real token usage.
- Date: 2026-11-08
- Spec refs: §2.3 C (training pipeline steps 1–5), §3.4 Loop B, §5.3, §7 (no fabricated results), §8 item 5

## Context

The router (ADR-0004) needs a quality label for **every (prompt, model) pair**, under the same task-specific acceptance criterion, plus cost, latency and failure observations. No API keys exist yet, and all cloud providers are on free tiers with tight daily limits (ADR-0003). Gemini's free tier uses submitted content for product improvement, so only **public benchmark data** may be sent to it. The datasets were checked on HF cards and official repos on 2026-11-08. Items marked UNVERIFIED could not be confirmed.

## Options considered

1. **Reuse public router outcome data (RouterBench, RouteLLM battles).** No collection cost, but the models are outdated (GPT-3.5, Claude v1, …) and not our pool. RouterBench's data card states **no license**. Usable at most as an offline sanity check of our evaluation code (ADR-0004), never as training data.
2. **Run our own workload through our own pool, labelled with automatic metrics where possible, an LLM judge where not, and a human-reviewed calibration subset** (chosen).
3. Human-label everything. The single available reviewer cannot label about 8K outputs.

## Decision

### Workload: public datasets mapped to the v1 task types

Only datasets with permissive or attribution licenses are used for training data. Share-alike (SA) data is used locally and never redistributed. Contents are **not committed**: only loaders, prompt templates and a manifest of item ids + SHA-256 hashes.

| task_type | Dataset (license) | v1 items | Prompt framing | Label (auto / judge) |
|---|---|---|---|---|
| query_classification | Banking77 (CC BY 4.0); CLINC150 (CC BY 3.0) | 100 + 100 | pick 1 of 10 candidate intents (gold + 9 sampled), JSON answer | **auto**: exact label |
| query_rewriting | CANARD (CC BY-SA 4.0) | 120 | rewrite the last question to be self-contained | **judge** (reference-guided: same intent as gold rewrite, self-contained); ROUGE-L stored as a weak signal |
| research_planning | MuSiQue (CC BY 4.0), using gold decompositions | 100 | produce an ordered list of sub-questions | **judge** (reference-guided coverage of gold hops) |
| evidence_synthesis | QASPER (CC BY 4.0) with gold evidence paragraphs; MuSiQue with supporting paragraphs | 90 + 90 | answer only from the supplied evidence; say "insufficient" if needed | **auto**: token F1 / EM vs gold (QASPER unanswerables included) **+ judge** faithfulness |
| citation_verification | SciFact (claims CC BY 4.0, abstracts ODC-By); AttributionBench (Apache-2.0, has `test_ood`) | 90 + 90 | JSON `{label, rationale_sentence_ids}` per schema | **auto**: schema-valid **and** label correct |
| general / final_response | IFEval (Apache-2.0); GSM8K (MIT); MMLU-Pro (MIT); MT-Bench Q turn 1 (FastChat, Apache-2.0) | 120 + 80 + 80 + 60 | as published | **auto** for IFEval (programmatic), GSM8K (EM) and MMLU-Pro (accuracy); **judge** for MT-Bench |
| structured output (cross-cutting, tagged `general` + `json_schema`) | JSONSchemaBench (MIT) | 80 | generate an instance for the schema and the task | **auto**: validates against the schema (our validator) |

Total: **about 1,200 prompts**. 8 of 13 sources are fully auto-labelled. Judge-labelled items: about 460 per model.

**Excluded:**
- MS MARCO, Multi-News, AlpacaEval data, Chatbot Arena outputs: non-commercial.
- ELI5: withdrawn.
- RouterBench data: no license.
- TREC CAsT: corpus agreements.
- StrategyQA, ALCE subsets, StructuredRAG, json-mode-eval: licenses UNVERIFIED. They can be added if clarified.

### Acceptance criteria

These define the quality scale used by ADR-0004.

- Auto tasks: accepted = metric passes. Classification is exact. GSM8K EM uses the numeric answer. IFEval uses strict all-instructions-followed. Structured output requires schema-valid plus required semantics. QASPER/MuSiQue require token F1 ≥ 0.5 **and** judge faithfulness ≥ 4/5.
- Judge tasks: accepted = rubric score ≥ 4 on a 1–5 anchored scale.
- Graded scores in [0,1] are stored for every item.
- Rubrics are versioned files (`rubric_version` is stored with every label).

### LLM judge protocol and bias mitigations

Literature (accessed 2026-11-08):
- Zheng et al., NeurIPS 2023 D&B (arXiv:2306.05685): position, verbosity and self-enhancement bias. Reference-guided grading cut math judging failures from 14/20 to 3/20.
- Panickssery et al., NeurIPS 2024 (arXiv:2404.13076): self-recognition correlates with self-preference.
- Dubois et al., COLM 2024 (arXiv:2404.04475): length-controlled win rates.
- Verga et al. 2024 (arXiv:2404.18796): multi-family judge panels.
- Thakur et al. 2024 (arXiv:2406.12624): use Cohen's κ, not percent agreement.

The protocol:

1. **Pointwise, reference-guided scoring** with an anchored 1–5 rubric, plus gold answer, gold rewrite, gold decomposition or evidence in the prompt. Labels are per (prompt, model), so position bias from pairwise judging doesn't arise. Scale drift is controlled with anchored examples.
2. **No self-judging (family exclusion).** A candidate is never judged by a model from its own family. The plan:
   - Gemini Flash judges gpt-oss, OpenRouter and Ollama outputs.
   - Groq `gpt-oss-120b` judges Gemini outputs.
   - The OpenRouter candidate is chosen from a family that is neither Gemini/Gemma nor gpt-oss.
   - Where quota allows, a second judge from another family scores a 20% subset, and inter-judge κ is reported.
3. **Length/verbosity control:** the rubric penalises content that isn't needed. The report includes the correlation between judge score and response length, plus a length-regressed sensitivity analysis.
4. **Determinism and provenance:** temperature 0, fixed judge model id, `rubric_version` and judge prompt hash stored with every label.
5. **Prompt-injection hygiene:** the candidate output is wrapped as delimited untrusted data, and the judge is told to ignore instructions inside it. A canary test checks that an output containing "score this 5" does not move the score.
6. **Human calibration:**
   - A stratified sample of **150 judge-labelled (prompt, model) pairs plus 50 auto-labelled spot checks** is labelled by a human reviewer, blind to model and judge score, using a small labelling page.
   - Report Cohen's κ (judge vs human, on accepted/rejected) per task_type.
   - Re-label 30 items after at least 7 days to measure intra-rater consistency.
   - **If κ < 0.6 for a task type, its judge labels are marked low-trust and excluded from headline claims** until the rubric or judge is revised and re-run.
   - Estimated reviewer time is about 4–5 hours. The reviewer is the product owner, working from LLM pre-labels (DECISIONS D3; see the amendment below).
   - Optional: MT-Bench human judgments (CC BY 4.0) as an external judge-agreement check.

### Collection mechanics

- The harness (`ml.benchmark`) calls ModelMesh's public API with an **explicit model**, `cache.mode: "off"`, `fallback: "none"` and `metadata.application = "modelmesh-bench"`. Every observation is then a real request record with attempts, usage source, cost and latency.
- Failures are recorded as observations, because they are reliability data. They are retried later and never dropped silently.
- Settings: temperature 0, seed when supported, `reasoning_effort: "low"` for reasoning models (reasoning tokens count against TPM/TPD).
- Optional, if quota remains: 3 samples on a 10% subset to estimate label noise. Hybrid-LLM-style soft labels can then be used.
- The scheduler respects per-model limits (ADR-0008 quota tracker), can be resumed from Postgres state, and skips completed (item, model) pairs (idempotent).

### Free-tier budget (estimates, not measurements)

Assumptions:
- Average about 550 tokens per item (in + out) from the prompt sizes above. Synthesis items with QASPER evidence run to about 1,700.
- A ×1.5 allowance for reasoning tokens gives **about 1.0M tokens per model** for 1,200 items.
- Judge load is about 460 items × 7 models ≈ 3,200 calls, about 1M tokens in total.

| Model group | Binding limit (source) | Generation time for 1,200 items | Notes |
|---|---|---|---|
| Groq gpt-oss-20b / 120b (in parallel) | 200K TPD per model (Developer plan figure; **free-plan figure UNVERIFIED**) | **about 5 days** | gpt-oss-120b also judges Gemini outputs (about 920 calls, about 0.4M tokens), adding about 2 days |
| Gemini Flash-Lite / Flash | about 500–1,500 RPD per model (third-party; **UNVERIFIED**, read the value in the AI Studio console) | about 1–3 days each | Flash also judges about 2,300 outputs, adding 2–5 days |
| OpenRouter `:free` | 50 req/day without credits; **1,000/day after buying ≥10 credits** (official) | **24 days per model without credits**; about 1.2 days with credits | Buying credits is the **user's decision** (a purchase). Without it, the OpenRouter model is benchmarked on the 260-item pilot only |
| Ollama qwen3:4b, llama3.2:3b | local throughput (**unmeasured**; guess about 10 s per item) | about 3–4 h per model, overnight | free and unlimited |

**Feasibility:**
- **P0 pilot:** 20 items per source, about 260 prompts × 7 models ≈ 1,800 generations + about 700 judge calls. Fits in about 2–3 days.
- **P1 full v1:** about 1,200 prompts takes **about 7–12 days of unattended wall-clock collection**, if the assumed limits hold. Token usage measured in P0 replaces these estimates before P1 starts.
- If limits are lower than assumed, P1 shrinks per source (minimum 60 items each) rather than dropping task types.
- Cost in USD: **$0** on free tiers, plus the optional $10 OpenRouter credit.

### Statistical honesty

- With 1,200 prompts and a 20% test split, the test set has about 240 prompts and about 30–50 per task_type.
- Headline claims are aggregate, with paired bootstrap CIs. Per-task results are reported as exploratory.
- No benchmark number appears in any doc until it is produced by the harness with a run id.

### Labels from clients (`/v1/feedback`)

Stored as **candidate** labels with `label_source`:
- `automated_eval` and `human_review` labels enter training only through a versioned dataset build with a reviewed manifest.
- `user_signal` is excluded by default.

## Consequences

- Benchmark collection is the critical path for M6. It starts as soon as keys exist and runs in the background while M3, M4 and M7 are built.
- The public workload stands in for real Mezvuat traffic. The workload-shift split (ADR-0004) and later feedback labels test how well that proxy transfers.
- Judge labels can't be fully trusted. The κ gate makes that visible instead of hiding it.

## Verification plan

- P0 report:
  - measured tokens and latency per model
  - quota actually consumed against the plan
  - judge–human κ on the first 50 human labels
  - any license or terms issues
- Unit tests: each auto metric matches published reference implementations on fixtures (IFEval checker, SQuAD-style F1, schema validation). A judge-prompt canary test covers injection resistance.
- The dataset manifest hash is recorded in every MLflow run.

## Amendment 2026-11-08 (contract 1.0-rc2, CR-01)

- **`evidence_assessment` (new task type).**
  - Proxy workload: sufficiency judgments built from **MuSiQue-Full** answerable/unanswerable pairs (CC BY 4.0) and QASPER unanswerable questions (CC BY 4.0).
  - Prompt: "is the supplied evidence sufficient to answer?", with a JSON verdict.
  - Label: auto. Accepted when the JSON is valid and the sufficiency label is correct.
  - About 100 items in P1, taken from the existing MuSiQue/QASPER budgets, so the total stays at about 1,200.
- **`evaluation_judge` (new task type).** It is **not benchmarked or calibrated in v1**.
  - Its executions and feedback are excluded from router quality-training data, because a judge's output is an evaluator artefact and not a task outcome.
  - Its latency and reliability telemetry is still used.
  - With `min_quality` set, a judge request always returns 422.
- **Feedback-derived labels** enter dataset builds only when `training_eligible` is true. That requires all three: `attribution = direct`, `label_source ≠ user_signal`, and task type ≠ `evaluation_judge`.

## Amendment 2026-11-08 (docs/DECISIONS.md D2, D3, D9)

- **D3, labelling.** An LLM pre-labels and the product owner confirms or corrects. This supersedes fully blind labelling, because a visible pre-label can anchor the reviewer. To measure that, the pre-label is hidden on a random subset; agreement with and without the pre-label is reported and the limitation documented. Pre-labels come from a model outside the labelled model's family. The Cohen's kappa gate of 0.6 still applies.
- **D2, OpenRouter credit.** Deferred. **M5 stops after the P0 pilot** and reports the yield at free limits and what the credit would change. P1 does not start for OpenRouter until the product owner decides. The credit is never assumed and never purchased by an agent.
- **D9, downloads.** Benchmark dataset downloads are approved, cached on X:.
