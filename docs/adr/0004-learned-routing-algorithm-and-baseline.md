# ADR-0004: Learned routing algorithm, baselines, metrics and data splits

- Status: Accepted as the experimental protocol. The *choice of production algorithm* is deferred to M6 results.
- Date: 2026-11-08
- Spec refs: §2.3 C (features, training pipeline, routing objective), §4.3 (safe fallback), §6 (trained router vs rules baseline, held-out data), §8 item 4
- Resolves contract open question 1: what `min_quality` / `predicted_quality` mean.

## Context

The router must choose, per request, among a small pool of about 6–8 models (ADR-0003). It optimises an explicit objective under hard constraints:

  min over m ∈ M_eligible of Ĉost(m,q)  s.t.  Q̂(m,q) ≥ Q_min,  L̂_p90(m,q) ≤ L_max

It has to beat a rules baseline on held-out queries **and** on a held-out workload distribution. Relevant literature (accessed 2026-11-08):

- **RouteLLM** (Ong et al., ICLR 2025, arXiv:2406.18665; code Apache-2.0). Binary strong/weak routing: similarity-weighted Bradley-Terry, matrix factorisation, BERT and causal-LLM classifiers. Metrics are APGR (area under quality vs %-strong-calls) and CPT(x%). Arena-only training data was near random on MMLU and needed "golden" labels.
- **RouterBench** (Hu et al., arXiv:2403.12031; code MIT, dataset card states **no license**). Its per-model quality regressors (kNN, MLP) score `λ·q̂ − cost`. Metrics: the non-decreasing convex hull and AIQ, compared against the "Zero router" (hull of single models) and an Oracle. Predictive routers did **not** significantly beat the Zero router overall.
- **Li, "When Simple kNN Beats Complex Learned Routers"** (arXiv:2505.12601), **"The Routing Plateau"** (arXiv:2606.07587: kNN in the top 2 on 5 benchmarks, trained routers +≤0.006 concordance) and **LLMRouterBench** (arXiv:2601.07206: many routers fail to beat Best-Single, and the embedding backbone barely matters).
- **Hybrid LLM** (Ding et al., ICLR 2024) uses soft labels P(small ≥ large) from repeated samples. **FrugalGPT** (TMLR 2024) and **AutoMix** (NeurIPS 2024) cover cascades. **RouterArena** (arXiv:2510.00202) adds robustness under paraphrase perturbation.
- **Not Diamond RoRF** (MIT): random forests over prompt embeddings. This is the closest tree-based evidence. **We found no paper that benchmarks GBDT routers specifically**, so a GBDT result here is a genuine experiment, not a replication.

Two implications. First, a kNN router is a mandatory strong baseline. Second, "the learned router wins" is not a given. It is the hypothesis M6 tests.

## Options considered (router model family)

| Family | Pros | Cons | Status |
|---|---|---|---|
| **Per-(query, model) quality predictor** P(acceptable \| x, m), then an explicit policy solver | Works with any policy and constraint without retraining. Calibrated probabilities map directly to `min_quality`. Handles a changing pool | Needs labels for every (query, model) pair | **Chosen design** |
| Pairwise strong-vs-weak (RouteLLM-style) | Strong for 2 models | Pool has about 7 models with non-total quality order. The threshold encodes one cost trade-off | Rejected as primary |
| Multi-class "best model" classifier | Simple inference | Label depends on λ/policy and constraints, so each policy needs a separate model. Infeasible constraints are hard to express | Stretch comparison only |
| Cascade (FrugalGPT/AutoMix) | Can beat single-shot routing | Adds latency (≥2 calls). Needs a reliable verifier. Interacts badly with streaming | Out of scope v1. Noted as a future policy |
| RL / bandits | Online adaptation | Spec: not needed initially. No unbiased offline evaluation without logging propensities | Out of scope v1 |

Within the chosen design, the M6 candidates are trained on identical features and splits:

- **L1 kNN**: cosine similarity on bge-small embeddings, optionally restricted to the same task_type. q̂(m) is the similarity-weighted acceptance of the neighbours for model m. k and the temperature are tuned on validation.
- **L2 LightGBM**: one global binary classifier over (request features ⊕ model features). This allows partial sharing across models and a prior for models with few labels.
- **L3 logistic regression per model**: a linear sanity baseline.

## Decision

### Quality scale (answers open question 1)

For each `task_type`, ADR-0005 defines a binary **acceptance criterion**, for example:
- classification: exact label correct
- citation verification: valid JSON **and** correct support label
- open-ended tasks: reference-guided judge score ≥ 4/5

`predicted_quality` = **calibrated estimate of P(response is acceptable under that task's criterion)**. Calibration uses isotonic regression fit on the validation split, per task_type, and is exported as a piecewise-linear table, so no scikit-learn is needed at runtime.

`min_quality: 0.7` therefore means: "only choose models for which, among requests like this, at least about 70% of responses are expected to pass the task's acceptance rubric." `predicted_quality_interval` is a 90% interval from split-conformal residuals on validation. It is `null` when there are fewer than 30 calibration points for that task_type. Graded scores (0..1) are stored as well, for analysis.

### Features

These are computed in `ml.features` and shared by online and offline code.

- **Request features:**
  - `task_type` (one-hot, 7)
  - log estimated input tokens, log system-prompt tokens, log "context" tokens (non-final messages)
  - message count and user-turn count
  - log `max_tokens`
  - `response_format` kind, plus JSON-schema complexity (property count, depth, required count, enum present)
  - tool count
  - content statistics: code-fence / digit / math-symbol fraction, question count, non-ASCII ratio
  - **embedding** of system prompt + final user message (bge-small-en-v1.5, 384-d, truncated to 512 tokens). Used raw for kNN; PCA to 32-d (fit on train only) for GBDT and LR.
- **Model features** (GBDT only): model one-hot, family, provider, context window, price tier, and target-encoded prior acceptance per task_type (computed out-of-fold on train only).
- **Excluded:** `metadata.application`, tags and end-user id. Excluding them prevents fitting the router to one client. Application is used only to *define* workload splits.
- **Token estimate:** `chars × a_m + b_m`, per model family, fitted on provider-reported usage (initially a = 0.25). This avoids a tokenizer dependency. It is reported as `estimated`.

### Other predictors (deliberately simple, not ML-heavy)

- **Latency:** per model, log-latency = α + β·input_tokens + γ·predicted_output_tokens, fit on recent telemetry. p50 and p90 come from empirical residual quantiles. Predicted output tokens = per-(model, task_type) median completion tokens.
- **Cost:** catalog price × (estimated input + predicted output tokens). `max_cost_usd` filtering uses an **upper bound** with output = `max_tokens` (or the catalog default), so it is conservative.
- **Failure probability:** rolling 15-minute error rate per model, plus breaker state (ADR-0008).

### Policy solver

This is explicit, with no hidden weighted score. It applies to the eligible set after the hard filters in ADR-0007, ADR-0008 and ADR-0010.

- `cost_optimized`: min Ĉ s.t. Q̂ ≥ Q_min, L̂_p90 ≤ L_max. Tie-break Q̂, then L̂. If the caller gave no `min_quality`, a default floor of 0.6 applies on calibrated task types only (see the 1.0 amendment).
- `latency_optimized`: min L̂_p50 s.t. Q̂ ≥ Q_min, Ĉ ≤ max_cost.
- `quality_balanced` (**changed from the draft's "max Q − λ·cost"**): **the cheapest model whose Q̂ is within δ = 0.05 of the best eligible Q̂**. This removes the undefined λ and its unit problem (quality per USD, when costs are about 1e-4 USD). δ is documented per policy and reported in `routing.reason`.
- `quality_max`: max Q̂. Tie-break Ĉ.
- `reliability_first`: min P̂_fail. Tie-break Ĉ.

The ranked list it returns is also the fallback order (ADR-0008).

### Baselines

All baselines are evaluated on the same splits.

- **B0a, static rules** (the spec's "rules-based baseline"). Hand-written before any benchmark results are seen, and committed with a timestamp:
  - task_type maps to a size tier: classification and rewriting → small; citation verification and general → medium; planning, synthesis and final response → large.
  - Escalate one tier when estimated input exceeds 8K tokens or the json_schema has more than 10 properties.
  - Within the tier, pick the cheapest eligible model; tie-break on p50 latency.
- **B0b, per-task lookup.** Data-driven, with no per-query features: for each task_type, the cheapest model whose *validation* acceptance rate is ≥ Q_min. The learned router must beat **B0b** to show that *per-query* routing adds value beyond task_type. This is the honest bar.
- **References:** Best-Single (best validation model for all queries), Cheapest-Single, Random, the Zero-router hull, and the **Oracle** (cheapest acceptable model per query, an upper bound).
- **Runtime safe fallback** (spec §4.3) is B0a when no learned artifact is loaded. It is **B0b** when the learned router is low-confidence, meaning the nearest training neighbour similarity is below τ (tuned so about 5% of validation queries trigger it), or the task_type or model is unseen in training. In both cases `routing.mode = "rules_fallback"`.

### Metrics

Reported for every router and baseline on each split, with 95% bootstrap CIs (2,000 resamples, paired for router-vs-baseline differences):

1. The **quality–cost curve** (acceptance rate vs mean cost per request) as λ or Q_min is swept, its non-decreasing convex hull, and **AIQ** (RouterBench), normalised over a shared cost range.
2. **Cost at fixed quality:** cost savings against Best-Single at matched acceptance (≥ 95% and 100% of Best-Single quality). The CPT-style "% calls to the most expensive model" is reported alongside.
3. **Regret against the Oracle:** quality regret at equal cost and cost regret at equal quality.
4. **Constraint fidelity:** for `cost_optimized` at Q_min ∈ {0.5, 0.6, 0.7, 0.8}, the realised acceptance among routed requests vs Q_min (violation rate), and the realised latency vs L_max.
5. **Calibration:** ECE (10 bins) and Brier score of Q̂ per task_type. A model with ECE above 0.05 cannot be promoted.
6. **Robustness:** routing-flip rate under paraphrase and typo perturbations of 100 test prompts (RouterArena-style).
7. **Overhead:** router p50 and p95 latency (ADR-0001 targets).

**Primary success criterion**, fixed now before any data exists: the selected learned router beats **B0b** on held-out-query test AIQ with a paired 95% CI excluding 0, **and** is not worse than B0b on the workload-shift split beyond a −0.01 AIQ non-inferiority margin. If it fails, the report says so. Rules (B0b) then stays the production default, and the learned router can only be enabled behind config. Nothing is spun.

### Data splits

All are built by a versioned dataset-build script with a committed manifest (item ids + SHA-256).

1. **Held-out queries:** 60/20/20 train/validation/test, stratified by task_type and dataset. Splits are **grouped**: all model outputs for one prompt stay together, and near-duplicate prompts (embedding cosine above 0.95) are clustered and kept in one split.
2. **Held-out workload distribution:**
   - (a) **Leave-one-dataset-out** within task types that have two or more sources: Banking77↔CLINC150; QASPER↔MuSiQue; SciFact↔AttributionBench (whose own `test_ood` split is also used).
   - (b) **Mixture shift:** train on the uniform mix and evaluate with importance weights on a "Mezvuat-like" mix (heavy synthesis and verification, per spec §3.3) and a "chat-like" mix (general-heavy).
   - (c) Later, a **temporal** holdout of real client traffic labelled through `/v1/feedback`. These are only labels whose `label_source` is not `user_signal`.
3. The **test set is touched once** per reported router version. Model selection and hyperparameters use validation only, and are logged in MLflow with the dataset manifest hash.

### Lifecycle

- A trained artifact (model file + feature spec + calibration table + manifest hash) moves through `candidate → shadow → active → retired`, stored in the `router_models` table (ADR-0010).
- `shadow`: decisions are computed and logged, never applied.
- Promotion to `active` needs an offline report meeting the criteria above, **plus explicit human approval** (an admin API call). Nothing retrains or promotes automatically (spec §3.4).
- If a catalog change adds a model with no benchmark labels, that model is ineligible for learned routing (rules only) until it is benchmarked.

## Consequences

- The router is interpretable: the policy solver is explicit, and every decision records Q̂, Ĉ, L̂ and the reason.
- Labels are needed for every (prompt, model) pair. This drives the benchmark budget in ADR-0005.
- With about 1,000 prompts, per-task test sets hold roughly 30–60 prompts, so per-task claims will have wide CIs. Headline claims will be aggregate only.

## Verification plan

- M6 first validates the training and evaluation code end to end on **synthetic data with a known optimal policy** (unit test: the Oracle is recovered and AIQ is computed correctly against hand-calculated values).
- Optional and offline only: if its license is clarified, a reproduction of RouterBench's kNN result on RouterBench data, used as a sanity check of the evaluation code. The data is never redistributed or committed.
- The M6 report includes every metric above, for all routers and baselines on all splits, the MLflow run ids and the manifest hash.

## Amendment 2026-11-08 (contract 1.0-rc2)

- The task_type one-hot grows from 7 to 9 values.
- In B0a, `evidence_assessment` maps to the small tier with native json_schema preferred, and `evaluation_judge` maps to the large tier. Judge calls are expected to name an explicit model.
- Requests with `evaluation_judge` are excluded from the router training set (ADR-0005 amendment).
- The calibration evidence described above is published per task_type at `GET /v1/routing/calibration`. A task type stays `calibrated: false` until a promoted router has at least 30 calibration points for it and ECE ≤ 0.05.

## Amendment 2026-11-08 (contract 1.0, CQ-1)

The default `Q_min = 0.6` for `cost_optimized` combined with "uncalibrated models are ineligible" would have made every such request return 422 until M6. Corrected rules (contract §2.1.1):

- An explicit `min_quality` is a hard constraint under every policy.
- The default floor is a policy default, not a hard constraint. It applies only to calibrated task types. If no eligible model meets it, it is dropped with a warning. A request that did not send `min_quality` never gets a 422 because of quality or missing calibration.
- On an uncalibrated task type, `cost_optimized` picks the minimum estimated cost with no floor. `quality_balanced` and `quality_max` have no Q̂ to compare, so they use B0a: `quality_balanced` takes the task_type's tier and `quality_max` the largest eligible tier, cheapest within it. `latency_optimized` and `reliability_first` don't use Q̂ and are unchanged.
- **B0a tier selection:** if the chosen tier has no eligible model, the nearest eligible tier is used (larger first). This matters for `local_only`, where only small models exist.
- In a calibrated task type, a model with no calibrated evidence is left out of the quality comparison and ranked after the models that have evidence.
- **Latency cold start:** the catalog holds a per-model latency prior, used until telemetry exists, so `max_latency_ms` can be evaluated on a fresh deployment.
