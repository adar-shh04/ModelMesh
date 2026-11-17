# P1 router evaluation: pre-registration

Frozen **2026-11-17**, while P1 was still collecting. When this was written, no P1 observation, judgement or label had been read by the person freezing it. The split was built from `data/manifests/full.json` and the item contents only. P0 outcomes were already known (`M6_P0_PRELIMINARY_REPORT.md`), and nothing selected on P0 carries over (ADR-0004 amendment of 2026-11-15).

This note fixes the P1 evaluation design before any P1 outcome is used. A change to anything below must be recorded here, with its date and reason, **before** the results it could affect are computed. A change made after looking at P1 results makes the P1 report exploratory, and the report must say so.

## Frozen artefacts

| | Value |
|---|---|
| Dataset manifest | `data/manifests/full.json`, sha256 `832ce76c96a5f59c0475073f0564387887401a964d1803612ff572dd54090c7c` (1,200 items) |
| Split manifest | `data/manifests/p1_router_split.json`, `split_manifest_sha256` `89aaf76128e13062547cccde4e05e9cdb12745cb7be93f8b034895009f949641` |
| **Test manifest** | `data/manifests/p1_test.json`, `test_manifest_sha256` **`c604a5889f36dae5f815a84d75433a972bb34c438671f8fa9c3cb9b4078ca36c`** (238 items) |
| Split sizes | train 723, validation 239, test 238 |
| Seed | 20261117 (split); the evaluation keeps M6's bootstrap convention with 2,000 resamples |
| Router features | feature version 2 (`modelmesh.ml.features`, 128-token embedding) |

The hashes are computed over the sorted `(item_id, item sha256[, split])` rows, so they do not depend on file formatting or line endings. `tests/unit/test_p1_split.py` asserts them, and `python -m modelmesh.ml.p1 verify-split` checks them. `python -m modelmesh.ml.p1 freeze-split` rebuilds the split from the items (about 6 minutes cold on one CPU thread, seconds from the embedding cache) and fails if the rebuild differs. It never rewrites a frozen manifest.

## How the split was built

The splitter is M6's (`modelmesh.ml.training.splits`), applied to all 1,200 items:

1. Items from one source question share a group. This includes the MuSiQue planning, sufficiency and synthesis variants of one question.
2. Near-duplicate prompts share a group. A pair counts when its prompt cosine (bge-small) is above 0.95 in **either** space:
   - the feature-version-1 grouping text (512 tokens, what M6 grouped on);
   - the feature-version-2 router text (128 tokens, what the P1 router embeds).
3. Groups are assigned 60/20/20 within each (task_type, source) stratum.

Two changes from the M6 P0 run were decided from item contents alone, before any outcome was read:

- **A pair is a near-duplicate only if its required response formats are near-identical too** (identical, or schema-text cosine above 0.95).
  - All 80 JSONSchemaBench items carry the same one-line instruction, and the item itself is the schema in `response_format`. On prompts alone they formed one 80-item "duplicate" group, so the M6 rule put none of them in test.
  - This condition can only split a prompt-based group. It never merges two.
- **Larger groups are placed first** (seeded random order, then stably sorted by size).
  - Classification prompts share a long template and chain into groups of 63 (banking77) and 43 (clinc150).
  - In random order such a group can arrive late and overshoot train. Under the M6 rule, clinc150 would have had 6 or 17 test items, depending on the other groups.

For comparison, the literal M6 rule gives train 778, validation 211 and test 211, with no JSONSchemaBench items in test and 10 banking77 test items. Both changes are optional parameters (`required_similarity`, `largest_first`); their defaults leave the P0 split unchanged.

Checks on the frozen split:
- No group crosses splits.
- The largest cross-split near-duplicate similarity, min(max prompt cosine, format similarity), is 0.9499.
- Prompt-only cosine reaches 1.0 across splits only between items with an identical instruction and a different schema.

Test items per stratum:

| Stratum | Test | | Stratum | Test |
|---|---|---|---|---|
| citation_verification / attributionbench | 18 | | general / ifeval | 24 |
| citation_verification / scifact | 18 | | general / jsonschemabench | 16 |
| evidence_assessment / musique | 16 | | general / mmlu_pro | 16 |
| evidence_synthesis / musique | 9 | | query_classification / banking77 | 18 |
| evidence_synthesis / qasper | 10 | | query_classification / clinc150 | 20 |
| final_response / mt_bench | 12 | | query_rewriting / qrecc | 24 |
| general / gsm8k | 16 | | research_planning / musique | 21 |

## Pool (D2)

- **Routed pool, five models, all 1,200 items:**
  - `gemini/gemini-3.1-flash-lite`
  - `groq/openai/gpt-oss-20b`
  - `groq/openai/gpt-oss-120b`
  - `ollama/llama3.2:3b`
  - `ollama/qwen3:4b`
- **Sparse, never routed or trained on:** `openrouter/nvidia/nemotron-3.5-super-vl:free` and `openrouter/liquid/lfm-3-1.7b:free`. They stay on the 280-item pilot set (50 requests a day).
- **Out of P1:** `gemini/gemini-3.6-flash`. Its pilot rows are kept and reported only in the P0 documents.
- **Data:** the run directories `full-2026-11-16` (cloud) and `full-2026-11-16-local` (Ollama), merged by `python -m modelmesh.ml.p1 merge`.
  - Dedupe on `(item_id, model)`, keeping the earliest observation. Every dropped row is reported.
  - OpenRouter rows come from `pilot-2026-11-11`, only for items whose pilot content hash equals the P1 item's.

## Labels

- **Automatic labels** follow ADR-0005. Recorded failures (502, 504, schema failure) count as not accepted. Cells without an observation are never labelled or imputed. A failed cell is charged the median successful cost of its model and task type, as in M6.
- **Judge-labelled task types:** query_rewriting, research_planning and final_response. evidence_synthesis needs both F1 ≥ 0.5 and judge faithfulness ≥ 4.
  - They use the ADR-0005 protocol: reference-guided pointwise 1–5 scoring at temperature 0, family exclusion, and an injection canary.
  - Each label records the judge prompt version and the prompt hash.
  - Which judge model scores which candidate family is fixed before the first judge call, using quota information only (`docs/milestones/P1_JUDGE_PLAN.md`).
- **Human review (D3):** 150 judge-labelled pairs plus 50 automatic spot checks.
  - The pre-label is **hidden by default**; it is shown on a random 30%, which measures anchoring.
  - Cohen's κ (judge vs human) is reported per task type, overall, and with and without the pre-label.
- **κ gate:** a judge-labelled task type with κ < 0.6 is **low-trust**. It is excluded from the primary analysis and the criterion, and reported in a sensitivity variant. If the judge or human step has not finished when the analysis runs, the judge task types are excluded from the primary analysis, and evidence_synthesis appears only in the F1-only sensitivity variant, as in P0.

## Analysis set and cut-off

- The P1 analysis runs once. It starts when the coordinator declares the collection cut-off.
  - The cut-off is decided from coverage, failure counts and projected finish only (`python -m modelmesh.ml.p1 status`, which reports no acceptance), never from acceptance or any router metric.
- **Items used:** those where all five pool models have a recorded outcome (complete cells), within each split.
  - The report gives how many frozen test items drop out for lack of a complete row, per stratum.
  - The test set is never refilled or re-drawn.
- Training and selection use train and validation only.
  - **The test split is evaluated once per reported router version.**
  - Crash-testing the pipeline uses validation as a stand-in test set, never test (P0 smoke-ran on test; P1 does not).

## Primary metric and success criterion (ADR-0004, unchanged)

- **Primary metric:** held-out-query test **AIQ**, normalised over a shared cost range, for the selected learned router and for B0b. The AIQ convention is the ADR-0004 amendment of 2026-11-15:
  - each router's upper hull includes the Cheapest-Single point and is flat to the right;
  - the cost range is [Cheapest-Single, most expensive single model] on the same items.
- **Criterion:** both conditions must hold.
  1. **Learned minus B0b**, held-out-query test AIQ: the paired bootstrap 95% CI (2,000 resamples) lies **above 0**.
  2. **Learned minus B0b**, leave-one-dataset-out workload shift: the 95% CI **lower bound is above −0.01**.
- The criterion is evaluated and reported **whether or not it passes**. If it fails, rules (B0b) stay the production default.
- **B0b** is estimated on train plus validation acceptance (stricter on the learned router). The validation-only lookup is reported alongside.
- **Selection** uses validation only:
  - Candidates: L1 kNN, L2 LightGBM and L3 logistic regression, on the M6 grids (`modelmesh.ml.evaluation.p0.GRIDS`), refitted on P1.
  - The family is chosen by validation AIQ of the uncalibrated scores, with log loss as the tie-break.
  - Isotonic calibration is fitted per task type on validation.
  - The low-confidence fallback to B0b triggers below the 5th percentile of validation nearest-neighbour similarity.
- **Leave-one-dataset-out folds:**
  - banking77 ↔ clinc150;
  - scifact ↔ attributionbench;
  - qasper ↔ musique (evidence_synthesis), included only if evidence_synthesis has judge labels that pass the κ gate.
- **Fold procedure.** As in M6:
  - each fold refits the selected configuration on the remaining items (grouped 75/25 train and validation) and predicts all items of the held-out source;
  - folds are pooled;
  - hyper-parameters are not re-selected.

## Secondary results (reported, not part of the criterion)

- **Mixture shift:** the M6 `mezvuat_like` and `chat_like` shares (`modelmesh.ml.evaluation.p0.MIXES`), applied as importance weights on the test set.
- **Other ADR-0004 metrics, with bootstrap CIs:**
  - cost at fixed quality against Best-Single;
  - Oracle regret;
  - constraint fidelity at Q_min ∈ {0.5, 0.6, 0.7, 0.8};
  - ECE and Brier per task type;
  - typo robustness;
  - router overhead.
- **Paraphrase robustness** on 100 test prompts needs provider calls and its own quota budget. It runs only if the coordinator schedules it, and the report says so either way.
- **Per-task results** are exploratory. Headline claims are aggregate.

## Pre-registered sensitivity variants

| Variant | Change from primary |
|---|---|
| failures_dropped | failed cells are missing instead of not accepted |
| f1_only_synthesis | evidence_synthesis labelled by F1 alone (only if its judge labels are missing or low-trust) |
| all_judge_tasks | judge task types included regardless of the κ gate |
| feature_v1_512 | router embedding over 512 tokens (feature version 1), same split |
| b0b_validation_only | B0b lookup from validation acceptance only |

None of these can turn a failed criterion into a pass: the criterion is read from the primary variant only.

## Sparse OpenRouter coverage

- The two `:free` models cover only the pilot items, in run order, which is not a random sample. 229 of the 280 pilot items have the same content hash in P1.
- They are reported **descriptively, never routed or trained on**, on the P1 items that both they and all five pool models have outcomes for. The table gives:
  - acceptance per model with bootstrap 95% CIs;
  - the share of Oracle choices;
  - n per task type.
- The report states their smaller coverage next to every number, as D2 requires, and makes no claim about how they would route on the full set.
- If the owner later buys credit or enables billing (D2), these models can be added to a later version of the analysis without changing this split. That version is reported as a separate router version.

## Amendment, 2026-11-17: collection incidents (no acceptance or router metric was looked at)

Two collection faults were found from `p1 status`-level data only (failure codes and attempt outcomes).

1. **Expired Groq key.** From 2026-11-16T21:07Z, after the gateway restarted, Groq rejected the benchmark's provider key (`expired_api_key`). The runner recorded 240 rows (120 per Groq model) whose only upstream outcome was `auth_error`. They measure our key, not the models. From this date the runner treats an attempt refused for authentication as a deferral and stops the lane (`auth_error`), so the cell stays pending. **Open, owner's decision:** whether the 240 recorded rows are moved to a quarantine file and re-collected with a new key, or kept as failures under the Labels rule above.
2. **Wedged circuit breaker.** A half-open provider breaker could hand out its single probe and never get it back when the model breaker was still open, which blocked every Ollama model with `circuit_open` until a restart. These were 503 deferrals, so no rows were recorded and no data is affected. Fixed in the gateway (the probe is returned, and an unanswered probe expires after 330 s). Collection only lost time.
