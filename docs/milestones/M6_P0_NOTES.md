## Caveats

These notes are hand-written and appended to the generated report. The only numbers in them come from this report's results file or from the first P0 run's results file (`git show 8266552:docs/milestones/M6_P0_results.json`).

- **Feature version 2: re-run on 2026-11-16.** The first P0 run (2026-11-15, feature version 1) embedded up to 512 tokens and measured a router p95 of 411.9 ms including embedding. This report is the same evaluation re-run after the embedding fix (ADR-0004 amendment of 2026-11-16).
  - What changed: at most 128 tokens are embedded, split between the system prompt and the user message; the embedder uses 4 ONNX threads instead of 1; offline and online use the same text.
  - What did not change: the inputs (the observations file has the same sha256, `3161868a…`; the dataset manifest is the same), the seed and the split (split manifest `76d073d77bd81eda…` in both runs).
  - Model selection, calibration and the test evaluation were all re-run.
- **Effect on quality (held-out test, primary variant).** The criterion still does not pass. No family's change is distinguishable from noise on 32 test prompts.

  | | Feature v1 | Feature v2 |
  |---|---|---|
  | Family selected on validation | logreg | kNN |
  | Learned AIQ | 0.834 | 0.787 |
  | Learned minus B0b | 0.017 [-0.023, 0.070] | -0.030 [-0.137, 0.068] |
  | kNN AIQ | 0.748 | 0.787 |
  | Logistic regression AIQ | 0.834 | 0.827 |
  | LightGBM AIQ | 0.809 | 0.843 |
  | Leave-one-dataset-out, learned minus B0b | -0.037 [-0.096, 0.014] | -0.041 [-0.097, 0.010] |

  - The drop in the selected router's AIQ is mostly a selection effect. On validation, kNN went from 0.815 to 0.835 and overtook logistic regression (0.825 in both runs).
  - On test, kNN and LightGBM went up and logistic regression went down by 0.007.
  - P0 cannot say whether the 128-token cap costs quality. P1 should report the 512-token feature as a sensitivity variant if latency allows.
  - The LightGBM hyper-parameters selected also changed (`min_child_samples` 5 instead of 20). This run limited OpenMP and BLAS to 2 threads so that it would load the machine less than the first run did, which may also change LightGBM results slightly.
- **Effect on latency.** Both runs measured 160 warm routing calls through the online path on the same machine.

  | | Feature v1 | Feature v2 |
  |---|---|---|
  | Including embedding, p50 / p95 | 71.6 / 411.9 ms | 63.5 / 138.1 ms |
  | Prompts of 2,048 characters or more, p95 | 455.9 ms (35 calls) | 172.8 ms (35 calls) |
  | Excluding embedding, p95 | 1.06 ms | 4.88 ms |

  - The 25 ms target **is still missed**.
  - Excluding embedding, p95 rose because the selected router is now kNN, which compares against every training embedding. It is still under the 10 ms target.
  - **Machine load during the v2 measurement.** It was taken while the P1 benchmark was running, including a local `llama-server` (Ollama) process that kept the CPU at about 80% load. On that loaded CPU, a separate probe (50 sampled pilot prompts per setting) measured a p50 of 81 ms at 1 thread and 39 ms at 4 threads, without the token cap.
  - So the absolute numbers mostly reflect contention. Whether the fix meets 25 ms on an idle machine is **not verified**.
- **Fix for the online/offline skew.** Feature version 1 had a skew: the gateway cut the embedded text at 2,048 characters, while training embedded up to 4,096. For 34 of the 280 pilot items, that changed which 512 tokens were embedded. In version 2, training calls the gateway's own `request_text`, so the two cannot differ.

- **Sample size.** The held-out-query test set is 32 prompts and the leave-one-dataset-out set is 80. ADR-0005 planned about 240 test prompts for P1. Every interval above is wide, and the per-task calibration rows rest on 16–48 cells.
- **Coverage is in run order, not random.** Which items each model reached depends on when its quota ran out. The four-model pool and the complete-cell restriction avoid imputation, but the items that survive are not a random sample of the pilot: `jsonschemabench`, the judge task types and every item gpt-oss-120b never reached are absent.
- **Failures as rejects mix reliability into quality.** Most failed cells are gemini-3.1-flash-lite's 502s (upstream, not item-dependent) and qwen3:4b's schema failures and timeouts (model-dependent). The failures-dropped variant shows how much this choice matters: B0b and the learned routers change rank between the two variants.
- **Calibration.** Per-task isotonic tables fitted on 4–12 validation items made ECE worse than the raw scores for most families and task types. No family meets the 0.05 promotion gate on P0. P1 needs at least 30 calibration points per task type (the ADR's own threshold for `calibrated: true`), or cross-fitted calibration.
- **Embedding latency.** See the feature-version-2 notes above. The remaining options are a smaller embedding model and running the embedding concurrently with the pre-routing work. A smaller model would need a new download, retraining and a second model in memory beside the semantic cache's. Neither option was tried.
- **Online versus offline cost and latency.** Offline routing uses the artifact's fitted token estimate, per-(model, task) output-token medians and latency model. The online Estimator still uses its own token estimate and the static `PREDICTED_OUTPUT_TOKENS` table; the artifact supplies only quality. A shadow decision can therefore differ from the offline replay of the same request on cost ties. Wiring the artifact's predictors into the Estimator is P1 work.
- **Conventions that shape AIQ.** Each router's hull includes the Cheapest-Single point and is flat to the right of its last vertex; the cost range runs from Cheapest-Single to the most expensive single model on the same items. Both Ollama models cost $0, so the left end of every curve is a free model. "Random" is the expected outcome of a uniform choice, not a sampled one.
- **Leave-one-dataset-out references.** Each fold's routers use only that fold's training data, but the pooled AIQ range and the Best-Single reference come from the main split's selection data, which includes some of the held-out sources' items. This affects the yardstick, not the routers.
- **Robustness** covers typos only, on the 32 test prompts. Paraphrases need an LLM, and this milestone made no provider calls.
- **The test set has now been evaluated three times:**
  - twice under feature version 1, by identical code (a 20-resample smoke run, then the reported run)
  - once under feature version 2 (this report)

  The feature change was chosen for latency, before the version-2 test results existed, and nothing was tuned on test. Still, P0's test set is spent: it must not be used to choose between feature versions.

## What was not verified

- **Integration tests did not run on 2026-11-15.** They have since passed, on 2026-11-16 with feature version 2 (`tests/integration/test_router_admin.py` included), against the test Redis and `modelmesh_test`. On 2026-11-15 Docker Desktop was stopped, so every test that needs Postgres or the test Redis failed to connect (203 errors or failures, 1 pure test passed in `tests/integration`). That includes the new `tests/integration/test_router_admin.py` (register, shadow mode leaves traffic unchanged and logs a decision, promotion needs a shadow period, approval and a passing non-preliminary report) and the round trip of migration `0002_router_models`.
- The shadow and active paths in `pipeline/chat.py` are covered by unit tests of their parts (`tests/unit/test_router_lifecycle.py`, `tests/unit/test_router_training.py`), not by a request through the running stack.
- The admin API document was regenerated from the code; no external OpenAPI linter was run.
- MLflow runs were written to the local SQLite store; the MLflow UI was not opened.
- No P0 artifact has been registered in a database, because none was running.

## What P1 needs

1. A complete matrix for the P1 pool. D2 was decided on 2026-11-14 (stay on free tiers, `docs/DECISIONS.md`): P1's full set runs on gemini-3.1-flash-lite, both Groq models and both Ollama models, so those five need an outcome on every item, including gpt-oss-20b, which P0 left out of routing for lack of coverage. The OpenRouter models stay on the pilot set and gemini-3.6-flash is dropped, so they belong in a separate, smaller comparison like the one above, not in the routed pool.
2. The judge pipeline run on the judge task types and on `evidence_synthesis` faithfulness, plus the human calibration and Cohen's kappa gate (ADR-0005, D3). Without it three of eight task types stay unlabelled and synthesis keeps an F1-only label.
3. **A new split and a frozen test manifest before any model selection.** Nothing chosen here (the family, the hyper-parameters, the 0.6 default floor behaviour, the low-confidence quantile) may be carried over as a P1 result; selection is re-run on P1 validation. The P0 split manifest (`data/manifests/p0_router_split.json`) is for P0 only.
4. At least 30 validation points per task type for calibration and conformal intervals, and about 100 test prompts for the paraphrase robustness check (which needs provider calls, so a quota budget).
5. A re-measurement of router latency with feature version 2 on an otherwise idle machine, and the 512-token feature as a sensitivity variant.
6. With Docker running: the full `scripts/check.sh`, including the router admin integration tests and the migration round trip.
