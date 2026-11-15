## Caveats

These notes are hand-written and appended to the generated report. They contain no new measurements.

- **Sample size.** The held-out-query test set is 32 prompts and the leave-one-dataset-out set is 80. ADR-0005 planned about 240 test prompts for P1. Every interval above is wide, and the per-task calibration rows rest on 16–48 cells.
- **Coverage is in run order, not random.** Which items each model reached depends on when its quota ran out. The four-model pool and the complete-cell restriction avoid imputation, but the items that survive are not a random sample of the pilot: `jsonschemabench`, the judge task types and every item gpt-oss-120b never reached are absent.
- **Failures as rejects mix reliability into quality.** Most failed cells are gemini-3.1-flash-lite's 502s (upstream, not item-dependent) and qwen3:4b's schema failures and timeouts (model-dependent). The failures-dropped variant shows how much this choice matters: B0b and the learned routers change rank between the two variants.
- **Calibration.** Per-task isotonic tables fitted on 4–12 validation items made ECE worse than the raw scores for most families and task types. No family meets the 0.05 promotion gate on P0. P1 needs at least 30 calibration points per task type (the ADR's own threshold for `calibrated: true`), or cross-fitted calibration.
- **Embedding latency.** The 10 ms budget without embedding is met by a wide margin, and the 25 ms budget with embedding is missed by a wide margin. The gateway embeds with fastembed on one thread (`threads=1`), and the router embeds the system prompt plus the final user message, which for citation and evidence prompts reaches the model's 512-token limit. The semantic-cache measurement in M4 (13.8 ms p95) used short texts. The machine was also running the evaluation itself while it measured. Options for P1, none tried here: embed a shorter text (needs retraining), give the embedder more threads, or reuse the semantic cache's embedding when both run on the same text.
- **Online versus offline cost and latency.** Offline routing uses the artifact's fitted token estimate, per-(model, task) output-token medians and latency model. The online Estimator still uses its own token estimate and the static `PREDICTED_OUTPUT_TOKENS` table; the artifact supplies only quality. A shadow decision can therefore differ from the offline replay of the same request on cost ties. Wiring the artifact's predictors into the Estimator is P1 work.
- **Conventions that shape AIQ.** Each router's hull includes the Cheapest-Single point and is flat to the right of its last vertex; the cost range runs from Cheapest-Single to the most expensive single model on the same items. Both Ollama models cost $0, so the left end of every curve is a free model. "Random" is the expected outcome of a uniform choice, not a sampled one.
- **Leave-one-dataset-out references.** Each fold's routers use only that fold's training data, but the pooled AIQ range and the Best-Single reference come from the main split's selection data, which includes some of the held-out sources' items. This affects the yardstick, not the routers.
- **Robustness** covers typos only, on the 32 test prompts. Paraphrases need an LLM, and this milestone made no provider calls.
- **The test set was touched twice** by identical evaluation code (a 20-resample smoke run, then the reported run). This is recorded above; nothing was tuned on it.

## What was not verified

- **Integration tests did not run.** Docker Desktop was stopped, so every test that needs Postgres or the test Redis failed to connect (203 errors or failures, 1 pure test passed in `tests/integration`). That includes the new `tests/integration/test_router_admin.py` (register, shadow mode leaves traffic unchanged and logs a decision, promotion needs a shadow period, approval and a passing non-preliminary report) and the round trip of migration `0002_router_models`.
- The shadow and active paths in `pipeline/chat.py` are covered by unit tests of their parts (`tests/unit/test_router_lifecycle.py`, `tests/unit/test_router_training.py`), not by a request through the running stack.
- The admin API document was regenerated from the code; no external OpenAPI linter was run.
- MLflow runs were written to the local SQLite store; the MLflow UI was not opened.
- No P0 artifact has been registered in a database, because none was running.

## What P1 needs

1. A complete matrix for the P1 pool. D2 was decided on 2026-11-14 (stay on free tiers, `docs/DECISIONS.md`): P1's full set runs on gemini-3.1-flash-lite, both Groq models and both Ollama models, so those five need an outcome on every item, including gpt-oss-20b, which P0 left out of routing for lack of coverage. The OpenRouter models stay on the pilot set and gemini-3.6-flash is dropped, so they belong in a separate, smaller comparison like the one above, not in the routed pool.
2. The judge pipeline run on the judge task types and on `evidence_synthesis` faithfulness, plus the human calibration and Cohen's kappa gate (ADR-0005, D3). Without it three of eight task types stay unlabelled and synthesis keeps an F1-only label.
3. **A new split and a frozen test manifest before any model selection.** Nothing chosen here (the family, the hyper-parameters, the 0.6 default floor behaviour, the low-confidence quantile) may be carried over as a P1 result; selection is re-run on P1 validation. The P0 split manifest (`data/manifests/p0_router_split.json`) is for P0 only.
4. At least 30 validation points per task type for calibration and conformal intervals, and about 100 test prompts for the paraphrase robustness check (which needs provider calls, so a quota budget).
5. A fix for the embedding share of router latency, then a re-measurement on an otherwise idle machine.
6. With Docker running: the full `scripts/check.sh`, including the router admin integration tests and the migration round trip.
