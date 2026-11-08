# Product Owner Decisions

Recorded 2026-11-08 at the end of Phase 1. These override any ADR or plan text that says otherwise.
Where an ADR conflicts with this file, update the ADR. Do not reinterpret the decision.

| # | Topic | Decision |
|---|---|---|
| D1 | Default per-key limits | 20 RPM, 50K TPM, $0.25/day (estimated cost). |
| D2 | OpenRouter credit (about $10) | Deferred. Run the pilot on free limits, then the product owner decides. |
| D3 | Human labelling | LLM pre-labels, the product owner confirms or corrects. |
| D4 | Data policy | Clients choose per request via `routing.data_policy`. Gemini free tier, Groq and OpenRouter stay classified as may-train or unknown until reviewed. |
| D5 | AGPL dependencies | Not in ModelMesh. |
| D6 | Repository licence | MIT. |
| D7 | Hosting and CI | GitHub with GitHub Actions. Commit locally; push only when the product owner says so. |
| D8 | Cloud demo | Not in this phase. Local Docker Compose only. |
| D9 | Installs and downloads | Approved: uv, Python 3.13 and benchmark datasets. Cache on X:. |

## What each decision requires

**D1 — Limits**
- These are defaults for newly created keys. Keys can be given explicit overrides.
- The benchmark harness, load tests and integration tests use explicitly configured keys, so the defaults don't distort their results. Record those overrides with each run.

**D2 — OpenRouter credit**
- Stop after the M5 pilot and report the yield at free limits, with what the credit would change. Do not assume the credit, and never make the purchase.

**D3 — Labelling**
- Pre-labels can anchor the reviewer. Measure this: hide the pre-label on a random subset, report the agreement, and document the limitation. Keep the Cohen's kappa gate from ADR-0005.

**D4 — Data policy**
- Never relax a request's `data_policy`, including during fallback.
- Mezvuat will send `any` for verified public-corpus workspaces and `local_only` for private ones, so the Ollama adapter is a required path, not an optional extra.

**D5 and D6 — Licence**
- Add an MIT `LICENSE`. Check that every dependency's licence is compatible.
