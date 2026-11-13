# M5 P0 pilot report

Run: `pilot-2026-11-11`, collected 2026-11-10 to 2026-11-13 through the public API (explicit model, cache off, fallback none). Raw observations are in `data/benchmark/pilot/pilot-2026-11-11/` (gitignored). This report gives only aggregates.

Pilot set: 280 prompts across 8 task types (general 80; query_classification, evidence_synthesis and citation_verification 40 each; query_rewriting, research_planning, evidence_assessment and final_response 20 each). Benchmark key overrides are in `key_overrides.json`, including the product owner's $5/day estimated-cost cap.

## Coverage and quality

The run is **incomplete**: 1,308 of the 2,240 model × prompt cells succeeded (58%). Failures are recorded, not dropped.

| Model | OK / 280 | Auto score* | p50 latency | p95 latency | Main errors |
|---|---|---|---|---|---|
| gemini/gemini-3.1-flash-lite | 250 | 0.81 | 9.5 s | 27.5 s | 27 × 502 all attempts failed |
| gemini/gemini-3.6-flash | 19 | 1.00 (n=19) | 6.0 s | 40.7 s | 252 deferrals, upstream capacity |
| groq/openai/gpt-oss-120b | 253 | 0.78 | 7.6 s | 9.0 s | none |
| groq/openai/gpt-oss-20b | 174 | 0.77 | 7.6 s | 112.7 s | 80 deferrals, upstream capacity |
| ollama/llama3.2:3b | 279 | 0.59 | 0.7 s | 7.9 s | 1 schema failure |
| ollama/qwen3:4b | 249 | 0.63 | 52.4 s | 106.7 s | 21 schema failures, 10 timeouts |
| openrouter/liquid/lfm-3-1.7b:free | 40 | 0.92 (n=37) | 6.7 s | 16.6 s | 3 schema failures |
| openrouter/nvidia/nemotron-3.5-super-vl:free | 44 | 0.93 (n=40) | 5.2 s | 8.9 s | none |

\* Mean automatic score over the auto-scored task types each model reached. **The scores can't be compared across rows**, because models reached different items. The OpenRouter and gemini-3.6-flash rows cover only `query_classification`, the first items in run order. Three task types (query_rewriting, research_planning, final_response) need the judge pipeline, which has not run yet.

Two points stand out:
- evidence_synthesis scores low for every model (0.30–0.48). The token-overlap metric probably understates quality there, and the judge should re-score it.
- The p50 latency of qwen3:4b is 52 s, because it is a thinking model on a 4 GB GPU.

## Free-tier yield per day

These are successful calls per model on the best full day (2026-11-12):

| Lane | Per day | Limit that applied |
|---|---|---|
| Groq (both models) | about 200 | daily quota (shared with Mezvuat pre-labelling) |
| Gemini flash-lite | about 85 | free-tier daily quota |
| Gemini 3.6 flash | about 5–15 | upstream capacity on the free tier |
| OpenRouter (both :free models) | 50 | 50 requests/day account cap, shared across all :free models |
| Ollama | 280+ | none, only local GPU time |

Estimated list-price cost for the whole pilot was $0.11. Actual spend was $0 (free tiers only), and the $5/day cap was never reached.

## Projection

Time to finish each piece at the measured rates. Gemini 3.6 flash is the bottleneck for "rest of pilot" without the credit; OpenRouter is the bottleneck for P1 without it.

| | Without OpenRouter credit | With $10 credit (1,000 free requests/day) |
|---|---|---|
| Rest of pilot, OpenRouter (476 calls) | about 10 days | under 1 day |
| Rest of pilot, Gemini 3.6 flash (261 calls) | about 20–25 days | same (the credit doesn't affect Gemini) |
| P1, OpenRouter (2 models × 1,200 prompts) | about 48 days | about 3 days |
| P1, Groq (2 × 1,200) | about 12 days | same |
| P1, Gemini flash-lite (1,200) | about 14 days | same |

OpenRouter's 1,000/day cap applies after at least $10 of credits have been bought ([pricing](https://openrouter.ai/pricing)). Failed requests count toward the cap, and the 20 RPM limit stays the same.

## What needs a decision

1. **OpenRouter credit (D2).** Without the credit, P1 could keep the two OpenRouter models on the 280-item pilot set only and run the other six on all 1,200 items. With it, all eight can cover the full set.
2. **Gemini 3.6 flash.** The free tier yields too little for P1 with or without the credit. Recommendation: keep it only on the pilot set, or drop it from P1.
