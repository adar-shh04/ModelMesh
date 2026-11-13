# M5 P0 pilot report

Run: `pilot-2026-11-11`, collected 2026-11-10 to 2026-11-13 (UTC) through the public API (explicit model, cache off, fallback none). Raw observations are in `data/benchmark/pilot/pilot-2026-11-11/` (gitignored). This report gives only aggregates.

Pilot set: 280 prompts across 8 task types (general 80; query_classification, evidence_synthesis and citation_verification 40 each; query_rewriting, research_planning, evidence_assessment and final_response 20 each). Benchmark key overrides are in `key_overrides.json`, including the product owner's $5/day estimated-cost cap.

## Coverage and quality

The run is **incomplete**: 1,381 of the 2,240 model × prompt cells succeeded (62%). Failures are recorded, not dropped.

Figures updated 2026-11-13 UTC after a resume on the Gemini, OpenRouter and Ollama lanes only (Groq was left out to keep its shared daily quota for Mezvuat). That session added 73 successes: 23 for gemini-3.6-flash and 25 for each OpenRouter model. Flash-lite and both Ollama models had no pending cells left, so the resume sent them nothing.

| Model | OK / 280 | Failed | Pending | Auto score* | p50 latency† | p95 latency† | Main errors |
|---|---|---|---|---|---|---|---|
| gemini/gemini-3.1-flash-lite | 250 | 30 | 0 | 0.81 | 2.1 s | 21.1 s | 27 × 502 all attempts failed, 2 × 504, 1 × 500 |
| gemini/gemini-3.6-flash | 42 | 10 | 228 | 0.97 (n=30) | 2.1 s | 25.8 s | 255 deferrals, upstream capacity; 5 × 504, 5 × 502 |
| groq/openai/gpt-oss-120b | 253 | 0 | 27 | 0.78 | 0.7 s | 1.7 s | none |
| groq/openai/gpt-oss-20b | 174 | 1 | 105 | 0.77 | 0.7 s | 59.9 s | 80 deferrals, upstream capacity; 1 × 500 |
| ollama/llama3.2:3b | 279 | 1 | 0 | 0.59 | 0.6 s | 7.9 s | 1 schema failure |
| ollama/qwen3:4b | 249 | 31 | 0 | 0.63 | 51.5 s | 106.9 s | 21 schema failures, 10 timeouts |
| openrouter/liquid/lfm-3-1.7b:free | 65 | 3 | 212 | 0.92 (n=37) | 2.7 s | 7.4 s | 3 schema failures, 2 deferrals |
| openrouter/nvidia/nemotron-3.5-super-vl:free | 69 | 0 | 211 | 0.93 (n=40) | 2.5 s | 22.8 s | 1 deferral |

\* Mean automatic score over the auto-scored task types each model reached. **The scores can't be compared across rows**, because models reached different items in run order. gemini-3.6-flash covers all 40 `query_classification` items (30 auto-scored successes) and 12 `query_rewriting` items. The OpenRouter models cover `query_classification`, all 20 `query_rewriting` items and 8–9 `research_planning` items. query_rewriting, research_planning and final_response need the judge pipeline, which has not run yet, so the new OpenRouter and gemini-3.6-flash successes did not change their automatic scores.

† Gateway-measured latency (`modelmesh.latency_ms.total`) over successful calls. The first version of this report used the runner's client-side latency, which also counts the runner's own pacing waits and capacity back-off; that is why it showed about 7.6 s for both Groq models (8 RPM pacing is a 7.5 s gap). Client-side p50 and p95 for comparison: flash-lite 9.5 s and 28.1 s, gemini-3.6-flash 6.7 s and 31.7 s, gpt-oss-120b 7.6 s and 9.0 s, gpt-oss-20b 7.6 s and 112.7 s, lfm 6.1 s and 11.4 s, nemotron 6.4 s and 23.7 s. Ollama is unpaced, so the two measures agree there.

Two points stand out:
- evidence_synthesis scores low for every model (0.30–0.48). The token-overlap metric probably understates quality there, and the judge should re-score it.
- The p50 latency of qwen3:4b is 51.5 s, because it is a thinking model on a 4 GB GPU.

## Free-tier yield per day

Successful calls per lane and UTC day, from the observations:

| Lane | Per day (measured) | Limit that applied |
|---|---|---|
| Groq (both models) | about 200 on the best day (206 on 2026-11-12) | daily quota (shared with Mezvuat pre-labelling) |
| Gemini flash-lite | 84–85 on full days (85 on 11-12, 84 on 11-13) | free-tier daily quota |
| Gemini 3.6 flash | 14, 0, 5 and 23 on the four days it ran (mean 10.5) | upstream capacity on the free tier (503, deferred) |
| OpenRouter (both :free models) | 50 (25 + 25 on 11-12 and on 11-13) | 50 requests/day account cap, shared across all :free models |
| Ollama | 280+ | none, only local GPU time |

On 2026-11-13 the gemini-3.6-flash lane stopped after 3 consecutive upstream-capacity deferrals and the OpenRouter lane stopped at its 50-request budget; neither was retried.

Estimated list-price cost of the successful calls in the whole pilot was $0.12 ($0.004 of it from the 2026-11-13 resume). Actual spend was $0 (free tiers only), and the $5/day cap was never reached.

## Projection

Time to finish each piece at the measured rates. Gemini 3.6 flash is the bottleneck for "rest of pilot" without the credit; OpenRouter is the bottleneck for P1 without it.

| | Without OpenRouter credit | With $10 credit (1,000 free requests/day) |
|---|---|---|
| Rest of pilot, OpenRouter (423 calls) | about 9 days | under 1 day |
| Rest of pilot, Gemini 3.6 flash (228 calls) | about 22 days at the measured mean (10 at the best day's 23) | same (the credit doesn't affect Gemini) |
| P1, OpenRouter (2 models × 1,200 prompts) | about 48 days | about 3 days |
| P1, Groq (2 × 1,200) | about 12 days | same |
| P1, Gemini flash-lite (1,200) | about 14 days | same |

OpenRouter's 1,000/day cap applies after at least $10 of credits have been bought ([pricing](https://openrouter.ai/pricing)). Failed requests count toward the cap, and the 20 RPM limit stays the same.

## What needs a decision

1. **OpenRouter credit (D2).** Without the credit, P1 could keep the two OpenRouter models on the 280-item pilot set only and run the other six on all 1,200 items. With it, all eight can cover the full set.
2. **Gemini 3.6 flash.** The free tier yields too little for P1 with or without the credit. Recommendation: keep it only on the pilot set, or drop it from P1.
