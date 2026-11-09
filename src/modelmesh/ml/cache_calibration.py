"""Calibrates the semantic-cache threshold and reports its false-hit rate (ADR-0006).

The pairs below are hand-written and small. They bound the error on this kind of question only: the result is a
sanity check that gives a defensible starting threshold, not a general guarantee.

    python -m modelmesh.ml.cache_calibration [--measure-redis redis://host:port/0]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import sys
import time

from modelmesh.cache.base import CacheRequest
from modelmesh.cache.embeddings import FastEmbedEmbedder
from modelmesh.cache.redis_cache import RedisResponseCache, lexical_guard

TARGET_FALSE_HIT_RATE = 0.005
SEED = 20261109

PARAPHRASES: list[tuple[str, str]] = [
    ("What is the refund policy for online orders?", "What's your refund policy for orders placed online?"),
    ("How do I reset my account password?", "How can I reset the password on my account?"),
    ("How long does standard shipping take?", "How long will standard shipping usually take?"),
    ("What are the main causes of inflation?", "What mainly causes inflation?"),
    ("Explain how photosynthesis works.", "Can you explain how photosynthesis works?"),
    ("What is the difference between TCP and UDP?", "How does TCP differ from UDP?"),
    ("How do vaccines train the immune system?", "In what way do vaccines train the immune system?"),
    ("What does the paper conclude about sleep and memory?", "What is the paper's conclusion about sleep and memory?"),
    ("Summarize the main findings of the report.", "Give a summary of the report's main findings."),
    ("Which methods did the authors use for data collection?", "What methods did the authors use to collect data?"),
    ("How is the sample size justified in this study?", "How does this study justify its sample size?"),
    ("What limitations do the authors acknowledge?", "Which limitations are acknowledged by the authors?"),
    ("What is the capital gains tax treatment of dividends?", "How are dividends treated for capital gains tax?"),
    ("How do I cancel my subscription?", "How can I cancel my subscription?"),
    ("Where can I download my invoices?", "Where do I go to download my invoices?"),
    ("What is the recommended daily intake of vitamin D?", "What's the recommended daily vitamin D intake?"),
    ("How does a transformer model use attention?", "How do transformer models use attention?"),
    ("What are the side effects of ibuprofen?", "What side effects does ibuprofen have?"),
    ("How can I improve the battery life of my laptop?", "How do I improve my laptop's battery life?"),
    ("What is the boiling point of water at sea level?", "At sea level, what is the boiling point of water?"),
    ("Who is responsible for approving the budget?", "Who approves the budget?"),
    ("What is the process for filing an appeal?", "What's the process to file an appeal?"),
    ("How does compound interest work?", "How does compound interest actually work?"),
    ("What are the eligibility criteria for the grant?", "What criteria determine eligibility for the grant?"),
    ("Why did the experiment fail to replicate?", "Why was the experiment not able to be replicated?"),
    ("Translate this sentence into French.", "Please translate this sentence into French."),
    ("What are the benefits of regular exercise?", "What benefits does regular exercise have?"),
    ("How do I connect to the office VPN?", "How do I connect to the VPN at the office?"),
    ("What does the contract say about termination?", "What does the contract state regarding termination?"),
    ("How accurate is the classifier on the test set?", "On the test set, how accurate is the classifier?"),
]

ENTITY_SWAPS: list[tuple[str, str]] = [
    ("What is the capital of France?", "What is the capital of Spain?"),
    ("What is the refund policy for online orders?", "What is the refund policy for in-store orders?"),
    ("How do I reset my account password?", "How do I reset my account email?"),
    ("How long does standard shipping take?", "How long does express shipping take?"),
    ("What are the main causes of inflation?", "What are the main causes of deflation?"),
    ("What is the difference between TCP and UDP?", "What is the difference between HTTP and HTTPS?"),
    ("What does the paper conclude about sleep and memory?", "What does the paper conclude about diet and memory?"),
    ("Summarize the main findings of the first report.", "Summarize the main findings of the second report."),
    ("Which methods did the authors use for data collection?", "Which methods did the authors use for data analysis?"),
    ("What is the capital gains tax treatment of dividends?", "What is the capital gains tax treatment of options?"),
    ("How do I cancel my subscription?", "How do I upgrade my subscription?"),
    ("Where can I download my invoices?", "Where can I download my receipts?"),
    ("What is the recommended daily intake of vitamin D?", "What is the recommended daily intake of vitamin C?"),
    ("What are the side effects of ibuprofen?", "What are the side effects of paracetamol?"),
    ("How can I improve the battery life of my laptop?", "How can I improve the battery life of my phone?"),
    ("What is the boiling point of water at sea level?", "What is the freezing point of water at sea level?"),
    ("Who is responsible for approving the budget?", "Who is responsible for approving the hiring plan?"),
    ("What are the eligibility criteria for the grant?", "What are the eligibility criteria for the loan?"),
    ("What are the benefits of regular exercise?", "What are the risks of regular exercise?"),
    ("How do I connect to the office VPN?", "How do I connect to the office printer?"),
    ("What does the contract say about termination?", "What does the contract say about renewal?"),
    ("How accurate is the classifier on the test set?", "How accurate is the classifier on the training set?"),
    ("Who wrote the first draft?", "Who wrote the final draft?"),
    ("What happened before the merger?", "What happened after the merger?"),
    ("What is the maximum dose for adults?", "What is the maximum dose for children?"),
]


def guard_negatives() -> list[tuple[str, str]]:
    """Near-misses the lexical guard exists for: flipped negation and changed numbers."""
    pairs: list[tuple[str, str]] = []
    for original, _ in PARAPHRASES:
        pairs.append(
            (original, original.replace("What is", "What is not", 1) if "What is" in original else "Not " + original)
        )
    for year in range(2015, 2030):
        pairs.append((f"What changed in the regulation in {year}?", f"What changed in the regulation in {year + 1}?"))
        pairs.append((f"How many employees were hired in Q1 {year}?", f"How many employees were hired in Q2 {year}?"))
    return pairs


def wilson_interval(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 1.0)
    p = hits / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def rate_at(pairs: list[tuple[float, bool]], threshold: float) -> int:
    """How many pairs would be served from the cache at this threshold, after the lexical guard."""
    return sum(1 for similarity, guard_equal in pairs if similarity >= threshold and guard_equal)


def choose_threshold(validation_negatives: list[tuple[float, bool]]) -> float:
    """The lowest threshold whose false-hit rate on validation negatives meets the target."""
    allowed = math.floor(TARGET_FALSE_HIT_RATE * len(validation_negatives))
    for threshold in (round(0.80 + 0.005 * step, 3) for step in range(41)):
        if rate_at(validation_negatives, threshold) <= allowed:
            return threshold
    return 1.0


def score_pairs(embedder: FastEmbedEmbedder, pairs: list[tuple[str, str]]) -> list[tuple[float, bool]]:
    scored = []
    for left, right in pairs:
        similarity = cosine(embedder.embed_sync(left), embedder.embed_sync(right))
        scored.append((similarity, lexical_guard(left) == lexical_guard(right)))
    return scored


def calibrate(model_name: str, cache_dir: str | None) -> dict[str, object]:
    embedder = FastEmbedEmbedder(model_name, cache_dir=cache_dir)
    rng = random.Random(SEED)  # noqa: S311 - sampling, not security
    negatives = [*ENTITY_SWAPS, *guard_negatives()]
    rng.shuffle(negatives)
    positives = list(PARAPHRASES)
    rng.shuffle(positives)
    split_negatives = len(negatives) // 2
    split_positives = len(positives) // 2
    validation_negative = score_pairs(embedder, negatives[:split_negatives])
    test_negative = score_pairs(embedder, negatives[split_negatives:])
    test_positive = score_pairs(embedder, positives[split_positives:])
    threshold = choose_threshold(validation_negative)
    false_hits = rate_at(test_negative, threshold)
    hits = rate_at(test_positive, threshold)
    false_low, false_high = wilson_interval(false_hits, len(test_negative))
    hit_low, hit_high = wilson_interval(hits, len(test_positive))
    started = time.perf_counter()
    timings = []
    for text, _ in PARAPHRASES * 3:
        begin = time.perf_counter()
        embedder.embed_sync(text)
        timings.append((time.perf_counter() - begin) * 1000)
    timings.sort()
    return {
        "threshold": threshold,
        "validation_negatives": len(validation_negative),
        "test_negatives": len(test_negative),
        "test_false_hits": false_hits,
        "test_false_hit_rate_95ci": [round(false_low, 4), round(false_high, 4)],
        "test_positives": len(test_positive),
        "test_hits": hits,
        "test_hit_rate_95ci": [round(hit_low, 4), round(hit_high, 4)],
        "embed_p95_ms_cpu": round(timings[int(0.95 * (len(timings) - 1))], 2),
        "calibration_seconds": round(time.perf_counter() - started, 1),
    }


async def measure_lookup_latency(redis_url: str, *, cache_dir: str | None, model_name: str) -> dict[str, float]:
    """p95 of exact and semantic lookups against real Redis, with the embedding warm."""
    from redis.asyncio import Redis

    redis = Redis.from_url(redis_url, decode_responses=True)
    embedder = FastEmbedEmbedder(model_name, cache_dir=cache_dir)
    await embedder.embed("warm up")
    cache = RedisResponseCache(redis, embedder)
    await cache.ensure_index()
    tenant = "ten_measure"
    payload = {"model": "m/x", "choices": [{"message": {"content": "x" * 500}}]}
    requests = [
        CacheRequest(tenant, "partition", f"question number {i}", "semantic", "general", None, None, False, 300)
        for i in range(100)
    ]
    for request in requests:
        await cache.store(request, payload)

    async def percentile(mode: str) -> float:
        timings: list[float] = []
        for request in requests:
            probe = CacheRequest(
                tenant,
                "partition",
                request.final_message + (" " if mode == "semantic" else ""),
                mode,
                "general",
                None,
                None,
                False,
                300,
            )
            begin = time.perf_counter()
            await cache.lookup(probe, eligible_model_ids={"m/x"})
            timings.append((time.perf_counter() - begin) * 1000)
        return statistics.quantiles(timings, n=20)[18]

    results = {"exact_lookup_p95_ms": await percentile("exact"), "semantic_lookup_p95_ms": await percentile("semantic")}
    await cache.invalidate(tenant, None, None)
    await redis.aclose()
    return {name: round(value, 2) for name, value in results.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--cache-dir")
    parser.add_argument("--measure-redis")
    args = parser.parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    report: dict[str, object] = calibrate(args.model, args.cache_dir)
    if args.measure_redis:
        report |= asyncio.run(
            measure_lookup_latency(args.measure_redis, cache_dir=args.cache_dir, model_name=args.model)
        )
    print(json.dumps(report, indent=1))  # noqa: T201 - command-line output


if __name__ == "__main__":
    main()
