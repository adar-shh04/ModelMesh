"""Renders the M6 preliminary report from ``docs/milestones/M6_P0_results.json``. Every number comes from there.

uv run python -m modelmesh.ml.evaluation.report
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "docs" / "milestones" / "M6_P0_results.json"
REPORT = ROOT / "docs" / "milestones" / "M6_P0_PRELIMINARY_REPORT.md"

# Older result files only carry platform.processor(); this name was read with Get-CimInstance Win32_Processor.
CPU_NAME = {"Intel64 Family 6 Model 141 Stepping 1, GenuineIntel": "11th Gen Intel(R) Core(TM) i5-11400H @ 2.70GHz"}

ROUTER_ORDER = [
    ("learned", "Learned (selected on validation)"),
    ("knn", "L1 kNN"),
    ("lightgbm", "L2 LightGBM"),
    ("logreg", "L3 logistic regression"),
    ("b0b", "B0b per-task lookup"),
    ("b0b_validation_only", "B0b, validation-only lookup"),
    ("b0a", "B0a static rules"),
    ("best_single", "Best-Single"),
    ("cheapest", "Cheapest-Single"),
    ("random", "Random (expected)"),
    ("zero_router", "Zero router (hull of single models)"),
    ("oracle", "Oracle"),
]


def ci(entry: Mapping[str, float] | None, digits: int = 3, scale: float = 1.0) -> str:
    if entry is None:
        return "n/a"
    point = entry["point"]
    if point is None or (isinstance(point, float) and math.isnan(point)):
        return "not reached"
    low, high = entry["low"] * scale, entry["high"] * scale
    return f"{point * scale:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def pct(entry: Mapping[str, float] | None) -> str:
    if entry is None or entry["point"] is None or math.isnan(entry["point"]):
        return "not reached"
    return f"{entry['point']:.0%} [{entry['low']:.0%}, {entry['high']:.0%}]"


def micro(value: float) -> str:
    return f"{value * 1e6:.1f}"


def aiq_table(block: Mapping[str, Any], label: str = "AIQ [95% CI]") -> list[str]:
    intervals = block["intervals"]
    ops = block.get("operating_points", {})
    lines = [f"| Router | {label} | Point at Q_min 0.6: mean cost (µUSD) | acceptance |", "|---|---|---|---|"]
    for key, name in ROUTER_ORDER:
        if f"aiq/{key}" not in intervals:
            continue
        op = ops.get(key)
        cost = micro(op["mean_cost_usd"]) if op else "-"
        acc = f"{op['acceptance']:.3f}" if op else "-"
        lines.append(f"| {name} | {ci(intervals[f'aiq/{key}'])} | {cost} | {acc} |")
    return lines


def diff_table(block: Mapping[str, Any]) -> list[str]:
    intervals = block["intervals"]
    lines = ["| Paired AIQ difference | Point [95% CI] |", "|---|---|"]
    for key in sorted(k for k in intervals if k.startswith("diff/")):
        lines.append(f"| {key[5:].replace('-', ' minus ')} | {ci(intervals[key])} |")
    return lines


def cost_table(block: Mapping[str, Any]) -> list[str]:
    intervals = block["intervals"]
    lines = [
        "| Router | Saving vs Best-Single at 95% of its quality | at 100% | Quality regret vs Oracle at Best-Single "
        "cost | Cost regret vs Oracle at 95% of Best-Single quality (µUSD) |",
        "|---|---|---|---|---|",
    ]
    for key in ("learned", "b0b", "b0a"):
        lines.append(
            f"| {dict(ROUTER_ORDER)[key]} | {pct(intervals.get(f'saving@0.95/{key}'))} | "
            f"{pct(intervals.get(f'saving@1.0/{key}'))} | {ci(intervals.get(f'quality_regret@bs_cost/{key}'))} | "
            f"{ci(intervals.get(f'cost_regret@0.95bs/{key}'), 1, 1e6)} |"
        )
    return lines


def coverage_table(coverage: Mapping[str, Any]) -> list[str]:
    tasks = sorted({t for per in coverage.values() for t in per})
    lines = ["| Model | " + " | ".join(tasks) + " |", "|---|" + "---|" * len(tasks)]
    for model, per in coverage.items():
        cells = [f"{per[t]['observed']}/{per[t]['items']}" if t in per else "-" for t in tasks]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return lines


def render(results: Mapping[str, Any]) -> str:
    primary = results["variants"]["primary"]
    f1 = results["variants"]["with_f1_only_synthesis"]
    dropped = results["variants"]["failures_dropped"]
    crit = results["criterion"]["primary"]
    held = primary["held_out_query"]
    lodo = primary["workload_shift_lodo"]
    overhead = primary["overhead"]
    cal = primary["calibration"][primary["selected"]]["_all"]
    machine = dict(overhead["machine"])
    machine["processor"] = machine.get("processor_name") or CPU_NAME.get(machine["processor"], machine["processor"])
    verdict = "passes" if crit["passes"] else "does not pass"
    out: list[str] = []
    add = out.append
    add("# M6 router evaluation: PRELIMINARY, P0 pilot data")
    add("")
    add(
        "> **Preliminary.** Built from the P0 pilot (`pilot-2026-11-11`), not from P1. The test sets hold "
        f"{held['n_items']} (held-out queries) and {lodo['n_items']} (leave-one-dataset-out) prompts, four models "
        "and only automatically labelled task types. Treat every number as a test of the pipeline and a first "
        "look, not as evidence about the router. The P0 split and the model selection made on it **must not be "
        "reused for P1**: P1 freezes its own test manifest before any model selection."
    )
    add("")
    add(f"Generated by `python -m modelmesh.ml.evaluation.report` from `docs/milestones/M6_P0_results.json` "
        f"(itself from `python -m modelmesh.ml.evaluation.p0`), feature version {results.get('feature_version', '1')}. Observations sha256 "
        f"`{results['observations_sha256'][:16]}…`, dataset manifest `{results['dataset_manifest_sha256'][:16]}…`, "
        f"split manifest `{primary['split_manifest_sha256'][:16]}…`, seed {results['seed']}, "
        f"{results['bootstrap_resamples']} bootstrap resamples. MLflow run (primary): `{primary['mlflow_run_id']}`.")  # fmt: skip
    add("")
    add("## Summary")
    add("")
    add(f"- **The ADR-0004 success criterion {verdict} on P0.** The selected learned router "
        f"({primary['selected']}, chosen on validation) against B0b on held-out-query test AIQ: "
        f"{ci(crit['held_out_query_diff'])} (needs a CI above 0). On the leave-one-dataset-out workload shift: "
        f"{ci(crit['workload_shift_diff'])} (needs a lower bound above -0.01).")  # fmt: skip
    add(f"- Held-out queries: learned AIQ {ci(held['intervals']['aiq/learned'])}, B0b "
        f"{ci(held['intervals']['aiq/b0b'])}, B0a {ci(held['intervals']['aiq/b0a'])}, Best-Single "
        f"{ci(held['intervals']['aiq/best_single'])}, Oracle {ci(held['intervals']['aiq/oracle'])}.")  # fmt: skip
    add(f"- Calibration of the selected router on test: ECE {cal['ece_calibrated']:.3f} after isotonic "
        f"({cal['ece_raw']:.3f} raw) over {cal['n_cells']} cells. The promotion gate is 0.05, so this router "
        "could not be promoted even if the criterion had passed.")  # fmt: skip
    ex, inc = overhead["excluding_embedding_ms"], overhead["including_embedding_ms"]
    add(f"- Router overhead p95: {ex['p95']:.2f} ms excluding embedding (target 10 ms, "
        f"{'met' if ex['p95'] < 10 else 'missed'}), {inc['p95']:.1f} ms including embedding (target 25 ms, "
        f"{'met' if inc['p95'] < 25 else 'missed'}), on {machine['processor']}, {machine['logical_cpus']} logical "
        f"CPUs, {machine.get('ram_gb', '?')} GB RAM, {machine['platform']}, Python {machine['python']}.")  # fmt: skip
    add("- Rules (B0b) stays the production default. The P0 artifact is preliminary; registered with "
        "`preliminary: true` (the default when the flag is absent), the promotion gate refuses it.")  # fmt: skip
    add("")
    add("## Reproduce")
    add("")
    add("```bash")
    add("uv sync --group ml   # caches on X: (UV_CACHE_DIR, HF_HOME, FASTEMBED_CACHE_PATH)")
    add(
        f"uv run python -m modelmesh.ml.evaluation.p0       # {results['elapsed_s'] / 60:.0f} min here; no provider calls"
    )
    add("uv run python -m modelmesh.ml.evaluation.report")
    add("```")
    add("")
    add("Inputs: `data/benchmark/pilot/items.jsonl` (checked against the committed `data/manifests/pilot.json`) and "
        "`data/benchmark/pilot/pilot-2026-11-11/observations.jsonl`, both gitignored. Later benchmark lanes still "
        "append to that file, so to reproduce this report point `MODELMESH_P0_OBSERVATIONS` at a copy of its first "
        f"bytes whose sha256 is `{results['observations_sha256'][:16]}…` (the P0 rows). Embeddings are cached in "
        "`data/benchmark/pilot/derived/`. MLflow uses a local SQLite store, `X:/modelmesh-cache/mlflow/mlflow.db` "
        "(the file-store backend is in maintenance mode in MLflow 3.18 and refuses to start without an override).")  # fmt: skip
    add("")
    add("## Data and labels")
    add("")
    add("Observations per model and task type (any recorded outcome, success or failure, out of the pilot items):")
    add("")
    out.extend(coverage_table(results["coverage"]))
    add("")
    add("Decisions:")
    add("")
    add(f"- **Pool: {', '.join(results['pool'])}.** These four are the models with an observation on every "
        "auto-labelled item they are evaluated on. gpt-oss-20b (95 of 180 auto-labelled items), gemini-3.6-flash and "
        "both OpenRouter models (the 40 `query_classification` items only) are left out of routing and reported "
        "separately below. A router choosing among models it has no labels for cannot be evaluated without imputing "
        "labels, which this report never does.")  # fmt: skip
    add("- **Items: complete cells only.** An item is used only when all pool models have a recorded outcome for it. "
        f"That leaves {primary['items']} items in the primary analysis (`jsonschemabench` drops out: gpt-oss-120b "
        "never reached it).")  # fmt: skip
    add("- **Labels: automatic only.** `query_rewriting`, `research_planning` and `final_response` need the judge "
        "pipeline, which has not run, so they carry no label and are excluded. `evidence_synthesis` needs token "
        "F1 ≥ 0.5 **and** judge faithfulness ≥ 4 under ADR-0005; the judge half is missing, so the primary analysis "
        "excludes it and a sensitivity variant adds it with the F1 half alone (an optimistic label).")  # fmt: skip
    add("- **Failures count as not accepted** (502 all attempts failed, 504 deadline, schema failure): the client "
        "got no acceptable answer from that call. A variant that drops failed cells instead is reported. Failed "
        "cells have no recorded cost; they are charged the median successful cost of that model and task type. "
        "That is the only imputed quantity, and it is a cost, not a label.")  # fmt: skip
    add("- B0b's lookup uses train plus validation acceptance (ADR-0004 says validation). P0 validation has 4 to 12 "
        "items per task type, and giving B0b the same data the learned router sees makes the comparison stricter. "
        "The validation-only lookup is reported alongside.")  # fmt: skip
    add("- Costs are the gateway's estimated list-price cost (actual spend was $0 on free tiers). Both Ollama models "
        "cost $0, so Cheapest-Single is the better of the two on selection data.")  # fmt: skip
    add("")
    add("## Splits")
    add("")
    counts = primary["split_counts"]
    add(f"- Held-out queries: grouped 60/20/20 within each (task type, source): train {counts['train']}, validation "
        f"{counts['validation']}, test {counts['test']}. Groups join items from one source question and prompts with "
        f"embedding cosine above 0.95. Asserted: {primary['leakage']['groups_crossing']} groups cross splits; the "
        f"highest train/validation/test cross-split cosine is {primary['leakage']['max_cross_split_cosine']:.4f}.")  # fmt: skip
    add("- The split manifest (item ids, content hashes, split) is committed as "
        "`data/manifests/p0_router_split.json`. It is P0-only.")  # fmt: skip
    add(f"- Workload shift (a), leave-one-dataset-out: {', '.join(lodo['folds'])}. Each fold refits the selected "
        "configuration on the remaining items (75/25 train/validation, grouped) and predicts the held-out source. "
        "The folds are pooled. Hyper-parameters are not re-selected.")  # fmt: skip
    add("- Workload shift (b), mixture shift: the held-out test set re-weighted to fixed task-type shares "
        "(Mezvuat-like: verification and assessment heavy; chat-like: general heavy). Shares were fixed in code "
        "before results were computed.")  # fmt: skip
    add("- The test set was evaluated by the same code twice: one smoke run with 20 resamples to catch crashes, then "
        "the reported run. Between them, only the overhead breakdown and a numerically equivalent isotonic "
        "implementation changed; no modelling or selection choice changed after the smoke run.")  # fmt: skip
    add("")
    add("## Model selection (validation only)")
    add("")
    add("| Family | Selected hyper-parameters | Validation AIQ (uncalibrated) | Validation log loss |")
    add("|---|---|---|---|")
    for algorithm, sel in primary["selection"].items():
        add(f"| {algorithm} | `{json.dumps(sel['hp'])}` | {sel['validation_aiq_raw']:.3f} | "
            f"{sel['validation_logloss']:.3f} |")  # fmt: skip
    add("")
    add(f"Selected: **{primary['selected']}**. Each family is calibrated per task type with isotonic regression on "
        "validation. The learned router falls back to the B0b lookup when the nearest training neighbour is below "
        "the 5th percentile of validation similarity, as the runtime does.")  # fmt: skip
    add("")
    add("## Held-out queries")
    add("")
    add(f"References: Cheapest `{primary['references']['cheapest']}`, most expensive "
        f"`{primary['references']['priciest']}` (upper end of the AIQ cost range), Best-Single "
        f"`{primary['references']['best_single']}`.")  # fmt: skip
    add("")
    out.extend(aiq_table(held))
    add("")
    out.extend(diff_table(held))
    add("")
    out.extend(cost_table(held))
    add("")
    add("Single models on the test items:")
    add("")
    add("| Model | Mean cost (µUSD) | Acceptance |")
    add("|---|---|---|")
    for model, s in held["singles"].items():
        add(f"| {model} | {micro(s['mean_cost_usd'])} | {s['acceptance']:.3f} |")
    add("")
    share = held["share_to_priciest_at_default_floor"]
    add("Share of calls to the most expensive model at Q_min 0.6: "
        + ", ".join(f"{k} {v:.0%}" for k, v in share.items()) + ".")  # fmt: skip
    add("")
    add("## Workload shift")
    add("")
    add(f"### Leave one dataset out ({lodo['n_items']} items)")
    add("")
    out.extend(aiq_table(lodo))
    add("")
    out.extend(diff_table(lodo))
    add("")
    out.extend(cost_table(lodo))
    add("")
    for mix, block in primary["mixture_shift"].items():
        add(f"### Mixture shift: {mix}")
        add("")
        out.extend(aiq_table(block, "Weighted AIQ [95% CI]"))
        add("")
        add(f"Learned minus B0b: {ci(block['intervals']['diff/learned-b0b'])}.")
        add("")
    add("## Constraint fidelity (selected router, held-out test)")
    add("")
    add("`cost_optimized` with an explicit `min_quality` (hard constraint; a request with no eligible model is "
        "rejected):")  # fmt: skip
    add("")
    add("| Q_min | Realised acceptance of served requests | Rejected | Below Q_min? |")
    add("|---|---|---|---|")
    for q_min, entry in primary["constraint_fidelity"]["quality"].items():
        add(f"| {q_min} | {ci(entry['realised_acceptance'])} | {pct(entry['rejected_fraction'])} | "
            f"{'yes' if entry['violated'] else 'no'} |")  # fmt: skip
    add("")
    add("With `max_latency_ms` = L_max (filter on predicted p90) and the default 0.6 floor:")
    add("")
    add("| L_max (ms) | Served | Realised latency above L_max |")
    add("|---|---|---|")
    for l_max, entry in primary["constraint_fidelity"]["latency"].items():
        add(f"| {l_max} | {entry['served']} | {pct(entry['exceed_fraction'])} |")
    add("")
    add("## Calibration (held-out test)")
    add("")
    add("| Family | Task type | Cells | ECE raw | ECE isotonic | Brier raw | Brier isotonic |")
    add("|---|---|---|---|---|---|---|")
    for algorithm, per in primary["calibration"].items():
        for task, entry in per.items():
            add(f"| {algorithm} | {task} | {entry['n_cells']} | {entry['ece_raw']:.3f} | {entry['ece_calibrated']:.3f} "
                f"| {entry['brier_raw']:.3f} | {entry['brier_calibrated']:.3f} |")  # fmt: skip
    add("")
    add("With 4 to 12 validation items per task type, isotonic tables are fitted on very few points; where the "
        "isotonic column is worse than raw, that is the expected small-sample overfit. Conformal intervals need "
        "30 calibration points per task type and are `null` below that.")  # fmt: skip
    add("")
    add("## Latency and cost predictors (held-out test)")
    add("")
    add("| Model | Latency cells | Median abs log error of p50 | p90 coverage | Predicted mean cost (µUSD) | "
        "Realised (µUSD) |")  # fmt: skip
    add("|---|---|---|---|---|---|")
    for model, entry in primary["predictors"].items():
        err = entry["latency_abs_log_error_median"]
        cov = entry["latency_p90_coverage"]
        add(f"| {model} | {entry['latency_cells']} | {'-' if err is None else f'{err:.2f}'} | "
            f"{'-' if cov is None else f'{cov:.0%}'} | {micro(entry['cost_mean_predicted_usd'])} | "
            f"{micro(entry['cost_mean_realised_usd'])} |")  # fmt: skip
    add("")
    add("p90 coverage counts timeouts as misses. The latency model is ADR-0004's log-linear fit on train telemetry.")
    add("")
    rob = primary["robustness"]
    add("## Robustness")
    add("")
    add(f"- {rob['perturbation']}; {rob['items']} test prompts, {rob['items_changed']:.0%} actually changed.")
    add(f"- Routing flip rate at Q_min 0.6: learned {pct(rob['learned_flip_rate'])}; B0a {rob['b0a_flip_rate']:.0%}; "
        "B0b 0% by construction (no per-query input).")  # fmt: skip
    add(f"- Paraphrase perturbation: {rob['paraphrase']}. ADR-0004 asks for 100 prompts; P0's test set is smaller.")
    add("")
    add("## Router overhead")
    add("")
    add(f"Measured over {overhead['calls']} routing calls through the online path (`ml.features`, the loaded "
        "artifact via `routing.learned`, the Estimator and the policy solver), warm.")  # fmt: skip
    add("")
    add("| | p50 (ms) | p95 (ms) | Target p95 |")
    add("|---|---|---|---|")
    add(f"| Excluding embedding | {ex['p50']:.2f} | {ex['p95']:.2f} | 10 ms |")
    add(f"| Including embedding ({overhead['embedding']}) | {inc['p50']:.1f} | {inc['p95']:.1f} | 25 ms |")
    add("")
    add("Including embedding, by prompt length before any cut: "
        + "; ".join(f"{k} chars: p95 {v['p95_ms']:.1f} ms ({v['calls']} calls)"
                    for k, v in overhead["including_embedding_by_text_chars"].items()) + ".")  # fmt: skip
    add("")
    add(f"Machine: {machine['processor']}, {machine['logical_cpus']} logical CPUs, {machine.get('ram_gb', '?')} GB "
        f"RAM, {machine['platform']}, Python {machine['python']}. Other work was running on the machine.")  # fmt: skip
    add("")
    add("## Sensitivity analyses")
    add("")
    fcrit = results["criterion"]["with_f1_only_synthesis"]
    add(f"**With F1-only evidence_synthesis labels** ({f1['items']} items, selected {f1['selected']}): learned "
        f"minus B0b on held-out queries {ci(fcrit['held_out_query_diff'])}; on leave-one-dataset-out (adds "
        f"QASPER and MuSiQue synthesis folds) {ci(fcrit['workload_shift_diff'])}; the criterion "
        f"{'passes' if fcrit['passes'] else 'does not pass'}.")  # fmt: skip
    add("")
    out.extend(aiq_table(f1["held_out_query"]))
    add("")
    add(f"**Failed cells dropped instead of rejected** ({dropped['items']} items remain, test "
        f"{dropped['split_counts']['test']}, selected {dropped['selected']}). Held-out queries only:")  # fmt: skip
    add("")
    out.extend(aiq_table(dropped["held_out_query"]))
    add("")
    add(f"Learned minus B0b: {ci(dropped['held_out_query']['intervals']['diff/learned-b0b'])}.")
    add("")
    sparse = results["sparse_models_query_classification"]
    add(f"## Models with sparse coverage (descriptive, {sparse['items']} query_classification items)")
    add("")
    add("All eight models have an outcome on these items, so they can be compared here, and only here. No router "
        "is trained on them.")  # fmt: skip
    add("")
    add("| Model | Acceptance | Share of Oracle choices |")
    add("|---|---|---|")
    for model, value in sparse["acceptance"].items():
        add(f"| {model} | {value:.3f} | {sparse['oracle_share'][model]:.0%} |")
    add("")
    add(f"Oracle acceptance on these items: {sparse['oracle_acceptance']:.3f}.")
    add("")
    add("## ADR-0004 criterion")
    add("")
    add("| Condition | Required | Observed | Met |")
    add("|---|---|---|---|")
    add(f"| Learned beats B0b on held-out-query AIQ | paired 95% CI above 0 | {ci(crit['held_out_query_diff'])} | "
        f"{'yes' if crit['held_out_query_ci_excludes_zero_and_positive'] else 'no'} |")  # fmt: skip
    add(f"| Not worse than B0b on workload shift | CI lower bound above -0.01 | {ci(crit['workload_shift_diff'])} | "
        f"{'yes' if crit['workload_shift_non_inferior_at_minus_0.01'] else 'no'} |")  # fmt: skip
    add("")
    add(f"Result on P0: the criterion **{verdict}**. As ADR-0004 requires, B0b remains the production default and the "
        "learned router can only run in shadow. With test sets this small the intervals are wide in both "
        "directions, so P0 can neither confirm nor rule out a real effect of the size ADR-0004 cares about.")  # fmt: skip
    add("")
    return "\n".join(out)


def main() -> None:
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    manifest = results["variants"]["primary"].get("split_manifest")
    if manifest is not None:
        target = ROOT / "data" / "manifests" / "p0_router_split.json"
        target.write_text(json.dumps(manifest, indent=1) + chr(10), encoding="utf-8", newline=chr(10))
    body = render(results)
    tail = (ROOT / "docs" / "milestones" / "M6_P0_NOTES.md").read_text(encoding="utf-8")
    REPORT.write_text(body + "\n" + tail, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
