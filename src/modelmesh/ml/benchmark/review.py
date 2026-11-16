"""Human confirmation of labels (ADR-0005, DECISIONS D3).

An LLM or automatic pre-label can be shown and the reviewer confirms or corrects it. The pre-label is **hidden by
default**: Mezvuat's labelling (2026-11-14) found that a shown pre-label anchors the reviewer heavily. It is shown on
a random ``SHOWN_FRACTION`` of rows only, so that the anchoring can still be measured (D3). Model identity is never
shown.
"""

from __future__ import annotations

import html
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from modelmesh.ml.benchmark.items import BenchItem
from modelmesh.ml.benchmark.metrics import cohen_kappa
from modelmesh.ml.benchmark.scoring import combined_acceptance, needs_judge, score_automatic

SHOWN_FRACTION = 0.3
JUDGE_SAMPLE = 150
AUTO_SAMPLE = 50


def build_review_set(
    items: Sequence[BenchItem],
    observations: Sequence[dict[str, Any]],
    judgements: Sequence[dict[str, Any]],
    *,
    seed: int,
    judge_sample: int = JUDGE_SAMPLE,
    auto_sample: int = AUTO_SAMPLE,
    shown_fraction: float = SHOWN_FRACTION,
) -> list[dict[str, Any]]:
    """A stratified sample of outputs with their pre-labels, ready for the review page."""
    rng = random.Random(seed)  # noqa: S311 - sampling, not security
    by_id = {item.item_id: item for item in items}
    scores = {(row["item_id"], row["model"]): row["score"] for row in judgements}
    judged: dict[str, list[dict[str, Any]]] = {}
    automatic: list[dict[str, Any]] = []
    for observation in observations:
        item = by_id.get(observation["item_id"])
        if item is None or not observation["ok"]:
            continue
        if needs_judge(item):
            if (item.item_id, observation["model"]) in scores:
                judged.setdefault(item.task_type, []).append(observation)
        else:
            automatic.append(observation)

    selected: list[dict[str, Any]] = []
    for rows in judged.values():
        quota = max(1, round(judge_sample * len(rows) / max(1, sum(len(group) for group in judged.values()))))
        selected.extend(rng.sample(rows, min(quota, len(rows))))
    selected.extend(rng.sample(automatic, min(auto_sample, len(automatic))))
    rng.shuffle(selected)

    review: list[dict[str, Any]] = []
    for position, observation in enumerate(selected):
        item = by_id[observation["item_id"]]
        automatic_score = score_automatic(item, observation["output"] or "")
        prelabel = combined_acceptance(item, automatic_score, scores.get((item.item_id, observation["model"])))
        review.append(
            {
                "review_id": position,
                "item_id": item.item_id,
                "model": observation["model"],
                "task_type": item.task_type,
                "prompt": "\n\n".join(m["content"] for m in item.messages),
                "reference": item.reference.get("gold") or item.reference.get("golds") or item.reference.get("label"),
                "output": observation["output"],
                "prelabel": prelabel,
                "prelabel_shown": rng.random() < shown_fraction,
            }
        )
    return review


def agreement_report(review: Sequence[dict[str, Any]], labels: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Cohen's kappa between pre-labels and the reviewer, overall, per task, and with the pre-label hidden or shown."""
    by_id = {row["review_id"]: row for row in review}
    pairs = [(by_id[label["review_id"]], bool(label["accepted"])) for label in labels if label["review_id"] in by_id]
    usable = [(row, human) for row, human in pairs if row["prelabel"] is not None]

    def kappa(rows: list[tuple[dict[str, Any], bool]]) -> dict[str, Any]:
        return {
            "n": len(rows),
            "kappa": cohen_kappa([bool(r["prelabel"]) for r, _ in rows], [human for _, human in rows]),
            "agreement": sum(bool(r["prelabel"]) == human for r, human in rows) / len(rows) if rows else None,
        }

    tasks = sorted({row["task_type"] for row, _ in usable})
    return {
        "overall": kappa(usable),
        "prelabel_shown": kappa([p for p in usable if p[0]["prelabel_shown"]]),
        "prelabel_hidden": kappa([p for p in usable if not p[0]["prelabel_shown"]]),
        "by_task": {task: kappa([p for p in usable if p[0]["task_type"] == task]) for task in tasks},
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def render_page(row: dict[str, Any], done: int, total: int) -> str:
    """One review form. Output is escaped because it is untrusted model text."""
    prelabel = ""
    if row["prelabel_shown"] and row["prelabel"] is not None:
        prelabel = f"<p><b>Pre-label:</b> {'acceptable' if row['prelabel'] else 'not acceptable'}</p>"
    reference = html.escape(json.dumps(row["reference"], ensure_ascii=False))
    return f"""<!doctype html><meta charset=utf-8><title>Review {done + 1}/{total}</title>
<body style="font-family:sans-serif;max-width:60rem;margin:2rem auto">
<h2>Item {done + 1} of {total} ({html.escape(row["task_type"])})</h2>
<h3>Prompt</h3><pre style="white-space:pre-wrap">{html.escape(row["prompt"])}</pre>
<h3>Reference</h3><pre style="white-space:pre-wrap">{reference}</pre>
<h3>Output</h3><pre style="white-space:pre-wrap">{html.escape(row["output"] or "")}</pre>
{prelabel}
<form method=post action=/label>
<input type=hidden name=review_id value={row["review_id"]}>
<button name=accepted value=1>Acceptable</button>
<button name=accepted value=0>Not acceptable</button>
</form></body>"""


def serve(review_path: Path, labels_path: Path, *, port: int) -> None:
    """A minimal local page for the reviewer. Binds to loopback only."""
    import uvicorn
    from fastapi import FastAPI, Form
    from fastapi.responses import HTMLResponse, RedirectResponse

    app = FastAPI()
    review = read_jsonl(review_path)

    @app.get("/", response_class=HTMLResponse)
    def next_item() -> str:
        done = {row["review_id"] for row in read_jsonl(labels_path)}
        for row in review:
            if row["review_id"] not in done:
                return render_page(row, len(done), len(review))
        return "<p>All items are labelled.</p>"

    @app.post("/label")
    def save(review_id: int = Form(), accepted: int = Form()) -> RedirectResponse:
        with labels_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"review_id": review_id, "accepted": bool(accepted)}) + "\n")
        return RedirectResponse("/", status_code=303)

    uvicorn.run(app, host="127.0.0.1", port=port)
