"""The benchmark's unit of work and the deterministic manifest that pins a dataset build."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1


@dataclass(frozen=True)
class BenchItem:
    item_id: str
    source: str
    task_type: str
    messages: list[dict[str, str]]
    label_kind: str  # how the output is scored; see scoring.py
    reference: dict[str, Any] = field(default_factory=dict)
    response_format: dict[str, Any] | None = None
    max_tokens: int = 512

    def content_hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def write_items(path: Path, items: list[BenchItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


def read_items(path: Path) -> list[BenchItem]:
    with path.open(encoding="utf-8") as handle:
        return [BenchItem(**json.loads(line)) for line in handle if line.strip()]


def manifest_for(items: list[BenchItem], *, seed: int, counts: dict[str, int]) -> dict[str, Any]:
    """Item ids and content hashes only: dataset contents are never committed."""
    rows = sorted((item.item_id, item.content_hash()) for item in items)
    digest = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
    return {
        "version": MANIFEST_VERSION,
        "seed": seed,
        "counts": counts,
        "manifest_sha256": digest,
        "items": [{"item_id": item_id, "sha256": content} for item_id, content in rows],
    }
