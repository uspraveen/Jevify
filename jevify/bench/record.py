"""BenchRecord: one System One question with ground truth, and its JSONL form.

On disk, ``state``, ``question`` and ``soft_label`` are stored as JSON strings
and ``label`` as a string. That keeps every source's file Arrow-compatible
(mixed string/object states and per-source criteria would otherwise break
schema inference on the Hub viewer). ``from_row``/``to_row`` hide this.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

Primitive = Literal["choice", "score", "noul"]
Split = Literal["train", "validation", "test"]


@dataclass
class BenchRecord:
    id: str
    source: str
    primitive: Primitive
    split: Split
    state: str | dict[str, Any] | list[Any]
    question: dict[str, Any]
    label: str | int
    soft_label: dict[str, float] | list[float] | float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- label helpers -------------------------------------------------------------
    def option_keys(self) -> list[str]:
        """Option keys (choice) or level indices as strings (score); ['0','1'] for noul."""
        if self.primitive == "choice":
            return list(self.question["criteria"].keys())
        if self.primitive == "score":
            return [str(i) for i in range(len(self.question["criteria"]))]
        return ["0", "1"]

    def label_index(self) -> int:
        keys = self.option_keys()
        return keys.index(str(self.label))

    # ---- (de)serialization -----------------------------------------------------------
    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "primitive": self.primitive,
            "split": self.split,
            "state": json.dumps(self.state, ensure_ascii=False),
            "question": json.dumps(self.question, ensure_ascii=False),
            "label": str(self.label),
            "soft_label": None if self.soft_label is None else json.dumps(self.soft_label),
            "meta": json.dumps(self.meta, ensure_ascii=False),
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "BenchRecord":
        soft = row.get("soft_label")
        label: str | int = row["label"]
        if row["primitive"] in ("score", "noul"):
            label = int(label)
        return cls(
            id=row["id"],
            source=row["source"],
            primitive=row["primitive"],
            split=row["split"],
            state=json.loads(row["state"]),
            question=json.loads(row["question"]),
            label=label,
            soft_label=None if soft in (None, "") else json.loads(soft),
            meta=json.loads(row.get("meta") or "{}"),
        )


def write_jsonl(path: Path, records: Iterable[BenchRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_row(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path, limit: int | None = None) -> Iterator[BenchRecord]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            if line.strip():
                yield BenchRecord.from_row(json.loads(line))
