"""Runner interface: records in, predictions out, in a runner-agnostic JSONL format."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..bench.record import BenchRecord


@dataclass
class Prediction:
    id: str
    primitive: str
    probabilities: dict[str, float] | None = None   # choice: by option key; score: by level index string
    p_yes: float | None = None                       # noul
    answer: str | float | None = None                # choice key / score expectation / p_yes
    confidence: float | None = None
    model: str = ""
    latency_ms: float | None = None
    usage: dict[str, int] | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Prediction":
        return cls(**row)


class Runner:
    name: str = "runner"

    def predict(self, records: Iterable[BenchRecord]) -> Iterator[Prediction]:
        raise NotImplementedError


def write_predictions(path: Path, preds: Iterable[Prediction]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p.to_row(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_predictions(path: Path) -> Iterator[Prediction]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield Prediction.from_row(json.loads(line))
