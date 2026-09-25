"""Runner interface: records in, predictions out, in a runner-agnostic JSONL format."""
from __future__ import annotations

import json
import sys
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


def write_predictions(path: Path, preds: Iterable[Prediction], *, append: bool = False,
                      progress_every: int = 500) -> int:
    """Stream predictions to JSONL as they arrive (flushed per line, so a crash
    loses at most one). ``append=True`` continues an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a" if append else "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p.to_row(), ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if progress_every and n % progress_every == 0:
                print(f"  {n} predictions written", file=sys.stderr, flush=True)
    return n


def read_predictions(path: Path) -> Iterator[Prediction]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield Prediction.from_row(json.loads(line))


def done_ids(path: Path) -> set[str]:
    """Ids already present in a predictions file (for --resume). Errored rows are retried.

    A run killed mid-write leaves a partial last line; it is cut off here, before anything is appended,
    or the next record would be glued onto it and the file could no longer be read."""
    if not path.exists():
        return set()
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        with path.open("r+b") as f:
            f.truncate(data.rfind(b"\n") + 1)
    return {p.id for p in read_predictions(path) if not p.error}
