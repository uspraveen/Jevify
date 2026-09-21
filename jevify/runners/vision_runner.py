"""Score vision records with a VLM, producing the same Prediction rows as any other runner.

Vision records carry PIL images on their state, so they are built in memory rather than read
from JSONL. Everything downstream — metrics, figures, the leaderboard — is unchanged, which
is the point: a System One question about an image is the same object as one about text.
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from ..bench.record import BenchRecord
from ..engine.calibrate import prior_correct
from ..engine.predict import Recipe
from ..engine.readout import softmax
from ..engine.vision import VisionScorer, split_images
from ..wire import choice_confidence, round_probabilities, score_confidence, score_expectation
from .base import Prediction


class VisionRunner:
    name = "vision"

    def __init__(self, scorer: VisionScorer, recipe: Recipe | None = None, *, want_prior: bool = True) -> None:
        self.scorer = scorer
        self.recipe = recipe or Recipe()
        self.want_prior = want_prior
        self._prior: dict[str, list[float]] = {}

    def predict(self, records: Sequence[BenchRecord], *, batch: int = 8) -> Iterator[Prediction]:
        recs = list(records)
        for start in range(0, len(recs), batch):
            chunk = recs[start:start + batch]
            items, rds = [], []
            for r in chunk:
                it, rd = self.scorer.item(r.state, r.question, mode=self.recipe.mode)
                items.append(it)
                rds.append(rd)
            t0 = time.perf_counter()
            scores = self.scorer.score_many(items)
            elapsed = (time.perf_counter() - t0) * 1000 / max(len(chunk), 1)
            for r, rd, sc in zip(chunk, rds, scores):
                extra = {"mode": rd.mode, "runs": [{"keys": rd.keys, "logscores": sc}], "prior": None}
                pred = self._finalize(r, rd.keys, sc)
                pred.id = r.id
                pred.model = self.scorer.model_id
                pred.latency_ms = elapsed
                pred.extra = extra
                yield pred

    def _finalize(self, r: BenchRecord, keys: list[str], logscores: list[float]) -> Prediction:
        prim = r.primitive
        T = self.recipe.temperature.get(prim, 1.0)
        ls = list(logscores)
        if prim == "noul" and self.recipe.bias.get("noul"):
            ls = [v + (self.recipe.bias["noul"] if k == "1" else 0.0) for k, v in zip(keys, ls)]
        probs = dict(zip(keys, softmax(ls, T)))
        if prim == "noul":
            p_yes = probs["1"]
            return Prediction(id="", primitive="noul", p_yes=p_yes, answer=p_yes)
        order = list(r.question["criteria"].keys()) if prim == "choice" else [str(i) for i in range(len(r.question["criteria"]))]
        vals = round_probabilities([probs[k] for k in order], 4)
        pm = dict(zip(order, vals))
        if prim == "choice":
            return Prediction(id="", primitive="choice", probabilities=pm, answer=max(pm, key=pm.get),
                              confidence=choice_confidence(vals))
        return Prediction(id="", primitive="score", probabilities=pm, answer=score_expectation(vals),
                          confidence=score_confidence(vals))


def write_vision_records(path: Path | str, records: Sequence[BenchRecord]) -> int:
    """Dump vision records as JSONL with every image replaced by its placeholder.

    ``BenchRecord.to_row`` JSON-encodes ``state`` itself, so a state holding PIL images
    has to be made text-only *before* the row is built. Overriding the ``state`` column
    on the resulting dict is too late — the encoder has already run and raised. This is
    what turns a scored vision run into a file that metrics and the Hub can read.
    """
    path = Path(path)
    lines = []
    for r in records:
        text_state, _images = split_images(r.state)
        lines.append(json.dumps(replace(r, state=text_state).to_row(), ensure_ascii=False))
    path.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
    return len(lines)


def build_vision_records(sources: Iterable[str], split: str, limit: int, seed: int = 20260921) -> list[BenchRecord]:
    """Sample vision records in memory (images cannot round-trip through the JSONL layout)."""
    from ..bench.adapters import VISION_REGISTRY
    from ..bench.build import sample_records

    out: list[BenchRecord] = []
    for name in sources:
        adapter = VISION_REGISTRY[name]()
        cap = limit or adapter.spec.caps.get(split, 0)
        out.extend(sample_records(adapter, split, cap, seed))
    return out
