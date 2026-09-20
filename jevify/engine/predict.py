"""The Jevified model: records (or wire requests) in, System One answers out.

``Tier0Engine`` renders each question, scores the allowed answers with an
``HFScorer``, and stores the *raw* log-scores on every prediction (plus the
content-free prior and any permutation runs) so calibration and debiasing are
offline experiments. ``finalize`` turns stored log-scores into wire answers
under a given recipe (temperature / prior weight / permutation averaging).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence

from ..bench.record import BenchRecord
from ..runners.base import Prediction
from ..wire import choice_confidence, round_probabilities, score_confidence, score_expectation
from .calibrate import average_permutations, prior_correct
from .readout import HFScorer, softmax
from .template import Rendered, render, to_chat


PRIMS = ("choice", "score", "noul")


@dataclass
class Recipe:
    """Everything that turns raw log-scores into an answer. Versioned with results.

    ``permutations`` and ``prior_weight`` may be a scalar (all primitives) or a per-primitive
    dict — they are applied offline, so nothing forces one setting on every primitive.
    ``bias`` is a Platt-style offset added to the log-score of "yes" for noul (the one
    primitive with a fixed answer set, where a scalar bias is meaningful)."""
    mode: str = "index"                 # choice readout: index | label
    chat: bool = True                   # wrap in the chat template when the model has one
    permutations: int | dict[str, int] = 1        # option orderings scored for choice (1 = presented order only)
    prior_weight: float | dict[str, float] = 0.0  # 0 = off; 1 = full PMI / contextual calibration
    temperature: dict[str, float] = field(default_factory=lambda: {p: 1.0 for p in PRIMS})
    bias: dict[str, float] = field(default_factory=lambda: {"noul": 0.0})

    def perm_for(self, prim: str) -> int:
        return self.permutations.get(prim, 1) if isinstance(self.permutations, dict) else int(self.permutations)

    def prior_for(self, prim: str) -> float:
        return self.prior_weight.get(prim, 0.0) if isinstance(self.prior_weight, dict) else float(self.prior_weight)

    def max_permutations(self) -> int:
        return max(self.permutations.values()) if isinstance(self.permutations, dict) else int(self.permutations)

    def as_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "chat": self.chat, "permutations": self.permutations,
                "prior_weight": self.prior_weight, "temperature": dict(self.temperature), "bias": dict(self.bias)}


class Tier0Engine:
    def __init__(self, scorer: HFScorer, recipe: Recipe | None = None, *, prior_cache: dict[str, list[float]] | None = None) -> None:
        self.scorer = scorer
        self.recipe = recipe or Recipe()
        self.chat = self.recipe.chat and getattr(scorer.tokenizer, "chat_template", None) is not None
        self._prior_cache: dict[str, list[float]] = prior_cache if prior_cache is not None else {}

    # ------------------------------------------------------------------ rendering
    def _prefix(self, rendered: Rendered) -> str:
        return to_chat(rendered.prefix, self.scorer.tokenizer) if self.chat else rendered.prefix

    def _renderings(self, state: Any, question: dict[str, Any]) -> list[Rendered]:
        ids = self.scorer.identifiers() if self.recipe.mode == "index" else None
        rs = [render(state, question, mode=self.recipe.mode, identifiers=ids)]
        if rs[0].primitive == "choice":
            for i in range(1, self.recipe.max_permutations()):
                rs.append(render(state, question, mode=self.recipe.mode, permutation_seed=1000 + i, identifiers=ids))
        return rs

    # ------------------------------------------------------------------ raw scoring
    def score_records(self, records: Iterable[BenchRecord], *, batch: int = 32, want_prior: bool = True) -> Iterator[Prediction]:
        """Yields one Prediction per record with raw log-scores in ``extra``."""
        recs = list(records)
        for start in range(0, len(recs), batch):
            chunk = recs[start:start + batch]
            items: list[tuple[str, list[str]]] = []
            plan: list[tuple[int, Rendered]] = []
            for i, r in enumerate(chunk):
                for rd in self._renderings(r.state, r.question):
                    items.append((self._prefix(rd), rd.candidates))
                    plan.append((i, rd))
            # content-free priors, one per distinct question rendering
            prior_keys: list[tuple[str, Rendered]] = []
            for i, r in enumerate(chunk):
                if not want_prior:
                    break
                rd = render(r.state, r.question, mode=self.recipe.mode, content_free=True,
                            identifiers=self.scorer.identifiers() if self.recipe.mode == "index" else None)
                key = f"{rd.question_hash}:{rd.mode}"
                if key not in self._prior_cache and all(k != key for k, _ in prior_keys):
                    prior_keys.append((key, rd))
                    items.append((self._prefix(rd), rd.candidates))
            t0 = time.perf_counter()
            scores = self.scorer.score_many(items)
            elapsed = (time.perf_counter() - t0) * 1000 / max(len(chunk), 1)
            for j, (key, rd) in enumerate(prior_keys):
                self._prior_cache[key] = scores[len(plan) + j]
            per_rec: dict[int, list[tuple[Rendered, list[float]]]] = {}
            for (i, rd), sc in zip(plan, scores[: len(plan)]):
                per_rec.setdefault(i, []).append((rd, sc))
            for i, r in enumerate(chunk):
                runs = per_rec[i]
                rd0 = runs[0][0]
                prior = self._prior_cache.get(f"{rd0.question_hash}:{rd0.mode}") if want_prior else None
                extra = {
                    "mode": rd0.mode,
                    "runs": [{"keys": rd.keys, "logscores": sc} for rd, sc in runs],
                    "prior": {"keys": render(r.state, r.question, mode=self.recipe.mode, content_free=True,
                                             identifiers=self.scorer.identifiers() if self.recipe.mode == "index" else None).keys,
                              "logscores": prior} if prior else None,
                }
                pred = finalize(r.primitive, r.question, extra, self.recipe)
                pred.id = r.id
                pred.model = self.scorer.model_id
                pred.latency_ms = elapsed
                pred.extra = extra
                yield pred


# ---------------------------------------------------------------------- recipe → answer

def finalize(primitive: str, question: dict[str, Any], extra: dict[str, Any], recipe: Recipe) -> Prediction:
    """Apply a recipe to stored raw log-scores. Pure; used offline for every recipe variant."""
    T = recipe.temperature.get(primitive, 1.0)
    prior = extra.get("prior")
    pw = recipe.prior_for(primitive)
    prob_maps: list[dict[str, float]] = []
    for run in extra["runs"][: max(1, recipe.perm_for(primitive))]:
        keys, ls = run["keys"], list(run["logscores"])
        if primitive == "noul" and recipe.bias.get("noul"):
            ls = [v + (recipe.bias["noul"] if k == "1" else 0.0) for k, v in zip(keys, ls)]
        if prior and pw > 0:
            if extra.get("mode") == "index":
                # identifiers carry a positional bias: align the content-free prior by position
                p0 = list(prior["logscores"])[: len(ls)]
            else:
                # labels/digits/yes-no carry a surface-form bias: align by key
                pmap = dict(zip(prior["keys"], prior["logscores"]))
                p0 = [pmap[k] for k in keys]
            ls = prior_correct(ls, p0, pw)
        prob_maps.append(dict(zip(keys, softmax(ls, T))))
    probs = average_permutations(prob_maps)

    if primitive == "noul":
        p_yes = probs["1"]
        return Prediction(id="", primitive="noul", p_yes=p_yes, answer=p_yes)
    if primitive == "choice":
        keys = list(question["criteria"].keys())
        vals = round_probabilities([probs[k] for k in keys], 4)
        pm = dict(zip(keys, vals))
        best = max(pm, key=pm.get)
        return Prediction(id="", primitive="choice", probabilities=pm, answer=best, confidence=choice_confidence(vals))
    keys = [str(i) for i in range(len(question["criteria"]))]
    vals = round_probabilities([probs[k] for k in keys], 4)
    return Prediction(id="", primitive="score", probabilities=dict(zip(keys, vals)),
                      answer=score_expectation(vals), confidence=score_confidence(vals))


def refinalize(records: Sequence[BenchRecord], preds: Sequence[Prediction], recipe: Recipe) -> list[Prediction]:
    """Re-derive answers for stored predictions under another recipe (offline)."""
    by_id = {r.id: r for r in records}
    out = []
    for p in preds:
        if p.error or not p.extra or p.id not in by_id:
            out.append(p)
            continue
        r = by_id[p.id]
        q = finalize(r.primitive, r.question, p.extra, recipe)
        q.id, q.model, q.latency_ms, q.extra = p.id, p.model, p.latency_ms, p.extra
        out.append(q)
    return out
