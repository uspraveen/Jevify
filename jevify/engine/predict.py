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
from .template import Rendered, render, to_chat, to_chat_tev1


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
    state_last: bool = False                       # question + options before the state: a cacheable prefix
    prior_weight: float | dict[str, float] = 0.0  # 0 = off; 1 = full PMI / contextual calibration
    temperature: dict[str, float] = field(default_factory=lambda: {p: 1.0 for p in PRIMS})
    bias: dict[str, float] = field(default_factory=lambda: {"noul": 0.0})
    # a model's overconfidence need not be constant in the number of options: Gemma-4-12B is
    # ECE 0.10 at K<=5 and 0.34 at K>30, Qwen3.5-9B the other way round, so one scalar per
    # primitive is fitted across two regimes and can be worse than none (FINDINGS 2.7). This
    # slope makes the temperature affine in log2(K/2) -- zero (the default) is the old behaviour.
    temp_k_slope: dict[str, float] = field(default_factory=dict)
    # the prompt format ("jevify" or "tev1", template.py). A fine-tuned model is scored in the format
    # it was trained on; everything after the log-scores is format-independent.
    prompt: str = "jevify"

    def temp_for(self, prim: str, k: int | None = None) -> float:
        T = self.temperature.get(prim, 1.0)
        slope = self.temp_k_slope.get(prim, 0.0) if self.temp_k_slope else 0.0
        if slope and k and k > 2:
            import math

            T = T + slope * math.log2(k / 2)
        return max(T, 0.05)

    def perm_for(self, prim: str) -> int:
        return self.permutations.get(prim, 1) if isinstance(self.permutations, dict) else int(self.permutations)

    def prior_for(self, prim: str) -> float:
        return self.prior_weight.get(prim, 0.0) if isinstance(self.prior_weight, dict) else float(self.prior_weight)

    def max_permutations(self) -> int:
        return max(self.permutations.values()) if isinstance(self.permutations, dict) else int(self.permutations)

    def as_dict(self) -> dict[str, Any]:
        d = {"mode": self.mode, "chat": self.chat, "permutations": self.permutations, "state_last": self.state_last,
             "prior_weight": self.prior_weight, "temperature": dict(self.temperature), "bias": dict(self.bias)}
        if self.temp_k_slope:          # omitted when unused, so old recipe.json files stay byte-identical
            d["temp_k_slope"] = dict(self.temp_k_slope)
        if self.prompt != "jevify":    # likewise
            d["prompt"] = self.prompt
        return d


class Tier0Engine:
    def __init__(self, scorer: HFScorer, recipe: Recipe | None = None, *, prior_cache: dict[str, list[float]] | None = None) -> None:
        self.scorer = scorer
        self.recipe = recipe or Recipe()
        self.chat = self.recipe.chat and getattr(scorer.tokenizer, "chat_template", None) is not None
        self._prior_cache: dict[str, list[float]] = prior_cache if prior_cache is not None else {}

    # ------------------------------------------------------------------ rendering
    def _prefix(self, rendered: Rendered) -> str:
        if rendered.fmt == "tev1":     # a format learned in one chat template is only meaningful inside it
            return to_chat_tev1(rendered.prefix, self.scorer.tokenizer)
        return to_chat(rendered.prefix, self.scorer.tokenizer) if self.chat else rendered.prefix

    def _render(self, state: Any, question: dict[str, Any], **kw: Any) -> Rendered:
        ids = self.scorer.identifiers() if self.recipe.mode == "index" else None
        return render(state, question, mode=self.recipe.mode, identifiers=ids, state_last=self.recipe.state_last,
                      fmt=self.recipe.prompt, **kw)

    def _renderings(self, state: Any, question: dict[str, Any]) -> list[Rendered]:
        rs = [self._render(state, question)]
        # in the Tev1 format yes/no are lettered options, so their order carries a position bias too
        permutable = ("choice", "noul") if self.recipe.prompt == "tev1" else ("choice",)
        if rs[0].primitive in permutable:
            for i in range(1, self.recipe.max_permutations()):
                rs.append(self._render(state, question, permutation_seed=1000 + i))
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
                rd = self._render(r.state, r.question, content_free=True)
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
                    "prior": {"keys": self._render(r.state, r.question, content_free=True).keys,
                              "logscores": prior} if prior else None,
                }
                if self.recipe.prompt != "jevify":   # absent for the default, so existing prediction files are unchanged
                    extra["prompt"] = self.recipe.prompt
                pred = finalize(r.primitive, r.question, extra, self.recipe)
                pred.id = r.id
                pred.model = self.scorer.model_id
                pred.latency_ms = elapsed
                pred.extra = extra
                yield pred


# ---------------------------------------------------------------------- recipe → answer

def finalize(primitive: str, question: dict[str, Any], extra: dict[str, Any], recipe: Recipe,
             *, round_to: int | None = 4) -> Prediction:
    """Apply a recipe to stored raw log-scores. Pure; used offline for every recipe variant.

    ``round_to`` is the wire format's precision (4 decimals, as the API reports). Pass ``None``
    when the distribution is an *input to fitting* rather than an answer: at K=151 a genuine
    probability of 1e-5 quantizes to exactly 0, and a fitter that takes its log then sees -27.6
    and inflates the temperature to explain it (this is why the high-K temperatures were wrong)."""
    T = recipe.temp_for(primitive, len(extra["runs"][0]["keys"]) if extra.get("runs") else None)
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
        vals = _round([probs[k] for k in keys], round_to)
        pm = dict(zip(keys, vals))
        best = max(pm, key=pm.get)
        return Prediction(id="", primitive="choice", probabilities=pm, answer=best, confidence=choice_confidence(vals))
    keys = [str(i) for i in range(len(question["criteria"]))]
    vals = _round([probs[k] for k in keys], round_to)
    return Prediction(id="", primitive="score", probabilities=dict(zip(keys, vals)),
                      answer=score_expectation(vals), confidence=score_confidence(vals))


def _round(vals: list[float], places: int | None) -> list[float]:
    return round_probabilities(vals, places) if places is not None else list(vals)


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
