"""Offline recipe search and multi-model comparison.

``search`` takes a run's *validation* predictions (raw log-scores stored by
the engine), fits temperatures and picks the best of a small variant grid
per primitive (permutation averaging on/off, prior correction on/off), then
applies the chosen recipe to the *test* predictions. Nothing is tuned on
test. Every choice is written to ``recipe.json`` next to the results so a
number is never separated from the recipe that produced it.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..bench.metrics import Report
from ..bench.record import BenchRecord
from ..engine.calibrate import fit_temperature
from ..engine.predict import Recipe, finalize, refinalize
from ..engine.readout import softmax
from .base import Prediction
from .cli import score as score_reports

PRIMS = ("choice", "score", "noul")


def _logscore_sets(records: Sequence[BenchRecord], preds: Sequence[Prediction], recipe: Recipe) -> dict[str, tuple[list[list[float]], list[int]]]:
    """Per primitive: (post-recipe log-scores aligned to the record's option
    order, label index). Temperature is fitted on these."""
    by_id = {r.id: r for r in records}
    out: dict[str, tuple[list[list[float]], list[int]]] = {p: ([], []) for p in PRIMS}
    for p in preds:
        r = by_id.get(p.id)
        if r is None or p.error or not p.extra:
            continue
        # finalize at T=1 to apply prior/permutation, then take log of the averaged probabilities
        probe = finalize(r.primitive, r.question, p.extra, replace(recipe, temperature={k: 1.0 for k in PRIMS}))
        keys = r.option_keys()
        if r.primitive == "noul":
            probs = [1 - probe.p_yes, probe.p_yes]
        else:
            probs = [probe.probabilities[k] for k in keys]
        ls = [float(np.log(max(v, 1e-12))) for v in probs]
        out[r.primitive][0].append(ls)
        out[r.primitive][1].append(r.label_index())
    return out


def fit_recipe(records: Sequence[BenchRecord], preds: Sequence[Prediction], base: Recipe) -> tuple[Recipe, dict[str, Any]]:
    """Grid over {permutations: 1 | all stored} x {prior_weight: 0 | 1}; per
    primitive pick the variant with the lowest validation NLL after its own
    temperature fit."""
    n_perm = max((len(p.extra["runs"]) for p in preds if p.extra), default=1)
    grid = [(perm, pw) for perm in sorted({1, n_perm}) for pw in (0.0, 1.0)]
    log: dict[str, Any] = {"grid": [], "chosen": {}}
    chosen: dict[str, tuple[int, float, float]] = {}
    for perm, pw in grid:
        cand = replace(base, permutations=perm, prior_weight=pw)
        sets = _logscore_sets(records, preds, cand)
        row = {"permutations": perm, "prior_weight": pw}
        for prim in PRIMS:
            ls, y = sets[prim]
            if not y:
                continue
            T = fit_temperature(ls, y)
            nll = float(np.mean([-np.log(max(softmax(s, T)[yy], 1e-12)) for s, yy in zip(ls, y)]))
            nll_raw = float(np.mean([-np.log(max(softmax(s, 1.0)[yy], 1e-12)) for s, yy in zip(ls, y)]))
            row[prim] = {"T": round(T, 4), "nll": round(nll, 4), "nll_raw": round(nll_raw, 4), "n": len(y)}
            if prim not in chosen or nll < chosen[prim][2]:
                chosen[prim] = (perm, pw, nll, T)
        log["grid"].append(row)
    # One recipe must serve all primitives for permutations/prior (they are rendering choices);
    # pick the (perm, pw) that wins the most primitives, then temperatures per primitive.
    votes = defaultdict(int)
    for prim, (perm, pw, _, _) in chosen.items():
        votes[(perm, pw)] += 1
    (perm, pw), _ = max(votes.items(), key=lambda kv: kv[1])
    final_sets = _logscore_sets(records, preds, replace(base, permutations=perm, prior_weight=pw))
    temps = {}
    for prim in PRIMS:
        ls, y = final_sets[prim]
        temps[prim] = round(fit_temperature(ls, y), 4) if y else 1.0
    recipe = replace(base, permutations=perm, prior_weight=pw, temperature=temps)
    log["chosen"] = {"permutations": perm, "prior_weight": pw, "temperature": temps,
                     "per_primitive_winners": {p: {"permutations": c[0], "prior_weight": c[1], "nll": round(c[2], 4)} for p, c in chosen.items()}}
    return recipe, log


def ablation(records: Sequence[BenchRecord], preds: Sequence[Prediction], recipe: Recipe) -> list[dict[str, Any]]:
    """Raw → +permutations → +prior → +temperature, each scored on the given records."""
    steps = [
        ("raw", replace(recipe, permutations=1, prior_weight=0.0, temperature={p: 1.0 for p in PRIMS})),
        ("+permutations", replace(recipe, prior_weight=0.0, temperature={p: 1.0 for p in PRIMS})),
        ("+prior", replace(recipe, temperature={p: 1.0 for p in PRIMS})),
        ("+temperature", recipe),
    ]
    rows = []
    for name, rc in steps:
        reps = score_reports(list(records), refinalize(records, preds, rc))
        agg = _aggregate(reps, records)
        rows.append({"step": name, **agg})
    return rows


def _aggregate(reps: dict[str, Report], records: Sequence[BenchRecord]) -> dict[str, float]:
    prim_of = {r.source: r.primitive for r in records}
    out: dict[str, float] = {}
    for prim in PRIMS:
        rs = [r for s, r in reps.items() if prim_of.get(s) == prim]
        if rs:
            out[f"{prim}_acc"] = round(float(np.mean([r.accuracy for r in rs])), 4)
            out[f"{prim}_ece"] = round(float(np.mean([r.ece for r in rs])), 4)
            out[f"{prim}_brier"] = round(float(np.mean([r.brier for r in rs])), 4)
    out["macro_acc"] = round(float(np.mean([r.accuracy for r in reps.values()])), 4)
    out["macro_ece"] = round(float(np.mean([r.ece for r in reps.values()])), 4)
    out["macro_brier"] = round(float(np.mean([r.brier for r in reps.values()])), 4)
    return out


def compare_table(named: dict[str, dict[str, Report]], prim_of: dict[str, str]) -> str:
    """Side-by-side markdown: rows = sources (+ macro), columns = model × (acc, ECE, Brier)."""
    models = list(named)
    sources = sorted({s for reps in named.values() for s in reps}, key=lambda s: (prim_of.get(s, ""), s))
    head = "| source | prim | " + " | ".join(f"{m} acc | {m} ECE | {m} Brier" for m in models) + " |"
    lines = [head, "|" + "---|" * (2 + 3 * len(models))]
    for s in sources:
        cells = []
        for m in models:
            r = named[m].get(s)
            cells += [f"{r.accuracy:.3f}", f"{r.ece:.3f}", f"{r.brier:.3f}"] if r else ["", "", ""]
        lines.append(f"| `{s}` | {prim_of.get(s, '')} | " + " | ".join(cells) + " |")
    cells = []
    for m in models:
        rs = list(named[m].values())
        cells += [f"**{np.mean([r.accuracy for r in rs]):.3f}**", f"**{np.mean([r.ece for r in rs]):.3f}**", f"**{np.mean([r.brier for r in rs]):.3f}**"]
    lines.append("| **macro** | | " + " | ".join(cells) + " |")
    return "\n".join(lines)
