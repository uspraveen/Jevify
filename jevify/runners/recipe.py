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
        probe = finalize(r.primitive, r.question, p.extra, replace(recipe, temperature={k: 1.0 for k in PRIMS}, bias={"noul": 0.0}))
        keys = r.option_keys()
        if r.primitive == "noul":
            probs = [1 - probe.p_yes, probe.p_yes]
        else:
            probs = [probe.probabilities[k] for k in keys]
        ls = [float(np.log(max(v, 1e-12))) for v in probs]
        out[r.primitive][0].append(ls)
        out[r.primitive][1].append(r.label_index())
    return out


def fit_temperature_and_bias(ls: list[list[float]], y: list[int]) -> tuple[float, float]:
    """Platt scaling for a fixed binary answer set: search a scalar bias on the 'yes' log-score
    (index 1) and fit the temperature for each; return the pair with the lowest NLL."""
    best = (float("inf"), 1.0, 0.0)
    for b in [x / 10 for x in range(-40, 41)]:
        shifted = [[s0, s1 + b] for s0, s1 in ls]
        T = fit_temperature(shifted, y)
        val = float(np.mean([-np.log(max(softmax(s, T)[yy], 1e-12)) for s, yy in zip(shifted, y)]))
        if val < best[0]:
            best = (val, T, b)
    return best[1], best[2]


def fit_recipe(records: Sequence[BenchRecord], preds: Sequence[Prediction], base: Recipe) -> tuple[Recipe, dict[str, Any]]:
    """Per primitive, grid over {permutations: 1 | all stored} x {prior_weight: 0 | 1}, fit the
    temperature (and, for noul, a bias) for each variant on validation, keep the lowest NLL."""
    n_perm = max((len(p.extra["runs"]) for p in preds if p.extra), default=1)
    grid = [(perm, pw) for perm in sorted({1, n_perm}) for pw in (0.0, 1.0)]
    log: dict[str, Any] = {"grid": [], "chosen": {}}
    best: dict[str, tuple[float, int, float, float, float]] = {}   # prim -> (nll, perm, pw, T, bias)
    for perm, pw in grid:
        cand = replace(base, permutations=perm, prior_weight=pw, temperature={p: 1.0 for p in PRIMS}, bias={"noul": 0.0})
        sets = _logscore_sets(records, preds, cand)
        row: dict[str, Any] = {"permutations": perm, "prior_weight": pw}
        for prim in PRIMS:
            ls, y = sets[prim]
            if not y:
                continue
            if prim == "noul":
                T, b = fit_temperature_and_bias(ls, y)
                scored = [[s0, s1 + b] for s0, s1 in ls]
            else:
                T, b = fit_temperature(ls, y), 0.0
                scored = ls
            nll = float(np.mean([-np.log(max(softmax(s, T)[yy], 1e-12)) for s, yy in zip(scored, y)]))
            nll_raw = float(np.mean([-np.log(max(softmax(s, 1.0)[yy], 1e-12)) for s, yy in zip(ls, y)]))
            row[prim] = {"T": round(T, 4), "bias": round(b, 3), "nll": round(nll, 4), "nll_raw": round(nll_raw, 4), "n": len(y)}
            if prim not in best or nll < best[prim][0]:
                best[prim] = (nll, perm, pw, T, b)
        log["grid"].append(row)
    recipe = replace(base,
                     permutations={p: best[p][1] if p in best else 1 for p in PRIMS},
                     prior_weight={p: best[p][2] if p in best else 0.0 for p in PRIMS},
                     temperature={p: round(best[p][3], 4) if p in best else 1.0 for p in PRIMS},
                     bias={"noul": round(best["noul"][4], 3) if "noul" in best else 0.0})
    log["chosen"] = {p: {"permutations": best[p][1], "prior_weight": best[p][2], "T": round(best[p][3], 4),
                         "bias": round(best[p][4], 3), "val_nll": round(best[p][0], 4)} for p in best}
    return recipe, log


def ablation(records: Sequence[BenchRecord], preds: Sequence[Prediction], recipe: Recipe) -> list[dict[str, Any]]:
    """raw → +permutations → +prior → +temperature/bias, each scored on the given records."""
    unit = {p: 1.0 for p in PRIMS}
    steps = [
        ("raw", replace(recipe, permutations=1, prior_weight=0.0, temperature=unit, bias={"noul": 0.0})),
        ("+permutations", replace(recipe, prior_weight=0.0, temperature=unit, bias={"noul": 0.0})),
        ("+prior", replace(recipe, temperature=unit, bias={"noul": 0.0})),
        ("+temperature/bias", recipe),
    ]
    rows = []
    for name, rc in steps:
        reps = score_reports(list(records), refinalize(records, preds, rc))
        rows.append({"step": name, **_aggregate(reps, records)})
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
