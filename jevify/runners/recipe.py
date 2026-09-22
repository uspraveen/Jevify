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
K_SLOPE_MARGIN = 0.01      # the K-dependent temperature must cut validation NLL by >=1% to be used


def _logscore_sets(records: Sequence[BenchRecord], preds: Sequence[Prediction], recipe: Recipe) -> dict[str, tuple[list[list[float]], list[int]]]:
    """Per primitive: (post-recipe log-scores aligned to the record's option
    order, label index). Temperature is fitted on these."""
    sets = _logscore_sets_k(records, preds, recipe)
    return {p: (v[0], v[1]) for p, v in sets.items()}


def _logscore_sets_k(records: Sequence[BenchRecord], preds: Sequence[Prediction],
                     recipe: Recipe) -> dict[str, tuple[list[list[float]], list[int], list[int]]]:
    """As ``_logscore_sets``, plus each row's option count -- what a K-dependent temperature needs."""
    by_id = {r.id: r for r in records}
    out: dict[str, tuple[list[list[float]], list[int], list[int]]] = {p: ([], [], []) for p in PRIMS}
    for p in preds:
        r = by_id.get(p.id)
        if r is None or p.error or not p.extra:
            continue
        # finalize at T=1 to apply prior/permutation, then take log of the averaged probabilities
        probe = finalize(r.primitive, r.question, p.extra, replace(recipe, temperature={k: 1.0 for k in PRIMS},
                                                                   temp_k_slope={}, bias={"noul": 0.0}), round_to=None)
        keys = r.option_keys()
        if r.primitive == "noul":
            probs = [1 - probe.p_yes, probe.p_yes]
        else:
            probs = [probe.probabilities[k] for k in keys]
        ls = [float(np.log(max(v, 1e-12))) for v in probs]
        out[r.primitive][0].append(ls)
        out[r.primitive][1].append(r.label_index())
        out[r.primitive][2].append(len(keys))
    return out


def _nll(ls: Sequence[Sequence[float]], y: Sequence[int], ks: Sequence[int], T: float, slope: float) -> float:
    import math

    tot = 0.0
    for s, yy, k in zip(ls, y, ks):
        t = max(T + (slope * math.log2(k / 2) if slope and k > 2 else 0.0), 0.05)
        tot += -math.log(max(softmax(list(s), t)[yy], 1e-12))
    return tot / max(len(y), 1)


def fit_temperature_k(ls: list[list[float]], y: list[int], ks: list[int], T0: float) -> tuple[float, float, float, float]:
    """Fit T(K) = T + slope * log2(K/2) by coordinate search around the scalar fit.

    Returns (T, slope, nll, nll_scalar). The slope is only worth using when the option counts
    actually vary, so a set with one K returns slope 0 and the scalar fit unchanged."""
    base = _nll(ls, y, ks, T0, 0.0)
    if len({min(k, 256) for k in ks}) < 2:
        return T0, 0.0, base, base
    best = (base, T0, 0.0)
    T = T0
    for _ in range(3):
        for slope in [x / 20 for x in range(-30, 31)]:            # -1.5 .. 1.5 per doubling
            v = _nll(ls, y, ks, T, slope)
            if v < best[0]:
                best = (v, T, slope)
        slope = best[2]
        for t in [x / 20 for x in range(1, 241)]:                  # 0.05 .. 12.0
            v = _nll(ls, y, ks, t, slope)
            if v < best[0]:
                best = (v, t, slope)
        T = best[1]
    return best[1], best[2], best[0], base


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


def fit_recipe(records: Sequence[BenchRecord], preds: Sequence[Prediction], base: Recipe,
               *, k_slope: bool = True) -> tuple[Recipe, dict[str, Any]]:
    """Per primitive, grid over {permutations: 1 | all stored} x {prior_weight: 0 | 1}, fit the
    temperature (and, for noul, a bias) for each variant on validation, keep the lowest NLL.

    ``k_slope`` also tries a temperature affine in log2(K/2) for the option primitives and keeps
    it only when it lowers validation NLL by more than ``K_SLOPE_MARGIN``; pass False for the
    single-scalar recipe every result before 2026-09-22 was fitted with."""
    n_perm = max((len(p.extra["runs"]) for p in preds if p.extra), default=1)
    grid = [(perm, pw) for perm in sorted({1, n_perm}) for pw in (0.0, 1.0)]
    log: dict[str, Any] = {"grid": [], "chosen": {}}
    best: dict[str, tuple[float, int, float, float, float, float]] = {}   # prim -> (nll, perm, pw, T, bias, slope)
    for perm, pw in grid:
        cand = replace(base, permutations=perm, prior_weight=pw, temperature={p: 1.0 for p in PRIMS},
                       temp_k_slope={}, bias={"noul": 0.0})
        sets = _logscore_sets_k(records, preds, cand)
        row: dict[str, Any] = {"permutations": perm, "prior_weight": pw}
        for prim in PRIMS:
            ls, y, ks = sets[prim]
            if not y:
                continue
            slope = 0.0
            if prim == "noul":
                T, b = fit_temperature_and_bias(ls, y)
                scored = [[s0, s1 + b] for s0, s1 in ls]
            else:
                T, b = fit_temperature(ls, y), 0.0
                scored = ls
            nll = float(np.mean([-np.log(max(softmax(s, T)[yy], 1e-12)) for s, yy in zip(scored, y)]))
            nll_raw = float(np.mean([-np.log(max(softmax(s, 1.0)[yy], 1e-12)) for s, yy in zip(ls, y)]))
            row[prim] = {"T": round(T, 4), "bias": round(b, 3), "nll": round(nll, 4), "nll_raw": round(nll_raw, 4), "n": len(y)}
            if k_slope and prim != "noul":
                # a temperature affine in log2(K/2), kept only if validation NLL improves by a
                # clear margin -- one more fitted parameter must earn its place
                Tk, sk, nll_k, _ = fit_temperature_k(scored, y, ks, T)
                row[prim].update({"T_k": round(Tk, 4), "k_slope": round(sk, 4), "nll_k": round(nll_k, 4)})
                if nll_k < nll * (1 - K_SLOPE_MARGIN):
                    T, slope, nll = Tk, sk, nll_k
                    row[prim]["k_slope_used"] = True
            if prim not in best or nll < best[prim][0]:
                best[prim] = (nll, perm, pw, T, b, slope)
        log["grid"].append(row)
    recipe = replace(base,
                     permutations={p: best[p][1] if p in best else 1 for p in PRIMS},
                     prior_weight={p: best[p][2] if p in best else 0.0 for p in PRIMS},
                     temperature={p: round(best[p][3], 4) if p in best else 1.0 for p in PRIMS},
                     temp_k_slope={p: round(best[p][5], 4) for p in best if best[p][5]},
                     bias={"noul": round(best["noul"][4], 3) if "noul" in best else 0.0})
    log["chosen"] = {p: {"permutations": best[p][1], "prior_weight": best[p][2], "T": round(best[p][3], 4),
                         "bias": round(best[p][4], 3), "k_slope": round(best[p][5], 4), "val_nll": round(best[p][0], 4)} for p in best}
    return recipe, log


def ablation(records: Sequence[BenchRecord], preds: Sequence[Prediction], recipe: Recipe) -> list[dict[str, Any]]:
    """raw → +permutations → +prior → +temperature/bias, each scored on the given records."""
    unit = {p: 1.0 for p in PRIMS}
    steps = [
        ("raw", replace(recipe, permutations=1, prior_weight=0.0, temperature=unit, temp_k_slope={}, bias={"noul": 0.0})),
        ("+permutations", replace(recipe, prior_weight=0.0, temperature=unit, temp_k_slope={}, bias={"noul": 0.0})),
        ("+prior", replace(recipe, temperature=unit, temp_k_slope={}, bias={"noul": 0.0})),
        ("+temperature/bias", replace(recipe, temp_k_slope={})),
    ]
    if recipe.temp_k_slope:
        steps.append(("+temperature(K)", recipe))
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
