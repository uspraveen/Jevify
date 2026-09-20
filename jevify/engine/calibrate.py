"""Post-hoc calibration on stored log-scores: no GPU needed.

- ``fit_temperature``: one scalar T per primitive (or per source) minimizing NLL
  on validation log-scores — the standard, hard-to-beat baseline.
- ``prior_correct``: contextual calibration / PMI. Subtract ``w * log p0`` where
  ``p0`` is the model's distribution over the same candidates given a
  content-free state. Removes surface-form and position priors.
- ``average_permutations``: mean probability per key across option orderings.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

from .readout import softmax


def nll(logscores: Sequence[Sequence[float]], labels: Sequence[int], temperature: float) -> float:
    total = 0.0
    for s, y in zip(logscores, labels):
        p = softmax(s, temperature)
        total -= math.log(max(p[y], 1e-12))
    return total / max(len(labels), 1)


def fit_temperature(logscores: Sequence[Sequence[float]], labels: Sequence[int],
                    lo: float = 0.05, hi: float = 20.0, iters: int = 60) -> float:
    """Golden-section search on log T. Convex enough in practice; bounded so a
    degenerate split cannot run away."""
    a, b = math.log(lo), math.log(hi)
    phi = (math.sqrt(5) - 1) / 2
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = nll(logscores, labels, math.exp(c)), nll(logscores, labels, math.exp(d))
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = nll(logscores, labels, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = nll(logscores, labels, math.exp(d))
    return math.exp((a + b) / 2)


def prior_correct(logscores: Sequence[float], prior_logscores: Sequence[float], weight: float = 1.0) -> list[float]:
    """logit_i - w * log p0_i, with p0 the normalized content-free distribution."""
    p0 = softmax(prior_logscores)
    return [s - weight * math.log(max(q, 1e-12)) for s, q in zip(logscores, p0)]


def average_permutations(prob_maps: Iterable[dict[str, float]]) -> dict[str, float]:
    acc: dict[str, float] = {}
    n = 0
    for pm in prob_maps:
        n += 1
        for k, v in pm.items():
            acc[k] = acc.get(k, 0.0) + v
    return {k: v / n for k, v in acc.items()}
