"""Calibration and accuracy metrics for System One predictions.

All functions take plain Python/numpy inputs so they work on any runner's
output. Conventions:

- ``probs``: (N, K) array of predicted distributions (choice/score) or (N,)
  array of P(yes) (noul).
- ``labels``: (N,) integer indices (choice/score) or {0,1} (noul).
- ``soft``: optional (N, K) human label distributions (calibration gold).
- ``confidence``: optional (N,) model-reported confidence used for selective
  prediction; defaults to max-probability (choice/score) or |p-0.5|*2 (noul).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

EPS = 1e-12


@dataclass
class Report:
    n: int
    accuracy: float
    nll: float
    brier: float
    ece: float
    mce: float
    selective_acc_at_90: float
    selective_acc_at_50: float
    aurc: float
    # ordinal-only (score)
    rps: float | None = None
    mae: float | None = None
    within_one: float | None = None
    # noul-only
    auroc: float | None = None
    # against human distributions (calibration gold)
    tvd_to_human: float | None = None
    kl_human_model: float | None = None
    # bookkeeping
    mean_confidence: float | None = None
    reliability: list[dict[str, float]] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- core pieces

def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> tuple[float, float, list[dict[str, float]]]:
    """Top-label ECE/MCE with equal-width bins plus the reliability table."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    table: list[dict[str, float]] = []
    n = len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        acc = float(correct[mask].mean())
        avg_conf = float(conf[mask].mean())
        gap = abs(acc - avg_conf)
        weight = mask.sum() / n
        ece += weight * gap
        mce = max(mce, gap)
        table.append({"lo": float(lo), "hi": float(hi), "count": int(mask.sum()), "accuracy": acc, "confidence": avg_conf})
    return float(ece), float(mce), table


def selective_accuracy(conf: np.ndarray, correct: np.ndarray, coverage: float) -> float:
    """Accuracy on the most-confident ``coverage`` fraction of examples."""
    k = max(1, int(round(coverage * len(conf))))
    idx = np.argsort(-conf, kind="stable")[:k]
    return float(correct[idx].mean())


def area_under_risk_coverage(conf: np.ndarray, correct: np.ndarray) -> float:
    """AURC: mean risk (error rate) over all coverage levels when ranking by confidence. Lower is better."""
    order = np.argsort(-conf, kind="stable")
    errors = 1.0 - correct[order]
    cum_risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    return float(cum_risk.mean())


def ranked_probability_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """RPS for ordinal outcomes: squared distance between cumulative distributions, normalized by K-1."""
    k = probs.shape[1]
    cum_p = np.cumsum(probs, axis=1)
    onehot = np.eye(k)[labels]
    cum_y = np.cumsum(onehot, axis=1)
    return float((((cum_p - cum_y) ** 2).sum(axis=1) / (k - 1)).mean())


def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    # rank-based (Mann–Whitney) with tie handling
    allv = np.concatenate([pos, neg])
    ranks = _average_ranks(allv)
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=float)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


# --------------------------------------------------------------------------- per-primitive reports

def report_categorical(probs: np.ndarray, labels: np.ndarray, *, ordinal: bool = False,
                       confidence: np.ndarray | None = None, soft: np.ndarray | None = None) -> Report:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n, k = probs.shape
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    p_true = np.clip(probs[np.arange(n), labels], EPS, 1.0)
    onehot = np.eye(k)[labels]
    conf = np.asarray(confidence, dtype=float) if confidence is not None else probs.max(axis=1)
    ece, mce, table = expected_calibration_error(probs.max(axis=1), correct)

    rep = Report(
        n=int(n),
        accuracy=float(correct.mean()),
        nll=float(-np.log(p_true).mean()),
        brier=float(((probs - onehot) ** 2).sum(axis=1).mean()),
        ece=ece,
        mce=mce,
        selective_acc_at_90=selective_accuracy(conf, correct, 0.9),
        selective_acc_at_50=selective_accuracy(conf, correct, 0.5),
        aurc=area_under_risk_coverage(conf, correct),
        mean_confidence=float(conf.mean()),
        reliability=table,
    )
    if ordinal:
        ev = (probs * np.arange(k)).sum(axis=1)
        rep.rps = ranked_probability_score(probs, labels)
        rep.mae = float(np.abs(ev - labels).mean())
        rep.within_one = float((np.abs(pred - labels) <= 1).mean())
    if soft is not None:
        soft = np.asarray(soft, dtype=float)
        rep.tvd_to_human = float(0.5 * np.abs(probs - soft).sum(axis=1).mean())
        rep.kl_human_model = float((soft * (np.log(np.clip(soft, EPS, 1)) - np.log(np.clip(probs, EPS, 1)))).sum(axis=1).mean())
    return rep


def report_noul(p_yes: np.ndarray, labels: np.ndarray, *, confidence: np.ndarray | None = None,
                soft: np.ndarray | None = None) -> Report:
    p = np.clip(np.asarray(p_yes, dtype=float), 0.0, 1.0)
    y = np.asarray(labels, dtype=int)
    pred = (p >= 0.5).astype(int)
    correct = (pred == y).astype(float)
    p_true = np.clip(np.where(y == 1, p, 1 - p), EPS, 1.0)
    top_conf = np.maximum(p, 1 - p)
    conf = np.asarray(confidence, dtype=float) if confidence is not None else top_conf
    ece, mce, table = expected_calibration_error(top_conf, correct)
    rep = Report(
        n=int(len(p)),
        accuracy=float(correct.mean()),
        nll=float(-np.log(p_true).mean()),
        brier=float(((p - y) ** 2).mean()),
        ece=ece,
        mce=mce,
        selective_acc_at_90=selective_accuracy(conf, correct, 0.9),
        selective_acc_at_50=selective_accuracy(conf, correct, 0.5),
        aurc=area_under_risk_coverage(conf, correct),
        auroc=auroc(p, y),
        mean_confidence=float(conf.mean()),
        reliability=table,
    )
    if soft is not None:
        s = np.clip(np.asarray(soft, dtype=float), 0.0, 1.0)
        rep.tvd_to_human = float(np.abs(p - s).mean())
        ps = np.stack([1 - p, p], axis=1)
        ss = np.stack([1 - s, s], axis=1)
        rep.kl_human_model = float((ss * (np.log(np.clip(ss, EPS, 1)) - np.log(np.clip(ps, EPS, 1)))).sum(axis=1).mean())
    return rep
