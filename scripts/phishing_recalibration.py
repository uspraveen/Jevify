"""Can a few labelled emails undo the phishing collapse? A per-deployment threshold, measured honestly.

Every fine-tuned model in this project (and Tev1) keeps ranking PhishNChips emails about as well as its
untrained base (AUROC) but calls almost all of them legitimate: the threshold moved, not the knowledge.
The repair a deployment would actually make: label a few of its own emails and shift the model's
log-odds for "phishing" by one number fitted on them. This measures that, on emails the shift never saw:

- draw n labelled emails (half phishing, half legitimate), fit one additive bias b on the model's
  phishing log-odds by maximum likelihood (the slope stays 1: one number, not a new classifier);
- score every *other* email with sigmoid(logit(p) + b): accuracy, recall, false-positive rate, ECE;
- repeat over many random draws and report the mean and the 5th-95th percentile, because with a
  handful of labels the draw matters as much as the model.

    python scripts/phishing_recalibration.py --records <community-bench root> --models <models.json> --out <dir>

``models.json`` is the list community_report.py takes (label, test predictions, optional recipe).
Writes report.json and tables.md. Nothing here changes a model; it measures what one fitted number buys.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from community_report import POSITIVE, auroc, load, p_positive  # noqa: E402
from jevify.bench.metrics import expected_calibration_error  # noqa: E402
from jevify.runners.cli import iter_records  # noqa: E402

CONFIGS = ["phishnchips_verdict", "phishnchips_noul", "phishnchips_click", "phishnchips_minimal"]
SIZES = (16, 32, 64, 128)
DRAWS = 200
EPS = 1e-4


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1 + np.tanh(0.5 * x))          # no overflow at any x


def fit_bias(z: np.ndarray, y: np.ndarray) -> float:
    """argmax_b of the log-likelihood of y under sigmoid(z + b). Its derivative, sum(y - sigmoid(z + b)),
    falls monotonically in b, so bisection finds the root exactly. (Plain Newton overshoots on a
    confidently one-sided model -- a first version did, and reported degenerate shifts.)"""
    lo, hi = -40.0, 40.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if np.sum(y - _sigmoid(z + mid)) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def metrics(z: np.ndarray, y: np.ndarray, b: float) -> dict:
    p = _sigmoid(z + b)
    flag = p >= 0.5
    correct = (flag == (y == 1)).astype(float)
    ece, _, _ = expected_calibration_error(np.maximum(p, 1 - p), correct)
    return {"acc": float(correct.mean()), "recall": float(flag[y == 1].mean()), "fpr": float(flag[y == 0].mean()),
            "ece": float(ece)}


def study(z: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict:
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    out = {"n": int(len(y)), "auroc": auroc(y, z), "as_served": metrics(z, y, 0.0),
           "oracle": metrics(z, y, fit_bias(z, y))}      # the best one number can do, fitted on everything
    for n in SIZES:
        runs = []
        for _ in range(DRAWS):
            cal = np.concatenate([rng.choice(pos, n // 2, replace=False), rng.choice(neg, n // 2, replace=False)])
            rest = np.setdiff1d(np.arange(len(y)), cal)
            b = fit_bias(z[cal], y[cal])
            runs.append({**metrics(z[rest], y[rest], b), "bias": b})
        agg = {}
        for k in ("acc", "recall", "fpr", "ece", "bias"):
            v = np.array([r[k] for r in runs])
            agg[k] = {"mean": float(v.mean()), "p5": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95))}
        out[f"n{n}"] = agg
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260924)
    a = ap.parse_args()
    recs = {r.id: r for r in iter_records(a.records, CONFIGS, "test", None)}
    rep: dict = {}
    for entry in json.loads(a.models.read_text(encoding="utf-8")):
        preds = [p for p in load(entry, "test", a.records, "test") if not p.error and p.id in recs]
        rep[entry["label"]] = {}
        for src in CONFIGS:
            ps = [p for p in preds if p.id.split("/")[0] == src]
            if not ps:
                continue
            rs = [recs[p.id] for p in ps]
            y = np.array([1 if (int(r.label) == 1 if r.primitive == "noul" else str(r.label) == POSITIVE[src]) else 0
                          for r in rs])
            s = np.clip(np.array([p_positive(p, r) for p, r in zip(ps, rs)]), EPS, 1 - EPS)
            rng = np.random.default_rng([a.seed, len(rep), CONFIGS.index(src)])
            rep[entry["label"]][src] = study(np.log(s / (1 - s)), y, rng)
        print(f"{entry['label']}: done", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    L = []
    for src in CONFIGS:
        L += [f"### {src}", "",
              "| model | AUROC | as served: acc / recall / FPR | " + " | ".join(f"{n} labelled: acc / recall / FPR (recall 5–95%)" for n in SIZES)
              + " | best single shift: acc / recall / FPR |",
              "|---|---|---|" + "---|" * len(SIZES) + "---|"]
        for m, per in rep.items():
            if src not in per:
                continue
            g = per[src]
            f = lambda d: f"{d['acc']:.3f} / {d['recall']:.3f} / {d['fpr']:.3f}"
            cells = [f"{g[f'n{n}']['acc']['mean']:.3f} / {g[f'n{n}']['recall']['mean']:.3f} / {g[f'n{n}']['fpr']['mean']:.3f} "
                     f"({g[f'n{n}']['recall']['p5']:.2f}–{g[f'n{n}']['recall']['p95']:.2f})" for n in SIZES]
            L.append(f"| {m} | {g['auroc']:.3f} | {f(g['as_served'])} | " + " | ".join(cells) + f" | {f(g['oracle'])} |")
        L.append("")
    (a.out / "tables.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
