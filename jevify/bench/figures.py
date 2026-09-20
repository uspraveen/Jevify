"""Figures for System One evaluations: reliability, calibration map, risk–coverage,
model-vs-human probability, latency, and multi-model comparisons.

Static SVG/PNG via matplotlib so they render in dataset cards and READMEs.
Colors follow a validated palette: one hue per primitive (blue/orange/aqua —
the three slots that pass all-pairs CVD checks), single-hue ramps for ordered
series, neutral ink for all text.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..runners.base import Prediction, read_predictions
from .record import BenchRecord, read_jsonl

PRIM_COLOR = {"choice": "#2a78d6", "score": "#eb6834", "noul": "#1baf7a"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#1c5cab", "#104281", "#0d366b"]


PROVENANCE = ""   # set by make_all, e.g. "Jev 1.13.0 · jev-bench v0.1.1 · 2026-09-21"


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "medium", "axes.titlelocation": "left",
        "legend.frameon": False, "svg.fonttype": "none", "axes.axisbelow": True,
        "xtick.major.size": 0, "ytick.major.size": 0, "font.family": "DejaVu Sans",
    })
    return plt


def _headline(fig, title: str, subtitle: str | None = None) -> None:
    """Left-aligned title hanging from the top edge, with a muted subtitle wrapped to the figure width."""
    import textwrap
    h, w = fig.get_figheight(), fig.get_figwidth()
    fig.text(0.012, 1 - 0.08 / h, title, fontsize=11, weight="semibold", color=INK, va="top", ha="left")
    if subtitle:
        wrapped = textwrap.fill(subtitle, width=int(w * 14.5))
        fig.text(0.012, 1 - 0.30 / h, wrapped, fontsize=8.2, color=INK2, va="top", ha="left", linespacing=1.35)


def _footer(fig, note: str | None = None) -> None:
    text = " · ".join(t for t in (PROVENANCE, note) if t)
    if text:
        fig.text(0.012, 0.06 / fig.get_figheight(), text, fontsize=7, color=INK2, va="bottom", ha="left")


def _layout(fig, subtitle_lines: int = 1) -> tuple[float, float, float, float]:
    """tight_layout rect leaving room for the headline (top) and footer (bottom), in inches."""
    h = fig.get_figheight()
    return (0, 0.22 / h, 1, 1 - (0.50 + 0.15 * (subtitle_lines - 1)) / h)


# --------------------------------------------------------------------------- data assembly

def load_eval(records_root: Path, preds_path: Path, split: str = "test") -> dict[str, dict[str, Any]]:
    """Per-source arrays: conf (top-label), correct, p_true, human-vs-model pairs, latency."""
    preds = {p.id: p for p in read_predictions(preds_path) if not p.error}
    out: dict[str, dict[str, Any]] = {}
    for path in sorted((records_root / "data").glob(f"*/{split}.jsonl")):
        src = path.parent.name
        rows = [(r, preds[r.id]) for r in read_jsonl(path) if r.id in preds]
        if not rows:
            continue
        prim = rows[0][0].primitive
        conf, correct, lat, hm_x, hm_y = [], [], [], [], []
        for r, p in rows:
            if prim == "noul":
                py = float(p.p_yes)
                conf.append(max(py, 1 - py)); correct.append(int((py >= 0.5) == bool(int(r.label))))
                if isinstance(r.soft_label, (int, float)):
                    hm_x.append(float(r.soft_label)); hm_y.append(py)
            else:
                keys = r.option_keys()
                probs = np.array([p.probabilities.get(k, 0.0) for k in keys])
                conf.append(float(probs.max())); correct.append(int(keys[int(probs.argmax())] == str(r.label)))
                if r.soft_label is not None:
                    sl = r.soft_label
                    human = [sl[k] for k in keys] if isinstance(sl, dict) else list(sl)
                    hm_x.extend(map(float, human)); hm_y.extend(map(float, probs))
            if p.latency_ms:
                lat.append(p.latency_ms)
        out[src] = {"primitive": prim, "conf": np.array(conf), "correct": np.array(correct, dtype=float),
                    "latency": np.array(lat), "human": np.array(hm_x), "model": np.array(hm_y), "n": len(rows)}
    return out


# --------------------------------------------------------------------------- figures

def fig_reliability(ev: dict[str, dict[str, Any]], out: Path, n_bins: int = 15, cols: int = 6, model: str = "") -> Path:
    plt = _mpl()
    names = sorted(ev, key=lambda s: (ev[s]["primitive"], s))
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.1 * cols, 2.25 * rows + 0.9), squeeze=False)
    edges = np.linspace(0, 1, n_bins + 1)
    for ax, src in zip(axes.flat, names):
        d = ev[src]; c = PRIM_COLOR[d["primitive"]]
        ax.plot([0, 1], [0, 1], color=GRID, lw=1, zorder=1)
        xs, ys, ns = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (d["conf"] > lo) & (d["conf"] <= hi) if lo > 0 else (d["conf"] >= lo) & (d["conf"] <= hi)
            if m.sum() >= 5:
                xs.append(d["conf"][m].mean()); ys.append(d["correct"][m].mean()); ns.append(m.sum())
        ns = np.array(ns, dtype=float)
        ax.plot(xs, ys, color=c, lw=1.5, zorder=2)
        ax.scatter(xs, ys, s=8 + 60 * ns / max(ns.max(), 1), color=c, zorder=3, linewidths=0.8, edgecolors=SURFACE)
        ece = _ece(d["conf"], d["correct"], n_bins)   # identical definition and binning to the metrics table
        ax.set_title(f"{src}\nECE {ece:.3f}  ·  n={d['n']:,}", loc="left")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.tick_params(length=0)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    fig.supxlabel("stated confidence (top-label probability)", color=INK2, fontsize=9)
    fig.supylabel("observed accuracy", color=INK2, fontsize=9)
    _legend_prims(fig, ev, loc="upper right")
    r = _layout(fig, 2)
    fig.tight_layout(rect=(0.02, r[1], 1, r[3]))
    _headline(fig, f"{model}: reliability diagrams, one per jev-bench config",
              "Diagonal = perfectly calibrated. Dot size = records in the bin (15 equal-width bins; bins with < 5 records hidden). A flat line means confidence carries no information.")
    _footer(fig)
    return _save(fig, out / "reliability")


def bootstrap_ci(conf: np.ndarray, correct: np.ndarray, n_boot: int = 300, seed: int = 0) -> tuple[tuple[float, float], tuple[float, float]]:
    """95% bootstrap intervals for (accuracy, ECE)."""
    rng = np.random.default_rng(seed)
    n = len(conf)
    accs, eces = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        accs.append(correct[idx].mean()); eces.append(_ece(conf[idx], correct[idx]))
    return (float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))), (float(np.percentile(eces, 2.5)), float(np.percentile(eces, 97.5)))


def fig_calibration_map(ev: dict[str, dict[str, Any]], out: Path, model: str) -> Path:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    pts = []
    for src, d in ev.items():
        acc = float(d["correct"].mean()); ece = _ece(d["conf"], d["correct"])
        (alo, ahi), (elo, ehi) = bootstrap_ci(d["conf"], d["correct"])
        c = PRIM_COLOR[d["primitive"]]
        ax.plot([alo, ahi], [ece, ece], color=c, lw=0.9, alpha=0.55, zorder=2)
        ax.plot([acc, acc], [elo, ehi], color=c, lw=0.9, alpha=0.55, zorder=2)
        ax.scatter(acc, ece, s=34, color=c, zorder=3, linewidths=0.8, edgecolors=SURFACE)
        pts.append((src, acc, ece, d["primitive"]))
    ymax = max(0.4, max(p[2] for p in pts) + 0.03)
    # ideal region: high accuracy, low calibration error
    ax.add_patch(plt.Rectangle((0.85, 0), 0.30, 0.07, facecolor="#1baf7a", alpha=0.08, edgecolor="none", zorder=0))
    ax.annotate("", xy=(1.03, 0.02), xytext=(0.62, 0.30), arrowprops={"arrowstyle": "-|>", "color": GRID, "lw": 1.0}, zorder=0)
    ax.text(0.615, 0.305, "better", fontsize=7.5, color=INK2, va="bottom")
    ax.set_xlabel("accuracy"); ax.set_ylabel("expected calibration error (lower is better)")
    ax.set_xlim(0.25, 1.12); ax.set_ylim(0, ymax)
    from matplotlib.lines import Line2D
    prims = [q for q in ("choice", "score", "noul") if any(d["primitive"] == q for d in ev.values())]
    ax.legend(handles=[Line2D([0], [0], marker="o", color=PRIM_COLOR[q], lw=0, markersize=6, label=q) for q in prims],
              loc="lower left", fontsize=8)
    fig.tight_layout(rect=_layout(fig, 2))
    _headline(fig, f"{model}: accuracy vs calibration, one point per jev-bench config",
              "Test splits: 1,000 records per config (2,000 civil_comments; 1,599 chaosnli). Bars are 95% bootstrap intervals. "
              "Down and to the right is better; the shaded corner (accuracy ≥ 0.85, ECE ≤ 0.07) is where a decision model earns its confidence.")
    _footer(fig)
    _annotate_without_overlap(fig, ax, [(src, acc, ece) for src, acc, ece, _ in pts])
    return _save(fig, out / "calibration_map")


def fig_vs_cardinality(ev: dict[str, dict[str, Any]], out: Path, model: str, k_of: dict[str, int]) -> Path | None:
    """Accuracy and ECE against the number of allowed answers, per primitive.
    Cross-dataset, so difficulty is confounded — a hypothesis view, not a result."""
    plt = _mpl()
    pts = [(src, k_of[src], float(d["correct"].mean()), _ece(d["conf"], d["correct"]), d["primitive"]) for src, d in ev.items() if src in k_of]
    if not pts:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for ax, name in zip(axes, ("accuracy", "expected calibration error")):
        col = 2 if name == "accuracy" else 3
        for src, k, acc, ece, prim in pts:
            ax.scatter(k, (acc, ece)[col - 2], s=30, color=PRIM_COLOR[prim], zorder=3, linewidths=0.8, edgecolors=SURFACE)
        ax.set_xscale("log"); ax.set_xlabel("number of allowed answers (K, log scale)"); ax.set_ylabel(name)
        ax.set_xticks([2, 3, 5, 10, 28, 60, 100, 151]); ax.set_xticklabels(["2", "3", "5", "10", "28", "60", "100", "151"])
        ax.set_xlim(1.7, 230); ax.tick_params(length=0)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0], [0], marker="o", color=PRIM_COLOR[q], lw=0, markersize=6, label=q) for q in ("choice", "score", "noul")],
               loc="upper right", ncol=3, fontsize=8, bbox_to_anchor=(0.99, 0.995))
    fig.tight_layout(rect=_layout(fig, 2))   # layout first, then measure and place labels
    _headline(fig, f"{model}: performance vs decision-set size, across configs",
              "Cross-dataset, so task difficulty is confounded with K — a hypothesis view. The within-item cardinality probe is the controlled version.")
    _footer(fig)
    for ax, col in zip(axes, (2, 3)):
        singles = [(src, k, (acc, ece)[col - 2]) for src, k, acc, ece, prim in pts if k > 2]
        two = [(acc, ece)[col - 2] for src, k, acc, ece, prim in pts if k == 2]
        if two:   # one label for the K=2 cluster, placed by the same overlap-avoiding routine
            singles.append((f"{len(two)} noul configs (K=2)", 2, float(min(two) if col == 2 else max(two))))
        _annotate_without_overlap(fig, ax, singles, fontsize=6.5)
    return _save(fig, out / "vs_cardinality")


def fig_risk_coverage(ev: dict[str, dict[str, Any]], out: Path, model: str = "") -> Path:
    plt = _mpl()
    prims = ["choice", "score", "noul"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    for ax, prim in zip(axes, prims):
        srcs = sorted([s for s in ev if ev[s]["primitive"] == prim], key=lambda s: -ev[s]["correct"].mean())
        ramp = BLUE_RAMP[1:] if prim == "choice" else BLUE_RAMP[2:]
        ends = []
        for i, src in enumerate(srcs):
            d = ev[src]
            order = np.argsort(-d["conf"], kind="stable")
            err = 1 - d["correct"][order]
            cov = np.arange(1, len(err) + 1) / len(err)
            risk = np.cumsum(err) / np.arange(1, len(err) + 1)
            color = ramp[min(i * len(ramp) // max(len(srcs), 1), len(ramp) - 1)]
            ax.plot(cov, risk, lw=1.4, color=color, zorder=2)
            ends.append((src, 1.0, float(risk[-1])))
        ax.set_title(prim, loc="left")
        ax.set_xlabel("coverage (fraction of records acted on)"); ax.set_xlim(0, 1.42); ax.set_ylim(0, None)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.tick_params(length=0)
        _end_labels(fig, ax, ends)
    axes[0].set_ylabel("error rate on the covered records")
    fig.tight_layout(rect=_layout(fig, 2))
    _headline(fig, f"{model}: risk–coverage curves",
              "Records ranked by stated confidence; x = fraction acted on, y = error rate among them. This is the curve a confidence-gated router lives on.")
    _footer(fig)
    return _save(fig, out / "risk_coverage")


def _end_labels(fig, ax, ends: list[tuple[str, float, float]], fontsize: float = 6.8) -> None:
    """Right-of-line labels, nudged vertically until no two overlap (measured)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed = []
    for src, x, y in sorted(ends, key=lambda e: e[2]):
        for dy in (0, 5, -5, 9, -9, 14, -14, 19, -19, 24, -24):
            ann = ax.annotate(src, (x, y), xytext=(4, dy), textcoords="offset points", fontsize=fontsize, color=INK2, va="center")
            box = ann.get_window_extent(renderer).expanded(1.0, 1.15)
            if not any(box.overlaps(b) for b in placed):
                placed.append(box)
                break
            ann.remove()
        else:
            placed.append(ax.annotate(src, (x, y), xytext=(4, 0), textcoords="offset points", fontsize=fontsize, color=INK2, va="center").get_window_extent(renderer))


def fig_human_vs_model(ev: dict[str, dict[str, Any]], out: Path, model: str, n_bins: int = 10) -> Path | None:
    plt = _mpl()
    srcs = [s for s in ev if len(ev[s]["human"])]
    if not srcs:
        return None
    fig, axes = plt.subplots(1, len(srcs), figsize=(3.6 * len(srcs), 3.5), squeeze=False)
    edges = np.linspace(0, 1, n_bins + 1)
    for ax, src in zip(axes[0], srcs):
        d = ev[src]; c = PRIM_COLOR[d["primitive"]]
        ax.plot([0, 1], [0, 1], color=GRID, lw=1, zorder=1)
        ax.scatter(d["human"], d["model"], s=4, color=c, alpha=0.12, zorder=2, linewidths=0)
        xs, ys, ns = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (d["human"] > lo) & (d["human"] <= hi) if lo > 0 else (d["human"] >= lo) & (d["human"] <= hi)
            if m.sum() >= 10:
                xs.append(d["human"][m].mean()); ys.append(d["model"][m].mean()); ns.append(int(m.sum()))
        ax.plot(xs, ys, color=c, lw=1.8, zorder=3)
        ax.scatter(xs, ys, s=22, color=c, zorder=4, linewidths=0.8, edgecolors=SURFACE)
        tvd = float(np.abs(d["human"] - d["model"]).mean())
        ax.set_title(f"{src}\nmean |model − human| per option = {tvd:.3f}  ·  {len(d['human']):,} pairs", loc="left")
        ax.set_xlabel("human probability (share of annotators)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.tick_params(length=0)
    axes[0][0].set_ylabel("model probability for the same option")
    fig.tight_layout(rect=_layout(fig, 2))
    _headline(fig, f"{model}: does the model's probability track how humans actually split?",
              "Each dot is one (item, option) pair on a calibration-gold config; the line is the mean model probability per human-probability bin. Diagonal = perfect agreement.")
    _footer(fig)
    return _save(fig, out / "human_vs_model")


def fig_latency(ev: dict[str, dict[str, Any]], out: Path, model: str) -> Path | None:
    plt = _mpl()
    lat = np.concatenate([d["latency"] for d in ev.values() if len(d["latency"])]) if any(len(d["latency"]) for d in ev.values()) else None
    if lat is None or not len(lat):
        return None
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    hi = np.percentile(lat, 99)
    ax.hist(lat[lat <= hi], bins=60, color=PRIM_COLOR["choice"], edgecolor=SURFACE, linewidth=0.4)
    p50, p90, p99 = np.percentile(lat, [50, 90, 99])
    for v, lab in ((p50, "p50"), (p90, "p90"), (p99, "p99")):
        ax.axvline(v, color=INK2, lw=0.8, ls=(0, (3, 3)))
        ax.annotate(f"{lab} {v:.0f} ms", (v, ax.get_ylim()[1]), xytext=(3, -10), textcoords="offset points", fontsize=7.5, color=INK2)
    ax.set_xlabel("client-observed latency per request (ms)"); ax.set_ylabel("requests")
    fig.tight_layout(rect=_layout(fig))
    _headline(fig, f"{model}: latency over {len(lat):,} requests, one question each",
              "Measured from a 2-core sandbox in us-east at concurrency 4; includes network. Top 1% trimmed for display.")
    _footer(fig)
    return _save(fig, out / "latency")


def fig_compare(evs: dict[str, dict[str, dict[str, Any]]], out: Path, metric: str = "accuracy") -> Path:
    """Grouped horizontal bars: one row per source, one bar per model. First model = reference."""
    plt = _mpl()
    models = list(evs)
    sources = sorted({s for ev in evs.values() for s in ev}, key=lambda s: (next(iter(evs.values())).get(s, {}).get("primitive", ""), s))
    vals = {m: [_metric(evs[m].get(s), metric) for s in sources] for m in models}
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(sources) * max(len(models), 2) / 2 + 1.5))
    h = 0.8 / len(models)
    ramp = [BLUE_RAMP[6], BLUE_RAMP[3], BLUE_RAMP[1], "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
    for i, m in enumerate(models):
        y = np.arange(len(sources)) + (i - (len(models) - 1) / 2) * h
        ax.barh(y, [v if v is not None else 0 for v in vals[m]], height=h * 0.92, color=ramp[i % len(ramp)], label=m, edgecolor=SURFACE, linewidth=0.5)
    ax.set_yticks(np.arange(len(sources))); ax.set_yticklabels(sources, fontsize=7.5)
    ax.invert_yaxis(); ax.set_xlabel(metric); ax.tick_params(length=0)
    ax.legend(loc="lower right", fontsize=7.5)
    ax.set_title(f"{metric} per jev-bench config", loc="left")
    fig.tight_layout()
    return _save(fig, out / f"compare_{metric}")


# --------------------------------------------------------------------------- helpers

def _metric(d: dict[str, Any] | None, metric: str) -> float | None:
    if d is None:
        return None
    if metric == "accuracy":
        return float(d["correct"].mean())
    if metric == "ece":
        return _ece(d["conf"], d["correct"])
    raise ValueError(metric)


def _ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def _legend_prims(fig, ev, loc="upper left"):
    from matplotlib.lines import Line2D
    prims = [p for p in ("choice", "score", "noul") if any(d["primitive"] == p for d in ev.values())]
    handles = [Line2D([0], [0], marker="o", color=PRIM_COLOR[p], lw=0, markersize=6, label=p) for p in prims]
    fig.legend(handles=handles, loc=loc, ncol=len(prims), fontsize=8, bbox_to_anchor=(0.99, 0.995) if loc == "upper right" else (0.01, 0.995))


def _annotate_without_overlap(fig, ax, pts: list[tuple[str, float, float]], fontsize: float = 7.2) -> None:
    """Place each label at the first candidate offset whose rendered bbox overlaps
    neither an already-placed label, a data point, nor the axes edge. Far offsets
    get a thin leader line. Measured with the real renderer, not guessed."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed = []
    point_boxes = []
    for _, x, y in pts:
        px, py = ax.transData.transform((x, y))
        point_boxes.append(_box(px - 5, py - 5, px + 5, py + 5))
    near = [(5, 3), (5, -9), (-5, 3), (-5, -9), (5, 11), (-5, 11)]
    far = [(14, 16), (14, -18), (-14, 16), (-14, -18), (22, 28), (22, -30), (-22, 28), (-22, -30), (30, 40), (-30, 40)]
    axes_box = ax.get_window_extent(renderer)
    for src, x, y in sorted(pts, key=lambda q: (q[1], q[2])):
        chosen = None
        for dx, dy in near + far:
            ha = "left" if dx > 0 else "right"
            kw = {}
            if (dx, dy) in far:
                kw["arrowprops"] = {"arrowstyle": "-", "color": GRID, "lw": 0.7, "shrinkA": 0, "shrinkB": 3}
            ann = ax.annotate(src, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize, color=INK2, ha=ha, **kw)
            box = ann.get_window_extent(renderer).expanded(1.06, 1.2)
            inside = box.x0 >= axes_box.x0 - 2 and box.x1 <= axes_box.x1 + 2 and box.y0 >= axes_box.y0 - 2 and box.y1 <= axes_box.y1 + 2
            clash = any(box.overlaps(b) for b in placed) or any(box.overlaps(b) for b in point_boxes if not _contains(b, x, y, ax))
            if inside and not clash:
                chosen = box
                break
            ann.remove()
        if chosen is None:   # nothing fit: fall back to the nearest offset
            ann = ax.annotate(src, (x, y), xytext=near[0], textcoords="offset points", fontsize=fontsize, color=INK2)
            chosen = ann.get_window_extent(renderer)
        placed.append(chosen)


def _box(x0, y0, x1, y1):
    from matplotlib.transforms import Bbox
    return Bbox([[x0, y0], [x1, y1]])


def _contains(box, x, y, ax) -> bool:
    px, py = ax.transData.transform((x, y))
    return box.x0 <= px <= box.x1 and box.y0 <= py <= box.y1


def _save(fig, stem: Path) -> Path:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return stem.with_suffix(".svg")


def make_all(records_root: Path, preds_path: Path, out: Path, model: str, split: str = "test", provenance: str | None = None) -> list[Path]:
    global PROVENANCE
    if provenance is None:
        import datetime as _dt
        manifest = records_root / "manifest.json"
        version = json.loads(manifest.read_text(encoding="utf-8")).get("version", "?") if manifest.exists() else "?"
        provenance = f"{model} · jev-bench v{version} · {_dt.date.today().isoformat()}"
    PROVENANCE = provenance
    ev = load_eval(records_root, preds_path, split)
    made = [fig_reliability(ev, out, model=model), fig_calibration_map(ev, out, model), fig_risk_coverage(ev, out, model)]
    k_of = _k_from_records(records_root, ev, split)
    for f in (fig_human_vs_model(ev, out, model), fig_latency(ev, out, model), fig_vs_cardinality(ev, out, model, k_of)):
        if f:
            made.append(f)
    return made


def _k_from_records(records_root: Path, ev: dict[str, Any], split: str) -> dict[str, int]:
    """Median number of allowed answers per source (2 for noul)."""
    out = {}
    for src in ev:
        path = records_root / "data" / src / f"{split}.jsonl"
        ks = [len(r.option_keys()) for r in read_jsonl(path, limit=200)]
        if ks:
            out[src] = int(np.median(ks))
    return out


def fig_cardinality_probe(probe_json: Path, out: Path, model: str) -> Path | None:
    """Within-item cardinality curves from `jevify-bench probe-report` output."""
    plt = _mpl()
    an = json.loads(Path(probe_json).read_text(encoding="utf-8"))
    card = an.get("per_config", {}).get("cardinality", {})
    if not card:
        return None
    by_src: dict[str, list[tuple[int, float, float, float]]] = defaultdict(list)
    for key, e in card.items():
        src, var = key.split("|")
        by_src[src].append((int(var[1:]), e["accuracy"], e["ece"], e["mean_p_gold"]))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    srcs = sorted(by_src, key=lambda s: -max(v[1] for v in by_src[s]))
    ramp = [BLUE_RAMP[7], BLUE_RAMP[5], BLUE_RAMP[3], BLUE_RAMP[1], "#eb6834"]
    for ax, (idx, name) in zip(axes, ((1, "accuracy"), (3, "mean P(gold)"), (2, "expected calibration error"))):
        ends = []
        for i, src in enumerate(srcs):
            pts = sorted(by_src[src])
            xs = [p[0] for p in pts]; ys = [p[idx] for p in pts]
            color = "#eb6834" if src == "go_emotions" else ramp[i % 4]
            ax.plot(xs, ys, marker="o", markersize=3.5, lw=1.5, color=color, zorder=3)
            ends.append((src, xs[-1], ys[-1]))
        ax.set_xscale("log"); ax.set_xticks([2, 5, 10, 25, 50, 100]); ax.set_xticklabels(["2", "5", "10", "25", "50", "100"])
        ax.set_xlim(1.7, 400); ax.set_xlabel("options offered (gold always included)"); ax.set_ylabel(name); ax.tick_params(length=0)
        ax.set_title(name, loc="left")
        _end_labels(fig, ax, ends)
    fig.tight_layout(rect=_layout(fig, 2))
    _headline(fig, f"{model}: the same items with more distractors — cardinality isolated from difficulty",
              "200 items per source; the gold option is always present, K−1 distractors drawn from the source's own labels. Past a source's label count the curve repeats the full set.")
    _footer(fig)
    return _save(fig, out / "probe_cardinality")
