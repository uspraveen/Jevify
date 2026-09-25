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
    # a long model name used to run off the right edge of the canvas; the title wraps now,
    # and the subtitle drops by however many lines the title took
    title_lines = textwrap.wrap(title, width=max(20, int(w * 9.3))) or [title]
    fig.text(0.012, 1 - 0.08 / h, chr(10).join(title_lines), fontsize=11, weight="semibold", color=INK,
             va="top", ha="left", linespacing=1.3)
    if subtitle:
        wrapped = textwrap.fill(subtitle, width=int(w * 14.5))
        fig.text(0.012, 1 - (0.30 + 0.17 * (len(title_lines) - 1)) / h, wrapped, fontsize=8.2, color=INK2,
                 va="top", ha="left", linespacing=1.35)


def _wrapped_lines(fig, title: str, subtitle: str | None) -> int:
    """How many text lines the headline will occupy, for reserving top margin."""
    import textwrap
    w = fig.get_figwidth()
    n = len(textwrap.wrap(title, width=max(20, int(w * 9.3))) or [title])
    if subtitle:
        n += len(textwrap.fill(subtitle, width=int(w * 14.5)).split(chr(10)))
    return n


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
    xlo = min(0.25, min(p[1] for p in pts) - 0.04)
    # Accuracy cannot exceed 1.0. The axis used to run to 1.12 and the "ideal" band to 1.15,
    # drawing a region no model can reach and inviting the eye to read the gap as headroom.
    ax.add_patch(plt.Rectangle((0.85, 0), 1.0 - 0.85, 0.07, facecolor="#1baf7a", alpha=0.08,
                               edgecolor="none", zorder=0))
    ax.set_xlabel("accuracy"); ax.set_ylabel("expected calibration error (lower is better)")
    ax.set_xlim(xlo, 1.0); ax.set_ylim(0, ymax)
    # direction cue in axes coordinates, so it lands in the same empty upper-right corner for
    # every model, with the word at the arrow's head rather than its tail
    ax.annotate("", xy=(0.88, 0.70), xytext=(0.64, 0.88), xycoords="axes fraction",
                textcoords="axes fraction", arrowprops={"arrowstyle": "-|>", "color": GRID, "lw": 1.0}, zorder=0)
    ax.text(0.895, 0.695, "better", fontsize=7.5, color=INK2, va="center", ha="left",
            transform=ax.transAxes)
    from matplotlib.lines import Line2D
    prims = [q for q in ("choice", "score", "noul") if any(d["primitive"] == q for d in ev.values())]
    ax.legend(handles=[Line2D([0], [0], marker="o", color=PRIM_COLOR[q], lw=0, markersize=6, label=q) for q in prims],
              loc="lower left", fontsize=8)

    # Labelling all 22 points crushed the high-accuracy corner into unreadable overlap. The
    # corner is the uniform "good" region, so its members are named in the subtitle and only
    # the points that carry information by position are labelled on the plot.
    in_corner = sorted(src for src, acc, ece, _ in pts if acc >= 0.85 and ece <= 0.07)
    labelled = [(src, acc, ece) for src, acc, ece, _ in pts if src not in set(in_corner)]
    title = f"{model}: accuracy vs calibration, one point per jev-bench config"
    subtitle = ("Test splits: 1,000 records per config (2,000 civil_comments; 1,599 chaosnli). Bars are 95% "
                "bootstrap intervals. Down and to the right is better; the shaded corner (accuracy ≥ 0.85, "
                "ECE ≤ 0.07) is where a decision model earns its confidence.")
    if in_corner:
        subtitle += "  Unlabelled, inside the corner: " + ", ".join(in_corner) + "."
    fig.tight_layout(rect=_layout(fig, _wrapped_lines(fig, title, subtitle) - 1))
    _headline(fig, title, subtitle)
    _footer(fig)
    # the unlabelled corner points are still markers: a label must not be drawn over them
    _annotate_without_overlap(fig, ax, labelled, obstacles=[(acc, ece) for src, acc, ece, _ in pts if src in set(in_corner)])
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


METRIC_LABEL = {"accuracy": ("accuracy", "higher is better"), "ece": ("expected calibration error", "lower is better"),
                "brier": ("Brier score", "lower is better")}
REFERENCE_COLOR = "#eb6834"          # Jev, everywhere in the project


def _ordered_sources(evs: dict[str, dict[str, Any]], prim_of: dict[str, str] | None) -> list[tuple[str, str]]:
    """(primitive, source) pairs in display order: grouped by primitive, then by name."""
    srcs = {s for ev in evs.values() for s in ev}

    def prim(src: str) -> str:
        if prim_of and src in prim_of:
            return prim_of[src]
        for ev in evs.values():
            if src in ev and isinstance(ev[src], dict) and ev[src].get("primitive"):
                return ev[src]["primitive"]
        return ""
    order = {"choice": 0, "score": 1, "noul": 2}
    return sorted(((prim(sname), sname) for sname in srcs), key=lambda t: (order.get(t[0], 9), t[1]))


def fig_compare(evs: dict[str, dict[str, dict[str, Any]]], out: Path, metric: str = "accuracy",
                prim_of: dict[str, str] | None = None) -> Path:
    """Grouped horizontal bars, one row per config: a model (or a few) against a reference.

    The first model is the reference and is drawn in the project's Jev colour so it reads
    the same here as on the models map. Meant for two to four models; a matrix of many
    models is a heatmap (``fig_metric_heatmap``), not a forest of bars.
    """
    plt = _mpl()
    models = list(evs)
    groups = _ordered_sources(evs, prim_of)
    sources = [sname for _, sname in groups]
    vals = {m: [_metric(evs[m].get(sname), metric) for sname in sources] for m in models}
    label, better = METRIC_LABEL.get(metric, (metric, ""))
    fig, ax = plt.subplots(figsize=(8.2, 0.30 * len(sources) * max(len(models), 2) / 2 + 2.2))
    h = 0.8 / len(models)
    palette = [REFERENCE_COLOR, BLUE_RAMP[5], BLUE_RAMP[2], "#1baf7a"]
    for i, m in enumerate(models):
        y = np.arange(len(sources)) + (i - (len(models) - 1) / 2) * h
        ax.barh(y, [v if v is not None else 0 for v in vals[m]], height=h * 0.9,
                color=palette[i % len(palette)], label=m, edgecolor=SURFACE, linewidth=0.5, zorder=3)
    # a thin rule between primitive groups, and the group's name just outside the right edge: inside
    # it, the name sat on the end of any bar near the top of the scale (ARC at 0.98)
    prev = None
    for i, (prim, _s) in enumerate(groups):
        if prim != prev:
            if i:
                ax.axhline(i - 0.5, color=GRID, lw=0.9, zorder=1)
            ax.text(1.01, i - 0.42, prim, transform=ax.get_yaxis_transform(), ha="left", va="top",
                    fontsize=7.2, color=INK2, style="italic")
            prev = prim
    ax.set_yticks(np.arange(len(sources))); ax.set_yticklabels(sources, fontsize=7.5)
    ax.invert_yaxis(); ax.set_xlabel(label); ax.tick_params(length=0)
    ax.set_xlim(left=0)
    ax.grid(True, axis="x", color=GRID, lw=0.7, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    # above the bars, not inside them: in the lower-right corner the legend covered the last row's
    # bars whenever they passed about half the scale, which on accuracy is nearly always
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=min(len(models), 2), fontsize=7.5,
              frameon=False, borderaxespad=0.0, handlelength=1.6)
    fig.tight_layout(rect=_layout(fig, 1))
    _headline(fig, f"{' vs '.join(models[1:] + [models[0]])}: {label} per jev-bench config",
              f"Same test records for every model. {better.capitalize()}.")
    _footer(fig)
    return _save(fig, out / f"compare_{metric}")


def fig_metric_heatmap(entries: list[dict[str, Any]], prim_of: dict[str, str], out: Path,
                       metric: str = "accuracy") -> Path:
    """Models x configs, one cell per (model, config), for the whole leaderboard.

    Grouped bars stop working past a handful of models -- the previous version of this
    chart was 4,800 pixels tall, showed six of thirteen models, and used three shades of
    blue nobody could tell apart. A matrix of magnitudes is a heatmap: one sequential hue,
    the value printed in each cell, rows in leaderboard order, columns grouped by primitive.
    Built from test_metrics.json, so it needs no predictions in memory.
    """
    plt = _mpl()
    from matplotlib.colors import LinearSegmentedColormap

    label, better = METRIC_LABEL.get(metric, (metric, ""))
    evs = {e["label"]: e["metrics"] for e in entries}
    groups = _ordered_sources(evs, prim_of)
    sources = [sname for _, sname in groups]
    models = [e["label"] for e in entries]
    grid = np.array([[_metric(evs[m].get(sname), metric) for sname in sources] for m in models], dtype=float)
    cmap = LinearSegmentedColormap.from_list("blue_seq", [SURFACE] + BLUE_RAMP, N=256)
    cmap.set_bad(GRID)
    vmax = float(np.nanmax(grid))
    vmin = 0.0 if metric != "accuracy" else float(max(0.0, np.nanmin(grid) - 0.05))

    fig, ax = plt.subplots(figsize=(0.46 * len(sources) + 3.6, 0.36 * len(models) + 2.4))
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(sources))); ax.set_xticklabels(sources, rotation=48, ha="right", fontsize=7.2)
    ax.set_yticks(np.arange(len(models))); ax.set_yticklabels(models, fontsize=7.6)
    ax.tick_params(length=0)
    ax.grid(False)                                       # the house grid would cross the cells
    for spine in ax.spines.values():
        spine.set_visible(False)
    # value in every cell; light ink on dark cells so the numbers are readable anywhere
    span = max(vmax - vmin, 1e-9)
    for i in range(len(models)):
        for j in range(len(sources)):
            v = grid[i, j]
            if np.isnan(v):
                continue
            dark = (v - vmin) / span > 0.55
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.3, color=SURFACE if dark else INK)
    # primitive group separators and labels above the columns
    prev, start = None, 0
    for j, (prim, _s) in enumerate(groups + [("", "")]):
        if prim != prev:
            if prev is not None:
                ax.text((start + j - 1) / 2, -0.9, prev, ha="center", va="bottom", fontsize=8, color=INK2, style="italic")
                if j < len(groups):
                    ax.axvline(j - 0.5, color=SURFACE, lw=2.2)
            prev, start = prim, j
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.012)
    cb.ax.tick_params(labelsize=7, length=0, colors=INK2)
    cb.outline.set_visible(False)
    fig.tight_layout(rect=_layout(fig, 1))
    _headline(fig, f"{label} per config, every model on jev-bench",
              f"Rows in leaderboard order; the first is Jev. Same test records for every model. {better.capitalize()}.")
    _footer(fig)
    return _save(fig, out / f"heatmap_{metric}")


# --------------------------------------------------------------------------- helpers

def _metric(d: dict[str, Any] | None, metric: str) -> float | None:
    """A config's metric from either a per-record eval (arrays) or a Report dict.

    Accepting the Report form means every comparison figure can be rebuilt from the
    published test_metrics.json files alone -- no predictions in memory, seconds not minutes.
    """
    if d is None:
        return None
    if "correct" in d:                                   # per-record eval from load_eval
        if metric == "accuracy":
            return float(d["correct"].mean())
        if metric == "ece":
            return _ece(d["conf"], d["correct"])
        raise ValueError(metric)
    v = d.get(metric)
    return None if v is None else float(v)


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


def _annotate_without_overlap(fig, ax, pts: list[tuple[str, float, float]], fontsize: float = 7.2,
                              obstacles: list[tuple[float, float]] | None = None) -> None:
    """Place each label at the first candidate offset whose rendered bbox overlaps
    neither an already-placed label, a data point, nor the axes edge. Far offsets
    get a thin leader line. Measured with the real renderer, not guessed.

    ``obstacles`` are extra data coordinates to keep labels off -- on a line chart, every
    plotted point, so a label does not land on a neighbouring series."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed = []
    leaders = []                  # ((x0, y0), (x1, y1)) of each leader drawn so far, in display pixels
    point_boxes = []
    # half-width of the largest marker drawn with labels (s = 74 pt², plus its edge) at the draw dpi;
    # a 5 px box let the arms of a plus marker sit on a neighbour's label unnoticed
    r = 0.5 * (74 ** 0.5 + 1.6) * fig.dpi / 72
    for _, x, y in pts:
        px, py = ax.transData.transform((x, y))
        point_boxes.append(_box(px - r, py - r, px + r, py + r))
    for x, y in obstacles or []:
        px, py = ax.transData.transform((x, y))
        point_boxes.append(_box(px - r, py - r, px + r, py + r))
    near = [(5, 3), (5, -9), (-5, 3), (-5, -9), (5, 11), (-5, 11)]
    # the last ring is only reached when everything nearer is taken: a tight cluster (five models within
    # 0.03 accuracy of each other on the leaderboard map) otherwise fell back to an overlapping spot
    far = [(14, 16), (14, -18), (-14, 16), (-14, -18), (22, 28), (22, -30), (-22, 28), (-22, -30), (30, 40), (-30, 40),
           (10, -44), (-10, -44), (40, -52), (-40, -52), (10, 50), (-10, 50), (45, 58), (-45, 58), (20, -66), (-20, -66)]
    # a label that had to move away from its point gets a thin line back to it; without
    # one, a displaced label in a crowded region reads as belonging to nothing
    leader = {"arrowstyle": "-", "color": INK2, "alpha": 0.45, "lw": 0.7, "shrinkA": 0, "shrinkB": 3}
    axes_box = ax.get_window_extent(renderer)
    for src, x, y in sorted(pts, key=lambda q: (q[1], q[2])):
        chosen = None
        for dx, dy in near + far:
            ha = "left" if dx > 0 else "right"
            kw = {}
            if (dx, dy) in far:
                kw["arrowprops"] = leader
            ann = ax.annotate(src, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize, color=INK2, ha=ha, **kw)
            box = _text_extent(ann, renderer).expanded(1.06, 1.2)
            inside = box.x0 >= axes_box.x0 - 2 and box.x1 <= axes_box.x1 + 2 and box.y0 >= axes_box.y0 - 2 and box.y1 <= axes_box.y1 + 2
            clash = any(box.overlaps(b) for b in placed) or any(box.overlaps(b) for b in point_boxes if not _contains(b, x, y, ax))
            # a label on an earlier leader line hides which point that leader belongs to
            clash = clash or any(_leader_crosses(box, b, [box], from_point=a) for a, b in leaders)
            seg = None
            if not clash and (dx, dy) in far:
                # a leader that runs past another marker, through another label, or across another
                # leader reads as pointing at the wrong thing
                others = placed + [b for b in point_boxes if not _contains(b, x, y, ax)]
                seg = ((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2), tuple(ax.transData.transform((x, y)))
                clash = _leader_crosses(box, seg[1], others) or any(_segments_cross(*seg, *l) for l in leaders)
            if inside and not clash:
                chosen = box
                if seg:
                    leaders.append(seg)
                break
            ann.remove()
        if chosen is None:
            # Nothing fit. Falling back to the first offset is what produced the collided
            # labels in the dense high-accuracy corner; pick the candidate that overlaps
            # least instead, so a crowded plot degrades rather than breaks.
            best = None
            for dx, dy in near + far:
                ha = "left" if dx > 0 else "right"
                kw = {"arrowprops": leader} if (dx, dy) in far else {}
                ann = ax.annotate(src, (x, y), xytext=(dx, dy), textcoords="offset points",
                                  fontsize=fontsize, color=INK2, ha=ha, **kw)
                box = _text_extent(ann, renderer).expanded(1.06, 1.2)
                cost = sum(_overlap_area(box, b) for b in placed + point_boxes if not _contains(b, x, y, ax))
                if not (box.x0 >= axes_box.x0 - 2 and box.x1 <= axes_box.x1 + 2
                        and box.y0 >= axes_box.y0 - 2 and box.y1 <= axes_box.y1 + 2):
                    cost += 1e6                       # never let a label leave the axes, on either axis
                seg = None
                if (dx, dy) in far:
                    # a crossed or misdirected leader misleads as much as a small overlap confuses
                    seg = ((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2), tuple(ax.transData.transform((x, y)))
                    cost += 400 * sum(_segments_cross(*seg, *l) for l in leaders)
                    cost += 400 * _leader_crosses(box, seg[1], placed + [b for b in point_boxes if not _contains(b, x, y, ax)])
                cost += 400 * sum(_leader_crosses(box, b, [box], from_point=a) for a, b in leaders)
                if best is None or cost < best[0]:
                    if best is not None:
                        best[1].remove()
                    best = (cost, ann, box, seg)
                else:
                    ann.remove()
            chosen = best[2]
            if best[3]:
                leaders.append(best[3])       # later labels must avoid this leader too
        placed.append(chosen)


def _overlap_area(a, b) -> float:
    dx = min(a.x1, b.x1) - max(a.x0, b.x0)
    dy = min(a.y1, b.y1) - max(a.y0, b.y0)
    return dx * dy if dx > 0 and dy > 0 else 0.0


def _box(x0, y0, x1, y1):
    from matplotlib.transforms import Bbox
    return Bbox([[x0, y0], [x1, y1]])


def _leader_crosses(text_box, point, boxes, from_point=None) -> bool:
    """Whether the straight leader from the label's centre (or ``from_point``) to ``point`` passes through
    any of ``boxes`` (sampled; the stretch inside ``text_box`` itself does not count)."""
    cx, cy = from_point if from_point is not None else ((text_box.x0 + text_box.x1) / 2, (text_box.y0 + text_box.y1) / 2)
    px, py = point
    if from_point is not None:           # an existing leader tested against a new label: every sample counts
        return any(any(b.x0 <= cx + t * (px - cx) <= b.x1 and b.y0 <= cy + t * (py - cy) <= b.y1 for b in boxes)
                   for t in np.linspace(0.0, 0.92, 24))
    for t in np.linspace(0.0, 0.92, 24):             # stop short of the point's own marker
        sx, sy = cx + t * (px - cx), cy + t * (py - cy)
        if text_box.x0 <= sx <= text_box.x1 and text_box.y0 <= sy <= text_box.y1:
            continue
        if any(b.x0 <= sx <= b.x1 and b.y0 <= sy <= b.y1 for b in boxes):
            return True
    return False


def _segments_cross(a, b, c, d) -> bool:
    """Whether segments ab and cd properly intersect (display coordinates)."""
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0


def _text_extent(ann, renderer):
    """The label's own text box. Annotation.get_window_extent also spans its leader line, whose bounding
    rectangle covers every neighbour between label and point -- so in a cluster no far offset could ever
    win, and every crowded label fell back to the least-bad overlap."""
    from matplotlib.text import Text
    ann.update_positions(renderer)
    return Text.get_window_extent(ann, renderer)


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


def _short_label(label: str) -> str:
    """'Qwen/Qwen3.5-2B (Tier 2 residual, lr 3e-05, soft labels)' -> 'Qwen3.5-2B T2 lr 3e-5 soft'.
    Readout fine-tunes sit in a tight cluster, so they get tags ('9B +coh'); the legend names the backbones."""
    if "(readout" in label:
        base = label.split(" (")[0].split("/")[-1].replace("Qwen3.5-", "").replace("gemma-4-", "Gemma ").replace("-it", "")
        return base + (" +coh" if "coherence" in label else "")
    s = label.split("/")[-1] if "/" in label.split(" (")[0] else label
    for a, b in ((" (TypeSafe API)", ""), (" (Tier 0)", ""), ("Tier 2 residual", "T2"), ("Tier 1 residual", "T1"),
                 ("Tier 1 replace", "T1 replace"), ("lr 3e-05", "lr 3e-5"), ("soft labels", "soft"),
                 ("6-epoch cap, mean of 3 seeds", "3 seeds"), ("Tev1-4B-experimental", "Tev1-4B"),
                 (", ", " "), (" (", " "), (")", "")):
        s = s.replace(a, b)
    return s


def fig_models_map(rows: list[dict[str, Any]], out: Path, title: str = "Jevified models vs Jev on jev-bench",
                   off_chart: list[dict[str, Any]] = ()) -> Path:
    """Model-level calibration map: macro accuracy vs macro ECE, one point per model.

    ``off_chart`` rows are named in the subtitle instead of drawn: one outlier (CLM-v0.1-8B at 0.34 /
    0.34) stretched both axes until the twenty models near Jev collapsed into one corner.
    """
    plt = _mpl()
    # wide, and zoomed to the data: twenty models sit between ECE 0.07 and 0.13, and a fixed 0-0.20
    # axis on a 7-inch canvas stacked their labels on top of each other
    fig, ax = plt.subplots(figsize=(11.0, 6.4))
    pts = []
    # marker shape is the primary encoding for tier (it survives print and colour blindness);
    # colour is the redundant one
    # validated over all pairs with the dataviz palette checker; Tier 2 was purple, which sat 12.9 ΔE
    # from Tier 0's blue for full colour vision (a hard fail), so it is magenta now
    style = {"API": ("#eb6834", "D", 74), "Tier 0": (PRIM_COLOR["choice"], "o", 42),
             "Tier 1": ("#1baf7a", "^", 62), "Tier 2": ("#c83c8c", "s", 54), "External": ("#6b6b00", "P", 66),
             # a readout fine-tune is also LoRA-trained: Tier 2's validated magenta, told apart by shape (no sixth hue
             # passes the all-pairs colour-vision check against these five)
             "Readout FT": ("#c83c8c", "*", 150)}
    for e in rows:
        s = e["summary"]["macro"]
        t = str(e.get("tier", ""))
        kind = ("API" if t == "API" else "External" if t == "External" else "Readout FT" if t == "Readout FT" else
                "Tier 2" if t.startswith("Tier 2") else "Tier 1" if t.startswith("Tier 1") else "Tier 0")
        color, marker, size = style[kind]
        ax.scatter(s["acc"], s["ece"], s=size, color=color, marker=marker, zorder=3, linewidths=0.8, edgecolors=SURFACE)
        pts.append((_short_label(e["label"]), s["acc"], s["ece"]))
    ax.set_xlabel("macro accuracy over 22 configs"); ax.set_ylabel("macro expected calibration error (lower is better)")
    # y stops a little above the worst model rather than at a fixed 0.30: the interesting
    # differences are a few hundredths of ECE and were squeezed into the bottom third
    xs, ys = [p[1] for p in pts], [p[2] for p in pts]
    ax.set_xlim(max(0.0, min(xs) - 0.04), min(1.0, max(xs) + 0.06))
    ax.set_ylim(max(0.0, min(ys) - 0.015), max(ys) + 0.02)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="D", color="#eb6834", lw=0, markersize=7, label="Jev 1.13.0 (API)"),
                       Line2D([0], [0], marker="o", color=PRIM_COLOR["choice"], lw=0, markersize=6, label="Jevified, Tier 0 (no training)"),
                       Line2D([0], [0], marker="^", color="#1baf7a", lw=0, markersize=7, label="Jevified, Tier 1 (trained heads)"),
                       Line2D([0], [0], marker="s", color="#c83c8c", lw=0, markersize=6.5, label="Jevified, Tier 2 (LoRA + heads)")]
                      + ([Line2D([0], [0], marker="P", color="#6b6b00", lw=0, markersize=7,
                                 label="other teams' models")]
                         if any(str(e.get("tier")) == "External" for e in rows) else [])
                      + ([Line2D([0], [0], marker="*", color="#c83c8c", lw=0, markersize=11,
                                 label="Jevified, readout fine-tuned + coherence (Qwen3.5 2B/4B/9B, Gemma E4B)")]
                         if any(str(e.get("tier")) == "Readout FT" for e in rows) else []),
              loc="lower left", fontsize=8)   # the empty corner: upper right now holds Jev and the Tier 2 rows
    sub = ("Every model scored on the same 22,773 test records. Tier 0 = zero training: prompt + logit readout + "
           "a recipe fitted on validation only. Down and to the right is better.")
    if off_chart:
        sub += " Off the chart: " + "; ".join(
            f"{e['label'].split(' (')[0]} (accuracy {e['summary']['macro']['acc']:.3f}, ECE {e['summary']['macro']['ece']:.3f})"
            for e in off_chart) + "."
    fig.tight_layout(rect=_layout(fig, _wrapped_lines(fig, title, sub) - 1))
    _headline(fig, title, sub)
    _footer(fig)
    _annotate_without_overlap(fig, ax, pts)
    return _save(fig, out / "models_map")


def fig_tier1_story(variants: list[dict[str, Any]], jev: dict[str, float], out: Path, model: str) -> Path:
    """The Tier 1 result: trained vs held-out sources, for each head variant.

    ``variants`` is an ordered list of {label, trained_acc, trained_ece, heldout_acc, heldout_ece}.
    """
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8))
    x = np.arange(len(variants))
    w = 0.38
    for ax, (metric, name, better) in zip(axes, (("acc", "accuracy", "higher is better"),
                                                 ("ece", "expected calibration error", "lower is better"))):
        tr = [v[f"trained_{metric}"] for v in variants]
        ho = [v[f"heldout_{metric}"] for v in variants]
        ax.bar(x - w / 2, tr, w, label="sources seen in training", color=BLUE_RAMP[5], edgecolor=SURFACE, linewidth=0.6)
        ax.bar(x + w / 2, ho, w, label="held-out sources", color="#eb6834", edgecolor=SURFACE, linewidth=0.6)
        for xi, (a, b) in enumerate(zip(tr, ho)):
            ax.text(xi - w / 2, a, f"{a:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
            ax.text(xi + w / 2, b, f"{b:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
        ref = jev[f"heldout_{metric}"]
        ax.axhline(ref, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=1, alpha=0.55)
        ax.set_xticks(x); ax.set_xticklabels([v["label"] for v in variants], fontsize=8)
        ax.set_ylabel(f"{name} ({better})"); ax.tick_params(length=0)
        top = max(max(tr), max(ho), ref) * 1.3
        ax.set_ylim(0, min(top, 1.0) if metric == "acc" else top)      # accuracy never past 1.0
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    # one legend for both panels, under them: with accuracy capped at 1.0 Jev's reference line
    # (0.835) runs through any legend placed inside the left panel
    fig.legend(handles=[Patch(color=BLUE_RAMP[5], label="sources seen in training"),
                        Patch(color="#eb6834", label="held-out sources"),
                        Line2D([0], [0], color=INK2, lw=1.0, ls=(0, (4, 3)), alpha=0.55, label="Jev 1.13.0, held-out sources")],
               loc="lower center", ncol=3, fontsize=7.8, frameon=False, bbox_to_anchor=(0.5, 0.30 / fig.get_figheight()))
    title = f"{model}: what trained decision heads buy, and where they cost"
    sub = ("A head that replaces the model's own scorer wins on sources it trained on and loses on sources it never saw — "
           "a result invisible to anyone holding out records instead of whole sources. A zero-initialized residual keeps the "
           "gain; on this seed it also holds held-out accuracy at Tier 0, but across five seeds it only halves the regression "
           "unless training stops early (FINDINGS 6.3a).")
    left, bottom, right, top = _layout(fig, _wrapped_lines(fig, title, sub))
    fig.tight_layout(rect=(left, bottom + 0.34 / fig.get_figheight(), right, top))
    _headline(fig, title, sub)
    _footer(fig)
    return _save(fig, out / "tier1_story")


def fig_tier1_per_source(deltas: list[tuple[str, bool, float, float]], out: Path, model: str) -> Path:
    """Per-source accuracy change vs Tier 0 for both head variants."""
    plt = _mpl()
    deltas = sorted(deltas, key=lambda d: (d[1], d[2]))
    fig, ax = plt.subplots(figsize=(8.4, 0.34 * len(deltas) + 1.9))
    y = np.arange(len(deltas))
    h = 0.38
    ax.barh(y + h / 2, [d[2] for d in deltas], h, color="#e34948", label="heads replace the LM scorer", edgecolor=SURFACE, linewidth=0.5)
    ax.barh(y - h / 2, [d[3] for d in deltas], h, color=BLUE_RAMP[4], label="heads residual on it", edgecolor=SURFACE, linewidth=0.5)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{d[0]}  ★" if d[1] else d[0] for d in deltas], fontsize=7.5)
    ax.invert_yaxis(); ax.tick_params(length=0)
    ax.set_xlabel("accuracy change vs Tier 0 on the same backbone")
    ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout(rect=_layout(fig, 2))
    _headline(fig, f"{model}: per-source effect of trained heads (★ = held out of training)",
              "Replacement collapses on knowledge (arc_challenge, mmlu) and on ordinal scales it never saw "
              "(measuring_hate_speech); it does not collapse on large option sets. The residual keeps the wins and recovers most of "
              "the losses — one seed; FINDINGS 6.3a has the spread.")
    _footer(fig)
    return _save(fig, out / "tier1_per_source")


def fig_instruct_vs_base(pairs: list[dict[str, Any]], out: Path) -> Path:
    """Raw vs calibrated ECE for instruct/base checkpoint pairs."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    y = np.arange(len(pairs))
    for i, p in enumerate(pairs):
        color = "#eb6834" if p["instruct"] else BLUE_RAMP[5]
        ax.annotate("", xy=(p["cal_ece"], i), xytext=(p["raw_ece"], i),
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.0, "shrinkA": 0, "shrinkB": 0})
        ax.scatter([p["raw_ece"]], [i], s=36, color=color, zorder=3, edgecolors=SURFACE, linewidths=0.8)
        ax.scatter([p["cal_ece"]], [i], s=36, color=color, zorder=3, marker="D", edgecolors=SURFACE, linewidths=0.8)
        ax.text(p["raw_ece"] + 0.006, i - 0.22, f"raw {p['raw_ece']:.3f}", fontsize=7, color=INK2)
        ax.text(p["cal_ece"] - 0.006, i - 0.22, f"{p['cal_ece']:.3f}", fontsize=7, color=INK2, ha="right")
    ax.set_yticks(y); ax.set_yticklabels([p["label"] for p in pairs], fontsize=8.5)
    ax.invert_yaxis(); ax.set_xlabel("expected calibration error"); ax.tick_params(length=0)
    ax.set_xlim(0, max(p["raw_ece"] for p in pairs) * 1.15)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="o", color="#eb6834", lw=0, markersize=6, label="instruction-tuned"),
                       Line2D([0], [0], marker="o", color=BLUE_RAMP[5], lw=0, markersize=6, label="base checkpoint"),
                       Line2D([0], [0], marker="D", color=INK2, lw=0, markersize=5, label="after calibration")],
              loc="lower right", fontsize=7.5)
    fig.tight_layout(rect=_layout(fig, 2))
    _headline(fig, "Instruction tuning costs calibration — but it is mostly a temperature problem",
              "Arrow start = raw ECE straight off the logits; arrow end = after one scalar per primitive fitted on validation splits. "
              "Instruct checkpoints start 1.3–2.1x worse and land close to their base counterparts.")
    _footer(fig)
    return _save(fig, out / "instruct_vs_base")


def _family_style(label: str, seen: dict[str, int]) -> tuple[str, str]:
    """Colour by model family, vary the dash within it: nine similar lines are easier to
    read as three families than as nine hues (and nine hues cannot pass the CVD gate)."""
    # five shades x five dashes: with five Qwen checkpoints the old 4 x 4 cycle gave the 1st
    # and 5th the same colour *and* dash (0.8B and 9B were indistinguishable)
    fams = [("Qwen", [BLUE_RAMP[7], BLUE_RAMP[5], BLUE_RAMP[3], BLUE_RAMP[2], BLUE_RAMP[6]]),
            ("gemma", ["#8c3413", "#b84a1e", "#eb6834", "#f19468"]),
            ("SmolLM", ["#1baf7a"]), ("K2", ["#4a3aa7", "#8a7fd0"])]
    dash_cycle = [(0, ()), (0, (5, 2)), (0, (1.5, 1.5)), (0, (6, 2, 1, 2)), (0, (3, 1, 1, 1, 1, 1))]
    for name, colors in fams:
        if label.lower().startswith(name.lower()):
            i = seen.get(name, 0)
            seen[name] = i + 1
            return colors[i % len(colors)], dash_cycle[i % len(dash_cycle)]
    i = seen.get("other", 0)
    seen["other"] = i + 1
    return ["#eda100", "#e87ba4", "#008300"][i % 3], (0, ())


def fig_recipe_ladder(models: dict[str, list[dict[str, Any]]], out: Path) -> Path:
    """Macro accuracy and ECE through the recipe steps, one line per model."""
    plt = _mpl()
    steps = ["raw", "+permutations", "+prior", "+temperature/bias"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
    seen: dict[str, int] = {}
    styles = {label: _family_style(label, seen) for label in sorted(models)}
    for ax, key, name in ((axes[0], "macro_acc", "macro accuracy"), (axes[1], "macro_ece", "macro expected calibration error")):
        for label in sorted(models):
            color, dash = styles[label]
            ys = [r[key] for r in models[label]]
            ax.plot(range(len(ys)), ys, marker="o", markersize=3.5, lw=1.6, color=color, ls=dash, zorder=3, label=label)
        ax.set_xticks(range(len(steps))); ax.set_xticklabels(steps, fontsize=7.5, rotation=12, ha="right")
        ax.set_ylabel(name); ax.tick_params(length=0); ax.set_xlim(-0.15, len(steps) - 0.85)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=7.8, bbox_to_anchor=(1.0, 0.52))
    fig.tight_layout(rect=(0, _layout(fig, 2)[1], 0.84, _layout(fig, 2)[3]))
    _headline(fig, "What each calibration step is worth, per model",
              "Every step is fitted on validation splits only. The contextual prior is a small-model fix: +6 to +12 Choice "
              "accuracy points below 1B, whatever the family, and ~0 from 2B up. Gemma needs the temperature most. "
              "A single fixed recipe would mis-rank these models.")
    _footer(fig)
    return _save(fig, out / "recipe_ladder")


def fig_confidence_vs_agreement(records, preds: dict[str, Any], out: Path, model: str, source: str = "chaosnli") -> Path:
    """Does the model's confidence know how contested an item is?

    x = share of human annotators on the majority label, y = the model's own confidence.
    A calibrated decision model should slope upward; a flat cloud means the confidence
    carries no information about ambiguity.
    """
    plt = _mpl()
    H, M, correct = [], [], []
    for r in records:
        p = preds.get(r.id)
        if p is None or p.error:
            continue
        keys = r.option_keys()
        H.append(max(r.soft_label[k] for k in keys))
        M.append(max(p.probabilities[k] for k in keys))
        correct.append(p.answer == str(r.label))
    H, M, correct = np.array(H), np.array(M), np.array(correct)
    r_p = float(np.corrcoef(H, M)[0, 1])
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, zorder=1)
    ax.scatter(H, M, s=7, color=PRIM_COLOR["choice"], alpha=0.16, linewidths=0, zorder=2)
    edges = np.linspace(H.min(), 1.0, 9)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (H >= lo) & (H < hi) if hi < 1.0 else (H >= lo)
        if m.sum() >= 15:
            xs.append(H[m].mean()); ys.append(M[m].mean())
    ax.plot(xs, ys, color="#eb6834", lw=2.2, zorder=4)
    ax.scatter(xs, ys, s=30, color="#eb6834", zorder=5, edgecolors=SURFACE, linewidths=0.8)
    ax.set_xlabel("human agreement (share of 100 annotators on the majority label)")
    ax.set_ylabel(f"{model} confidence")
    ax.set_xlim(0.3, 1.0); ax.set_ylim(0, 1.02); ax.tick_params(length=0)
    ax.set_title(f"one point per item  ·  Pearson r = {r_p:.3f}", loc="left")
    ax.annotate("confidence = human agreement", xy=(0.86, 0.86), xytext=(0.72, 0.52), fontsize=7.5, color=INK2,
                ha="center", arrowprops={"arrowstyle": "-", "color": GRID, "lw": 0.8})

    ax = axes[1]
    bands = [(0.33, 0.5, "humans split\n(<50%)"), (0.5, 0.7, "contested\n(50–70%)"),
             (0.7, 0.9, "clear\n(70–90%)"), (0.9, 1.01, "consensus\n(≥90%)")]
    labels, conf, acc, ns = [], [], [], []
    for lo, hi, name in bands:
        m = (H >= lo) & (H < hi)
        if m.sum() < 10:
            continue
        labels.append(name + chr(10) + f"n={int(m.sum()):,}"); conf.append(M[m].mean()); acc.append(correct[m].mean()); ns.append(int(m.sum()))
    x = np.arange(len(labels)); w = 0.38
    ax.bar(x - w / 2, conf, w, color=PRIM_COLOR["choice"], label="model confidence", edgecolor=SURFACE, linewidth=0.6)
    ax.bar(x + w / 2, acc, w, color="#eb6834", label="model accuracy", edgecolor=SURFACE, linewidth=0.6)
    for xi, (c, a, n) in enumerate(zip(conf, acc, ns)):
        ax.text(xi - w / 2, c, f"{c:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
        ax.text(xi + w / 2, a, f"{a:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.8)
    ax.set_ylim(0, 1.12); ax.set_ylabel("mean over items"); ax.tick_params(length=0)
    ax.legend(loc="upper left", fontsize=7.5)
    ax.set_title("confidence stays flat while accuracy collapses", loc="left")

    fig.tight_layout(rect=_layout(fig, 2))
    _headline(fig, f"{model}: does its confidence know when humans disagree?",
              f"{source} — every item labelled by 100 annotators. A decision model's confidence should fall as human "
              f"agreement falls. Here it does not: r = {r_p:.3f}, and the model is more confident than the human "
              f"majority on {(M > H).mean():.0%} of items.")
    _footer(fig)
    return _save(fig, out / "confidence_vs_agreement")


def fig_latency_ladder(rows: list[dict[str, Any]], jev: dict[str, Any], ladder: list, out: Path, device: str) -> Path:
    """Single-request latency against answer-set size, as three small multiples.

    One panel per question a deployer actually asks -- which size, which tier, which recipe --
    each with at most five series and a legend, Jev's measured API round trip drawn in every
    panel as the reference. K is on a log axis because it spans 2 to 151; latency is linear
    from zero, so a flat floor looks flat and a doubling looks like a doubling.
    """
    plt = _mpl()
    from matplotlib.lines import Line2D

    ks = [k for _, _, k in ladder]

    def series(r):
        return [r["single"][src]["p50_ms"] for src, _, _ in ladder]

    sizes = [r for r in rows if r["tier"] == "Tier 0" and r["permutations"] == 1]
    base = next((r for r in sizes if "2B" in r["model"]), sizes[0] if sizes else None)
    base_name = base["model"].split(" (")[0] if base else ""
    tiers = [r for r in rows if r["model"].startswith(base_name) and r["permutations"] == 1]
    perms = [r for r in rows if r["tier"] == "Tier 0" and any(x in r["model"] for x in ("2B", "4B"))]

    tier_name = {"Tier 0": "Tier 0 (readout only)", "Tier 1": "Tier 1 (+ decision heads)", "Tier 2": "Tier 2 (+ merged LoRA)"}
    panels = [
        ("Backbone size  ·  Tier 0, one option order", sizes,
         lambda r: r["model"].split(" (")[0], [BLUE_RAMP[1], BLUE_RAMP[3], BLUE_RAMP[5], BLUE_RAMP[7]], lambda r: "-"),
        (f"Tier  ·  {base_name}, one option order", tiers,
         lambda r: tier_name.get(r["tier"], r["tier"]), [BLUE_RAMP[4], "#1baf7a", "#7b4fbf"], lambda r: "-"),
        ("Recipe  ·  one vs two option orders averaged", perms,
         lambda r: f"{r['model'].split(' (')[0]}, {r['permutations']} order{'s' if r['permutations'] > 1 else ''}",
         [BLUE_RAMP[2], BLUE_RAMP[2], BLUE_RAMP[6], BLUE_RAMP[6]], lambda r: "-" if r["permutations"] > 1 else (0, (2.5, 2))),
    ]
    jys = [jev[src]["p50_ms"] if src in jev else np.nan for src, _, _ in ladder] if jev else None
    ymax = max([max(series(r)) for r in rows] + ([np.nanmax(jys)] if jys is not None else [])) * 1.08

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.9), sharey=True)
    for ax, (title, rs, name, colors, style) in zip(axes, panels):
        handles = []
        for i, r in enumerate(rs):
            color, ls = colors[i % len(colors)], style(r)
            ax.plot(ks, series(r), marker="o", markersize=4.2, lw=1.8, color=color, ls=ls, zorder=3)
            handles.append(Line2D([0], [0], color=color, ls=ls, marker="o", markersize=4.2, lw=1.8, label=name(r)))
        if jys is not None:
            ax.plot(ks, jys, marker="D", markersize=4.6, lw=1.8, color="#eb6834", zorder=3)
            handles.append(Line2D([0], [0], color="#eb6834", marker="D", markersize=4.6, lw=1.8, label="Jev 1.13.0 API round trip"))
        ax.set_xscale("log")
        ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks], fontsize=8); ax.minorticks_off()
        ax.set_xlim(ks[0] * 0.8, ks[-1] * 1.2)
        ax.set_ylim(0, ymax)
        ax.set_xlabel("answer-set size K (log scale)", fontsize=9)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK)
        ax.grid(True, which="major", color=GRID, lw=0.7, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.legend(handles=handles, loc="upper left", fontsize=7.6, frameon=False, handlelength=3.2)
    axes[0].set_ylabel("median latency per request (ms)", fontsize=9)
    fig.tight_layout(rect=(0, 0.20 / fig.get_figheight(), 1, 1 - 0.62 / fig.get_figheight()))
    _headline(fig, f"Latency per request vs answer-set size, one {device}",
              "One record at a time through the served path, median of 30 real records per K. Every option is scored and the option text is in "
              "the prompt, so prefill grows with K. Jev's line is its API round trip as observed by our client, network included.")
    _footer(fig)
    return _save(fig, out / "latency_ladder")
