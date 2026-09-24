"""Figures for the Tev1 comparison and the community benchmarks, in the house style.

    python scripts/tev1_figures.py --comparison results/tev1-comparison/comparison.json --community results/community/report.json

Both are dot plots: a row per subset (or benchmark), a marker per model, accuracy and ECE side by
side on their own axes (never one twin axis). Colours follow the model, never its rank; Jev keeps
the project's reference orange. The tables the dots come from sit next to each figure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jevify.bench import figures as F  # noqa: E402

# validated with the dataviz palette checker over ALL pairs (every series in a dot plot sits beside
# every other): lightness, chroma and normal-vision separation pass; the one colour-blind pair in the
# 6-8 band (magenta / blue) also differs in marker shape; the lighter marks are below 3:1 contrast, so
# every figure ships with its table. At most six series per figure; the rest live in the tables.
COLOR = {"jev": "#eb6834", "base-ours": "#1c5cab", "tev1-raw": "#9a86e0", "t2": "#1baf7a", "tev1": "#c83c8c",
         "base-tev1": "#6b6b00", "clm": "#6b6b00"}
MARKER = {"jev": "D", "base-ours": "o", "tev1-raw": "v", "t2": "s", "tev1": "v", "base-tev1": "P", "clm": "P"}
ORDER = ["jev", "base-ours", "tev1-raw", "t2", "tev1", "base-tev1", "clm"]


def style_of(label: str) -> str:
    l = label.lower()
    if l.startswith("jev "):
        return "jev"
    if "clm" in l:
        return "clm"
    if "tev1" in l and "as shipped" in l:
        return "tev1-raw"
    if "tev1-4b" in l or l.startswith("tev1"):
        return "tev1"
    if "tier 2" in l:
        return "t2"
    if "qwen3.5-4b" in l and "tev1 prompt" in l:
        return "base-tev1"
    if "qwen3.5-4b" in l:
        return "base-ours"
    return "base-ours"


def dot_panels(rows: list[str], series: dict[str, dict[str, tuple[float, float | None]]], title: str, sub: str,
               out: Path, stem: str, xlim_acc=(0.3, 1.0), ci: dict | None = None) -> Path:
    plt = F._mpl()
    from matplotlib.lines import Line2D

    # a fixed order (Jev first, on top of every row) and at most six series: the palette is only
    # validated for six, and the colour follows the model, never its position
    kept, seen = {}, set()
    for lab, vals in series.items():                  # the first model of each style is drawn; the rest are table-only
        if style_of(lab) in seen:
            print(f"  (table only, not drawn: {lab})")
            continue
        seen.add(style_of(lab))
        kept[lab] = vals
    series = dict(sorted(kept.items(), key=lambda kv: ORDER.index(style_of(kv[0]))))
    assert len(series) <= 6, list(series)
    n = len(series)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 0.52 * len(rows) + 2.4), sharey=True)
    y = np.arange(len(rows))[::-1].astype(float)
    offs = np.linspace(0.28, -0.28, n) if n > 1 else [0.0]      # first series on top, as in the legend
    handles = []
    for (label, vals), off in zip(series.items(), offs):
        st = style_of(label)
        for ax, key in zip(axes, ("acc", "ece")):
            xs = [vals.get(r, (np.nan, np.nan))[0 if key == "acc" else 1] for r in rows]
            if key == "acc" and ci and label in ci:
                for yi, r in zip(y, rows):
                    lo_hi = ci[label].get(r)
                    if lo_hi:
                        ax.plot(lo_hi, [yi + off, yi + off], color=COLOR[st], lw=1.2, alpha=0.55, zorder=2)
            # clip_on=False: a perfect score sits on the axis edge, and the axis never runs past 1.0
            ax.scatter(xs, y + off, s=36, marker=MARKER[st], color=COLOR[st], edgecolors=F.SURFACE, linewidths=0.8,
                       zorder=3, clip_on=False)
        handles.append(Line2D([0], [0], marker=MARKER[st], color=COLOR[st], lw=0, markersize=6.5, label=label))
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(rows, fontsize=8.2)
    # the axis starts where the data does (intervals included): a marker drawn outside it reads as
    # belonging to the row label. It never runs past 1.0.
    lows = [v[0] for vals in series.values() for v in vals.values()]
    if ci:
        lows += [lh[0] for lab in series for lh in ci.get(lab, {}).values()]
    lo = min(xlim_acc[0], np.floor((min(lows) - 0.03) * 10) / 10) if lows else xlim_acc[0]
    axes[0].set_xlim(max(0.0, lo), xlim_acc[1])
    axes[0].set_xlabel("accuracy (higher is better)")
    axes[1].set_xlabel("expected calibration error (lower is better)")
    axes[1].set_xlim(left=0)
    for ax in axes:
        ax.grid(axis="y", visible=False)
        for yi in y[:-1]:
            ax.axhline(yi - 0.5, color=F.GRID, lw=0.6, zorder=0)
    fig.legend(handles=handles, loc="lower center", ncol=min(3, n), fontsize=7.8, frameon=False,
               bbox_to_anchor=(0.5, 0.28 / fig.get_figheight()))
    legend_h = 0.22 * ((n + 2) // 3) + 0.12
    left, bottom, right, top = F._layout(fig, F._wrapped_lines(fig, title, sub))
    fig.tight_layout(rect=(left, bottom + legend_h / fig.get_figheight(), right, top))
    F._headline(fig, title, sub)
    F._footer(fig)
    return F._save(fig, out / stem)


def fig_subsets(comparison: Path) -> Path:
    d = json.loads(comparison.read_text(encoding="utf-8"))
    order = ["all 22 configs", "6 held-out sources", "16 Jevify-trained sources", "4 Tev1-trained tasks",
             "other 18 (new to Tev1)", "K <= 24 options", "K > 24 options"]
    # CLM runs its own server on another backbone: it stays in the table, the figure is about Tev1
    series = {lab: {r: (row[r]["acc"], row[r]["ece"]) for r in order if row.get(r)} for lab, row in d["table"].items()
              if style_of(lab) != "clm"}
    title = "Tev1-4B on jev-bench, beside Jev and the Jevified models"
    sub = ("Macro over the configs in each row. \"As shipped\" is Tev1's own answer (one option order, no calibration); "
           "\"+ recipe\" is the same readout after the Tier 0 recipe fitted on jev-bench validation. Every marker but "
           "Jev's is the same base model, Qwen3.5-4B; the olive one is that base, untrained, in Tev1's prompt.")
    return dot_panels(order, series, title, sub, comparison.parent, "tev1_subsets", xlim_acc=(0.55, 1.0))


def fig_community(report: Path) -> Path:
    d = json.loads(report.read_text(encoding="utf-8"))
    rows = [("phishnchips_verdict", "phishing: verdict"), ("phishnchips_noul", "phishing: as a Noul"),
            ("phishnchips_click", "phishing: \"click?\""), ("phishnchips_minimal", "phishing: \"classify\""),
            ("tool_risk", "tool-call risk (60)"), ("ticket_routing", "ticket routing (27)")]
    names = [n for _, n in rows]
    series, ci = {}, {}
    for m in d["models"]:
        if style_of(m["label"]) == "base-tev1":          # the prompt-format control belongs to the jev-bench story
            continue
        series[m["label"]] = {n: (m["configs"][s]["accuracy"], m["configs"][s]["ece"]) for s, n in rows if s in m["configs"]}
        ci[m["label"]] = {n: tuple(m["configs"][s]["wilson95"]) for s, n in rows if s in m["configs"]}
    title = "Three Jev benchmarks written by other people"
    sub = ("PhishNChips (2,000 emails; four wordings of the same decision), agent tool-call risk and support-ticket "
           "routing, scored exactly as their authors built them. Bars are 95% Wilson intervals; the two small "
           "benchmarks separate only the outlier, CLM-v0.1-8B served by its own code.")
    return dot_panels(names, series, title, sub, report.parent, "community", xlim_acc=(0.4, 1.0), ci=ci)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison", type=Path, default=None)
    ap.add_argument("--community", type=Path, default=None)
    ap.add_argument("--provenance", default="jev-bench v0.1.1 · 2026-09-24")
    a = ap.parse_args()
    F.PROVENANCE = a.provenance
    if a.comparison:
        print(fig_subsets(a.comparison))
    if a.community:
        print(fig_community(a.community))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
