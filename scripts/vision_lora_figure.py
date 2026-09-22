"""Where should a VLM be allowed to move? Three LoRA scopes against the untrained readout.

Reads results/qwen3vl-2b (Tier 0, recipe-fitted) and results/qwen3vl-2b-t2-{vision,decoder,both}
(Tier 2, each recipe-fitted on its own validation predictions) and draws accuracy and ECE per
source, with the trained source marked. A-OKVQA is the source the adapters trained on; POPE and
AI2D never appeared in training, so their bars are the generalization claim.

    python scripts/vision_lora_figure.py --out results/figures
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jevify.bench import figures as F  # noqa: E402

ARMS = [("qwen3vl-2b", "Tier 0 (no training)", F.BLUE_RAMP[1]),
        ("qwen3vl-2b-t2-vision", "LoRA on the vision tower", F.BLUE_RAMP[3]),
        ("qwen3vl-2b-t2-decoder", "LoRA on the decoder", F.BLUE_RAMP[5]),
        ("qwen3vl-2b-t2-both", "LoRA on both", F.BLUE_RAMP[7])]
SOURCES = [("aokvqa", "A-OKVQA\n(choice · trained on)"), ("pope", "POPE\n(noul · held out)"), ("ai2d", "AI2D\n(choice · held out)")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "figures")
    a = ap.parse_args()

    arms = []
    for run, label, color in ARMS:
        p = a.results / run / "test_metrics.json"
        if not p.exists():
            print(f"skip {run}: no test_metrics.json", file=sys.stderr)
            continue
        m = json.loads(p.read_text(encoding="utf-8"))
        meta = json.loads((a.results / run / "run.json").read_text(encoding="utf-8")) if (a.results / run / "run.json").exists() else {}
        arms.append({"run": run, "label": label, "color": color, "metrics": m, "meta": meta})
    if len(arms) < 2:
        print("need Tier 0 and at least one LoRA arm", file=sys.stderr)
        return 1

    plt = F._mpl()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.0))
    x = np.arange(len(SOURCES))
    n = len(arms)
    w = 0.8 / n
    for ax, (metric, name, better) in zip(axes, (("accuracy", "accuracy", "higher is better"),
                                                 ("ece", "expected calibration error", "lower is better"))):
        top = 0.0
        for j, arm in enumerate(arms):
            vals = [arm["metrics"].get(src, {}).get(metric, float("nan")) for src, _ in SOURCES]
            pos = x - 0.4 + w * (j + 0.5)
            ax.bar(pos, vals, w * 0.92, color=arm["color"], edgecolor=F.SURFACE, linewidth=0.6, label=arm["label"], zorder=3)
            for xi, v in zip(pos, vals):
                if np.isfinite(v):
                    ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=6.6, color=F.INK2, rotation=90 if n > 3 else 0)
                    top = max(top, v)
        ax.set_xticks(x); ax.set_xticklabels([lbl for _, lbl in SOURCES], fontsize=8)
        ax.set_ylabel(f"{name} ({better})", fontsize=9); ax.tick_params(length=0)
        # accuracy is bounded: never draw an axis past 1.0 (a defect this project has shipped before)
        ax.set_ylim(0, min(1.0, top * 1.42) if metric == "accuracy" else top * (1.42 if n > 3 else 1.3))
        ax.grid(True, axis="y", color=F.GRID, lw=0.6, zorder=0); ax.grid(False, axis="x")
    # the tall POPE bars leave no corner free inside the axes: one legend for both panels, under them
    handles, labels = axes[0].get_legend_handles_labels()
    h_in = fig.get_figheight()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7.6, frameon=False,
               bbox_to_anchor=(0.5, 0.30 / h_in), handlelength=1.6, columnspacing=1.6)

    tr = next((arm for arm in arms if arm["run"].endswith("-vision")), None)
    n_train = (tr or arms[-1])["meta"].get("n_train", 0)
    title = "Qwen3-VL-2B: what moving the vision tower buys, against moving the decoder"
    sub = (f"Each LoRA (rank 16) is trained on the readout itself over {n_train:,} A-OKVQA records, early-stopped on A-OKVQA "
           "validation, and then given its own calibration recipe. POPE and AI2D never appeared in training. "
           "A tower-scoped adapter can only change what the model sees; a decoder-scoped one only how it decides.")
    left, bottom, right, top = F._layout(fig, F._wrapped_lines(fig, title, sub))
    fig.tight_layout(rect=(left, bottom + 0.34 / h_in, right, top))       # room for the legend row
    F._headline(fig, title, sub)
    F.PROVENANCE = f"jev-bench v0.1.1 · {datetime.date.today().isoformat()}"
    F._footer(fig)
    print(F._save(fig, a.out / "vision_lora_scopes"))

    # the table the docs quote
    rows = ["| arm | " + " | ".join(f"{src} acc / ECE" for src, _ in SOURCES) + " | macro acc | macro ECE | best epoch |",
            "|---|" + "---|" * (len(SOURCES) + 3)]
    for arm in arms:
        m = arm["metrics"]
        accs = [m[s]["accuracy"] for s, _ in SOURCES if s in m]; eces = [m[s]["ece"] for s, _ in SOURCES if s in m]
        cells = [f"{m[s]['accuracy']:.3f} / {m[s]['ece']:.3f}" if s in m else "—" for s, _ in SOURCES]
        rows.append(f"| {arm['label']} | " + " | ".join(cells) + f" | {np.mean(accs):.3f} | {np.mean(eces):.3f} | "
                    f"{arm['meta'].get('best_epoch', '—')} |")
    (a.out / "vision_lora_scopes.md").write_text(chr(10).join(rows) + chr(10), encoding="utf-8")
    print(chr(10).join(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
