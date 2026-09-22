"""One figure for the readout-vs-generation measurements: per model, what a request costs when the
answer is read from one position against when it is decoded, same prompt, same GPU.

    python scripts/latency_generate_figure.py --out results/latency
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

# (file tag, label, the K=4 choice source to show)
PANELS = [("qwen35-9b", "Qwen3.5-9B\ntext, K=4", "arc_challenge"),
          ("gemma4-12b", "Gemma-4-12B\ntext, K=4", "arc_challenge"),
          ("qwen3vl-8b", "Qwen3-VL-8B\nimage, K=4", "aokvqa"),
          ("qwen35-9b-vision", "Qwen3.5-9B\nimage, K=4", "aokvqa"),
          ("gemma4-12b-vision", "Gemma-4-12B\nimage, K=4", "aokvqa")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    ap.add_argument("--latency", type=Path, default=ROOT / "results" / "latency")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "latency")
    a = ap.parse_args()

    cols, device, new_tokens = [], "", None
    for tag, label, src in PANELS:
        p = a.latency / f"latency_generate_{tag}.json"
        if not p.exists():
            print(f"skip {tag}", file=sys.stderr)
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        device = d["device"]; new_tokens = d["new_tokens"]
        g = d["rows"][src]
        cols.append({"label": label, "readout": g["readout_p50_ms"], "gen": [g[f"generate_{n}_p50_ms"] for n in new_tokens],
                     "per_tok": g["ms_per_generated_token"], "prompt": g["prompt_tokens_median"]})
    if not cols:
        return 1

    plt = F._mpl()
    fig, ax = plt.subplots(figsize=(11.0, 5.3))
    series = [("readout (one forward pass)", F.PRIM_COLOR["choice"])] + \
             [(f"generate {n} token{'s' if n > 1 else ''}", F.BLUE_RAMP[i]) for i, n in zip((1, 4, 7), new_tokens)]
    x = np.arange(len(cols)); n_s = len(series); w = 0.8 / n_s
    top = 0
    for j, (name, color) in enumerate(series):
        vals = [c["readout"] if j == 0 else c["gen"][j - 1] for c in cols]
        pos = x - 0.4 + w * (j + 0.5)
        ax.bar(pos, vals, w * 0.9, color=color, edgecolor=F.SURFACE, linewidth=0.6, label=name, zorder=3)
        for xi, v in zip(pos, vals):
            ax.text(xi, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=6.8, color=F.INK2, rotation=90)
            top = max(top, v)
    for xi, c in zip(x, cols):
        ax.text(xi, top * 1.24, f"{c['per_tok']:.0f} ms / generated token", ha="center", va="bottom", fontsize=7.4, color=F.INK)
    ax.set_xticks(x); ax.set_xticklabels([c["label"] for c in cols], fontsize=8)
    ax.set_ylabel("median latency per request (ms)", fontsize=9); ax.tick_params(length=0)
    ax.set_ylim(0, top * 1.38)
    ax.grid(True, axis="y", color=F.GRID, lw=0.6, zorder=0); ax.grid(False, axis="x")
    handles, labels = ax.get_legend_handles_labels()
    h_in = fig.get_figheight()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7.6, frameon=False,
               bbox_to_anchor=(0.5, 0.30 / h_in), handlelength=1.6, columnspacing=1.6)

    title = f"Reading the answer against decoding it, same model, same prompt, one {device}"
    sub = ("A Jevified answer is the prefill alone: the prompt goes through once and every allowed answer's probability is read "
           "from the last position. Decoding the same answer as text pays the prefill and then one decoder step per token "
           "(Hugging Face eager, greedy). A one-word answer is a wash; anything longer is not.")
    left, bottom, right, top_ = F._layout(fig, F._wrapped_lines(fig, title, sub))
    fig.tight_layout(rect=(left, bottom + 0.34 / h_in, right, top_))
    F._headline(fig, title, sub)
    F.PROVENANCE = f"jev-bench v0.1.1 · {datetime.date.today().isoformat()}"
    F._footer(fig)
    print(F._save(fig, a.out / "latency_generate"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
