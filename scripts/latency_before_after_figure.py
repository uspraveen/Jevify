"""Before/after: research path vs serving path on the same A40, same models, same records."""
import json, datetime, os, sys
os.chdir("C:/dev/Jevify"); sys.path.insert(0, "C:/dev/Jevify")
from pathlib import Path
import numpy as np
from jevify.bench import figures as F
from matplotlib.lines import Line2D

plt = F._mpl()
before = json.load(open("results/latency/latency.json", encoding="utf-8"))
after = json.load(open("results/latency/latency_vllm.json", encoding="utf-8"))
ladder = [tuple(x) for x in before["ladder"]]; ks = [k for _, _, k in ladder]
jev = before["jev_api"]

def row(rows, sub, perms):
    return next(r for r in rows if sub in r["model"] and r["permutations"] == perms and r["tier"] == "Tier 0")
def ys(r): return [r["single"][s]["p50_ms"] for s, _, _ in ladder]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.6, 4.9), gridspec_kw={"width_ratios": [1.45, 1]})
sizes = [("0.8B", F.BLUE_RAMP[2]), ("2B", F.BLUE_RAMP[4]), ("4B", F.BLUE_RAMP[7])]
handles = []
for name, color in sizes:
    b, a = row(before["rows"], f"Qwen3.5-{name} ", 1), row(after["rows"], f"Qwen3.5-{name} ", 1)
    ax.plot(ks, ys(b), color=color, ls=(0, (2.5, 2)), lw=1.6, marker="o", markersize=3.8, zorder=3)
    ax.plot(ks, ys(a), color=color, ls="-", lw=2.0, marker="o", markersize=4.2, zorder=4)
    handles.append(Line2D([0], [0], color=color, lw=2, label=f"Qwen3.5-{name}"))
jys = [jev[s]["p50_ms"] for s, _, _ in ladder]
ax.plot(ks, jys, color="#eb6834", marker="D", markersize=4.6, lw=1.8, zorder=3)
handles += [Line2D([0], [0], color="#eb6834", marker="D", lw=1.8, markersize=4.6, label="Jev 1.13.0 API round trip"),
            Line2D([0], [0], color=F.INK2, ls=(0, (2.5, 2)), lw=1.6, label="before: research path (HF eager)"),
            Line2D([0], [0], color=F.INK2, ls="-", lw=2.0, label="after: serving path (vLLM)")]
ax.set_xscale("log"); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks], fontsize=8); ax.minorticks_off()
ax.set_xlim(ks[0] * 0.8, ks[-1] * 1.2); ax.set_ylim(0, 420)
ax.set_xlabel("answer-set size K (log scale)", fontsize=9); ax.set_ylabel("median latency per request (ms)", fontsize=9)
ax.set_title("Single request, one option order  ·  dashed → solid is the same model on the same GPU", loc="left", fontsize=9.5, color=F.INK)
ax.grid(True, color=F.GRID, lw=0.7, zorder=0)
for side in ("top", "right"): ax.spines[side].set_visible(False)
ax.tick_params(colors=F.INK2, labelsize=8)
ax.legend(handles=handles, loc="upper left", fontsize=7.6, frameon=False, handlelength=3.0, ncol=2)

# throughput
names = [n for n, _ in sizes]
tb = [row(before["rows"], f"Qwen3.5-{n} ", 1)["batched"]["records_per_s"] for n in names]
ta = [row(after["rows"], f"Qwen3.5-{n} ", 1)["batched"]["records_per_s"] for n in names]
x = np.arange(len(names)); w = 0.36
bx.bar(x - w / 2, tb, w, color=F.BLUE_RAMP[1], edgecolor=F.SURFACE, label="before (HF eager)", zorder=3)
bx.bar(x + w / 2, ta, w, color=F.BLUE_RAMP[5], edgecolor=F.SURFACE, label="after (vLLM)", zorder=3)
for i, (b, a) in enumerate(zip(tb, ta)):
    bx.text(i - w / 2, b + 1, f"{b:.0f}", ha="center", va="bottom", fontsize=8, color=F.INK2)
    bx.text(i + w / 2, a + 1, f"{a:.0f}  ({a / b:.1f}×)", ha="center", va="bottom", fontsize=8, color=F.INK)
bx.set_xticks(x); bx.set_xticklabels([f"Qwen3.5-{n}" for n in names], fontsize=9)
bx.set_ylabel("records / s at batch 16", fontsize=9); bx.set_ylim(0, max(ta) * 1.45)
bx.set_title("Batched throughput, mixed K", loc="left", fontsize=9.5, color=F.INK)
bx.grid(True, axis="y", color=F.GRID, lw=0.7, zorder=0)
for side in ("top", "right"): bx.spines[side].set_visible(False)
bx.tick_params(colors=F.INK2, labelsize=8, length=0)
bx.legend(loc="upper left", fontsize=8, frameon=False)

title = f"Same model, same {before['device']}: what a serving path changes"
sub = ("The research path pays ~120 µs per prompt token in unfused kernels and per-request Python; vLLM brings that to ~48 µs and halves "
       "the fixed floor, so the option list stops dominating. Jev's line is its API round trip as observed by our client, network included.")
fig.tight_layout(rect=F._layout(fig, F._wrapped_lines(fig, title, sub) - 1))
F._headline(fig, title, sub)
F.PROVENANCE = f"jev-bench v0.1.1 · {datetime.date.today().isoformat()}"
F._footer(fig)
print(F._save(fig, Path("results/latency") / "latency_before_after"))
