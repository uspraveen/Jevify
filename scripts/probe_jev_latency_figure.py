"""Server-side time vs billed input tokens across every probe condition, with the fitted line."""
import json, datetime, os
os.chdir("C:/dev/Jevify")
from pathlib import Path
import numpy as np
from jevify.bench import figures as F

plt = F._mpl()
r1 = json.load(open("results/jev-latency-probe/probe.json", encoding="utf-8"))
r2 = json.load(open("results/jev-latency-probe/probe_round2.json", encoding="utf-8"))
rows = [r for r in r2["rows"] if r.get("server_ms") is not None and r["status"] == 200 and r["block"] in ("F", "G", "H")]

kinds = {"F": ("state length, K=1 (Noul)", F.PRIM_COLOR["noul"]), "G": ("options, K = 2 / 64 / 255", F.PRIM_COLOR["choice"]),
         "H": ("small request, alone / 8 in flight", F.INK2)}
fig, ax = plt.subplots(figsize=(8.4, 4.9))
xs_all, ys_all = [], []
for b, (label, color) in kinds.items():
    pts = [(r["input_tokens"], r["server_ms"]) for r in rows if r["block"] == b]
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    ax.scatter(xs, ys, s=14, color=color, alpha=0.55, edgecolors="none", label=label, zorder=3)
    if b in ("F", "G"):
        xs_all.extend(xs); ys_all.extend(ys)
xs_all, ys_all = np.array(xs_all), np.array(ys_all)
slope, floor = np.polyfit(xs_all, ys_all, 1)
grid = np.linspace(0, xs_all.max() * 1.02, 100)
ax.plot(grid, floor + slope * grid, color=F.INK, lw=1.4, ls=(0, (5, 3)), zorder=4)
ax.text(0.03, 0.93, f"fit: {floor:.0f} ms + {slope*1000:.1f} µs per input token" + chr(10) + f"≈ {1/slope:.0f}k tokens / s, linear to 27k tokens",
        transform=ax.transAxes, fontsize=9, color=F.INK, va="top", ha="left")
ax.set_xlabel("input tokens, as billed by the API"); ax.set_ylabel("server time per request, ms")
ax.set_xlim(0, xs_all.max() * 1.04); ax.set_ylim(0, max(ys_all) * 1.1)
ax.grid(True, color=F.GRID, lw=0.7, zorder=0)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(loc="lower right", fontsize=8, frameon=False)
title = "Jev 1.13.0: a fixed floor plus a per-token cost; options are just tokens"
sub = ("Every request sequential on one connection, conditions interleaved; the server's own upstream time (x-envoy-upstream-service-time), network excluded. "
       "Options (K=2 to 255, unique text every request) fall on the same line as state tokens: no cache, no window, no batching wait.")
fig.tight_layout(rect=F._layout(fig, F._wrapped_lines(fig, title, sub) - 1))
F._headline(fig, title,
            sub)
F.PROVENANCE = f"jev-bench v0.1.1 · {datetime.date.today().isoformat()}"
F._footer(fig)
out = F._save(fig, Path("results/jev-latency-probe") / "server_time_vs_tokens")
print(out, f"floor={floor:.1f} ms slope={slope*1000:.2f} us/token")
