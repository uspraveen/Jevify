"""Does a temperature that varies with the option count beat one scalar per primitive?

Pure re-scoring: every run already stores raw log-scores, so this refits the recipe on each
model's *validation* predictions with and without the K slope and scores both on test. No GPU,
no re-inference. Writes a table and a figure.

    python scripts/recipe_k_sweep.py --records runs/jev-bench --out results/recipe-k
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
from jevify.engine.predict import Recipe, refinalize  # noqa: E402
from jevify.runners.base import read_predictions  # noqa: E402
from jevify.runners.cli import iter_records, score  # noqa: E402
from jevify.runners.recipe import fit_recipe  # noqa: E402


def macro(reports, sources=None) -> dict[str, float]:
    rs = [r for s, r in reports.items() if sources is None or s in sources]
    return {"acc": float(np.mean([r.accuracy for r in rs])), "ece": float(np.mean([r.ece for r in rs])),
            "brier": float(np.mean([r.brier for r in rs]))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    ap.add_argument("--records", type=Path, default=ROOT / "data" / "jev-bench")
    ap.add_argument("--runs", type=Path, default=ROOT / "runs", help="where <run>/validation_predictions.jsonl lives")
    ap.add_argument("--models", default="", help="comma-separated run ids; default: every run with validation predictions")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "recipe-k")
    a = ap.parse_args()

    runs = [m for m in a.models.split(",") if m] or sorted(
        p.parent.name for p in a.runs.glob("*/validation_predictions.jsonl"))
    if not runs:
        print("no runs with validation predictions", file=sys.stderr)
        return 1
    test_recs = list(iter_records(a.records, None, "test", None))
    val_recs = list(iter_records(a.records, None, "validation", None))
    k_of = {r.source: len(r.option_keys()) for r in test_recs}
    prim_of = {r.source: r.primitive for r in test_recs}
    big = sorted({s for s, k in k_of.items() if k > 30 and prim_of[s] == "choice"})
    small = sorted({s for s, k in k_of.items() if k <= 5 and prim_of[s] == "choice"})

    rows = []
    for run in runs:
        vp = a.runs / run / "validation_predictions.jsonl"
        tp = a.runs / run / "test_predictions.jsonl"
        if not (vp.exists() and tp.exists()):
            print(f"skip {run}: missing predictions", file=sys.stderr)
            continue
        val_preds, test_preds = list(read_predictions(vp)), list(read_predictions(tp))
        mode = next((p.extra.get("mode") for p in val_preds if p.extra), "index")
        base = Recipe(mode="label" if mode == "label" else "index")
        out: dict[str, dict] = {}
        for tag, ks in (("scalar", False), ("k_slope", True)):
            recipe, log = fit_recipe(val_recs, val_preds, base, k_slope=ks)
            reps = score(test_recs, refinalize(test_recs, test_preds, recipe))
            out[tag] = {"recipe": recipe.as_dict(), "macro": macro(reps), "big_k": macro(reps, big),
                        "small_k": macro(reps, small), "chosen": log["chosen"]}
        row = {"run": run, **out}
        row["slope"] = out["k_slope"]["recipe"].get("temp_k_slope", {})
        rows.append(row)
        s, k = out["scalar"], out["k_slope"]
        print(f"{run:22s} macro ECE {s['macro']['ece']:.3f} -> {k['macro']['ece']:.3f}   "
              f"K>30 {s['big_k']['ece']:.3f} -> {k['big_k']['ece']:.3f}   K<=5 {s['small_k']['ece']:.3f} -> {k['small_k']['ece']:.3f}   "
              f"slope {row['slope'] or '—'}", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "recipe_k.json").write_text(json.dumps({"big_k_sources": big, "small_k_sources": small, "rows": rows}, indent=1),
                                         encoding="utf-8")
    md = ["# One temperature per primitive, or one that varies with the option count?", "",
          "Each model's recipe refitted on its own validation predictions both ways and scored on test. "
          f"`K>30` = {', '.join('`' + s + '`' for s in big)}; `K<=5` = {', '.join('`' + s + '`' for s in small)}. "
          "The slope is kept only when it cuts validation NLL by >=1%.", "",
          "| model | macro ECE scalar | macro ECE T(K) | K>30 scalar | K>30 T(K) | K<=5 scalar | K<=5 T(K) | macro acc | slope kept |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        s, k = r["scalar"], r["k_slope"]
        md.append(f"| `{r['run']}` | {s['macro']['ece']:.3f} | **{k['macro']['ece']:.3f}** | {s['big_k']['ece']:.3f} | "
                  f"**{k['big_k']['ece']:.3f}** | {s['small_k']['ece']:.3f} | {k['small_k']['ece']:.3f} | "
                  f"{k['macro']['acc']:.3f} | {', '.join(f'{p} {v:+.2f}' for p, v in r['slope'].items()) or '—'} |")
    (a.out / "recipe_k.md").write_text(chr(10).join(md) + chr(10), encoding="utf-8")
    print(chr(10).join(md))

    # figure: macro ECE before/after, ordered by the gain
    plt = F._mpl()
    order = sorted(rows, key=lambda r: r["scalar"]["macro"]["ece"] - r["k_slope"]["macro"]["ece"], reverse=True)
    fig, ax = plt.subplots(figsize=(11.0, 0.42 * len(order) + 2.4))
    y = np.arange(len(order)); h = 0.38
    ax.barh(y + h / 2, [r["scalar"]["macro"]["ece"] for r in order], h, color=F.BLUE_RAMP[2],
            edgecolor=F.SURFACE, linewidth=0.6, label="one temperature per primitive", zorder=3)
    ax.barh(y - h / 2, [r["k_slope"]["macro"]["ece"] for r in order], h, color=F.BLUE_RAMP[6],
            edgecolor=F.SURFACE, linewidth=0.6, label="temperature affine in log2(K/2)", zorder=3)
    for yi, r in zip(y, order):
        ax.text(r["scalar"]["macro"]["ece"] + 0.002, yi + h / 2, f"{r['scalar']['macro']['ece']:.3f}", va="center", fontsize=6.8, color=F.INK2)
        ax.text(r["k_slope"]["macro"]["ece"] + 0.002, yi - h / 2, f"{r['k_slope']['macro']['ece']:.3f}", va="center", fontsize=6.8, color=F.INK)
    ax.set_yticks(y); ax.set_yticklabels([r["run"] for r in order], fontsize=8)
    ax.invert_yaxis(); ax.tick_params(length=0)
    ax.set_xlabel("macro expected calibration error on test (lower is better)", fontsize=9)
    ax.grid(True, axis="x", color=F.GRID, lw=0.6, zorder=0); ax.grid(False, axis="y")
    ax.legend(loc="lower right", fontsize=8)
    title = "One temperature per primitive is fitted across two regimes"
    sub = ("Every model stores its raw log-scores, so both recipes are fitted on the same validation predictions and scored on "
           "the same test records -- no re-inference. The slope is kept only where it cuts validation NLL by at least 1%, "
           "so a model whose miscalibration does not depend on the option count keeps exactly its old recipe.")
    fig.tight_layout(rect=F._layout(fig, F._wrapped_lines(fig, title, sub)))
    F._headline(fig, title, sub)
    F.PROVENANCE = f"jev-bench v0.1.1 · {datetime.date.today().isoformat()}"
    F._footer(fig)
    print(F._save(fig, a.out / "recipe_k"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
