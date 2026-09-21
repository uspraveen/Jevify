"""Does starving the vision encoder cost accuracy, calibration, or both?

Every VLM deployment picks a pixel budget, usually by copying a default. The budget is
treated as a speed/accuracy dial. For a System One model the interesting question is
different: when the encoder is given less to work with, does the model *know* it knows
less? A model that loses accuracy and gains uncertainty is behaving well under
degradation. A model that loses accuracy and holds its confidence is the dangerous case
— and on POPE, which asks whether an object is present, that case is hallucination.

The sweep holds the records, the prompt and the decoder fixed and moves only the budget,
recording the token count each budget actually produced rather than the one requested.

    python scripts/vision_budget.py --model-id Qwen/Qwen3-VL-2B-Instruct \\
        --budgets 64,128,256,512,1024 --limit 400 --out runs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jevify.bench.figures import GRID, INK, INK2, SURFACE  # noqa: E402
from jevify.runners.base import read_predictions  # noqa: E402
from jevify.runners.cli import score  # noqa: E402
from jevify.bench.record import read_jsonl  # noqa: E402

PATCH = 28 * 28              # Qwen-family pixel budgets are counted in 28x28 patches
SOURCE_COLOR = {"pope": "#1baf7a", "aokvqa": "#2a78d6", "ai2d": "#eb6834"}


def run_budgets(model_id: str, base: str, out: Path, budgets: list[int], limit: int,
                batch: int, sources: str, trust_remote_code: bool) -> None:
    from jevify.train import run_vision

    for b in budgets:
        run_id = f"{base}-px{b}"
        if (out / run_id / "run.json").exists():
            print(f"[{run_id}] already done, skipping", flush=True)
            continue
        run_vision(model_id, run_id, out / run_id,
                   sources=[s for s in sources.split(",") if s] or None,
                   split="test", limit=limit, batch=batch, max_pixels=b * PATCH,
                   trust_remote_code=trust_remote_code)


def collect(out: Path, base: str, budgets: list[int]) -> list[dict]:
    rows = []
    for b in budgets:
        d = out / f"{base}-px{b}"
        meta_path, preds_path, recs_path = d / "run.json", d / "test_predictions.jsonl", d / "test_records.jsonl"
        if not (meta_path.exists() and preds_path.exists() and recs_path.exists()):
            print(f"  skipping {d.name}: incomplete", file=sys.stderr)
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        preds = list(read_predictions(preds_path))
        recs = list(read_jsonl(recs_path))
        reports = score(recs, preds)
        tokens = (meta.get("budget") or {}).get("image_tokens") or {}
        row = {"budget_patches": b, "max_pixels": b * PATCH,
               "image_tokens_median": tokens.get("median"),
               "wall_s": meta.get("wall_s"), "n": meta.get("n"),
               "sources": {src: {"n": r.n, "accuracy": r.accuracy, "ece": r.ece,
                                 "brier": r.brier, "mean_confidence": r.mean_confidence}
                           for src, r in sorted(reports.items())}}
        vals = list(row["sources"].values())
        row["macro_accuracy"] = sum(v["accuracy"] for v in vals) / len(vals) if vals else None
        row["macro_ece"] = sum(v["ece"] for v in vals) / len(vals) if vals else None
        rows.append(row)
    return rows


def markdown(rows: list[dict], model_id: str) -> str:
    srcs = sorted({s for r in rows for s in r["sources"]})
    out = [f"# Vision encoder budget sweep — {model_id}", "",
           "Same records, same prompt, same decoder; only the pixel budget moves.",
           "`tokens` is the median image-token count the budget actually produced.", "",
           "| budget (patches) | tokens | " + " | ".join(f"{s} acc / ECE" for s in srcs)
           + " | macro acc | macro ECE |",
           "|---|---|" + "---|" * (len(srcs) + 2)]
    for r in rows:
        cells = []
        for s in srcs:
            v = r["sources"].get(s)
            cells.append(f"{v['accuracy']:.3f} / {v['ece']:.3f}" if v else "—")
        out.append(f"| {r['budget_patches']} | {r['image_tokens_median'] or '—'} | "
                   + " | ".join(cells)
                   + f" | {r['macro_accuracy']:.3f} | {r['macro_ece']:.3f} |")
    return "\n".join(out) + "\n"


def figure(rows: list[dict], model_id: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    srcs = sorted({s for r in rows for s in r["sources"]})
    x = [r["image_tokens_median"] or r["budget_patches"] for r in rows]

    # where do distinct budgets stop buying distinct tokens?
    seen: dict[int, list[int]] = {}
    for r, xi in zip(rows, x):
        seen.setdefault(xi, []).append(r["budget_patches"])
    collapsed = {k: v for k, v in seen.items() if len(v) > 1}
    sat = min(collapsed) if collapsed else None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    for ax, metric, title, better in (
            (axes[0], "accuracy", "Accuracy", "higher is better"),
            (axes[1], "ece", "Calibration error (ECE)", "lower is better")):
        ax.set_facecolor(SURFACE)
        for s in srcs:
            ys = [r["sources"].get(s, {}).get(metric) for r in rows]
            pts = [(a, b) for a, b in zip(x, ys) if b is not None]
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", markersize=7, linewidth=2,
                    color=SOURCE_COLOR.get(s, INK2), label=s, zorder=3)
            ax.annotate(s, (pts[-1][0], pts[-1][1]), textcoords="offset points", xytext=(8, 0),
                        color=SOURCE_COLOR.get(s, INK2), fontsize=9, va="center")
        ax.set_xscale("log")
        # tick at the measured token counts, not at decades: those are the x values that exist
        ticks = sorted(set(x))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(int(t)) for t in ticks], fontsize=9)
        ax.minorticks_off()
        # two budgets collapsing onto one token count is the finding, not a plotting artifact
        if sat is not None:
            ax.axvline(sat, color=INK2, linewidth=1, linestyle=(0, (4, 3)), alpha=0.45, zorder=1)
        ax.set_xlabel("median image tokens per record (log)", color=INK2, fontsize=9)
        ax.set_title(f"{title} — {better}", color=INK, fontsize=11, loc="left")
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
    fig.suptitle(f"What the vision encoder's budget buys — {model_id}", color=INK, fontsize=13, x=0.01, ha="left")
    if sat is not None:
        budgets = " and ".join(f"{b}" for b in sorted(collapsed[sat]))
        fig.text(0.01, 0.905, f"Dashed line: the budget saturates. {budgets} patches both yield the same "
                              f"{int(sat)} median tokens, and score the same — the images are smaller than "
                              f"the budget.", color=INK2, fontsize=9, ha="left")
    fig.tight_layout(rect=(0, 0, 0.98, 0.90))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--base", default="", help="run-id prefix (default: derived from the model)")
    ap.add_argument("--out", type=Path, default=ROOT / "runs")
    ap.add_argument("--budgets", default="64,128,256,512,1024", help="in 28x28 patches")
    ap.add_argument("--limit", type=int, default=400, help="records per source")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--sources", default="")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--reports", type=Path, default=ROOT / "reports" / "vision-budget")
    a = ap.parse_args()

    base = a.base or a.model_id.split("/")[-1].lower().replace("-instruct", "")
    budgets = [int(b) for b in a.budgets.split(",") if b]
    if not a.report_only:
        run_budgets(a.model_id, base, a.out, budgets, a.limit, a.batch, a.sources, a.trust_remote_code)

    rows = collect(a.out, base, budgets)
    if not rows:
        print("no completed runs to report", file=sys.stderr)
        return 1
    a.reports.mkdir(parents=True, exist_ok=True)
    (a.reports / "results.json").write_text(json.dumps({"model_id": a.model_id, "rows": rows}, indent=1),
                                            encoding="utf-8")
    md = markdown(rows, a.model_id)
    (a.reports / "README.md").write_text(md, encoding="utf-8")
    print(md)
    figure(rows, a.model_id, a.reports / "vision_budget.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
