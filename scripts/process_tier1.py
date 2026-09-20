"""Score a Tier 1 run, split by held-out vs trained sources, against Jev and its own Tier 0.

    python scripts/process_tier1.py --run-id qwen35-2b-t1 --tier0 qwen35-2b [--push]

The held-out sources are the honest comparison: the heads never saw them in training, so
that number answers "would this work on a question nobody trained for?" — which is the only
regime in which a Tier 1 model is comparable to Jev at all.

For fairness the Tier 0 recipe is *refitted* using only the training sources' validation
splits before it is scored on the held-out sources; otherwise its temperature has seen data
the heads never did.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VENV = ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin")
MODAL = str(VENV / "modal")

from jevify.bench.figures import make_all  # noqa: E402
from jevify.engine.predict import Recipe, refinalize  # noqa: E402
from jevify.runners.base import read_predictions, write_predictions  # noqa: E402
from jevify.runners.cli import iter_records, markdown_table, score  # noqa: E402
from jevify.runners.recipe import fit_recipe  # noqa: E402


def pull(run_id: str, out: Path, names: list[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        subprocess.run([MODAL, "volume", "get", "jevify-runs", f"{run_id}/{name}", str(out / Path(name).name), "--force"],
                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")


def macro(reports: dict, sources: list[str]) -> dict[str, float]:
    rs = [r for s, r in reports.items() if s in sources]
    if not rs:
        return {}
    return {"n_sources": len(rs), "acc": float(np.mean([r.accuracy for r in rs])), "ece": float(np.mean([r.ece for r in rs])),
            "brier": float(np.mean([r.brier for r in rs])), "sel90": float(np.mean([r.selective_acc_at_90 for r in rs]))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--tier0", default=None, help="run id of the same backbone at Tier 0")
    ap.add_argument("--records", type=Path, default=ROOT / "data" / "jev-bench")
    ap.add_argument("--label", default=None)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--no-pull", action="store_true")
    a = ap.parse_args()

    runs = ROOT / "runs" / a.run_id
    if not a.no_pull:
        pull(a.run_id, runs, ["test_predictions.jsonl", "run.json"])
    meta = json.loads((runs / "run.json").read_text())
    held = meta["heldout_sources"]
    label = a.label or f"{meta['model_id']} (Tier 1)"
    results = ROOT / "results" / a.run_id
    results.mkdir(parents=True, exist_ok=True)

    test_recs = list(iter_records(a.records, None, "test", None))
    all_sources = sorted({r.source for r in test_recs})
    trained = [s for s in all_sources if s not in held]
    t1_preds = list(read_predictions(runs / "test_predictions.jsonl"))
    t1_reports = score(test_recs, t1_preds)

    rows: list[dict] = [{"model": label, "variant": "Tier 1 heads", **{f"heldout_{k}": v for k, v in macro(t1_reports, held).items()},
                         **{f"trained_{k}": v for k, v in macro(t1_reports, trained).items()}}]

    # Jev on the same records
    jev_preds = list(read_predictions(a.records / "results" / "jev-1.13.0" / "test_predictions.jsonl"))
    jev_reports = score(test_recs, jev_preds)
    rows.append({"model": "Jev 1.13.0", "variant": "API (zero-shot)", **{f"heldout_{k}": v for k, v in macro(jev_reports, held).items()},
                 **{f"trained_{k}": v for k, v in macro(jev_reports, trained).items()}})

    # the same backbone at Tier 0, with its recipe refitted without the held-out sources
    if a.tier0:
        t0_dir = ROOT / "runs" / a.tier0
        val_preds = [p for p in read_predictions(t0_dir / "validation_predictions.jsonl") if p.id.split("/")[0] in trained]
        val_recs = [r for r in iter_records(a.records, trained, "validation", None)]
        base = Recipe(mode="index")
        fair_recipe, log = fit_recipe(val_recs, val_preds, base)
        t0_raw = list(read_predictions(t0_dir / "test_predictions.jsonl"))
        t0_fair = refinalize(test_recs, t0_raw, fair_recipe)
        t0_reports = score(test_recs, t0_fair)
        rows.append({"model": meta["model_id"], "variant": "Tier 0 (recipe refit w/o held-out)",
                     **{f"heldout_{k}": v for k, v in macro(t0_reports, held).items()},
                     **{f"trained_{k}": v for k, v in macro(t0_reports, trained).items()}})
        (results / "tier0_fair_recipe.json").write_text(json.dumps({"recipe": fair_recipe.as_dict(), "search": log}, indent=1), encoding="utf-8")

    NL = chr(10)
    cols = ["model", "variant", "held-out acc", "held-out ECE", "held-out Brier", "trained acc", "trained ECE", "trained Brier"]
    md = [f"# {label}: Tier 1 vs Tier 0 vs Jev", "",
          f"Heads trained on {meta['n_train']:,} records from {len(trained)} sources (max {meta['max_slots']} options per record), "
          f"early-stopped on {meta['n_val']:,} validation records; epoch {meta['best_epoch']}.", "",
          f"**Held-out sources** (never seen in training): {', '.join('`' + s + '`' for s in held)}. "
          "`chaosnli` has no train split, so it is held out by construction.", "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        md.append("| " + " | ".join([r["model"], r["variant"],
                                     f"{r.get('heldout_acc', float('nan')):.3f}", f"{r.get('heldout_ece', float('nan')):.3f}",
                                     f"{r.get('heldout_brier', float('nan')):.3f}", f"{r.get('trained_acc', float('nan')):.3f}",
                                     f"{r.get('trained_ece', float('nan')):.3f}", f"{r.get('trained_brier', float('nan')):.3f}"]) + " |")
    md += ["", "## Per-config (Tier 1)", "", markdown_table(t1_reports, {r.source: r.primitive for r in test_recs}, label)]
    (results / "tier1_report.md").write_text(NL.join(md) + NL, encoding="utf-8")
    (results / "test_report.md").write_text(markdown_table(t1_reports, {r.source: r.primitive for r in test_recs}, label) + NL, encoding="utf-8")
    (results / "test_metrics.json").write_text(json.dumps({k: v.as_dict() for k, v in t1_reports.items()}, indent=1), encoding="utf-8")
    (results / "summary.json").write_text(json.dumps({"rows": rows, "heldout": held, "meta": meta}, indent=1), encoding="utf-8")
    (results / "run.json").write_text(json.dumps(meta, indent=1))
    write_predictions(results / "test_predictions.jsonl", t1_preds, progress_every=0)
    make_all(a.records, results / "test_predictions.jsonl", results / "figures", label, "test")
    print(NL.join(md[:12]))
    if a.push:
        from huggingface_hub import HfApi

        HfApi(token=os.environ.get("HF_TOKEN")).upload_folder(folder_path=str(results), path_in_repo=f"results/{a.run_id}",
                                                             repo_id="Praveenrajus/jev-bench", repo_type="dataset",
                                                             commit_message=f"results: {a.run_id} (Tier 1)")
        print("pushed results/" + a.run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
