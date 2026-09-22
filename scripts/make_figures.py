"""Regenerate every cross-run figure from the results on disk.

    python scripts/make_figures.py [--push]

Writes results/figures/{tier1_story,tier1_per_source,instruct_vs_base,recipe_ladder,confidence_vs_agreement}.{png,svg}
and refreshes the leaderboard's own figures.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jevify.bench import figures as F  # noqa: E402

PAIRS = [("google/gemma-4-E2B-it", "gemma4-e2b-it", True), ("google/gemma-4-E2B", "gemma4-e2b", False),
         ("Qwen/Qwen3.5-0.8B", "qwen35-0.8b", True), ("Qwen/Qwen3.5-0.8B-Base", "qwen35-0.8b-base", False)]


def ablation_rows(run: str) -> list[dict]:
    path = ROOT / "results" / run / "ablation.md"
    if not path.exists():
        return []
    rows, in_test = [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## test"):
            in_test = True
            continue
        if line.startswith("## ") and in_test:
            break
        if in_test and line.startswith("| ") and not line.startswith("| step") and "---" not in line:
            cells = [c.strip() for c in line.strip("|").split("|")]
            rows.append({"step": cells[0], "macro_acc": float(cells[1]), "macro_ece": float(cells[2])})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=ROOT / "data" / "jev-bench")
    ap.add_argument("--tier1", default="qwen35-2b-t1r")
    ap.add_argument("--tier1-replace", default="qwen35-2b-t1")
    ap.add_argument("--tier0", default="qwen35-2b")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    out = ROOT / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    import datetime as _dt
    manifest = json.loads((a.records / "manifest.json").read_text(encoding="utf-8"))
    F.PROVENANCE = f"jev-bench v{manifest.get('version', '?')} · {_dt.date.today().isoformat()}"
    made = []

    summary_path = ROOT / "results" / a.tier1 / "summary.json"
    rep_summary_path = ROOT / "results" / a.tier1_replace / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        label = summary["meta"]["model_id"]
        pick = lambda rows, want: next(r for r in rows if want in r["variant"])  # noqa: E731
        t0row = pick(summary["rows"], "Tier 0")
        jev = pick(summary["rows"], "API")
        variants = [{"label": "Tier 0" + chr(10) + "(no training)", **{k: t0row[k] for k in t0row if k.startswith(("trained_", "heldout_"))}}]
        if rep_summary_path.exists():
            rep = pick(json.loads(rep_summary_path.read_text(encoding="utf-8"))["rows"], "Tier 1")
            variants.append({"label": "Tier 1" + chr(10) + "heads replace", **{k: rep[k] for k in rep if k.startswith(("trained_", "heldout_"))}})
        res = pick(summary["rows"], "Tier 1")
        variants.append({"label": "Tier 1" + chr(10) + "heads residual", **{k: res[k] for k in res if k.startswith(("trained_", "heldout_"))}})
        made.append(F.fig_tier1_story(variants, jev, out, label))
        t0 = json.loads((ROOT / "results" / a.tier0 / "test_metrics.json").read_text(encoding="utf-8"))
        res = json.loads((ROOT / "results" / a.tier1 / "test_metrics.json").read_text(encoding="utf-8"))
        rep_path = ROOT / "results" / a.tier1_replace / "test_metrics.json"
        rep = json.loads(rep_path.read_text(encoding="utf-8")) if rep_path.exists() else res
        held = set(summary["heldout"]) | {"chaosnli"}
        deltas = [(s, s in held, rep[s]["accuracy"] - t0[s]["accuracy"], res[s]["accuracy"] - t0[s]["accuracy"])
                  for s in t0 if s in res and s in rep]
        made.append(F.fig_tier1_per_source(deltas, out, label))

    pairs = []
    for label, run, instruct in PAIRS:
        rows = ablation_rows(run)
        if rows:
            pairs.append({"label": label, "instruct": instruct, "raw_ece": rows[0]["macro_ece"], "cal_ece": rows[-1]["macro_ece"]})
    if pairs:
        made.append(F.fig_instruct_vs_base(pairs, out))

    models = {}
    for d in sorted((ROOT / "results").glob("*/")):
        run_json = d / "run.json"
        if not run_json.exists():
            continue
        meta = json.loads(run_json.read_text())
        if meta.get("tier") or meta.get("modality") == "vision":     # a text-benchmark figure: Tier 0 text runs only
            continue
        rows = ablation_rows(d.name)
        if len(rows) == 4:
            models[meta.get("model_id", d.name).split("/")[-1]] = rows
    if models:
        made.append(F.fig_recipe_ladder(models, out))

    # Jev's confidence against human agreement on ChaosNLI: the figure behind headline
    # finding 1. Generated here so it has a reproducible source like every other figure.
    jev_preds = a.records / "results" / "jev-1.13.0" / "test_predictions.jsonl"
    chaos = a.records / "data" / "chaosnli" / "test.jsonl"
    if jev_preds.exists() and chaos.exists():
        from jevify.bench.record import read_jsonl
        from jevify.runners.base import read_predictions
        recs = list(read_jsonl(chaos))
        preds = {q.id: q for q in read_predictions(jev_preds) if q.id.startswith("chaosnli/")}
        made.append(F.fig_confidence_vs_agreement(recs, preds, out, "Jev 1.13.0"))

    # Every figure lives with the data that produced it: per-model figures under
    # results/<run>/figures, cross-model ones under results/leaderboard, probes under
    # results/jev-1.13.0/probes. This folder holds only the cross-cutting figures generated
    # above. It used to also receive *copies* of headline figures from those other places,
    # which left duplicates -- and once, a stale Tier 1 chart -- where readers expected the
    # canonical file. Nothing is copied here any more; results/README.md is the index.

    for m in made:
        print(m)
    if a.push and made:
        from huggingface_hub import HfApi

        HfApi(token=os.environ.get("HF_TOKEN")).upload_folder(folder_path=str(out), path_in_repo="results/figures",
                                                             repo_id="Praveenrajus/jev-bench", repo_type="dataset",
                                                             commit_message="figures: tier-1 story, instruct vs base, recipe ladder")
        print("pushed results/figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
