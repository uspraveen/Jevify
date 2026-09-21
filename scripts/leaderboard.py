"""Aggregate every results/<run>/ (plus the Jev baseline) into a leaderboard table and figures.

    python scripts/leaderboard.py [--push]

Writes results/leaderboard/{leaderboard.md,leaderboard.json,compare_accuracy.png,compare_ece.png}
and, with --push, uploads them and refreshes the dataset card's Jevified-models section.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jevify.bench import figures as F  # noqa: E402
from jevify.bench.figures import fig_compare, load_eval  # noqa: E402

PRIMS = ("choice", "score", "noul")


def summarize(metrics: dict, prim_of: dict[str, str]) -> dict:
    out = {}
    for prim in PRIMS + ("macro",):
        rs = [m for s, m in metrics.items() if prim == "macro" or prim_of.get(s) == prim]
        if rs:
            out[prim] = {"acc": float(np.mean([r["accuracy"] for r in rs])), "ece": float(np.mean([r["ece"] for r in rs])),
                         "brier": float(np.mean([r["brier"] for r in rs])), "sel90": float(np.mean([r["selective_acc_at_90"] for r in rs]))}
    gold = [m["tvd_to_human"] for s, m in metrics.items() if m.get("tvd_to_human") is not None]
    out["tvd_human"] = float(np.mean(gold)) if gold else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=ROOT / "data" / "jev-bench")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    manifest = json.loads((a.records / "manifest.json").read_text(encoding="utf-8"))
    prim_of = {k: v["primitive"] for k, v in manifest["sources"].items() if "error" not in v}

    entries: list[dict] = []
    jev_dir = a.records / "results" / "jev-1.13.0"
    entries.append({"run": "jev-1.13.0", "label": "Jev 1.13.0 (TypeSafe API)", "tier": "API", "params": "undisclosed",
                    "metrics": json.loads((jev_dir / "test_metrics.json").read_text(encoding="utf-8")),
                    "preds": jev_dir / "test_predictions.jsonl", "cost": None, "gpu": None})
    for d in sorted((ROOT / "results").glob("*/")):
        if not (d / "test_metrics.json").exists() or d.name == "leaderboard":
            continue
        run = json.loads((d / "run.json").read_text()) if (d / "run.json").exists() else {}
        recipe = json.loads((d / "recipe.json").read_text(encoding="utf-8")) if (d / "recipe.json").exists() else {}
        tier = f"Tier {run.get('tier', 0)}" + ("" if not run.get("tier") else (" residual" if run.get("residual") else " replace"))
        entries.append({"run": d.name, "label": run.get("model_id", d.name) + ("" if not run.get("tier") else f" [{d.name}]"),
                        "tier": tier, "params": "",
                        "metrics": json.loads((d / "test_metrics.json").read_text(encoding="utf-8")),
                        "preds": d / "test_predictions.jsonl", "cost": run.get("est_cost_usd"), "gpu": run.get("gpu"),
                        "chat": run.get("chat_applied"), "recipe": recipe.get("recipe")})
    rows = []
    for e in entries:
        s = summarize(e["metrics"], prim_of)
        e["summary"] = s
        rows.append(e)
    rows.sort(key=lambda e: (e["tier"] == "API" and -2 or 0, -e["summary"]["macro"]["acc"]))

    NL = chr(10)
    cols = ["model", "tier", "macro acc", "macro ECE", "macro Brier", "sel@90", "choice acc", "score acc", "noul acc", "TVD→human", "GPU", "test cost"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for e in rows:
        s = e["summary"]
        lines.append("| " + " | ".join([
            f"**{e['label']}**" if e["tier"] == "API" else e["label"], e["tier"],
            f"{s['macro']['acc']:.3f}", f"{s['macro']['ece']:.3f}", f"{s['macro']['brier']:.3f}", f"{s['macro']['sel90']:.3f}",
            f"{s['choice']['acc']:.3f}", f"{s['score']['acc']:.3f}", f"{s['noul']['acc']:.3f}",
            f"{s['tvd_human']:.3f}" if s["tvd_human"] is not None else "",
            e["gpu"] or "", f"${e['cost']:.2f}" if e.get("cost") else ""]) + " |")
    table = NL.join(lines)
    out = ROOT / "results" / "leaderboard"
    out.mkdir(parents=True, exist_ok=True)
    (out / "leaderboard.md").write_text(table + NL, encoding="utf-8")
    (out / "leaderboard.json").write_text(json.dumps([{k: v for k, v in e.items() if k not in ("metrics", "preds")} for e in rows], indent=1, default=str), encoding="utf-8")
    evs = {}
    for e in rows:
        if Path(e["preds"]).exists():
            evs[e["label"] if e["tier"] != "API" else "Jev 1.13.0"] = load_eval(a.records, Path(e["preds"]), "test")
    import datetime as _dt
    F.PROVENANCE = f"jev-bench v{manifest.get('version', '?')} · {_dt.date.today().isoformat()}"
    F.fig_models_map(rows, out)
    if evs and len(evs) <= 6:
        for metric in ("accuracy", "ece"):
            fig_compare(evs, out, metric)
    print(table)
    if a.push:
        from huggingface_hub import HfApi

        HfApi(token=os.environ.get("HF_TOKEN")).upload_folder(folder_path=str(out), path_in_repo="results/leaderboard",
                                                             repo_id="Praveenrajus/jev-bench", repo_type="dataset",
                                                             commit_message="results: leaderboard refresh")
        print("pushed results/leaderboard")
    return 0


if __name__ == "__main__":
    sys.exit(main())
