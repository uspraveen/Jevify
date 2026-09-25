"""Aggregate every results/<run>/ (plus the Jev baseline) into a leaderboard table and figures.

    python scripts/leaderboard.py [--push]

Writes results/leaderboard/{leaderboard.md,leaderboard.json,compare_accuracy.png,compare_ece.png}
and, with --push, uploads them and refreshes the dataset card's Jevified-models section.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jevify.bench import figures as F  # noqa: E402

PRIMS = ("choice", "score", "noul")
# checkpoints trained by other teams, scored here with the Tier 0 readout and a recipe fitted on validation
EXTERNAL = {"togethercomputer/Tev1-4B-experimental": "Together AI, LoRA SFT of Qwen3.5-4B"}
# the published readout fine-tunes (FINDINGS 18; seed 0), read from results/post-training/models/<slug>/. The figures
# draw only the coherence arm of each backbone, so four new points join the map instead of fifteen.
READOUT = [("2b-lora-supervised", "Qwen/Qwen3.5-2B", "readout LoRA"), ("2b-lora-coherence", "Qwen/Qwen3.5-2B", "readout LoRA + coherence"),
           ("4b-lora-supervised-seed-0", "Qwen/Qwen3.5-4B", "readout LoRA"), ("4b-lora-coherence-seed-0", "Qwen/Qwen3.5-4B", "readout LoRA + coherence"),
           ("9b-lora-supervised", "Qwen/Qwen3.5-9B", "readout LoRA"), ("9b-lora-coherence", "Qwen/Qwen3.5-9B", "readout LoRA + coherence"),
           ("gemma-4-e4b-it-lora-supervised", "google/gemma-4-E4B-it", "readout LoRA"),
           ("gemma-4-e4b-it-lora-coherence", "google/gemma-4-E4B-it", "readout LoRA + coherence"),
           ("qwen3-5-4b-base-lora-supervised", "Qwen/Qwen3.5-4B-Base", "readout LoRA"),
           ("qwen3-5-4b-base-lora-coherence", "Qwen/Qwen3.5-4B-Base", "readout LoRA + coherence"),
           ("qwen3-5-2b-base-lora-supervised", "Qwen/Qwen3.5-2B-Base", "readout LoRA"),
           ("qwen3-5-2b-base-lora-coherence", "Qwen/Qwen3.5-2B-Base", "readout LoRA + coherence"),
           ("gemma-4-e4b-base-lora-supervised", "google/gemma-4-E4B", "readout LoRA"),
           ("2b-full-ft-supervised-lr-1e-6", "Qwen/Qwen3.5-2B", "readout full fine-tune"),
           ("2b-full-ft-coherence-lr-1e-6", "Qwen/Qwen3.5-2B", "readout full fine-tune + coherence")]
IN_FIGURES = {"2b-lora-coherence", "4b-lora-coherence-seed-0", "9b-lora-coherence", "gemma-4-e4b-it-lora-coherence"}


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
    clm_dir = ROOT / "results" / "clm-8b"
    if (clm_dir / "test_metrics.json").exists():
        # another team's open System One model, served by its own code: a row beside Jev, not a Jevified one
        entries.append({"run": "clm-8b", "label": "CLM-v0.1-8B (Contrastive-LM, self-hosted)", "tier": "External",
                        "params": "Qwen3-8B frozen + 20M head",
                        "metrics": json.loads((clm_dir / "test_metrics.json").read_text(encoding="utf-8")),
                        "preds": clm_dir / "test_predictions.jsonl", "cost": None, "gpu": None})
    for d in sorted((ROOT / "results").glob("*/")):
        # a Jevified run always carries run.json; results/jev-1.13.0 (the baseline, added
        # explicitly above) and results/leaderboard do not, and neither is a model row
        if not (d / "test_metrics.json").exists() or not (d / "run.json").exists():
            continue
        if re.search(r"-s\d+$", d.name):
            # seed replicates (<run>-s1, -s2, ...) are reported as a spread in their own
            # section, not as five rows of the same model
            continue
        run = json.loads((d / "run.json").read_text())
        if run.get("modality") == "vision":
            # vision runs score different sources (POPE, A-OKVQA, AI2D); averaging them into the
            # text leaderboard would compare nothing with nothing. They have their own card section.
            continue
        recipe = json.loads((d / "recipe.json").read_text(encoding="utf-8")) if (d / "recipe.json").exists() else {}
        suffix = "" if not run.get("tier") else (" residual" if run.get("residual") else " replace")
        tier = f"Tier {run.get('tier', 0)}{suffix}"
        if run.get("model_id") in EXTERNAL:
            # someone else's fine-tuned checkpoint read through our Tier 0 readout: not "no training"
            tier = "External"
        # two runs of the same tier on the same backbone must not share a label: name the
        # hyperparameters that differ from the defaults (a low-LR or soft-label Tier 2 arm)
        variant = []
        if run.get("tier") == 2 and run.get("lora_lr") not in (None, 1e-4):
            variant.append(f"lr {run['lora_lr']:g}")
        if run.get("soft_labels"):
            variant.append("soft labels")
        shown = tier + (", " + ", ".join(variant) if variant else "")
        # the same checkpoint read in two prompt formats is two rows: name the non-default one, and
        # always name it for another team's checkpoint, which was trained on a format of its own
        fmt = (recipe.get("recipe") or {}).get("prompt", "jevify")
        fmt_label = "" if fmt == "jevify" and run.get("model_id") not in EXTERNAL else f" ({fmt.capitalize()} prompt)"
        entries.append({"run": d.name, "label": run.get("model_id", d.name) + ("" if not run.get("tier") else f" ({shown})") + fmt_label,
                        "tier": tier, "params": "",
                        "metrics": json.loads((d / "test_metrics.json").read_text(encoding="utf-8")),
                        "preds": d / "test_predictions.jsonl", "cost": run.get("est_cost_usd"),
                        # rented runs record the Modal GPU name; runs on our own hardware record the device
                        "gpu": run.get("gpu") or (run.get("device", "").replace("NVIDIA ", "") or None),
                        "chat": run.get("chat_applied"), "recipe": recipe.get("recipe")})
    # a model reported only as seed replicates (<run>-s0 ... -sN with no <run>/) gets one row: the per-source mean over
    # its seeds. Without this, the 9B Tier 1 row existed only as a hand edit of the README and a regeneration dropped it.
    groups: dict[str, list[Path]] = {}
    for d in sorted((ROOT / "results").glob("*/")):
        m = re.match(r"(.+)-s\d+$", d.name)
        if m and (d / "test_metrics.json").exists() and (d / "run.json").exists():
            groups.setdefault(m.group(1), []).append(d)
    for base, ds in groups.items():
        if len(ds) < 2 or (ROOT / "results" / base / "run.json").exists():
            continue                          # a single-run row already stands for this model
        run = json.loads((ds[0] / "run.json").read_text())
        ms = [json.loads((d / "test_metrics.json").read_text(encoding="utf-8")) for d in ds]
        num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
        mean = {src: {k: (float(np.mean([m[src][k] for m in ms])) if all(num(m[src].get(k)) for m in ms) else ms[0][src].get(k))
                      for k in ms[0][src]} for src in ms[0] if all(src in m for m in ms)}
        tier = f"Tier {run.get('tier', 0)}" + ("" if not run.get("tier") else (" residual" if run.get("residual") else " replace"))
        cap = f"{run['epochs']}-epoch cap, " if run.get("tier") == 1 and run.get("epochs") == 6 else ""
        # drawn only when it is the backbone's sole row at this tier (the 9B Tier 1); a capped-schedule ablation of a
        # model already on the map stays in the table and off the figure
        drawn = not any(x["label"].split(" (")[0] == run.get("model_id") and x["tier"] == tier for x in entries)
        entries.append({"run": base, "label": f"{run.get('model_id', base)} ({tier}, {cap}mean of {len(ds)} seeds)", "tier": tier,
                        "params": "", "metrics": mean, "preds": None, "cost": None, "in_figures": drawn,
                        "gpu": (run.get("device", "").replace("NVIDIA ", "") or None)})
    for slug, model, variant in READOUT:
        d = ROOT / "results" / "post-training" / "models" / slug
        if (d / "test_metrics.json").exists():
            entries.append({"run": f"post-training/{slug}", "label": f"{model} ({variant})", "tier": "Readout FT", "params": "",
                            "metrics": json.loads((d / "test_metrics.json").read_text(encoding="utf-8")),
                            "preds": None, "cost": None, "gpu": None, "in_figures": slug in IN_FIGURES})
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
    import datetime as _dt
    F.PROVENANCE = f"jev-bench v{manifest.get('version', '?')} · {_dt.date.today().isoformat()}"
    # CLM sits at 0.34 / 0.34, far from every other model; drawn, it squeezes the rest into one corner
    drawn = [e for e in rows if e.get("in_figures", True)]
    F.fig_models_map([e for e in drawn if e["run"] != "clm-8b"], out,
                     off_chart=[e for e in drawn if e["run"] == "clm-8b"])
    # per-config detail for every model at once: a heatmap built from each run's
    # test_metrics.json. Nothing here reads a predictions file -- the previous grouped bar
    # chart loaded all of them and then only drew when there were six models or fewer,
    # which left a stale six-model chart published under a thirteen-model leaderboard.
    heat_rows = [{"label": "Jev 1.13.0" if e["tier"] == "API" else e["label"].split("/")[-1], "metrics": e["metrics"]} for e in drawn]
    for metric in ("accuracy", "ece"):
        F.fig_metric_heatmap(heat_rows, prim_of, out, metric)
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
