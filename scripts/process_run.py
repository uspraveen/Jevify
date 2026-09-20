"""Turn a finished Tier 0 run into published results.

    python scripts/process_run.py --run-id qwen35-0.8b [--label "Qwen3.5-0.8B (Tier 0)"] [--push]

1. pull <run>-validation and <run> predictions from the jevify-runs Modal volume
2. jevify-run recipe: fit temperatures / permutation / prior on validation, apply to test,
   write recipe.json, ablation.md, test_report.md, test_metrics.json, figures/
3. compare against Jev 1.13.0 on the same records (compare.md + bar charts)
4. optionally upload results/<run>/ to the jev-bench dataset repo
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin")
MODAL, JRUN = str(VENV / "modal"), str(VENV / "jevify-run")


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    return subprocess.run(cmd, cwd=ROOT, env=env, text=True, encoding="utf-8", errors="replace", **kw)


def pull(run_id: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for remote, local in ((f"{run_id}-validation/validation_predictions.jsonl", "validation_predictions.jsonl"),
                          (f"{run_id}/test_predictions.jsonl", "test_predictions.jsonl"),
                          (f"{run_id}/run.json", "run.json"), (f"{run_id}-validation/run.json", "run_validation.json")):
        r = sh([MODAL, "volume", "get", "jevify-runs", remote, str(out / local), "--force"], capture_output=True)
        if r.returncode != 0:
            print(f"  missing on volume: {remote}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--records", type=Path, default=ROOT / "data" / "jev-bench")
    ap.add_argument("--jev", type=Path, default=ROOT / "data" / "jev-bench" / "results" / "jev-1.13.0" / "test_predictions.jsonl")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--no-pull", action="store_true")
    a = ap.parse_args()
    runs = ROOT / "runs" / a.run_id
    if not a.no_pull:
        pull(a.run_id, runs)
    label = a.label or a.run_id
    results = ROOT / "results" / a.run_id
    results.mkdir(parents=True, exist_ok=True)
    r = sh([JRUN, "recipe", "--records", str(a.records), "--val", str(runs / "validation_predictions.jsonl"),
            "--test", str(runs / "test_predictions.jsonl"), "--out", str(results), "--label", label], capture_output=True)
    print(r.stdout[-3000:], r.stderr[-1500:] if r.returncode else "", sep="\n")
    if r.returncode:
        return r.returncode
    # side-by-side with Jev on the same records
    sh([JRUN, "compare", "--records", str(a.records), "--preds", str(a.jev), str(results / "test_predictions.jsonl"),
        "--labels", "Jev 1.13.0", label, "--out", str(results / "vs_jev")], capture_output=True)
    for name in ("run.json", "run_validation.json"):
        if (runs / name).exists():
            (results / name).write_text((runs / name).read_text())
    if a.push:
        from huggingface_hub import HfApi

        HfApi(token=os.environ.get("HF_TOKEN")).upload_folder(folder_path=str(results), path_in_repo=f"results/{a.run_id}",
                                                             repo_id="Praveenrajus/jev-bench", repo_type="dataset",
                                                             commit_message=f"results: {a.run_id} (Tier 0)")
        print("pushed to Praveenrajus/jev-bench results/" + a.run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
