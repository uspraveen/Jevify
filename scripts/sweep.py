"""Run the Tier 0 sweep on Modal: one job per (model, split), sequential, resumable.

    python scripts/sweep.py --plan scripts/sweep_plan.json [--smoke] [--only qwen35-0.8b,k2-0.9b]

Each plan entry: {"run_id", "model_id", "gpu", "trust_remote_code"?, "chat"?}. A completed run
is detected by runs/<run_id>/<split>_predictions.jsonl next to run.json, so re-running the
sweep only does what is missing. A ledger of wall time and estimated cost is appended to
runs/ledger.jsonl after every job.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODAL = str(ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin") / "modal")


def run_job(entry: dict, split: str, limit: int, permutations: int, batch: int, out: Path) -> dict | None:
    run_id = entry["run_id"] + ("-smoke" if limit and limit <= 10 else "") + ("" if split == "test" else f"-{split}")
    local = out / run_id
    if (local / f"{split}_predictions.jsonl").exists() and (local / "run.json").exists():
        print(f"[skip] {run_id} exists", flush=True)
        return json.loads((local / "run.json").read_text())
    cmd = [MODAL, "run", "modal_app.py::main", "--model-id", entry["model_id"], "--run-id", run_id, "--split", split,
           "--limit", str(limit), "--permutations", str(permutations), "--batch", str(batch), "--out", str(out)]
    if entry.get("trust_remote_code"):
        cmd.append("--trust-remote-code")
    if entry.get("chat") is False:
        cmd += ["--chat", "false"]
    env = {**os.environ, "JEVIFY_GPU": entry["gpu"], "PYTHONIOENCODING": "utf-8"}
    print(f"[run ] {run_id} on {entry['gpu']} ({split}, limit={limit})", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = "\n".join(proc.stdout.splitlines()[-6:] + proc.stderr.splitlines()[-6:])
    info = None
    if (local / "run.json").exists():
        info = json.loads((local / "run.json").read_text())
    status = "ok" if proc.returncode == 0 and info else "FAILED"
    print(f"[{status}] {run_id} in {time.time() - t0:.0f}s" + (f" est ${info['est_cost_usd']}" if info else f"\n{tail}"), flush=True)
    with (out / "ledger.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": run_id, "model_id": entry["model_id"], "gpu": entry["gpu"], "split": split, "status": status,
                            "wall_s": round(time.time() - t0, 1), "est_cost_usd": info.get("est_cost_usd") if info else None,
                            "scorer_stats": info.get("scorer_stats") if info else None}) + "\n")
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, default=ROOT / "scripts" / "sweep_plan.json")
    ap.add_argument("--only", default="", help="comma-separated run_ids")
    ap.add_argument("--smoke", action="store_true", help="5 records per source, test split only")
    ap.add_argument("--val-limit", type=int, default=200)
    ap.add_argument("--permutations", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", type=Path, default=ROOT / "runs")
    a = ap.parse_args()
    plan = json.loads(a.plan.read_text())
    only = {s for s in a.only.split(",") if s}
    total = 0.0
    for entry in plan:
        if only and entry["run_id"] not in only:
            continue
        if a.smoke:
            info = run_job(entry, "test", 5, a.permutations, a.batch, a.out)
            total += (info or {}).get("est_cost_usd") or 0
            continue
        for split, limit in (("validation", a.val_limit), ("test", 0)):
            info = run_job(entry, split, limit, a.permutations, a.batch, a.out)
            total += (info or {}).get("est_cost_usd") or 0
    print(f"estimated GPU cost this invocation: ${total:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
