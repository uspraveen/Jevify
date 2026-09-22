"""Package a Jevified model and publish it to the Hub so anyone can load it.

    python scripts/publish_model.py --run-id qwen35-4b-t1r --repo Praveenrajus/jevify-qwen3.5-4b [--push]
    python scripts/publish_model.py --run-id qwen35-4b-t2 --from runs/qwen35-4b-t2 --repo Praveenrajus/jevify-qwen3.5-4b-t2 --push

A Jevified model is a *small* artifact: the backbone stays on the Hub as-is, and this
repo carries only what Jevify adds — the trained decision heads (a few MB), a LoRA adapter
if the backbone was allowed to move (Tier 2; merged into the weights at load, so it serves
at the backbone's own speed), the recipe, the benchmark numbers it earned, and a config
naming the backbone it attaches to. Loading it pulls the backbone from its own repo, so no
weights are duplicated or relicensed. ``--from`` packages a run directory that is already
local (a run trained on one GPU rather than on Modal).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VENV = ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin")
MODAL = str(VENV / "modal")

CARD = """---
library_name: jevify
pipeline_tag: text-classification
tags: [jevify, system-one, decision-model, calibration, jev]
base_model: {backbone}
license: apache-2.0
datasets: [Praveenrajus/jev-bench]
---

# {name}

A **System One decision model**: it does not write text. It reads a `state`, answers typed
questions, and returns calibrated probability distributions your code can branch on.

| primitive | question | answer |
|---|---|---|
| `choice` | which of these K options? | probabilities over the options + `confidence` |
| `score` | where on these K ordered levels? | probabilities over levels, expected `score`, `confidence` |
| `noul` | is this true? | a single `P(yes)` |

This repo holds only what Jevify adds to `{backbone}`: **{n_params:,} parameters** of trained
decision heads ({size_mb:.1f} MB){adapter_note} plus the calibration recipe. The backbone is pulled
from its own repo at load time, so nothing is duplicated or relicensed.

## Results on [jev-bench]({bench_url})

Scored on all {n_test:,} test records, against TypeSafe's Jev 1.13.0 on the identical records.

{results_table}

{tier_note}

## Use it

```bash
pip install git+https://github.com/uspraveen/Jevify
```

```python
from jevify import load_jevified

model = load_jevified("{repo}")
answer = model.ask(
    state={{"ticket": "I was charged twice for order A-104, please refund the duplicate."}},
    questions={{
        "dept": {{"type": "choice", "instructions": "Which team should handle `ticket`?",
                 "criteria": {{"billing": "Payments and refunds", "shipping": "Delivery problems", "other": None}}}},
        "refund": {{"type": "noul", "instructions": "Does `ticket` ask for a refund?"}},
        "anger": {{"type": "score", "instructions": "How angry is the customer?",
                  "criteria": ["calm", "annoyed", "furious"]}},
    }},
)
print(answer["dept"]["choice"], answer["dept"]["confidence"])
print(answer["refund"]["noul"])
```

### As a drop-in for the TypeSafe API

```bash
jevify-serve --model {repo} --port 8000
```

```bash
TYPESAFE_BASE_URL=http://localhost:8000 python your_existing_typesafe_code.py
```

The official `typesafe-sdk` works against this unchanged — that is a test in the repo.

## How it was built

{how_built}

Training optimizes strictly proper scoring rules directly — log score for Choice and Noul,
ranked probability score for the ordinal Score — so calibration is the objective rather than a
post-hoc repair. No reinforcement learning is involved: with a differentiable head the
calibration objective is just a loss.

Six sources were **held out of training entirely** so generalization to unseen question types is
measured rather than assumed. Full method, findings and limitations:
[github.com/uspraveen/Jevify](https://github.com/uspraveen/Jevify) ·
[FINDINGS.md](https://github.com/uspraveen/Jevify/blob/main/docs/FINDINGS.md)

## Limitations

- English-first, text only, following the benchmark it was tuned on.
- Ordinal (`score`) questions on scales unlike those in training are the weakest case.
- The heads are trained on jev-bench's own train splits, so "held out" means held-out *source*,
  not a wholly different data universe.
- This run is one seed. The seed study in FINDINGS §6.3a found held-out accuracy to move by ±0.05 across
  seeds for a 2B residual head, so differences of that size between models are noise, not signal.
"""


def sh(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tier0", default=None, help="Tier 0 run of the same backbone, for the comparison row")
    ap.add_argument("--records", type=Path, default=ROOT / "data" / "jev-bench")
    ap.add_argument("--from", dest="src", type=Path, default=None,
                    help="a local run directory (heads/, lora/, run.json) instead of the Modal volume")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()

    out = ROOT / "artifacts" / a.run_id
    (out / "heads").mkdir(parents=True, exist_ok=True)
    wanted = ["heads/heads.pt", "heads/heads.json", "run.json"]
    if a.src is not None:
        import shutil

        if (a.src / "lora" / "adapter_config.json").exists():
            # peft writes a placeholder model card beside the adapter; the repo has its own
            wanted += ["lora/" + f.name for f in (a.src / "lora").iterdir() if f.is_file() and f.name != "README.md"]
        for name in wanted:
            (out / name).parent.mkdir(parents=True, exist_ok=True)
            if not (a.src / name).exists():
                print(f"missing: {a.src / name}", file=sys.stderr)
                return 1
            shutil.copy(a.src / name, out / name)
    else:
        for name in wanted:
            r = sh([MODAL, "volume", "get", "jevify-runs", f"{a.run_id}/{name}", str(out / name), "--force"])
            if r.returncode != 0 and not (out / name).exists():
                print(f"missing: {name}", file=sys.stderr)
                return 1
    meta = json.loads((out / "run.json").read_text())
    heads_info = json.loads((out / "heads" / "heads.json").read_text())
    backbone = meta["model_id"]

    import torch

    state = torch.load(out / "heads" / "heads.pt", map_location="cpu")
    n_params = sum(v.numel() for v in state.values())
    size_mb = (out / "heads" / "heads.pt").stat().st_size / 1e6
    lora_dir = out / "lora"
    has_lora = (lora_dir / "adapter_config.json").exists()
    adapter_note = ""
    if has_lora:
        lora_mb = sum(f.stat().st_size for f in lora_dir.iterdir() if f.is_file()) / 1e6
        adapter_note = (f" and a rank-{meta.get('lora_r', 16)} LoRA adapter ({lora_mb:.0f} MB, "
                        f"{meta.get('lora_trainable', 0):,} parameters) that is merged into the backbone at load")

    # the config a loader needs: which backbone, which layer, which recipe
    cfg = {"jevify_version": 1, "tier": meta.get("tier", 1), "backbone": backbone,
           "layer": meta.get("layer", -1), "chat": meta.get("chat_applied", True),
           "residual": meta.get("residual", True), "head_config": heads_info["config"],
           "heldout_sources": meta.get("heldout_sources", []),
           "trained_on": "Praveenrajus/jev-bench", "n_train": meta.get("n_train"),
           "lm_weight": meta.get("lm_weight"), "best_epoch": meta.get("best_epoch")}
    if has_lora:
        cfg["lora"] = {"r": meta.get("lora_r", 16), "lr": meta.get("lora_lr"), "epochs": meta.get("epochs"),
                       "trainable": meta.get("lora_trainable"), "merged_at_load": True}
    (out / "jevify_config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")

    # results table straight from the scored run
    summary_path = ROOT / "results" / a.run_id / "summary.json"
    rows = json.loads(summary_path.read_text(encoding="utf-8"))["rows"] if summary_path.exists() else []
    lines = ["| model | held-out sources: acc / ECE | sources seen in training: acc / ECE |", "|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} — {r['variant']} | {r.get('heldout_acc', 0):.3f} / {r.get('heldout_ece', 0):.3f} "
                     f"| {r.get('trained_acc', 0):.3f} / {r.get('trained_ece', 0):.3f} |")
    lb = ROOT / "results" / "leaderboard" / "leaderboard.json"
    tier_note = ""
    if lb.exists():
        board = json.loads(lb.read_text(encoding="utf-8"))
        me = next((e for e in board if e["run"] == a.run_id), None)
        jev = next((e for e in board if e["run"] == "jev-1.13.0"), None)
        if me and jev:
            m, j = me["summary"], jev["summary"]
            tier_note = (f"Across the whole benchmark: **macro accuracy {m['macro']['acc']:.3f}** against Jev's "
                         f"{j['macro']['acc']:.3f}, **ECE {m['macro']['ece']:.3f}** against {j['macro']['ece']:.3f}, and "
                         f"**{m['tvd_human']:.3f}** mean distance to human label distributions against Jev's "
                         f"{j['tvd_human']:.3f} — lower is better, and that last number is the one a calibration claim "
                         f"rests on.")
    how = (f"Decision heads read the backbone's hidden state at each option's own line, so they score what an option "
           f"*means* rather than how likely its identifier token is. They are applied as a **residual on the model's own "
           f"log-score** — `score_i = w·lm_i + f(...)` with `f` zero-initialized — so training starts exactly at the "
           f"untrained baseline and can only add to it. Trained on {meta.get('n_train', 0):,} records from 16 sources "
           f"with at most {meta.get('max_slots', 16)} options each, which is what keeps the head usable at any K.")
    if has_lora:
        NL = chr(10)
        how += (f"{NL}{NL}This is a **Tier 2** model: a rank-{meta.get('lora_r', 16)} LoRA on the backbone's attention and MLP "
                f"projections was trained *jointly* with the heads (LoRA learning rate {meta.get('lora_lr')}, "
                f"{meta.get('epochs')} epochs, best epoch {meta.get('best_epoch')} by validation loss), so the backbone "
                f"can learn to put the judgment at the option's position instead of only being read there. The adapter "
                f"is merged into the weights when the model loads, so inference costs exactly what the plain backbone costs.")
    card = CARD.format(name=a.repo.split("/")[-1], backbone=backbone, n_params=n_params, size_mb=size_mb,
                       adapter_note=adapter_note,
                       n_test=meta.get("n_test", 22773), bench_url="https://huggingface.co/datasets/Praveenrajus/jev-bench",
                       results_table=chr(10).join(lines), tier_note=tier_note, repo=a.repo, how_built=how)
    (out / "README.md").write_text(card, encoding="utf-8")
    print(f"packaged {out}  ({n_params:,} head params, {size_mb:.1f} MB)")

    if a.push:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(a.repo, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=a.repo, repo_type="model",
                          ignore_patterns=["run.json"],
                          commit_message="Jevified decision heads" + (" + LoRA adapter" if has_lora else ""))
        print(f"pushed https://huggingface.co/{a.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
