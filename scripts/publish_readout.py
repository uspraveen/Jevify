"""Package a readout fine-tune (LoRA adapter or full fine-tune) as a loadable Jevify model repo.

A readout fine-tune trains the model on its own decision readout -- the restricted distribution over the allowed
answers at the answer position -- with a proper scoring rule, optionally plus a coherence penalty (the sure loss of
the question's paraphrase / negation / complement family). What ships is what `load_jevified` needs: the fitted
Tier 0 recipe, the adapter (merged at load) or the full weights, and a model card whose every number is read from
the study's result table (pt.json, one row per model; results/post-training/).

Runs where the training output lives (no Hub token needed); upload the folder separately.

    python scripts/publish_readout.py --run runs/rl/4b-coh-s0-save --repo Praveenrajus/jevify-qwen3.5-4b-readout-coh \\
        --pt runs/agg/out/pt.json --label "4B LoRA, + coherence, seed 0" \\
        --compare "Qwen3.5-4B" --compare "4B LoRA, supervised, seed 0" --compare "Jev 1.13.0 (API)"
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HELDOUT = ["clinc150", "arc_challenge", "yelp5", "measuring_hate_speech", "fever_evidence", "strategyqa_grounded"]
MODEL_FILES = (".json", ".safetensors", ".jinja", ".txt", ".model", ".tiktoken")


def resolved_commit(model_id: str) -> str | None:
    """The commit of the backbone snapshot this run loaded (from the local Hugging Face cache)."""
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
    snaps = hub / f"models--{model_id.replace('/', '--')}" / "snapshots"
    shas = sorted(p.name for p in snaps.iterdir()) if snaps.exists() else []
    return shas[0] if len(shas) == 1 else None


def f3(x, sign=False):
    return "—" if x is None else (f"{x:+.3f}" if sign else f"{x:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, required=True, help="training output: train.json, results/, adapter/ or model/")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pt", type=Path, required=True, help="the study's pt.json")
    ap.add_argument("--label", required=True, help="this model's row label in pt.json")
    ap.add_argument("--compare", action="append", default=[],
                    help="other pt.json rows for the card's tables, LABEL or LABEL=display name")
    ap.add_argument("--verification", type=Path, default=None, help="verify_pub output (JSON) to report on the card")
    ap.add_argument("--branch", default="main", help="the Hub branch this folder is meant for (e.g. seed1)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    train = json.loads((a.run / "train.json").read_text(encoding="utf-8"))
    recipe = json.loads((a.run / "results" / "recipe.json").read_text(encoding="utf-8"))
    rec = recipe.get("recipe", recipe)
    rows = {r["label"]: r for r in json.loads(a.pt.read_text(encoding="utf-8"))}
    me = rows[a.label]
    full = bool(train.get("full"))
    base = train["model"]
    out = a.out or (ROOT / "artifacts" / (a.repo.split("/")[-1] + ("" if a.branch == "main" else f"@{a.branch}")))
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    if full:
        # linked, not copied: a full checkpoint is gigabytes and this folder only stages the upload
        for f in (a.run / "model").iterdir():
            if f.is_file() and f.name.endswith(MODEL_FILES):
                (out / f.name).symlink_to(f.resolve())
    else:
        (out / "lora").mkdir()
        for f in (a.run / "adapter").iterdir():
            if f.is_file() and f.name != "README.md":
                shutil.copy(f, out / "lora" / f.name)
    (out / "results").mkdir()
    for src, dst in (("results/test_metrics.json", "test_metrics.json"), ("results/recipe.json", "recipe.json"),
                     ("coh/coherence.json", "coherence.json"), ("probes/probes.json", "probes.json"), ("tags/tags.json", "tags.json"),
                     ("train.json", "train.json")):
        if (a.run / src).exists():
            shutil.copy(a.run / src, out / "results" / dst)
    (out / "results" / "summary.json").write_text(json.dumps(me, indent=1), encoding="utf-8")

    commit = None if full else resolved_commit(base)
    arm = train.get("arm")
    beta = train.get("beta") if arm == "coh" else 0.0
    cfg = {"jevify_version": 1, "tier": 2, "method": "readout-full" if full else "readout-lora",
           "backbone": "." if full else base, "backbone_revision": commit, "base_model": base, "chat": True,
           "recipe": {"mode": rec.get("mode", "index"), "permutations": rec.get("permutations", 1),
                      "prior_weight": rec.get("prior_weight", 0.0), "temperature": rec.get("temperature"),
                      "bias": rec.get("bias"), "state_last": rec.get("state_last", False), "prompt": rec.get("prompt", "jevify")},
           "training": {"objective": "proper scoring rule on the readout" + (f" + {beta} x coherence (sure loss)" if beta else ""),
                        "arm": arm, "coherence_beta": beta, "seed": train.get("seed"), "lr": train.get("lr"),
                        "epochs": train.get("epochs"), "best_epoch": train.get("best_epoch"), "best_val_loss": train.get("best_val_sup"),
                        "records_per_source_cap": train.get("cap"), "n_train_families": train.get("n_train"),
                        "full_fine_tune": full, "precision": "fp32 master weights, bf16 autocast, gradient checkpointing" if full else "bf16",
                        "trained_on": "Praveenrajus/jev-bench train splits of the 16 sources that are not held out",
                        "heldout_sources": HELDOUT},
           "evaluated_on": "Praveenrajus/jev-bench (test, 22,773 records)"}
    if not full:
        ad = json.loads((out / "lora" / "adapter_config.json").read_text(encoding="utf-8"))
        cfg["lora"] = {"r": ad.get("r"), "alpha": ad.get("lora_alpha"), "target_modules": ad.get("target_modules"),
                       "trainable": train.get("trainable"), "merged_at_load": True}
    (out / "jevify_config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")

    # ------------------------------------------------------------------ model card
    comp = []
    for c in a.compare:
        lab, _, shown = c.partition("=")
        if lab in rows:
            comp.append((shown or lab, rows[lab]))
    allrows = [("**this model**", me)] + comp
    ver = json.loads(a.verification.read_text(encoding="utf-8")) if a.verification else None
    if ver:
        (out / "results" / "verification.json").write_text(json.dumps(ver, indent=1), encoding="utf-8")
    ver_line = ("\n**Reproduction check.** Loading this folder with `load_jevified` and re-scoring "
                f"{ver['records']} jev-bench test records from six sources reproduced the training run's own test predictions: "
                f"{ver['choice_answer_flips']} changed choice answers, mean largest |Δp| {ver['mean_max_abs_dp']:.3f}, "
                f"max {ver['max_abs_dp']:.3f}{' (the adapter is merged into bf16 weights at load)' if not full else ''}.\n"
                if ver else "")

    def table(cols):
        L = ["| model | " + " | ".join(c[0] for c in cols) + " |", "|---|" + "---|" * len(cols)]
        for lab, r in allrows:
            L.append(f"| {lab} | " + " | ".join(f3(r.get(c[1])) for c in cols) + " |")
        return "\n".join(L)

    what = (f"every parameter of `{base}` fine-tuned" if full else
            f"a rank-{cfg['lora']['r']} LoRA ({(train.get('trainable') or 0):,} parameters) on `{base}`, merged into the weights at load")
    title = a.title or f"{base.split('/')[-1]}, readout fine-tuned ({'full' if full else 'LoRA'}, {'+ coherence' if beta else 'supervised'})"
    seed_note = (f"\n\n> This is the **`{a.branch}`** branch (seed {train.get('seed')}, lr {train.get('lr')}); `main` holds the "
                 f"recommended configuration." if a.branch != "main" else "")
    usage_rev = "" if a.branch == "main" else f', revision="{a.branch}"'
    load_line = (f'model = load_jevified("{a.repo}")' if a.branch == "main" else
                 f'from huggingface_hub import snapshot_download\nmodel = load_jevified(snapshot_download("{a.repo}"{usage_rev}))')
    readme = f"""---
license: apache-2.0
base_model: {base}
library_name: jevify
tags: [jevify, system-one, decision-model, calibration, coherence]
datasets: [Praveenrajus/jev-bench]
---

# {title}{seed_note}

A **System One decision model**: it does not write text. It reads a `state`, answers typed questions (`choice`, `score`,
`noul`), and returns calibrated probability distributions your code can branch on. This repo is {what}.

**How it was trained (readout fine-tuning).** The model is trained on its own *decision readout* — the distribution over
the allowed answers read at the answer position, one forward pass, no decoding — with the primitive's proper scoring
rule{f", plus a **coherence penalty** (weight {beta}): every training question comes with automatically derived siblings (the options as yes/no questions, the negation, the threshold questions of a scale), and the de Finetti sure loss of the family's answers is penalised, so the model's answers to related questions stay mutually consistent" if beta else ""}.
Options are shuffled per family. Training data: the train splits of the 16 non-held-out jev-bench sources
({train.get('n_train'):,} families, at most {train.get('cap')} records per source); lr {train.get('lr')}, {train.get('epochs')} epochs,
best epoch by validation loss (epoch {train.get('best_epoch')}), seed {train.get('seed')}{", fp32 master weights with bf16 autocast" if full else ""}.
A Tier 0 recipe (temperature per primitive, Noul bias, option-order permutations) was then fitted on validation splits.
The six held-out sources (`{"`, `".join(HELDOUT)}`) never appeared in training.

```python
from jevify import load_jevified

{load_line}
model.ask({{"text": "The battery lasted two days on a single charge."}},
          {{"q": {{"type": "noul", "instructions": "Is the review positive?"}}}})
```

`jevify-serve --model {a.repo}` serves it as a drop-in for the TypeSafe SDK (`TYPESAFE_BASE_URL=http://localhost:8000`).
{"The backbone is pulled from its own repo at load, pinned to commit `" + str(commit) + "`." if commit else ("The weights are in this repo." if full else "")}

## Results

Every number is on the jev-bench **test** splits (22,773 records) or the study's other test suites, scored the same way
for every model; rows below this model are references from the same study.

**Decisions and calibration**

{table([("acc", "acc"), ("ECE", "ece"), ("Brier", "brier"), ("held-out acc", "heldout_acc"), ("TVD to human labels", "tvd")])}

**Coherence and invariance** — sure loss: mean d² over 4,749 question families (0 = perfectly coherent); order flip: how
often the top answer changes when options are shuffled; tag TVD: how much the distribution moves when option tags change
from A–J to other identifiers.

{table([("sure loss", "sure_loss"), ("share incoherent", "frac_incoherent"), ("order flip", "order_flip"), ("tag TVD", "tag_tvd"),
         ("K=2→max acc drop", "cardinality_drop")])}

**Out of distribution** — stated rules (LegalBench, rule given in the question), none-of-the-above when the gold option is
removed, injected-instruction hijack rate, and three community Jev benchmarks.

{table([("stated rule", "rule_acc"), ("'none' when gone", "nota_none_when_gone"), ("hijack", "hijack"), ("phishing AUROC", "phish_auroc"),
         ("tool risk", "tool_acc")])}

{ver_line}
`results/` holds the raw files: `test_metrics.json` (per source), `recipe.json`, `coherence.json`, `probes.json`,
`tags.json`, `train.json` (the training log) and `summary.json` (this model's row of the study table).

## Limitations

- One training seed per repo branch; out-of-distribution numbers in particular vary between identical runs, so compare
  arms across seeds before drawing conclusions.
- The phishing benchmark's decision threshold shifts after fine-tuning (ranking, AUROC, is preserved); a one-number
  log-odds shift fitted on a handful of labelled emails repairs it.
- English only; the recipe was fitted on jev-bench validation splits and may need refitting on a very different domain.

Method, benchmark and findings: [github.com/uspraveen/Jevify](https://github.com/uspraveen/Jevify)
([docs/FINDINGS.md](https://github.com/uspraveen/Jevify/blob/main/docs/FINDINGS.md)).
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"prepared {out} ({size / 1e6:.1f} MB): {sorted(p.name for p in out.iterdir())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
