"""Publish a Tier 0 Jevified model: a backbone pointer plus its fitted recipe, no weights.

A Tier 0 model *is* its recipe -- which readout mode, how many option orders, the
temperature per primitive, the Noul bias -- fitted on validation splits only. Publishing it
as a repo makes it loadable by name (``load_jevified("Praveenrajus/jevify-qwen3-vl-2b")``),
servable (``jevify-serve --model ...``), and citable with its results table, exactly like a
Tier 1 or Tier 2 repo. This is also how a vision-language model ships: the repo says
``modality: vision`` and the loader picks the processor path.

    python scripts/publish_recipe.py --results results/qwen3vl-2b --repo Praveenrajus/jevify-qwen3-vl-2b \\
        --title "Qwen3-VL-2B, Jevified (Tier 0)" --push
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, required=True, help="results/<run> with recipe.json, run.json, test_metrics.json")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()

    recipe = json.loads((a.results / "recipe.json").read_text(encoding="utf-8"))
    rec = recipe.get("recipe", recipe)
    meta = json.loads((a.results / "run.json").read_text(encoding="utf-8"))
    metrics = json.loads((a.results / "test_metrics.json").read_text(encoding="utf-8"))
    backbone = meta["model_id"]
    modality = meta.get("modality", "text")

    out = ROOT / "artifacts" / a.repo.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"jevify_version": 1, "tier": 0, "backbone": backbone, "modality": modality,
           "chat": meta.get("chat_applied", True), "max_pixels": meta.get("max_pixels"),
           "recipe": {"mode": rec.get("mode", "index"), "permutations": rec.get("permutations", 1),
                      "prior_weight": rec.get("prior_weight", 0.0), "temperature": rec.get("temperature"),
                      "bias": rec.get("bias"), "state_last": rec.get("state_last", False)},
           "trust_remote_code": meta.get("trust_remote_code", False),
           "evaluated_on": "Praveenrajus/jev-bench", "sources": sorted(metrics)}
    if modality == "vision":
        cfg["vision_stage"] = meta.get("vision_stage")
    (out / "jevify_config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")

    accs = [v["accuracy"] for v in metrics.values()]; eces = [v["ece"] for v in metrics.values()]
    rows = ["| source | primitive | n | acc | ECE | Brier |", "|---|---|---|---|---|---|"]
    prim = {"pope": "noul", "aokvqa": "choice", "ai2d": "choice"}
    for src, v in sorted(metrics.items()):
        rows.append(f"| `{src}` | {prim.get(src, '')} | {v['n']} | {v['accuracy']:.3f} | {v['ece']:.3f} | {v['brier']:.3f} |")
    title = a.title or f"{backbone.split('/')[-1]}, Jevified (Tier 0)"
    what = ("a vision-language model: `state` may carry images (PIL, path, URL, data URI or bytes) and the same three "
            "typed questions are asked about them" if modality == "vision" else "a language model")
    example_state = ('{"image": "https://example.com/photo.jpg", "question": "Is there a cat?"}' if modality == "vision"
                     else '{"text": "The battery lasted two days on a single charge."}')
    readme = f'''---
license: apache-2.0
base_model: {backbone}
tags: [jevify, system-one, calibrated-decisions{", vision-language" if modality == "vision" else ""}]
---

# {title}

A [Jevify](https://github.com/uspraveen/Jevify) **Tier 0** model: {backbone} used as {what}, with **no training** —
one forward pass, the probability of every allowed answer read from a single position, and a calibration recipe
fitted on validation splits only. This repo carries the recipe and a pointer to the backbone; nothing else is needed.

```python
from jevify import load_jevified

model = load_jevified("{a.repo}")
model.ask({example_state},
          {{"q": {{"type": "noul", "instructions": "Is the answer to `question` yes, based on `image`?"}}}})
```

Serve it as a drop-in for the TypeSafe SDK: `jevify-serve --model {a.repo}` then `TYPESAFE_BASE_URL=http://localhost:8000`.

## Results on jev-bench test splits

Macro accuracy **{sum(accs)/len(accs):.3f}**, macro ECE **{sum(eces)/len(eces):.3f}** over {len(metrics)} sources.

{chr(10).join(rows)}

Recipe: mode `{cfg["recipe"]["mode"]}`, permutations `{cfg["recipe"]["permutations"]}`, temperature
`{cfg["recipe"]["temperature"]}`, Noul bias `{cfg["recipe"]["bias"]}`{f", pixel budget {cfg['max_pixels']}" if cfg.get("max_pixels") else ""}.
Predictions, metrics and figures: `results/{a.results.name}/` on
[Praveenrajus/jev-bench](https://huggingface.co/datasets/Praveenrajus/jev-bench). Findings and method:
[docs/FINDINGS.md](https://github.com/uspraveen/Jevify/blob/main/docs/FINDINGS.md).
'''
    (out / "README.md").write_text(readme, encoding="utf-8")
    print(f"prepared {out}: {list(p.name for p in out.iterdir())}")
    if a.push:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(a.repo, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=a.repo, repo_type="model",
                          commit_message=f"{title}: Tier 0 recipe")
        print(f"pushed https://huggingface.co/{a.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
