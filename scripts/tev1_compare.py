"""Tev1 against Jev and the Jevified models on jev-bench, by subset.

    python scripts/tev1_compare.py --records <jev-bench root> --out results/tev1-comparison \
        "Jev 1.13.0=results/jev-1.13.0" "Tev1 (as shipped)=raw:runs/tev1-4b/test_predictions.jsonl" ...

An entry is ``label=results dir`` (its test_metrics.json) or ``label=raw:<predictions>`` (raw Tier 0
log-scores, scored as the model ships: one option order, temperature 1, no prior, no bias).
Subsets: all configs; the six held-out sources (no Jevified model trained on them); the four
jev-bench sources whose *train* splits are in Tev1's data (MNLI, BoolQ, Banking77, SST-5 -- item
overlap is zero by construction on both sides, but the task is not new to Tev1); configs with at
most 24 options (the range Tev1 was trained on) and those with more.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jevify.bench.record import read_jsonl  # noqa: E402

HELDOUT = ["clinc150", "arc_challenge", "yelp5", "measuring_hate_speech", "fever_evidence", "strategyqa_grounded"]
TEV1_TRAINED = ["mnli", "boolq", "banking77", "sst5"]


def metrics_for(spec: str, records: Path) -> dict[str, dict]:
    if spec.startswith("raw:"):
        from jevify.engine.predict import Recipe, refinalize
        from jevify.runners.base import read_predictions
        from jevify.runners.cli import iter_records, score

        preds = list(read_predictions(Path(spec[4:])))
        prompt = next((p.extra.get("prompt", "jevify") for p in preds if p.extra), "jevify")
        recs = list(iter_records(records, sorted({p.id.split("/")[0] for p in preds}), "test", None))
        final = refinalize(recs, preds, Recipe(permutations=1, prompt=prompt))
        return {k: v.as_dict() for k, v in score(recs, final).items()}
    return json.loads((Path(spec) / "test_metrics.json").read_text(encoding="utf-8"))


def option_counts(records: Path) -> dict[str, int]:
    out = {}
    for path in sorted((records / "data").glob("*/test.jsonl")):
        r = next(read_jsonl(path, limit=1))
        out[path.parent.name] = len(r.option_keys())
    return out


def agg(m: dict[str, dict], sources: list[str]) -> dict:
    rs = [m[s] for s in sources if s in m]
    if not rs:
        return {}
    tv = [r["tvd_to_human"] for r in rs if r.get("tvd_to_human") is not None]
    return {"n": len(rs), "acc": float(np.mean([r["accuracy"] for r in rs])), "ece": float(np.mean([r["ece"] for r in rs])),
            "brier": float(np.mean([r["brier"] for r in rs])), "tvd": float(np.mean(tv)) if tv else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("entries", nargs="+")
    a = ap.parse_args()
    K = option_counts(a.records)
    sources = sorted(K)
    subsets = {
        "all 22 configs": sources,
        "6 held-out sources": HELDOUT,
        "16 Jevify-trained sources": [s for s in sources if s not in HELDOUT],
        "4 Tev1-trained tasks": TEV1_TRAINED,
        "other 18 (new to Tev1)": [s for s in sources if s not in TEV1_TRAINED],
        "K <= 24 options": [s for s in sources if K[s] <= 24],
        "K > 24 options": [s for s in sources if K[s] > 24],
    }
    models = {}
    for e in a.entries:
        label, spec = e.split("=", 1)
        models[label] = metrics_for(spec, a.records)
    table = {lab: {name: agg(m, subs) for name, subs in subsets.items()} for lab, m in models.items()}
    a.out.mkdir(parents=True, exist_ok=True)
    lines = ["| model | " + " | ".join(subsets) + " |", "|---|" + "---|" * len(subsets)]
    for lab, row in table.items():
        cells = [f"{row[n]['acc']:.3f} / {row[n]['ece']:.3f}" if row[n] else "" for n in subsets]
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    lines += ["", "*accuracy / ECE, macro over the configs in each column*", "",
              "| model | TVD to human distributions (all configs that have them) | Brier, all configs |", "|---|---|---|"]
    for lab, row in table.items():
        t = row["all 22 configs"]
        lines.append(f"| {lab} | {t['tvd']:.3f} | {t['brier']:.3f} |" if t.get("tvd") is not None else f"| {lab} | | {t['brier']:.3f} |")
    per = ["", "| config | K | " + " | ".join(models) + " |", "|---|---|" + "---|" * len(models)]
    for s in sources:
        per.append(f"| `{s}` | {K[s]} | " + " | ".join(f"{m[s]['accuracy']:.3f} / {m[s]['ece']:.3f}" if s in m else "" for m in models.values()) + " |")
    (a.out / "comparison.md").write_text("\n".join(lines + ["", "Per config, accuracy / ECE:"] + per) + "\n", encoding="utf-8")
    (a.out / "comparison.json").write_text(json.dumps({"subsets": subsets, "option_counts": K, "table": table,
                                                        "per_config": {lab: {s: {k: m[s][k] for k in ("accuracy", "ece", "brier")}
                                                                             for s in m} for lab, m in models.items()}}, indent=1),
                                           encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
