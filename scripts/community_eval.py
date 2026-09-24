"""Answer the community benchmarks (scripts/build_community.py) with a Jevified model.

    # a Tier 0 model: raw log-scores from `python -m jevify.train tier0 --bench <community root>`,
    # finished with the recipe that model was fitted with on *jev-bench* validation
    python scripts/community_eval.py apply --records <root> --split test --preds <raw> --recipe results/<run>/recipe.json --out <file>

    # a model with heads (Tier 1 / Tier 2): a run directory or a published repo, as load_jevified takes it
    python scripts/community_eval.py heads --records <root> --split test --model <dir or repo> --out <file>

Nothing is fitted on these benchmarks: they are other people's test sets, so every model answers
with the recipe or heads it already had.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jevify.engine.predict import Recipe, refinalize  # noqa: E402
from jevify.runners.base import read_predictions, write_predictions  # noqa: E402
from jevify.runners.cli import iter_records  # noqa: E402


def recipe_from(path: Path) -> Recipe:
    d = json.loads(path.read_text(encoding="utf-8"))
    d = d.get("recipe", d)
    return Recipe(mode=d.get("mode", "index"), chat=d.get("chat", True), permutations=d.get("permutations", 1),
                  state_last=bool(d.get("state_last", False)), prior_weight=d.get("prior_weight", 0.0),
                  temperature=d.get("temperature") or {}, bias=d.get("bias") or {"noul": 0.0},
                  temp_k_slope=d.get("temp_k_slope") or {}, prompt=d.get("prompt", "jevify"))


def cmd_apply(a) -> int:
    preds = list(read_predictions(a.preds))
    sources = sorted({p.id.split("/")[0] for p in preds})
    recs = list(iter_records(a.records, sources, a.split, None))
    recipe = recipe_from(a.recipe)
    raw_prompt = next((p.extra.get("prompt", "jevify") for p in preds if p.extra), "jevify")
    if raw_prompt != recipe.prompt:
        raise SystemExit(f"recipe is for the {recipe.prompt!r} prompt, the log-scores are from {raw_prompt!r}")
    out = refinalize(recs, preds, recipe)
    missing = {r.id for r in recs} - {p.id for p in out if not p.error}
    if missing:
        raise SystemExit(f"{len(missing)} records have no prediction, e.g. {sorted(missing)[:3]}")
    write_predictions(a.out, out, progress_every=0)
    print(f"{a.out}: {len(out)} predictions under {a.recipe}")
    return 0


def cmd_heads(a) -> int:
    from jevify.load import load_jevified
    from jevify.engine.heads import predict_rows
    from jevify.train import predictions_from_probs

    model = load_jevified(str(a.model))
    if model.heads is None:
        raise SystemExit(f"{a.model} has no heads; use `apply` for a Tier 0 model")
    recs = list(iter_records(a.records, None, a.split, None))
    dists: dict[str, dict[str, float]] = {}
    for start in range(0, len(recs), a.chunk):
        chunk = recs[start:start + a.chunk]
        rows = model._extractor.extract(chunk, batch_size=a.batch)
        dists.update(predict_rows(model.heads, rows, device=model.engine.scorer.device))
        print(f"  {min(start + a.chunk, len(recs))}/{len(recs)}", flush=True)
    label = a.label or str(a.model)
    preds = predictions_from_probs(dists, recs, label)
    if len(preds) != len(recs):
        raise SystemExit(f"{len(recs) - len(preds)} records have no prediction")
    write_predictions(a.out, preds, progress_every=0)
    print(f"{a.out}: {len(preds)} predictions from {model!r}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("apply")
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--preds", type=Path, required=True)
    p.add_argument("--recipe", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    h = sub.add_parser("heads")
    h.add_argument("--records", type=Path, required=True)
    h.add_argument("--split", default="test")
    h.add_argument("--model", required=True)
    h.add_argument("--out", type=Path, required=True)
    h.add_argument("--label", default=None)
    h.add_argument("--batch", type=int, default=8)
    h.add_argument("--chunk", type=int, default=512)
    a = ap.parse_args()
    return cmd_apply(a) if a.cmd == "apply" else cmd_heads(a)


if __name__ == "__main__":
    raise SystemExit(main())
