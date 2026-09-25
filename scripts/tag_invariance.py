"""Does the tag on each option steer the answer? Identifier invariance of the Tier 0 readout.

The readout scores one token per option: the option's tag (A, B, C ...) after the answer cue. If the
distribution a model gives depends on *which* tags are used, the tag itself carries a prior -- a habit
of answering "A", letter or digit frequency -- and that bias sits under every number read out this way.
Same items, same model, one option order; only the tags change:

    upper    A B C D ...   (the default)          lower    a b c d ...
    shifted  P Q R S ...                          digits1  1 2 3 4 ...
    digits0  0 1 2 3 ...                          symbols  ! @ # $ ...

A tag set is used only if every tag it needs is a single token after the answer cue for this model's
tokenizer (the check the scorer itself applies); otherwise it is reported as skipped.

Per tag set: accuracy, ECE, and against the default set the mean total variation distance (TVD) between
the two distributions for the same item, and how often the top answer flips. Raw (temperature 1) and with
the model's fitted temperature for that primitive. Choice and Score sources with at most ten options
(Noul has no tags).

    python scripts/tag_invariance.py --model <hf id> [--adapter <lora dir>] [--revision <sha>]
        --records <jev-bench root> --recipe <recipe.json> --out <dir> [--n 150]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jevify.bench.metrics import expected_calibration_error  # noqa: E402
from jevify.bench.record import read_jsonl  # noqa: E402

SOURCES = ["arc_challenge", "mmlu", "mnli", "chaosnli", "sst5", "yelp5", "helpsteer2_helpfulness",
           "helpsteer2_verbosity", "measuring_hate_speech", "stsb"]
TAGS = {
    "upper": list("ABCDEFGHIJ"),
    "shifted": list("PQRSTUVWXY"),
    "lower": list("abcdefghij"),
    "digits1": list("123456789"),
    "digits0": list("0123456789"),
    "symbols": list("!@#$%&*+=?"),
}


def single_token(sc, tags: list[str], context: str | None) -> bool:
    from jevify.engine.readout import _common_prefix_len
    from jevify.engine.template import CUE

    probe = context if context is not None else "Allowed answers: A, B" + chr(10) + CUE
    base = sc._encode_raw(probe)
    for t in tags:
        ids = sc._encode_raw(probe + t)
        if len(ids) - _common_prefix_len(base, ids) != 1 or ids[: _common_prefix_len(base, ids)] != base[: _common_prefix_len(base, ids)]:
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="a LoRA adapter directory, merged into the model")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--recipe", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--label", default=None)
    ap.add_argument("--no-chat", action="store_true", help="skip the chat template (a base checkpoint without one)")
    a = ap.parse_args()

    import torch

    from jevify.engine.predict import Recipe, Tier0Engine
    from jevify.engine.readout import HFScorer
    from jevify.engine.template import TEV1_ASSISTANT_PREFIX

    rj = json.loads(a.recipe.read_text(encoding="utf-8"))
    rec = rj.get("recipe", rj)
    prompt, chat = rec.get("prompt", "jevify"), rec.get("chat", True) and not a.no_chat
    temps = rec.get("temperature", {"choice": 1.0, "score": 1.0})
    sc = HFScorer(a.model, dtype=torch.bfloat16, revision=a.revision)
    if a.adapter:
        from peft import PeftModel
        sc.model = PeftModel.from_pretrained(sc.model, a.adapter).merge_and_unload().eval()
    eng = Tier0Engine(sc, Recipe(mode="index", chat=chat, permutations=1, prompt=prompt))
    context = TEV1_ASSISTANT_PREFIX if prompt == "tev1" else None

    recs = []
    for s in SOURCES:
        p = a.records / "data" / s / "test.jsonl"
        if p.exists():
            rs = [r for r in read_jsonl(p) if len(r.option_keys()) <= 10]
            recs.extend(rs[: a.n])
    kmax = max(len(r.option_keys()) for r in recs)

    default_ids = list(sc.identifiers(context=context) if context else sc.identifiers())
    results, per = {}, {}
    for name, tags in TAGS.items():
        need = [r for r in recs if len(r.option_keys()) <= len(tags)]
        if not single_token(sc, tags[:kmax], context):
            results[name] = {"skipped": "a tag is not a single token after the cue for this tokenizer"}
            continue
        if context:
            sc._identifiers_by_context[context] = tags + default_ids[len(tags):]
        else:
            sc._identifiers = tags + default_ids[len(tags):]
        out = {}
        for p in eng.score_records(need, batch=a.batch, want_prior=False):
            if p.error or not p.extra or not p.extra.get("runs"):
                continue
            run = p.extra["runs"][0]
            out[p.id] = (list(map(str, run["keys"])), np.array(run["logscores"], dtype=float))
        per[name] = out
        print(f"{name}: {len(out)} items", flush=True)
    if context:                                  # restore the scorer's own identifiers
        sc._identifiers_by_context[context] = default_ids
    else:
        sc._identifiers = default_ids

    by_id = {r.id: r for r in recs}

    def dist(keys, ls, T):
        z = ls / T
        z = z - z.max()
        p = np.exp(z)
        p /= p.sum()
        return dict(zip(keys, p))

    def stats(name, temp: bool):
        rows = per[name]
        conf, correct = [], []
        for i, (keys, ls) in rows.items():
            r = by_id[i]
            d = dist(keys, ls, temps.get(r.primitive, 1.0) if temp else 1.0)
            top = max(d, key=d.get)
            conf.append(d[top]); correct.append(float(top == str(r.label)))
        ece, _, _ = expected_calibration_error(np.array(conf), np.array(correct))
        o = {"n": len(rows), "acc": float(np.mean(correct)), "ece": float(ece)}
        if name != "upper" and "upper" in per:
            tv, flips = [], []
            for i, (keys, ls) in rows.items():
                if i not in per["upper"]:
                    continue
                r = by_id[i]
                T = temps.get(r.primitive, 1.0) if temp else 1.0
                d0, d1 = dist(*per["upper"][i], T), dist(keys, ls, T)
                tv.append(0.5 * sum(abs(d0[k] - d1.get(k, 0.0)) for k in d0))
                flips.append(float(max(d0, key=d0.get) != max(d1, key=d1.get)))
            o.update({"tvd_vs_upper": float(np.mean(tv)), "flip_vs_upper": float(np.mean(flips)), "pairs": len(tv)})
        return o

    for name in per:
        results[name] = {"raw": stats(name, False), "tempered": stats(name, True)}
    summary = {"model": a.label or a.model, "adapter": a.adapter, "prompt": prompt, "temperatures": temps,
               "sources": sorted({by_id[i].source for i in per.get("upper", {})}), "results": results}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "tags.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for name, v in results.items():
        if "skipped" in v:
            print(f"{name:8s} skipped: {v['skipped']}")
        else:
            r, t = v["raw"], v["tempered"]
            print(f"{name:8s} acc {r['acc']:.3f}  ece raw {r['ece']:.3f} / T {t['ece']:.3f}  "
                  + (f"TVD vs upper {r['tvd_vs_upper']:.3f} (T {t['tvd_vs_upper']:.3f})  flips {r['flip_vs_upper']:.3f}" if "tvd_vs_upper" in r else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
