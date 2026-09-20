"""Controlled behavioral probes derived from jev-bench test records.

Each probe rewrites existing records into variants that isolate one factor,
so the comparison is within-item (same state, same gold), not across datasets.
Variants are written in the standard ``data/<config>/test.jsonl`` layout under
a probe root, so ``jevify-run api`` scores them unchanged; ``analyze`` then
groups results by probe, source and variant.

    jevify-bench probe --records data/jev-bench --out data/jev-probes --n 200
    jevify-run api --records data/jev-probes --out preds/jev-probes.jsonl
    jevify-bench probe-report --records data/jev-probes --preds preds/jev-probes.jsonl --out results/jev-1.13.0/probes
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..runners.base import Prediction, read_predictions
from .record import BenchRecord, read_jsonl, write_jsonl

SEED = 20260921
CARDINALITY_SOURCES = ["clinc150", "banking77", "ledgar", "massive", "go_emotions"]
CARDINALITY_K = [2, 5, 10, 25, 50, 100, 0]          # 0 = all options
ORDER_SOURCES = ["mmlu", "arc_challenge", "mnli", "banking77", "clinc150", "ledgar", "go_emotions", "chaosnli"]
RENAME_SOURCES = ["banking77", "clinc150", "massive"]
DISTRACTOR_SOURCES = ["mnli", "arc_challenge", "mmlu"]
DISTRACTORS = {"unrelated_a": "The state is about cooking recipes.", "unrelated_b": "The state is written in French.",
               "none_of_the_above": "None of the other options applies."}
PRIMITIVE_NOUL_SOURCES = ["boolq", "paws", "civil_comments"]
PRIMITIVE_SCORE_SOURCES = ["sst5", "yelp5", "helpsteer2_helpfulness"]


def _cfg(probe: str, source: str, variant: str) -> str:
    return f"{probe}__{source}__{variant}"


def _parse_cfg(cfg: str) -> tuple[str, str, str]:
    probe, source, variant = cfg.split("__", 2)
    return probe, source, variant


def _take(records_root: Path, source: str, n: int) -> list[BenchRecord]:
    return list(read_jsonl(records_root / "data" / source / "test.jsonl", limit=n))


def _rewrite(r: BenchRecord, cfg: str, question: dict[str, Any], label: str | int, **meta: Any) -> BenchRecord:
    return BenchRecord(id=f"{cfg}/test/{r.id.split('/')[-1]}", source=cfg, primitive=question["type"], split="test",
                      state=r.state, question=question, label=label, soft_label=None,
                      meta={**r.meta, "probe_origin": r.id, **meta})


# --------------------------------------------------------------------------- generators

def gen_cardinality(root: Path, n: int, rng: random.Random) -> Iterable[tuple[str, list[BenchRecord]]]:
    for src in CARDINALITY_SOURCES:
        recs = _take(root, src, n)
        for K in CARDINALITY_K:
            out = []
            for r in recs:
                keys = list(r.question["criteria"].keys())
                gold = str(r.label)
                if K and K < len(keys):
                    others = [k for k in keys if k != gold]
                    chosen = rng.sample(others, K - 1) + [gold]
                    rng.shuffle(chosen)
                else:
                    chosen = keys[:]
                q = dict(r.question, criteria={k: r.question["criteria"][k] for k in chosen})
                out.append(_rewrite(r, _cfg("cardinality", src, f"K{K or len(keys)}"), q, gold, K=len(chosen)))
            yield _cfg("cardinality", src, f"K{K or len(keys)}"), out


def gen_order(root: Path, n: int, rng: random.Random) -> Iterable[tuple[str, list[BenchRecord]]]:
    for src in ORDER_SOURCES:
        recs = _take(root, src, n)
        base, shuf = [], []
        for r in recs:
            keys = list(r.question["criteria"].keys())
            base.append(_rewrite(r, _cfg("order", src, "original"), r.question, r.label))
            perm = keys[:]
            rng.shuffle(perm)
            if perm == keys and len(keys) > 1:
                perm = keys[::-1]
            q = dict(r.question, criteria={k: r.question["criteria"][k] for k in perm})
            shuf.append(_rewrite(r, _cfg("order", src, "shuffled"), q, r.label))
        yield _cfg("order", src, "original"), base
        yield _cfg("order", src, "shuffled"), shuf


def gen_rename(root: Path, n: int, rng: random.Random) -> Iterable[tuple[str, list[BenchRecord]]]:
    """Opaque keys, descriptions kept (the humanized name); and opaque keys with NO description."""
    for src in RENAME_SOURCES:
        recs = _take(root, src, n)
        keep, blind = [], []
        for r in recs:
            keys = list(r.question["criteria"].keys())
            mapping = {k: f"option_{i + 1}" for i, k in enumerate(keys)}
            desc = {mapping[k]: (r.question["criteria"][k] or k.replace("_", " ")) for k in keys}
            keep.append(_rewrite(r, _cfg("rename", src, "opaque_keys"), dict(r.question, criteria=desc), mapping[str(r.label)], key_map=mapping))
            blind.append(_rewrite(r, _cfg("rename", src, "opaque_no_desc"), dict(r.question, criteria={mapping[k]: None for k in keys}), mapping[str(r.label)], key_map=mapping))
        yield _cfg("rename", src, "opaque_keys"), keep
        yield _cfg("rename", src, "opaque_no_desc"), blind


def gen_distractor(root: Path, n: int, rng: random.Random) -> Iterable[tuple[str, list[BenchRecord]]]:
    for src in DISTRACTOR_SOURCES:
        recs = _take(root, src, n)
        base = [_rewrite(r, _cfg("distractor", src, "original"), r.question, r.label) for r in recs]
        inj = [_rewrite(r, _cfg("distractor", src, "injected"), dict(r.question, criteria={**r.question["criteria"], **DISTRACTORS}), r.label) for r in recs]
        yield _cfg("distractor", src, "original"), base
        yield _cfg("distractor", src, "injected"), inj


def gen_primitive(root: Path, n: int, rng: random.Random) -> Iterable[tuple[str, list[BenchRecord]]]:
    for src in PRIMITIVE_NOUL_SOURCES:
        recs = _take(root, src, n)
        as_noul = [_rewrite(r, _cfg("primitive", src, "noul"), r.question, int(r.label)) for r in recs]
        as_choice = []
        for r in recs:
            q = {"type": "choice", "instructions": r.question.get("instructions"), "criteria": {"yes": None, "no": None}}
            as_choice.append(_rewrite(r, _cfg("primitive", src, "choice_yes_no"), q, "yes" if int(r.label) else "no"))
        yield _cfg("primitive", src, "noul"), as_noul
        yield _cfg("primitive", src, "choice_yes_no"), as_choice
    for src in PRIMITIVE_SCORE_SOURCES:
        recs = _take(root, src, n)
        as_score = [_rewrite(r, _cfg("primitive", src, "score"), r.question, int(r.label)) for r in recs]
        as_choice = []
        for r in recs:
            levels = r.question["criteria"]
            q = {"type": "choice", "instructions": r.question.get("instructions"), "criteria": {f"level_{i}": d for i, d in enumerate(levels)}}
            as_choice.append(_rewrite(r, _cfg("primitive", src, "choice_levels"), q, f"level_{int(r.label)}"))
        yield _cfg("primitive", src, "score"), as_score
        yield _cfg("primitive", src, "choice_levels"), as_choice


GENERATORS = {"cardinality": gen_cardinality, "order": gen_order, "rename": gen_rename,
              "distractor": gen_distractor, "primitive": gen_primitive}


def generate(records_root: Path, out: Path, which: list[str], n: int, seed: int = SEED) -> dict[str, int]:
    rng = random.Random(seed)
    counts = {}
    for name in which:
        for cfg, recs in GENERATORS[name](records_root, n, rng):
            counts[cfg] = write_jsonl(out / "data" / cfg / "test.jsonl", recs)
    (out / "manifest.json").write_text(json.dumps({"name": "jev-probes", "seed": seed, "n_per_source": n, "configs": counts}, indent=1))
    return counts


# --------------------------------------------------------------------------- analysis

def _probs(r: BenchRecord, p: Prediction) -> tuple[list[str], np.ndarray]:
    keys = r.option_keys()
    if r.primitive == "noul":
        return keys, np.array([1 - p.p_yes, p.p_yes])
    return keys, np.array([p.probabilities.get(k, 0.0) for k in keys])


def _ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def analyze(records_root: Path, preds_path: Path) -> dict[str, Any]:
    preds = {p.id: p for p in read_predictions(preds_path) if not p.error}
    per_cfg: dict[str, list[tuple[BenchRecord, Prediction]]] = defaultdict(list)
    for path in sorted((records_root / "data").glob("*/test.jsonl")):
        for r in read_jsonl(path):
            if r.id in preds:
                per_cfg[path.parent.name].append((r, preds[r.id]))
    summary: dict[str, Any] = defaultdict(dict)
    for cfg, rows in per_cfg.items():
        probe, src, variant = _parse_cfg(cfg)
        conf, correct, p_gold = [], [], []
        for r, p in rows:
            keys, pr = _probs(r, p)
            gi = keys.index(str(r.label))
            conf.append(float(pr.max())); correct.append(float(pr.argmax() == gi)); p_gold.append(float(pr[gi]))
        conf, correct = np.array(conf), np.array(correct)
        summary[probe][f"{src}|{variant}"] = {"n": len(rows), "accuracy": float(correct.mean()), "ece": _ece(conf, correct),
                                              "mean_p_gold": float(np.mean(p_gold)), "mean_conf": float(conf.mean())}
    # paired comparisons for order / distractor / primitive
    paired: dict[str, Any] = {}
    for probe, a, b in (("order", "original", "shuffled"), ("distractor", "original", "injected"),
                        ("primitive", "noul", "choice_yes_no"), ("primitive", "score", "choice_levels")):
        for src in sorted({_parse_cfg(c)[1] for c in per_cfg if c.startswith(probe + "__")}):
            ca, cb = _cfg(probe, src, a), _cfg(probe, src, b)
            if ca not in per_cfg or cb not in per_cfg:
                continue
            A = {r.meta["probe_origin"]: (r, p) for r, p in per_cfg[ca]}
            B = {r.meta["probe_origin"]: (r, p) for r, p in per_cfg[cb]}
            flips, tvds, leaked, pa, pb = [], [], [], [], []
            for oid in A.keys() & B.keys():
                ra, qa = A[oid]; rb, qb = B[oid]
                ka, va = _probs(ra, qa); kb, vb = _probs(rb, qb)
                if probe == "distractor":
                    leaked.append(float(sum(vb[kb.index(d)] for d in DISTRACTORS if d in kb)))
                    vb_common = np.array([vb[kb.index(k)] for k in ka]); vb_common = vb_common / max(vb_common.sum(), 1e-9)
                    tvds.append(0.5 * float(np.abs(va - vb_common).sum()))
                    flips.append(float(ka[int(va.argmax())] != ka[int(vb_common.argmax())]))
                elif probe == "primitive":
                    # align by gold-index position: noul keys ['1','0'] vs choice ['yes','no']; score digits vs level_i
                    map_b = {k: i for i, k in enumerate(kb)}
                    if b == "choice_yes_no":
                        vb_al = np.array([vb[map_b["no"]], vb[map_b["yes"]]])   # ka order is ['1','0'] → reorder to ['0','1']
                        va_al = np.array([va[ka.index("0")], va[ka.index("1")]])
                    else:
                        vb_al = np.array([vb[map_b[f"level_{i}"]] for i in range(len(ka))]); va_al = va
                    tvds.append(0.5 * float(np.abs(va_al - vb_al).sum()))
                    flips.append(float(va_al.argmax() != vb_al.argmax()))
                    pa.append(float(va_al[1] if b == "choice_yes_no" else va_al.max())); pb.append(float(vb_al[1] if b == "choice_yes_no" else vb_al.max()))
                else:
                    vb_al = np.array([vb[kb.index(k)] for k in ka])
                    tvds.append(0.5 * float(np.abs(va - vb_al).sum()))
                    flips.append(float(va.argmax() != vb_al.argmax()))
            entry = {"n": len(tvds), "argmax_flip_rate": float(np.mean(flips)), "mean_tvd": float(np.mean(tvds))}
            if leaked:
                entry["mean_prob_on_distractors"] = float(np.mean(leaked))
            paired[f"{probe}|{src}|{a}->{b}"] = entry
    return {"per_config": dict(summary), "paired": paired}


def report_md(an: dict[str, Any]) -> str:
    NL = chr(10)
    out = ["# Behavioral probes", ""]
    card = an["per_config"].get("cardinality", {})
    if card:
        out += ["## Cardinality: same items, gold kept, K options", "",
                "| source | K | accuracy | ECE | mean P(gold) | mean confidence |", "|---|---|---|---|---|---|"]
        for key in sorted(card, key=lambda k: (k.split("|")[0], int(k.split("|K")[1]))):
            src, var = key.split("|"); e = card[key]
            out.append(f"| `{src}` | {var[1:]} | {e['accuracy']:.3f} | {e['ece']:.3f} | {e['mean_p_gold']:.3f} | {e['mean_conf']:.3f} |")
        out.append("")
    ren = an["per_config"].get("rename", {})
    if ren:
        out += ["## Label renaming: opaque option keys", "", "| source | variant | accuracy | ECE | mean P(gold) |", "|---|---|---|---|---|"]
        for key in sorted(ren):
            src, var = key.split("|"); e = ren[key]
            out.append(f"| `{src}` | {var} | {e['accuracy']:.3f} | {e['ece']:.3f} | {e['mean_p_gold']:.3f} |")
        out.append("")
    for probe, title in (("order", "Order sensitivity: original vs shuffled option order"),
                         ("distractor", "Distractor injection: three irrelevant options added"),
                         ("primitive", "Primitive ablation: the same question through a different primitive")):
        rows = {k: v for k, v in an["paired"].items() if k.startswith(probe + "|")}
        if not rows:
            continue
        out += [f"## {title}", "", "| source | comparison | n | argmax flip rate | mean TVD between answers | P on distractors |", "|---|---|---|---|---|---|"]
        for key in sorted(rows):
            _, src, cmp_ = key.split("|"); e = rows[key]
            out.append(f"| `{src}` | {cmp_} | {e['n']} | {e['argmax_flip_rate']:.3f} | {e['mean_tvd']:.3f} | {e.get('mean_prob_on_distractors', '')} |")
        acc = an["per_config"].get(probe, {})
        if acc:
            out += ["", "| config | accuracy | ECE |", "|---|---|---|"]
            for key in sorted(acc):
                src, var = key.split("|"); e = acc[key]
                out.append(f"| `{src}` / {var} | {e['accuracy']:.3f} | {e['ece']:.3f} |")
        out.append("")
    return NL.join(out)
