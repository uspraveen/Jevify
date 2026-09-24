"""Score every model on the community benchmarks and write results/community/.

    python scripts/community_report.py --records <community root> --models models.json --out results/community \
        [--together <tev1 repo>/evaluation/public-third-party]

``models.json`` is a list of {"label", "test", "unlabelled"?, "recipe"?}: ``test``/``unlabelled`` are
prediction files; with ``recipe`` they hold raw Tier 0 log-scores and are finished under that recipe
(``"identity"`` = the readout as the model ships it: one option order, temperature 1, no prior).

Per benchmark: accuracy with a Wilson interval, ECE and Brier (jev-bench's own definitions, so
numbers are comparable with it), and what each source's author cared about --
phishing recall / false-positive rate / AUROC, tool-risk accuracy by difficulty and misses made
with high confidence, and on the tickets whether confidence drops on the six that are ambiguous
by construction. Paired exact McNemar tests against Jev on the same items. With ``--together``,
the answers Together's hosted Tev1 gave on the same records are compared with the local readout.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jevify.engine.predict import Recipe, refinalize  # noqa: E402
from jevify.runners.base import read_predictions  # noqa: E402
from jevify.runners.cli import iter_records, score  # noqa: E402

PHISHING = ["phishnchips_verdict", "phishnchips_noul", "phishnchips_click", "phishnchips_minimal"]
POSITIVE = {"phishnchips_verdict": "phishing", "phishnchips_click": "do_not_click", "phishnchips_minimal": "phishing"}
CONFIDENT = 0.9


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (mid - half, mid + half)


def mcnemar(a_only: int, b_only: int) -> float:
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(allv):                      # average ranks over ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load(entry: dict, key: str, records_root: Path, split: str) -> list:
    path = entry.get(key)
    if not path:
        return []
    preds = list(read_predictions(Path(path)))
    if entry.get("recipe"):
        recs = list(iter_records(records_root, sorted({p.id.split("/")[0] for p in preds}), split, None))
        prompt = next((p.extra.get("prompt", "jevify") for p in preds if p.extra), "jevify")
        if entry["recipe"] == "identity":
            recipe = Recipe(permutations=1, prompt=prompt)
        else:
            from community_eval import recipe_from   # the same reader the per-model step used
            recipe = recipe_from(Path(entry["recipe"]))
        preds = refinalize(recs, preds, recipe)
    return preds


def p_max(p) -> float:
    if p.primitive == "noul":
        return max(p.p_yes, 1 - p.p_yes)
    return max(p.probabilities.values())


def correct(p, r) -> bool:
    if r.primitive == "noul":
        return (p.p_yes >= 0.5) == (int(r.label) == 1)
    return str(p.answer) == str(r.label)


def p_positive(p, r) -> float:
    """P(phishing / do-not-click) -- the score an AUROC ranks by."""
    if r.primitive == "noul":
        return p.p_yes
    return p.probabilities[POSITIVE[r.source]]


def model_rows(label: str, preds: list, unl: list, recs: dict, unl_recs: dict) -> dict:
    by_src: dict[str, list] = {}
    for p in preds:
        if p.error:
            continue
        by_src.setdefault(p.id.split("/")[0], []).append(p)
    reports = score(list(recs.values()), [p for p in preds if not p.error])
    out: dict = {"label": label, "configs": {}}
    for src, ps in sorted(by_src.items()):
        rs = [recs[p.id] for p in ps]
        ok = [correct(p, r) for p, r in zip(ps, rs)]
        k, n = sum(ok), len(ok)
        rep = reports[src]
        row = {"n": n, "correct": k, "accuracy": k / n, "wilson95": wilson(k, n), "ece": rep.ece, "brier": rep.brier,
               "correct_ids": [p.id for p, c in zip(ps, ok) if c]}
        if src in PHISHING:
            y = np.array([1 if (int(r.label) == 1 if r.primitive == "noul" else str(r.label) == POSITIVE[src]) else 0 for r in rs])
            s = np.array([p_positive(p, r) for p, r in zip(ps, rs)])
            flag = s >= 0.5
            row.update({"recall": float(flag[y == 1].mean()), "false_positive_rate": float(flag[y == 0].mean()),
                        "flag_rate": float(flag.mean()), "auroc": auroc(y, s)})
        if src == "tool_risk":
            diff = {}
            for d in ("clear", "ambiguous", "adversarial"):
                sel = [c for c, r in zip(ok, rs) if r.meta.get("difficulty") == d]
                diff[d] = {"n": len(sel), "accuracy": sum(sel) / len(sel) if sel else float("nan")}
            row["by_difficulty"] = diff
        wrong_conf = [p_max(p) for p, c in zip(ps, ok) if not c]
        row["misses_confident"] = sum(v >= CONFIDENT for v in wrong_conf)
        row["miss_pmax"] = sorted(round(v, 3) for v in wrong_conf) if n < 100 else None
        row["mean_pmax"] = float(np.mean([p_max(p) for p in ps]))
        out["configs"][src] = row
    if unl:
        out["tickets_ambiguous_mean_pmax"] = float(np.mean([p_max(p) for p in unl]))
        out["tickets_ambiguous"] = {p.id: {"answer": p.answer, "p_max": round(p_max(p), 3)} for p in unl}
    return out


def markdown(report: dict) -> str:
    """The tables results/community/README.md quotes: one per benchmark, every model a row."""
    ms = report["models"]
    out = []
    head = "| model | accuracy [95% CI] | ECE | Brier | {extra} | wrong at p ≥ 0.9 | vs Jev (only this / only Jev, McNemar p) |"
    sep = "|---|---|---|---|---|---|---|"

    def base(r):
        lo, hi = r["wilson95"]
        vs = r.get("vs_jev")
        v = f"{vs['only_this']} / {vs['only_jev']}, p = {vs['mcnemar_p']:.2g}" if vs else "—"
        return f"{r['accuracy']:.3f} [{lo:.3f}, {hi:.3f}]", f"{r['ece']:.3f}", f"{r['brier']:.3f}", v

    for src, title in (("phishnchips_verdict", "Phishing: the verdict question (the one Together scored)"),
                       ("phishnchips_noul", "Phishing: the same question as a Noul"),
                       ("phishnchips_click", "Phishing: \"should the user click?\""),
                       ("phishnchips_minimal", "Phishing: \"classify this email\" (no descriptions)")):
        out += [f"### {title}", "", head.format(extra="recall / false-positive rate / flag rate / AUROC"), sep]
        for m in ms:
            r = m["configs"].get(src)
            if r:
                acc, ece, bri, v = base(r)
                ex = f"{r['recall']:.3f} / {r['false_positive_rate']:.3f} / {r['flag_rate']:.3f} / {r['auroc']:.3f}"
                out.append(f"| {m['label']} | {acc} | {ece} | {bri} | {ex} | {r['misses_confident']} | {v} |")
        out.append("")
    out += ["### Agent tool-call risk (60 hand-labelled calls)", "",
            head.format(extra="clear / ambiguous / adversarial accuracy"), sep]
    for m in ms:
        r = m["configs"].get("tool_risk")
        if r:
            acc, ece, bri, v = base(r)
            d = r["by_difficulty"]
            ex = " / ".join(f"{d[k]['accuracy']:.2f} (n={d[k]['n']})" for k in ("clear", "ambiguous", "adversarial"))
            out.append(f"| {m['label']} | {acc} | {ece} | {bri} | {ex} | {r['misses_confident']} | {v} |")
    out += ["", "### Support-ticket routing (27 unambiguous tickets, 6 deliberately ambiguous)", "",
            head.format(extra="mean top probability: clear → ambiguous tickets"), sep]
    for m in ms:
        r = m["configs"].get("ticket_routing")
        if r:
            acc, ece, bri, v = base(r)
            amb = m.get("tickets_ambiguous_mean_pmax")
            ex = f"{r['mean_pmax']:.3f} → {amb:.3f}" if amb is not None else ""
            out.append(f"| {m['label']} | {acc} | {ece} | {bri} | {ex} | {r['misses_confident']} | {v} |")
    th = report.get("together_hosted_tev1")
    if th:
        out += ["", "### Local Tev1 against Together's hosted Tev1", ""]
        for lab, a in th["agreement_with_local_readout"].items():
            out.append(f"- {lab}: the same answer on **{a['agree']} of {a['n']}** records "
                       + "; ".join(f"{s} {x[0]}/{x[1]}" for s, x in a["by_suite"].items()))
        out.append("- hosted accuracy (Together's own run): " + "; ".join(
            f"{s} {v['correct']}/{v['n']}" for s, v in th["hosted_accuracy"].items()))
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--together", type=Path, default=None)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    entries = json.loads(a.models.read_text(encoding="utf-8"))
    recs = {r.id: r for r in iter_records(a.records, None, "test", None)}
    unl_recs = {r.id: r for r in iter_records(a.records, None, "unlabelled", None)}
    models = []
    for e in entries:
        preds = load(e, "test", a.records, "test")
        unl = load(e, "unlabelled", a.records, "unlabelled")
        missing = set(recs) - {p.id for p in preds if not p.error}
        if missing:
            raise SystemExit(f"{e['label']}: {len(missing)} records unanswered, e.g. {sorted(missing)[:3]}")
        m = model_rows(e["label"], preds, unl, recs, unl_recs)
        m["raw_answers"] = {p.id: (p.p_yes if p.primitive == "noul" else p.answer) for p in preds}
        models.append(m)
    jev = next((m for m in models if m["label"].startswith("Jev")), None)
    for m in models:
        for src, row in m["configs"].items():
            if jev is None or m is jev:
                continue
            ours, theirs = set(row["correct_ids"]), set(jev["configs"][src]["correct_ids"])
            row["vs_jev"] = {"only_this": len(ours - theirs), "only_jev": len(theirs - ours),
                             "mcnemar_p": mcnemar(len(ours - theirs), len(theirs - ours))}
    report: dict = {"models": [{k: v for k, v in m.items() if k != "raw_answers"} for m in models]}
    for m in report["models"]:
        for row in m["configs"].values():
            row.pop("correct_ids", None)
    if a.together:
        hosted = [json.loads(l) for l in (a.together / "qwen.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        hosted = {r["id"]: r for r in hosted if r.get("ok")}
        idmap = {"phishing": "phishnchips_verdict", "tool_risk": "tool_risk", "ticket_routing": "ticket_routing"}
        agree = {}
        for m in models:
            if not m["label"].startswith("Tev1"):
                continue
            same = total = 0
            per = {}
            for hid, r in hosted.items():
                suite, key = hid.split(":", 1)
                ours = m["raw_answers"].get(f"{idmap[suite]}/test/{key}")
                if ours is None:
                    continue
                total += 1
                same += ours == r["prediction"]
                per.setdefault(suite, [0, 0])
                per[suite][0] += ours == r["prediction"]
                per[suite][1] += 1
            agree[m["label"]] = {"agree": same, "n": total, "by_suite": per}
        hosted_acc = {}
        for suite in ("phishing", "tool_risk", "ticket_routing"):
            rows = [r for hid, r in hosted.items() if hid.startswith(suite + ":")]
            hosted_acc[suite] = {"n": len(rows), "correct": sum(r["correct"] for r in rows)}
        report["together_hosted_tev1"] = {"agreement_with_local_readout": agree, "hosted_accuracy": hosted_acc}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (a.out / "tables.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({m["label"]: {s: round(r["accuracy"], 3) for s, r in m["configs"].items()} for m in report["models"]}, indent=1))
    if "together_hosted_tev1" in report:
        print(json.dumps(report["together_hosted_tev1"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
