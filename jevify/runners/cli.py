"""CLI: run a runner over bench records, and score predictions against them.

    jevify-run api    --records data/jev-bench --out preds/jev.jsonl [--sources a,b] [--split test] [--limit 50] [--model jev-latest]
    jevify-run report --records data/jev-bench --preds preds/jev.jsonl [--md report.md]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np

from ..bench.metrics import Report, report_categorical, report_noul
from ..bench.record import BenchRecord, read_jsonl
from .base import Prediction, done_ids, read_predictions, write_predictions


def iter_records(root: Path, sources: list[str] | None, split: str, limit: int | None) -> Iterator[BenchRecord]:
    data = root / "data"
    for src_dir in sorted(p for p in data.iterdir() if p.is_dir()):
        if sources and src_dir.name not in sources:
            continue
        path = src_dir / f"{split}.jsonl"
        if path.exists():
            yield from read_jsonl(path, limit=limit)


def score(records: list[BenchRecord], preds: list[Prediction]) -> dict[str, Report]:
    by_id = {p.id: p for p in preds}
    groups: dict[str, list[tuple[BenchRecord, Prediction]]] = defaultdict(list)
    for r in records:
        p = by_id.get(r.id)
        if p is None or p.error:
            continue
        groups[r.source].append((r, p))
    out: dict[str, Report] = {}
    for src, pairs in groups.items():
        recs, ps = zip(*pairs)
        prim = recs[0].primitive
        if prim == "noul":
            p_yes = np.array([p.p_yes for p in ps])
            labels = np.array([int(r.label) for r in recs])
            soft = np.array([r.soft_label for r in recs]) if all(isinstance(r.soft_label, (int, float)) for r in recs) else None
            out[src] = report_noul(p_yes, labels, soft=soft)
        else:
            keys_per = [r.option_keys() for r in recs]
            probs = np.array([[p.probabilities.get(k, 0.0) for k in keys] for p, keys in zip(ps, keys_per)], dtype=object)
            # option counts can vary within a source (ARC); pad to the widest
            width = max(len(k) for k in keys_per)
            mat = np.zeros((len(recs), width))
            for i, (row, keys) in enumerate(zip(probs, keys_per)):
                mat[i, : len(keys)] = np.asarray(list(row), dtype=float)
            labels = np.array([r.label_index() for r in recs])
            conf = np.array([p.confidence if p.confidence is not None else max(mat[i]) for i, p in enumerate(ps)])
            soft = None
            if all(r.soft_label is not None for r in recs):
                soft = np.zeros_like(mat)
                for i, (r, keys) in enumerate(zip(recs, keys_per)):
                    sl = r.soft_label
                    vals = [sl[k] for k in keys] if isinstance(sl, dict) else list(sl)
                    soft[i, : len(vals)] = vals
            out[src] = report_categorical(mat, labels, ordinal=(prim == "score"), confidence=conf, soft=soft)
    return out


def markdown_table(reports: dict[str, Report], primitives: dict[str, str], model: str) -> str:
    cols = ["source", "prim", "n", "acc", "ECE", "Brier", "NLL", "sel@90", "sel@50", "AURC", "RPS", "MAE", "AUROC", "TVD→human"]
    lines = [f"**model:** `{model}`", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]

    def f(x: float | None, d: int = 3) -> str:
        return "" if x is None else f"{x:.{d}f}"

    for src, r in sorted(reports.items()):
        lines.append("| " + " | ".join([
            f"`{src}`", primitives.get(src, ""), str(r.n), f(r.accuracy), f(r.ece), f(r.brier), f(r.nll, 2),
            f(r.selective_acc_at_90), f(r.selective_acc_at_50), f(r.aurc), f(r.rps), f(r.mae, 2), f(r.auroc), f(r.tvd_to_human),
        ]) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="jevify-run")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("api", help="run records through a System One HTTP API")
    a.add_argument("--records", type=Path, required=True, help="jev-bench root (contains data/<source>/<split>.jsonl)")
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--sources", default="")
    a.add_argument("--split", default="test")
    a.add_argument("--limit", type=int, default=None, help="records per source")
    a.add_argument("--model", default="jev-latest")
    a.add_argument("--base-url", default=None)
    a.add_argument("--concurrency", type=int, default=8)
    a.add_argument("--resume", action="store_true", help="skip ids already in --out; append new predictions")

    r = sub.add_parser("report", help="score predictions against records")
    r.add_argument("--records", type=Path, required=True)
    r.add_argument("--preds", type=Path, required=True)
    r.add_argument("--split", default="test")
    r.add_argument("--md", type=Path, default=None, help="write a markdown report here")
    r.add_argument("--json", type=Path, default=None, help="write full metrics (incl. reliability bins) here")
    r.add_argument("--figures", type=Path, default=None, help="write reliability/calibration/coverage/human figures here")
    r.add_argument("--label", default=None, help="model label for figure titles (default: model from predictions)")

    rc = sub.add_parser("recipe", help="fit a recipe on validation log-scores, apply to test, write results")
    rc.add_argument("--records", type=Path, required=True)
    rc.add_argument("--val", type=Path, required=True, help="validation predictions with raw log-scores")
    rc.add_argument("--test", type=Path, required=True, help="test predictions with raw log-scores")
    rc.add_argument("--out", type=Path, required=True, help="results dir (test_predictions.jsonl, recipe.json, reports, figures)")
    rc.add_argument("--label", default=None)

    cp = sub.add_parser("compare", help="side-by-side table and figures for several prediction files")
    cp.add_argument("--records", type=Path, required=True)
    cp.add_argument("--preds", type=Path, nargs="+", required=True)
    cp.add_argument("--labels", nargs="+", default=None)
    cp.add_argument("--out", type=Path, required=True)
    cp.add_argument("--split", default="test")

    args = ap.parse_args(argv)
    if args.cmd == "recipe":
        return _cmd_recipe(args)
    if args.cmd == "compare":
        return _cmd_compare(args)
    if args.cmd == "api":
        from .jev_api import SystemOneAPIRunner

        sources = [s for s in args.sources.split(",") if s] or None
        recs = list(iter_records(args.records, sources, args.split, args.limit))
        skip = done_ids(args.out) if args.resume else set()
        if skip:
            recs = [r for r in recs if r.id not in skip]
            print(f"resuming: {len(skip)} done, {len(recs)} to go", file=sys.stderr)
        print(f"{len(recs)} records -> {args.model} @ {args.base_url or 'default'}", file=sys.stderr)
        runner = SystemOneAPIRunner(model=args.model, base_url=args.base_url, concurrency=args.concurrency)
        write_predictions(args.out, runner.predict(recs), append=bool(skip))
        preds = list(read_predictions(args.out))
        errs = sum(1 for p in preds if p.error)
        lat = [p.latency_ms for p in preds if p.latency_ms]
        med = f"; median latency {np.median(lat):.0f} ms" if lat else ""
        print(f"{args.out}: {len(preds)} predictions ({errs} errors){med}", file=sys.stderr)
        return 0

    preds = list(read_predictions(args.preds))
    sources = sorted({p.id.split("/")[0] for p in preds})
    recs = list(iter_records(args.records, sources, args.split, None))
    reports = score(recs, preds)
    prims = {r.source: r.primitive for r in recs}
    model = next((p.model for p in preds if p.model), "?")
    md = markdown_table(reports, prims, model)
    print(md)
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(md + "\n", encoding="utf-8")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({k: v.as_dict() for k, v in reports.items()}, indent=1), encoding="utf-8")
    if args.figures:
        from ..bench.figures import make_all

        for f in make_all(args.records, args.preds, args.figures, args.label or model, args.split):
            print("figure:", f, file=sys.stderr)
    return 0


def _cmd_recipe(args) -> int:
    from ..engine.predict import Recipe, refinalize
    from .recipe import ablation, fit_recipe

    val_preds = list(read_predictions(args.val))
    test_preds = list(read_predictions(args.test))
    sources = sorted({p.id.split("/")[0] for p in val_preds})
    val_recs = list(iter_records(args.records, sources, "validation", None))
    test_recs = list(iter_records(args.records, sorted({p.id.split("/")[0] for p in test_preds}), "test", None))
    mode = next((p.extra.get("mode") for p in val_preds if p.extra), "index")
    base = Recipe(mode="label" if mode == "label" else "index")
    recipe, log = fit_recipe(val_recs, val_preds, base)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "recipe.json").write_text(json.dumps({"recipe": recipe.as_dict(), "search": log}, indent=1), encoding="utf-8")
    abl_val = ablation(val_recs, val_preds, recipe)
    abl_test = ablation(test_recs, test_preds, recipe)
    cols = ["step", "macro_acc", "macro_ece", "macro_brier", "choice_acc", "choice_ece", "score_acc", "score_ece", "noul_acc", "noul_ece"]
    def table(rows):
        head = '| ' + ' | '.join(cols) + ' |'
        sep = '|' + '---|' * len(cols)
        body = ["| " + " | ".join(str(r.get(c, '')) for c in cols) + " |" for r in rows]
        return chr(10).join([head, sep, *body])
    (args.out / 'ablation.md').write_text(chr(10).join(['## validation', '', table(abl_val), '', '## test', '', table(abl_test), '']), encoding='utf-8')
    final = refinalize(test_recs, test_preds, recipe)
    out_preds = args.out / "test_predictions.jsonl"
    write_predictions(out_preds, final, progress_every=0)
    reports = score(test_recs, final)
    prims = {r.source: r.primitive for r in test_recs}
    model = args.label or next((p.model for p in final if p.model), "?")
    md = markdown_table(reports, prims, model)
    (args.out / "test_report.md").write_text(md + chr(10), encoding="utf-8")
    (args.out / "test_metrics.json").write_text(json.dumps({k: v.as_dict() for k, v in reports.items()}, indent=1), encoding="utf-8")
    from ..bench.figures import make_all
    make_all(args.records, out_preds, args.out / "figures", model, "test")
    print(json.dumps(log["chosen"], indent=1))
    print(table(abl_test))
    return 0


def _cmd_compare(args) -> int:
    from ..bench.figures import fig_compare, load_eval
    from .recipe import compare_table

    labels = args.labels or [p.parent.name for p in args.preds]
    named, evs = {}, {}
    prims: dict[str, str] = {}
    for label, path in zip(labels, args.preds):
        preds = list(read_predictions(path))
        recs = list(iter_records(args.records, sorted({p.id.split("/")[0] for p in preds}), args.split, None))
        prims.update({r.source: r.primitive for r in recs})
        named[label] = score(recs, preds)
        evs[label] = load_eval(args.records, path, args.split)
    args.out.mkdir(parents=True, exist_ok=True)
    md = compare_table(named, prims)
    (args.out / "compare.md").write_text(md + chr(10), encoding="utf-8")
    for metric in ("accuracy", "ece"):
        fig_compare(evs, args.out, metric)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
