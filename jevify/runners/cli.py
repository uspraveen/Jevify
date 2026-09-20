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
from .base import Prediction, read_predictions, write_predictions


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

    r = sub.add_parser("report", help="score predictions against records")
    r.add_argument("--records", type=Path, required=True)
    r.add_argument("--preds", type=Path, required=True)
    r.add_argument("--split", default="test")
    r.add_argument("--md", type=Path, default=None, help="write a markdown report here")
    r.add_argument("--json", type=Path, default=None, help="write full metrics (incl. reliability bins) here")

    args = ap.parse_args(argv)
    if args.cmd == "api":
        from .jev_api import SystemOneAPIRunner

        sources = [s for s in args.sources.split(",") if s] or None
        recs = list(iter_records(args.records, sources, args.split, args.limit))
        print(f"{len(recs)} records -> {args.model} @ {args.base_url or 'default'}", file=sys.stderr)
        runner = SystemOneAPIRunner(model=args.model, base_url=args.base_url, concurrency=args.concurrency)
        preds = list(runner.predict(recs))
        n = write_predictions(args.out, preds)
        errs = sum(1 for p in preds if p.error)
        lat = [p.latency_ms for p in preds if p.latency_ms]
        print(f"wrote {n} predictions ({errs} errors); median latency {np.median(lat):.0f} ms" if lat else f"wrote {n}", file=sys.stderr)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
