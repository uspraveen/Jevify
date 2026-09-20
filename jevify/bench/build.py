"""Build jev-bench: sample each source, write JSONL per (source, split), emit a
manifest and dataset card, optionally push to the Hugging Face Hub.

    jevify-bench build --out data/jev-bench [--sources banking77,boolq] [--push user/jev-bench]
    jevify-bench list

Designed to run on a small box: sources are processed one at a time and the
HF cache is evicted between them when ``--evict-cache`` is set.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from ..wire import parse_question
from .adapters import REGISTRY, specs
from .adapters._base import DEFAULT_SEED, Adapter, stratified_order, uniform_order
from .record import BenchRecord, Split, write_jsonl

SPLITS: tuple[Split, ...] = ("test", "validation", "train")
STRATIFY_MAX_CLASSES = 12   # with --stratify, balance only small label spaces; large-K stays uniform


def sample_records(adapter: Adapter, split: Split, cap: int, seed: int, *, stratify: bool = False) -> list[BenchRecord]:
    """Uniform random sample by default so each split keeps the source's natural
    label distribution: a calibration benchmark must not shift base rates.
    ``stratify=True`` balances small label spaces instead (useful for training
    mixes, wrong for evaluation)."""
    if cap <= 0:
        return []
    rows = adapter.load(split)
    n = len(rows)
    if n == 0:
        return []
    order = _order(adapter, rows, n, seed) if stratify else uniform_order(n, seed)
    out: list[BenchRecord] = []
    for i in order:
        rec = adapter.convert(rows[i], split, i)
        if rec is None:
            continue
        parse_question(rec.question)   # every emitted question must be a valid wire question
        _check_label(rec)
        out.append(rec)
        if len(out) >= cap:
            break
    return out


def _order(adapter: Adapter, rows: Any, n: int, seed: int) -> list[int]:
    col = adapter.label_column
    if col is not None:
        try:
            labels = rows[col] if hasattr(rows, "column_names") else [r[col] for r in rows]
        except (KeyError, TypeError):
            labels = None
        if labels is not None and len(set(map(str, labels))) <= STRATIFY_MAX_CLASSES:
            return stratified_order(list(labels), seed)
    return uniform_order(n, seed)


def _check_label(rec: BenchRecord) -> None:
    keys = rec.option_keys()
    if str(rec.label) not in keys:
        raise ValueError(f"{rec.id}: label {rec.label!r} not in answer space {keys[:5]}...")
    if rec.soft_label is not None and rec.primitive == "choice":
        assert set(rec.soft_label) <= set(keys), rec.id


def build(out: Path, sources: list[str], *, seed: int, evict_cache: bool, caps_override: dict[str, int],
          stratify: bool = False) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"sources": {}}
    manifest.update({"name": "jev-bench", "version": "0.1.1", "seed": seed, "sampling": "stratified" if stratify else "natural",
                     "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                     "git_commit": _git_commit()})
    for name in sources:
        adapter_cls = REGISTRY[name]
        spec = adapter_cls.spec
        t0 = time.time()
        print(f"[{name}] {spec.hf_id or spec.notes[:40]} ...", flush=True)
        try:
            adapter = adapter_cls()
            entry: dict[str, Any] = {"primitive": spec.primitive, "hf_id": spec.hf_id, "hf_config": spec.hf_config,
                                     "license": spec.license, "domain": spec.domain, "task_family": spec.task_family,
                                     "k": spec.k, "has_soft_labels": spec.has_soft_labels, "description": spec.description,
                                     "notes": spec.notes, "counts": {}, "files": {}}
            for split in SPLITS:
                cap = caps_override.get(split, spec.caps.get(split, 0))
                recs = sample_records(adapter, split, cap, seed, stratify=stratify)
                if not recs:
                    continue
                path = out / "data" / name / f"{split}.jsonl"
                write_jsonl(path, recs)
                entry["counts"][split] = len(recs)
                entry["files"][split] = str(path.relative_to(out)).replace(os.sep, "/")
                if spec.k is None:
                    entry["k"] = "variable"
                print(f"   {split:10s} {len(recs):6d}", flush=True)
            manifest["sources"][name] = entry
            print(f"   done in {time.time() - t0:.0f}s", flush=True)
        except Exception:
            print(f"   FAILED:\n{traceback.format_exc()}", flush=True)
            manifest["sources"][name] = {"error": traceback.format_exc().splitlines()[-1]}
        manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        if evict_cache:
            _evict_hf_cache()
    (out / "README.md").write_text(dataset_card(manifest, baselines=collect_baselines(out)), encoding="utf-8")
    return manifest


def collect_baselines(out: Path) -> list[tuple[str, str]]:
    """(model, markdown table) for every results/<model>/test_report.md under the build dir."""
    found = []
    for rep in sorted((out / "results").glob("*/test_report.md")):
        found.append((rep.parent.name, rep.read_text(encoding="utf-8").strip()))
    return found


def dataset_card(manifest: dict[str, Any], baselines: list[tuple[str, str]] | None = None) -> str:
    srcs = {k: v for k, v in manifest["sources"].items() if "error" not in v}
    configs = [{"config_name": n, "data_files": [{"split": sp, "path": pth} for sp, pth in e["files"].items()]} for n, e in srcs.items()]
    header = {
        "pretty_name": "jev-bench",
        "license": "other", "license_name": "mixed-see-manifest",
        "license_link": "https://github.com/uspraveen/Jevify/blob/main/docs/DATASETS.md",
        "language": ["en"], "task_categories": ["text-classification"], "size_categories": ["100K<n<1M"],
        "tags": ["calibration", "system-one", "decision-model", "jevify", "benchmark", "human-label-distributions"],
        "configs": configs,
    }
    import yaml  # PyYAML ships with huggingface_hub

    total = sum(sum(e["counts"].values()) for e in srcs.values())
    n_test = sum(e["counts"].get("test", 0) for e in srcs.values())
    gold = [n for n, e in srcs.items() if e["has_soft_labels"]]
    version = manifest.get("version", "0.1")
    hero = baselines[0][0] if baselines else None
    NL = chr(10)

    def source_rows(prim: str) -> str:
        rows = []
        for n, e in srcs.items():
            if e["primitive"] != prim:
                continue
            counts = " / ".join(f"{e['counts'].get(sp, 0):,}" for sp in ("test", "validation", "train"))
            rows.append(f"| `{n}` | {e['k']} | {e['domain']} | {e['description']} | {counts} | {'**yes**' if e['has_soft_labels'] else ''} | {e['license']} |")
        return NL.join(rows)

    def table(prim: str, title: str) -> str:
        return NL.join([f"#### {title}", "", "| config | K | domain | question | test / val / train | human distribution | license |",
                        "|---|---|---|---|---|---|---|", source_rows(prim), ""])

    hero_block = ""
    if hero:
        hero_block = NL.join([
            f"![accuracy vs calibration, one point per config](results/{hero}/figures/calibration_map.png)", "",
            f"*{hero} on every test record: crisp, grounded decisions land in the accurate-and-calibrated corner; "
            "ordinal ratings and anything humans disagree about do not.*", ""])

    gold_list = ", ".join(f"`{g}`" for g in gold)
    audit_dir = hero or "jev-1.13.0"
    lines = [
        "---", yaml.safe_dump(header, sort_keys=False).strip(), "---", "",
        '<div align="center">', "", "# jev-bench", "",
        "**Real human-labeled data, reformatted into System One questions — with human label *distributions* wherever they exist.**", "",
        f"`{len(srcs)}` configs · `{total:,}` rows · `{n_test:,}` test records · `{len(gold)}` calibration-gold configs · v{version}", "",
        "[Repo & engine](https://github.com/uspraveen/Jevify) · "
        "[Source rationale](https://github.com/uspraveen/Jevify/blob/main/docs/DATASETS.md) · "
        "[What we verified about Jev's API](https://github.com/uspraveen/Jevify/blob/main/docs/JEV_CONTRACT.md) · "
        "[Other independent Jev evaluations](https://github.com/OmniJev/awesome-jev)", "",
        "</div>", "", hero_block,
        "## Why this exists", "",
        "A *System One* model (TypeSafe's [Jev](https://typesafe.ai), or any open model Jevified by the engine in this repo)",
        "does not write text. It reads a `state`, answers typed questions, and returns **probability distributions your code**",
        "**can branch on**. The product claim is calibration: an answer given 0.8 should be right about 80% of the time.", "",
        f"Most benchmarks can only check the argmax. jev-bench checks the *distribution* — {len(gold)} configs carry the human",
        f"vote shares behind each label ({gold_list}), so \"calibrated\" is measured against how humans actually split,",
        "not only against a single hard label.", "",
        "## The three primitives", "",
        "| primitive | the question | what comes back | example config |", "|---|---|---|---|",
        "| `choice` | which of these K options? | probabilities over options + `confidence` | `clinc150` (151 intents incl. out-of-scope) |",
        "| `score` | where on these K ordered levels? | probabilities over levels, expected `score`, `confidence` | `helpsteer2_helpfulness` (0–4 Likert) |",
        "| `noul` | is this true? | a single `P(yes)` | `civil_comments` (toxic? with annotator share) |", "",
        "Every row is one `(state, question, label)` triple in **exactly the wire format a System One model consumes** — send",
        "`state` and `question` to `POST /v1/systemone` as-is.", "",
        "```python", "from datasets import load_dataset", "import json", "",
        'ds = load_dataset("Praveenrajus/jev-bench", "chaosnli", split="test")', "row = ds[0]",
        'state, question = json.loads(row["state"]), json.loads(row["question"])',
        'label, human = row["label"], json.loads(row["soft_label"])   # human = {"entailment": 0.63, "neutral": 0.37, ...}',
        "```", "",
        "`state`, `question` and `soft_label` are JSON strings so every config shares one stable schema; `label` is a string",
        "(option key for choice, level index for score, `\"0\"`/`\"1\"` for noul). The [jevify](https://github.com/uspraveen/Jevify)",
        "package gives you `BenchRecord.from_row`, the metrics (ECE, Brier, RPS, selective accuracy, TVD to human), the API",
        "runner and the figures.", "",
        "## Sources", "",
        f"Natural label distributions everywhere (uniform random samples, seed {manifest['seed']}); a calibration benchmark must not",
        "shift base rates. Splits: test ≤ 1,000 per config (2,000 for `civil_comments`, all of ChaosNLI), validation ≤ 500,",
        "train ≤ 8,000 so Tier 1/2 recipes and temperature scaling have in-distribution data without touching test.", "",
        table("choice", "Choice"), table("score", "Score"), table("noul", "Noul"),
        "Licenses are those of the upstream datasets; this repackaging adds no restrictions. Per-source provenance is in",
        "`manifest.json`.",
        baselines_section(baselines or []), "",
        probes_section(hero, manifest.get("_probes_summary")), "",
        "## Label audit", "",
        "Every weak result was checked by reading samples of the model's errors. Verdicts, examples and the two v0.1.1 fixes that",
        f"came out of it are in [`results/{audit_dir}/README.md`](results/{audit_dir}/README.md).", "",
        "## Changelog", "",
        "- **v0.1.1** — `helpsteer2_verbosity` levels replaced with NVIDIA's verbatim length scale (v0.1 misdescribed them);",
        "  `go_emotions` rebuilt from raw per-rater votes with soft labels; other configs unchanged.",
        "- **v0.1** — initial release.", "",
        f"Built by [`jevify-bench`](https://github.com/uspraveen/Jevify) at commit `{manifest.get('git_commit', '?')[:10]}`, {manifest['built_at']}.", "",
    ]
    return NL.join(lines)


def probes_section(model: str | None, summary: dict[str, Any] | None) -> str:
    if not model or not summary:
        return ""
    NL = chr(10)
    c = summary
    return NL.join([
        "## Behavioral probes", "",
        "Within-item experiments: the same state and gold answer, one factor changed. 200 items per source. "
        f"Full write-up in [`results/{model}/probes/README.md`](results/{model}/probes/README.md).", "",
        f"![cardinality probe](results/{model}/figures/probe_cardinality.png)", "",
        f"- **Decision-set size is a cost, not a cliff**: {c['card_crisp']}",
        f"- **Ambiguity is the cliff**: {c['card_ambig']}",
        f"- **Option order**: argmax flips {c['order']} — modest, and scaling with ambiguity, not K.",
        f"- **Opaque keys with descriptions kept**: accuracy unchanged ({c['rename']}); without descriptions: chance. Jev reads semantics, not key strings.",
        f"- **Distractor injection**: ≤{c['distractor']} of probability mass leaks to nonsense options.",
        f"- **Primitive geometry matters**: the same yes/no question is better calibrated as Noul than as a 2-way Choice ({c['prim_noul']}); "
        f"Score beats an unordered Choice over the same levels ({c['prim_score']}).", ""])


def baselines_section(baselines: list[tuple[str, str]]) -> str:
    if not baselines:
        return ""
    NL = chr(10)
    parts = ["", "## Baselines", "",
             "Every test record, one request each, scored by `jevify-run`. Predictions, full metrics with reliability bins, "
             "and figures live under `results/<model>/`. Columns: accuracy, top-label ECE, Brier, NLL, selective accuracy "
             "at 90% / 50% coverage, AURC, RPS and MAE (ordinal), AUROC (noul), total variation distance to the human "
             "label distribution. NLL is inflated on high-K configs because the API rounds probabilities to 0.01; read "
             "Brier and ECE as the proper scores.", ""]
    for model, table in baselines:
        base = f"results/{model}/figures"
        parts += [f"### {model}", "", table, "",
                  "**Reliability diagrams** — stated confidence vs observed accuracy per config. Flat lines mean the "
                  "confidence carries no information.", "", f"![reliability]({base}/reliability.png)", "",
                  "**Model vs human probability** on the calibration-gold configs — the axis that separates a decision "
                  "model from a classifier.", "", f"![model vs human]({base}/human_vs_model.png)", "",
                  "**Risk–coverage** — the error rate a confidence-gated router actually gets at each coverage.", "",
                  f"![risk coverage]({base}/risk_coverage.png)", "",
                  "**Performance vs decision-set size** (cross-dataset, so difficulty is confounded — a hypothesis view).", "",
                  f"![vs cardinality]({base}/vs_cardinality.png)", ""]
    return NL.join(parts)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _evict_hf_cache() -> None:
    for var in ("HF_HUB_CACHE", "HF_DATASETS_CACHE"):
        p = os.environ.get(var)
        if p and Path(p).exists():
            shutil.rmtree(p, ignore_errors=True)
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    for sub in ("hub", "datasets"):
        shutil.rmtree(home / sub, ignore_errors=True)


def push(out: Path, repo_id: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=repo_id, repo_type="dataset",
                      commit_message=f"jev-bench build {dt.datetime.now(dt.timezone.utc):%Y-%m-%d}")
    return f"https://huggingface.co/datasets/{repo_id}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="jevify-bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the benchmark")
    b.add_argument("--out", type=Path, default=Path("data/jev-bench"))
    b.add_argument("--sources", default="all", help="comma-separated config names or 'all'")
    b.add_argument("--seed", type=int, default=DEFAULT_SEED)
    b.add_argument("--caps", default="", help="override caps, e.g. test=200,train=0")
    b.add_argument("--evict-cache", action="store_true", help="delete the HF cache after each source (small disks)")
    b.add_argument("--stratify", action="store_true", help="balance small label spaces (training mixes only; distorts base rates)")
    b.add_argument("--push", default="", help="dataset repo id to upload to, e.g. Praveenrajus/jev-bench")
    sub.add_parser("list", help="list sources")
    pr = sub.add_parser("probe", help="generate controlled behavioral probes from test records")
    pr.add_argument("--records", type=Path, required=True)
    pr.add_argument("--out", type=Path, required=True)
    pr.add_argument("--which", default="cardinality,order,rename,distractor,primitive")
    pr.add_argument("--n", type=int, default=200, help="items per source")
    prr = sub.add_parser("probe-report", help="analyze probe predictions")
    prr.add_argument("--records", type=Path, required=True, help="probe root (from `probe`)")
    prr.add_argument("--preds", type=Path, required=True)
    prr.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.cmd == "probe":
        from .probes import generate

        counts = generate(args.records, args.out, [w for w in args.which.split(",") if w], args.n)
        print(f"{len(counts)} probe configs, {sum(counts.values())} records -> {args.out}")
        return 0
    if args.cmd == "probe-report":
        from .probes import analyze, report_md

        an = analyze(args.records, args.preds)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "probes.json").write_text(json.dumps(an, indent=1), encoding="utf-8")
        md = report_md(an)
        (args.out / "probes.md").write_text(md, encoding="utf-8")
        print(md)
        return 0

    if args.cmd == "list":
        for s in specs():
            print(f"{s.name:24s} {s.primitive:7s} K={s.k or 'var':<9} {s.hf_id or 'download':40s} {s.license}")
        return 0

    sources = list(REGISTRY) if args.sources == "all" else [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = [s for s in sources if s not in REGISTRY]
    if unknown:
        print(f"unknown sources: {unknown}", file=sys.stderr)
        return 2
    caps = {k: int(v) for k, v in (kv.split("=") for kv in args.caps.split(",") if kv)}
    manifest = build(args.out, sources, seed=args.seed, evict_cache=args.evict_cache, caps_override=caps, stratify=args.stratify)
    failed = [k for k, v in manifest["sources"].items() if "error" in v]
    print(f"\nbuilt {len(manifest['sources']) - len(failed)} sources" + (f", FAILED: {failed}" if failed else ""))
    if args.push:
        print("pushed:", push(args.out, args.push))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
