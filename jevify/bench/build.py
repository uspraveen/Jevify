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
    manifest.update({"name": "jev-bench", "version": "0.1", "seed": seed, "sampling": "stratified" if stratify else "natural",
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
    configs = []
    for name, e in srcs.items():
        files = [{"split": s, "path": p} for s, p in e["files"].items()]
        configs.append({"config_name": name, "data_files": files})
    header = {
        "pretty_name": "jev-bench",
        "license": "other",
        "license_name": "mixed-see-manifest",
        "license_link": "https://github.com/uspraveen/Jevify/blob/main/docs/DATASETS.md",
        "language": ["en"],
        "task_categories": ["text-classification"],
        "tags": ["calibration", "system-one", "decision-model", "jevify"],
        "configs": configs,
    }
    import yaml  # PyYAML ships with huggingface_hub

    rows = "\n".join(
        f"| `{n}` | {e['primitive']} | {e['k']} | {e['domain']} | " + " / ".join(f"{s}={c}" for s, c in e["counts"].items())
        + f" | {'yes' if e['has_soft_labels'] else ''} | {e['license']} |"
        for n, e in srcs.items()
    )
    total = sum(sum(e["counts"].values()) for e in srcs.values())
    return f"""---
{yaml.safe_dump(header, sort_keys=False).strip()}
---

# jev-bench

Real, human-labeled data reformatted into **System One questions**: every row is one
`(state, question, label)` triple in the exact wire format a System One model (TypeSafe's
Jev, or anything Jevified) consumes. Three primitives:

- **choice** — pick one of K labeled options → probabilities over options
- **score** — place the state on K ordered levels → probabilities over levels
- **noul** — an absolute yes/no judgment → P(yes)

Where the source provides one, `soft_label` carries the **human label distribution**
(ChaosNLI, Civil Comments, Measuring Hate Speech) so calibration can be measured against
human uncertainty rather than only against hard labels.

Built by [`jevify-bench`](https://github.com/uspraveen/Jevify) (seed {manifest['seed']},
commit `{manifest.get('git_commit', '?')[:10]}`, {manifest['built_at']}). {total:,} rows.

## Row format

`state`, `question` and `soft_label` are JSON-encoded strings (so every config has one
stable schema); `label` is a string (an option key for choice, a level index for score,
`"0"`/`"1"` for noul). Decode with `jevify.bench.record.BenchRecord.from_row`.

## Sources

| config | primitive | K | domain | rows | soft labels | license |
|---|---|---|---|---|---|---|
{rows}

Licenses are those of the upstream datasets; this repackaging adds no restrictions.
See `manifest.json` for per-source provenance and `docs/DATASETS.md` in the repo for the
selection rationale.
{baselines_section(baselines or [])}"""


def baselines_section(baselines: list[tuple[str, str]]) -> str:
    if not baselines:
        return ""
    intro = ("Test-split results produced by `jevify-run`; predictions and full metrics live under "
             "`results/<model>/`. Columns: accuracy, top-label ECE, Brier, NLL, selective accuracy at "
             "90%/50% coverage, AURC, RPS and MAE (ordinal), AUROC (noul), total variation distance to "
             "human label distributions.")
    parts = ["", "## Baselines", "", intro, ""]
    for model, table in baselines:
        parts += [f"### {model}", "", table, ""]
    return chr(10).join(parts)


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
    args = ap.parse_args(argv)

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
