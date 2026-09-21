"""How fast is a Jevified model, and what does each tier cost at inference?

A System One model's pitch is that a typed question is answered in one forward pass,
not a generation loop. That claim has a number, and it depends on three things a user
actually chooses: the backbone's size, the tier (heads cost a tiny MLP; a merged LoRA
costs nothing), and the size of the answer set K -- every option is scored, so K is the
axis latency should be plotted against.

Two measurements, both on one GPU, both with ``torch.cuda.synchronize`` around the call:

  single   one record at a time through the same ``ask`` path the API serves -- what a
           caller feels; median and p90 over N real jev-bench records per K
  batched  records/s at a fixed batch over a mixed set -- what an offline job gets

Jev's own number is its client-observed API round trip from the baseline run, on the
same sources. It is a different thing (network + their serving stack) and is labelled
as such; it is the number a user comparing the two would actually experience.

    python scripts/latency.py --tier0 Qwen/Qwen3.5-0.8B,Qwen/Qwen3.5-2B,Qwen/Qwen3.5-4B \\
        --jevified Praveenrajus/jevify-qwen3.5-2b --jevified-dir runs/qwen35-2b-t2 --out results/latency
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# one real source per K, so the ladder is real prompts and real option sets, not synthetic
LADDER = [("boolq", "noul", 2), ("arc_challenge", "choice", 4), ("sst5", "score", 5), ("go_emotions", "choice", 27),
          ("massive", "choice", 60), ("banking77", "choice", 77), ("clinc150", "choice", 151)]


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_ladder(root: Path, n: int, seed: int = 0):
    from jevify.bench.record import read_jsonl
    rng = np.random.default_rng(seed)
    out = {}
    for src, _prim, _k in LADDER:
        recs = list(read_jsonl(root / "data" / src / "test.jsonl"))
        pick = rng.choice(len(recs), size=min(n, len(recs)), replace=False)
        out[src] = [recs[i] for i in sorted(pick)]
    return out


def time_single(model, recs, warmup: int = 3) -> dict:
    """Median / p90 ms for one record at a time through ``ask``."""
    for r in recs[:warmup]:
        model.ask(r.state, {"q": r.question})
    times = []
    for r in recs:
        _sync(); t0 = time.perf_counter()
        model.ask(r.state, {"q": r.question})
        _sync(); times.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": float(np.median(times)), "p90_ms": float(np.percentile(times, 90)), "n": len(times)}


def time_batched(model, recs, batch: int) -> dict:
    """Records/s through the engine at a fixed batch (Tier 0 path; heads add negligible cost)."""
    engine = model.engine
    list(engine.score_records(recs[:batch], batch=batch))                 # warm up
    _sync(); t0 = time.perf_counter()
    n = sum(1 for _ in engine.score_records(recs, batch=batch))
    _sync(); dt = time.perf_counter() - t0
    return {"records_per_s": n / dt, "n": n, "batch": batch}


def jev_latency(root: Path, ladder) -> dict:
    """Jev's client-observed round trip per source from the baseline predictions."""
    from jevify.runners.base import read_predictions
    want = {src for src, _, _ in LADDER}
    by_src: dict[str, list[float]] = {s: [] for s in want}
    for p in read_predictions(root / "results" / "jev-1.13.0" / "test_predictions.jsonl"):
        src = p.id.split("/")[0]
        if src in want and p.latency_ms:
            by_src[src].append(p.latency_ms)
    return {src: {"p50_ms": float(np.median(v)), "p90_ms": float(np.percentile(v, 90)), "n": len(v)}
            for src, v in by_src.items() if v}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", type=Path, default=ROOT / "data" / "jev-bench")
    ap.add_argument("--tier0", default="Qwen/Qwen3.5-0.8B,Qwen/Qwen3.5-2B,Qwen/Qwen3.5-4B",
                    help="comma-separated backbones to measure at Tier 0")
    ap.add_argument("--jevified", default="", help="comma-separated published Tier 1 repos")
    ap.add_argument("--jevified-dir", default="", help="comma-separated local training dirs (Tier 1/2)")
    ap.add_argument("--n", type=int, default=30, help="records per K for the single-request measurement")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--permutations", default="1,2", help="Tier 0 option-order permutations to measure")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "latency")
    ap.add_argument("--engine", default="hf", choices=["hf", "vllm"], help="Tier 0 scorer: research path or vLLM")
    ap.add_argument("--state-last", action="store_true", help="question + options before the state (cacheable prefix)")
    ap.add_argument("--tag", default="", help="suffix for output files")
    a = ap.parse_args()

    import torch
    from jevify.load import JevifiedModel, load_jevified
    from jevify.engine.readout import HFScorer

    ladder = load_ladder(a.records, a.n)
    mixed = [r for recs in ladder.values() for r in recs]
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    rows = []

    def measure(label: str, tier: str, model, perms: int):
        row = {"model": label, "tier": tier, "permutations": perms, "single": {}, "batched": None}
        for src, prim, k in LADDER:
            row["single"][src] = {"primitive": prim, "k": k, **time_single(model, ladder[src])}
            print(f"  {label:34s} P={perms} {src:14s} K={k:3d}  p50 {row['single'][src]['p50_ms']:7.1f} ms", flush=True)
        row["batched"] = time_batched(model, mixed, a.batch)
        print(f"  {label:34s} batched {row['batched']['records_per_s']:.1f} rec/s @ {a.batch}", flush=True)
        rows.append(row)

    for mid in [m for m in a.tier0.split(",") if m]:
        if a.engine == "vllm":
            from jevify.engine.vllm_readout import VLLMScorer
            scorer = VLLMScorer(mid)
        else:
            scorer = HFScorer(mid, dtype=torch.bfloat16)
        for perms in [int(x) for x in a.permutations.split(",") if x]:
            model = JevifiedModel(scorer, {"backbone": mid, "chat": True,
                                           "recipe": {"permutations": perms, "state_last": a.state_last}})
            label = f"{mid.split('/')[-1]} (Tier 0{', vLLM' if a.engine == 'vllm' else ''}{', state last' if a.state_last else ''})"
            measure(label, "Tier 0", model, perms)
        del scorer, model; torch.cuda.empty_cache()
    for spec in [x for x in (a.jevified + "," + a.jevified_dir).split(",") if x]:
        model = load_jevified(spec)
        tier = "Tier 2" if model.config.get("lora") else "Tier 1"
        perms = model.engine.recipe.permutations
        measure(f"{model.backbone.split('/')[-1]} ({tier})", tier, model, perms)
        del model; torch.cuda.empty_cache()

    jev = jev_latency(a.records, ladder)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"latency{a.tag}.json").write_text(json.dumps({"device": device, "n_per_k": a.n, "batch": a.batch, "rows": rows,
                                                   "engine": a.engine, "state_last": a.state_last,
                                                   "jev_api": jev, "ladder": LADDER}, indent=1), encoding="utf-8")
    md = markdown(rows, jev, device, a.batch)
    (a.out / f"latency{a.tag}.md").write_text(md, encoding="utf-8")
    print(md)
    from jevify.bench.figures import fig_latency_ladder
    if not a.tag:
        fig_latency_ladder(rows, jev, LADDER, a.out, device)
    return 0


def markdown(rows, jev, device, batch) -> str:
    ks = [k for _, _, k in LADDER]
    out = [f"# Inference latency on one {device}", "",
           f"Single request = one record through the served `ask` path, median over N per K. Batched = records/s at batch {batch} "
           "over the same mixed set. Jev = its client-observed API round trip on the same sources (network included).", "",
           "| model | tier | perms | " + " | ".join(f"K={k}" for k in ks) + f" | rec/s @{batch} |",
           "|---|---|---|" + "---|" * (len(ks) + 1)]
    for r in rows:
        cells = [f"{r['single'][src]['p50_ms']:.0f} ms" for src, _, _ in LADDER]
        out.append(f"| {r['model']} | {r['tier']} | {r['permutations']} | " + " | ".join(cells)
                   + f" | {r['batched']['records_per_s']:.1f} |")
    if jev:
        cells = [f"{jev[src]['p50_ms']:.0f} ms" if src in jev else "—" for src, _, _ in LADDER]
        out.append("| Jev 1.13.0 (API round trip) | API | — | " + " | ".join(cells) + " | — |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
