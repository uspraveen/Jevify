"""How fast is a Jevified vision model, measured the way the text ladder was?

One record at a time through the served `ask` path -- the image goes through the processor,
the answer is read from one position -- median over N real records per source, plus batched
throughput. Image tokens are the cost axis here rather than K: a photo at the default budget
is ~280 tokens before a word of the question is read, so the pixel budget is reported too.

    python scripts/latency_vision.py --model-id Qwen/Qwen3-VL-2B-Instruct --n 30 --out results/latency
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


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-pixels", type=int, default=0, help="pixel budget; 0 = the processor's default")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "latency")
    ap.add_argument("--tag", default="_vision")
    a = ap.parse_args()

    import torch
    from jevify.engine.vision import VisionScorer
    from jevify.engine.vision_backbone import image_tokens
    from jevify.engine.vision import split_images
    from jevify.load import JevifiedModel
    from jevify.runners.vision_runner import VisionRunner, build_vision_records

    scorer = VisionScorer(a.model_id, dtype=torch.bfloat16, batch_size=a.batch, max_pixels=a.max_pixels or None)
    model = JevifiedModel(scorer, {"backbone": a.model_id, "chat": True, "modality": "vision",
                                   "recipe": {"mode": "index", "permutations": 1}})
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    rows = {}
    for src, prim, k in (("pope", "noul", 2), ("aokvqa", "choice", 4), ("ai2d", "choice", 4)):
        recs = build_vision_records([src], "test", a.n)
        for r in recs[:3]:                                              # warm-up
            model.ask(r.state, {"q": r.question})
        times, toks = [], []
        for r in recs:
            _sync(); t0 = time.perf_counter()
            model.ask(r.state, {"q": r.question})
            _sync(); times.append((time.perf_counter() - t0) * 1000)
            toks.append(scorer.last_input_tokens)
        img_toks = [image_tokens(scorer.processor, im) for r in recs[:12] for im in split_images(r.state)[1]]
        rows[src] = {"primitive": prim, "k": k, "p50_ms": float(np.median(times)), "p90_ms": float(np.percentile(times, 90)),
                     "input_tokens_median": float(np.median(toks)), "image_tokens_median": float(np.median(img_toks)), "n": len(times)}
        print(f"  {src:8s} K={k}  p50 {rows[src]['p50_ms']:6.1f} ms  p90 {rows[src]['p90_ms']:6.1f}  input tok {rows[src]['input_tokens_median']:.0f} "
              f"(image {rows[src]['image_tokens_median']:.0f})", flush=True)
    mixed = [r for src in ("pope", "aokvqa", "ai2d") for r in build_vision_records([src], "test", a.n)]
    runner = VisionRunner(scorer)
    list(runner.predict(mixed[: a.batch], batch=a.batch))
    _sync(); t0 = time.perf_counter()
    n = sum(1 for _ in runner.predict(mixed, batch=a.batch))
    _sync(); rps = n / (time.perf_counter() - t0)
    print(f"  batched {rps:.1f} rec/s @ {a.batch}", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"latency{a.tag}.json").write_text(json.dumps({"device": device, "model": a.model_id, "max_pixels": a.max_pixels or None,
                                                            "rows": rows, "batched": {"records_per_s": rps, "batch": a.batch}}, indent=1),
                                                encoding="utf-8")
    md = [f"# Vision latency on one {device} — {a.model_id}", "",
          "One record at a time through the served `ask` path, median of N real records; the image goes through the processor "
          "and the answer is read from one position.", "",
          "| source | primitive | K | p50 ms | p90 ms | input tokens | of which image |", "|---|---|---|---|---|---|---|"]
    for src, r in rows.items():
        md.append(f"| {src} | {r['primitive']} | {r['k']} | {r['p50_ms']:.0f} | {r['p90_ms']:.0f} | {r['input_tokens_median']:.0f} | {r['image_tokens_median']:.0f} |")
    md += ["", f"Batched: **{rps:.1f} records/s** at batch {a.batch}, mixed sources."]
    (a.out / f"latency{a.tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
