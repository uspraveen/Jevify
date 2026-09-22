"""The readout against normal generation, same model, same prompt, same GPU.

A Jevified answer is one forward pass: the prompt is prefilled and the probability of every
allowed answer is read from the last position. "Normal" use of the same model decodes an
answer token by token after the same prefill. This measures both on the same records so the
saving is a number, not a slogan: the readout costs the prefill; every generated token costs
one more decoder step on top.

    python scripts/latency_generate.py --model-id Qwen/Qwen3.5-9B --n 20 --out results/latency --tag _qwen35-9b
    python scripts/latency_generate.py --model-id Qwen/Qwen3-VL-8B-Instruct --vision --n 20 --tag _qwen3vl-8b
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

TEXT_SOURCES = [("boolq", "noul", 2), ("arc_challenge", "choice", 4), ("clinc150", "choice", 151)]
VISION_SOURCES = [("pope", "noul", 2), ("aokvqa", "choice", 4)]


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(fn, warm: int = 2, reps: int = 1):
    for _ in range(warm):
        fn()
    _sync(); t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    _sync()
    return (time.perf_counter() - t0) * 1000 / reps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--new-tokens", default="1,8,32")
    ap.add_argument("--records", type=Path, default=ROOT / "data" / "jev-bench")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "latency")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    import torch
    from jevify.load import JevifiedModel

    new_tokens = [int(x) for x in a.new_tokens.split(",") if x]
    if a.vision:
        from jevify.engine.vision import VisionScorer
        from jevify.runners.vision_runner import build_vision_records

        scorer = VisionScorer(a.model_id, dtype=torch.bfloat16, batch_size=4)
        model = JevifiedModel(scorer, {"backbone": a.model_id, "chat": True, "modality": "vision",
                                       "recipe": {"mode": "index", "permutations": 1}}, None)
        sources = VISION_SOURCES
        recs_of = {src: build_vision_records([src], "test", a.n) for src, _, _ in sources}
    else:
        from jevify.bench.record import read_jsonl
        from jevify.engine.readout import HFScorer

        scorer = HFScorer(a.model_id, dtype=torch.bfloat16)
        model = JevifiedModel(scorer, {"backbone": a.model_id, "chat": True, "recipe": {"permutations": 1}}, None)
        sources = TEXT_SOURCES
        rng = np.random.default_rng(0)
        recs_of = {}
        for src, _, _ in sources:
            allr = list(read_jsonl(a.records / "data" / src / "test.jsonl"))
            pick = rng.choice(len(allr), size=min(a.n, len(allr)), replace=False)
            recs_of[src] = [allr[i] for i in sorted(pick)]
    tok = scorer.tokenizer
    dev = scorer.device
    rows = {}
    for src, prim, k in sources:
        recs = recs_of[src]
        readout, prefill, gen = [], [], {n: [] for n in new_tokens}
        ptoks = []
        for r in recs:
            # the readout: exactly what `ask` does
            readout.append(_timed(lambda: model.ask(r.state, {"q": r.question})))
            # the same prompt, as the model would see it for generation
            if a.vision:
                it, _rd = scorer.item(r.state, r.question, mode="index")
                kw = {"text": [it.prefix], "return_tensors": "pt"}
                if it.images:
                    kw["images"] = [it.images]
                enc = scorer.processor(**kw)
            else:
                rd = model.engine._renderings(r.state, r.question)[0]
                enc = tok(model.engine._prefix(rd), return_tensors="pt", add_special_tokens=True)
            enc = {kk: (v.to(dev) if hasattr(v, "to") else v) for kk, v in enc.items()}
            ptoks.append(int(enc["attention_mask"].sum()) if "attention_mask" in enc else int(enc["input_ids"].shape[-1]))
            with torch.inference_mode():
                prefill.append(_timed(lambda: scorer.model(**enc, use_cache=False)))
                for n in new_tokens:
                    gen[n].append(_timed(lambda: scorer.model.generate(**enc, max_new_tokens=n, min_new_tokens=n, do_sample=False,
                                                                       pad_token_id=scorer.pad_id)))
        rows[src] = {"primitive": prim, "k": k, "n": len(recs), "prompt_tokens_median": float(np.median(ptoks)),
                     "readout_p50_ms": float(np.median(readout)), "prefill_only_p50_ms": float(np.median(prefill)),
                     **{f"generate_{n}_p50_ms": float(np.median(gen[n])) for n in new_tokens}}
        g = rows[src]
        per_tok = (g[f"generate_{new_tokens[-1]}_p50_ms"] - g[f"generate_{new_tokens[0]}_p50_ms"]) / max(new_tokens[-1] - new_tokens[0], 1)
        rows[src]["ms_per_generated_token"] = float(per_tok)
        print(f"  {src:14s} K={k:3d} prompt {g['prompt_tokens_median']:.0f} tok | readout {g['readout_p50_ms']:.0f} ms | prefill {g['prefill_only_p50_ms']:.0f} | "
              + " | ".join(f"gen{n} {g[f'generate_{n}_p50_ms']:.0f}" for n in new_tokens) + f" | {per_tok:.1f} ms/token", flush=True)

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"latency_generate{a.tag}.json").write_text(json.dumps({"device": device, "model": a.model_id, "vision": a.vision,
                                                                       "new_tokens": new_tokens, "rows": rows}, indent=1), encoding="utf-8")
    md = [f"# Readout vs generation on one {device} — {a.model_id}", "",
          "Same records, same prompt (chat template included), one request at a time, medians. The readout is Jevify's `ask`; "
          "`prefill` is one forward pass over the prompt with no answer read; `generate N` is Hugging Face greedy decoding of N tokens.", "",
          "| source | K | prompt tokens | readout | prefill only | " + " | ".join(f"generate {n}" for n in new_tokens) + " | per generated token |",
          "|---|---|---|---|---|" + "---|" * len(new_tokens) + "---|"]
    for src, g in rows.items():
        md.append(f"| {src} | {g['k']} | {g['prompt_tokens_median']:.0f} | **{g['readout_p50_ms']:.0f} ms** | {g['prefill_only_p50_ms']:.0f} ms | "
                  + " | ".join(f"{g[f'generate_{n}_p50_ms']:.0f} ms" for n in new_tokens) + f" | {g['ms_per_generated_token']:.1f} ms |")
    (a.out / f"latency_generate{a.tag}.md").write_text(chr(10).join(md) + chr(10), encoding="utf-8")
    print(chr(10).join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
