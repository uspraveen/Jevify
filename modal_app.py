"""Modal jobs for Jevify: Tier 0 scoring of any HF checkpoint on jev-bench.

    JEVIFY_GPU=L4 modal run modal_app.py::tier0 --model-id Qwen/Qwen3.5-0.8B --run-id qwen35-0.8b --limit 50

Weights are cached in a Volume so nothing downloads twice; predictions stream
to a second Volume and are pulled back with ``fetch``. Every run writes a
``run.json`` with wall time and the GPU used so cost can be accounted.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

GPU = os.environ.get("JEVIFY_GPU", "L4")
RATE_PER_HOUR = {"L4": 0.80, "A10G": 1.10, "L40S": 1.95, "A100-40GB": 2.10, "A100-80GB": 2.50, "H100": 3.95, "H200": 4.54, "T4": 0.59}

app = modal.App("jevify")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers>=5.0", "accelerate", "huggingface_hub", "numpy", "pydantic>=2.5", "httpx", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf"})
    .add_local_python_source("jevify")
)
hf_cache = modal.Volume.from_name("jevify-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("jevify-runs", create_if_missing=True)
BENCH = "Praveenrajus/jev-bench"


def _bench_root() -> Path:
    from huggingface_hub import snapshot_download

    root = Path("/runs/jev-bench")
    snapshot_download(BENCH, repo_type="dataset", local_dir=str(root), token=os.environ.get("HF_TOKEN"),
                      allow_patterns=["data/*/test.jsonl", "data/*/validation.jsonl", "manifest.json"])
    return root


@app.function(image=image, gpu=GPU, volumes={"/hf": hf_cache, "/runs": runs}, secrets=[modal.Secret.from_name("huggingface")],
              timeout=4 * 3600, memory=32768)
def tier0(model_id: str, run_id: str, sources: str = "", split: str = "test", limit: int = 0, mode: str = "index",
          chat: bool = True, permutations: int = 2, batch: int = 16, cand_chunk: int = 64, trust_remote_code: bool = False,
          resume: bool = True) -> dict:
    import torch

    from jevify.bench.record import read_jsonl
    from jevify.engine.predict import Recipe, Tier0Engine
    from jevify.engine.readout import HFScorer
    from jevify.runners.base import done_ids, write_predictions

    root = _bench_root()
    out_dir = Path(f"/runs/{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{split}_predictions.jsonl"
    want = [s for s in sources.split(",") if s] or None
    recs = []
    for path in sorted((root / "data").glob(f"*/{split}.jsonl")):
        if want and path.parent.name not in want:
            continue
        recs.extend(read_jsonl(path, limit=limit or None))
    skip = done_ids(out) if resume and out.exists() else set()
    todo = [r for r in recs if r.id not in skip]
    print(f"[{run_id}] {model_id} on {GPU}: {len(todo)} records ({len(skip)} already done)", flush=True)

    t0 = time.time()
    scorer = HFScorer(model_id, dtype=torch.bfloat16, batch_size=batch, cand_chunk=cand_chunk,
                      trust_remote_code=trust_remote_code, hf_token=os.environ.get("HF_TOKEN"))
    t_load = time.time() - t0
    engine = Tier0Engine(scorer, Recipe(mode=mode, chat=chat, permutations=permutations))
    n = write_predictions(out, engine.score_records(todo, batch=batch), append=bool(skip), progress_every=250)
    elapsed = time.time() - t0
    info = {"run_id": run_id, "model_id": model_id, "gpu": GPU, "split": split, "sources": want, "limit": limit,
            "recipe": engine.recipe.as_dict(), "chat_applied": engine.chat, "n_new": n, "n_total": len(recs),
            "load_s": round(t_load, 1), "wall_s": round(elapsed, 1),
            "est_cost_usd": round(elapsed / 3600 * RATE_PER_HOUR.get(GPU, 2.5), 3),
            "torch": torch.__version__, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "scorer_stats": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in scorer.stats.items()},
            "attn_impl": getattr(scorer.model.config, "_attn_implementation", None), "dtype": str(next(scorer.model.parameters()).dtype)}
    (out_dir / "run.json").write_text(json.dumps(info, indent=1))
    runs.commit()
    print(json.dumps(info), flush=True)
    return info


@app.function(image=image, volumes={"/runs": runs}, timeout=600)
def fetch(run_id: str, name: str = "test_predictions.jsonl") -> bytes:
    runs.reload()
    return Path(f"/runs/{run_id}/{name}").read_bytes()


@app.function(image=image, volumes={"/runs": runs}, timeout=600)
def ls(run_id: str = "") -> list[str]:
    runs.reload()
    base = Path("/runs") / run_id
    return sorted(str(p.relative_to("/runs")) + (f"  {p.stat().st_size}" if p.is_file() else "/") for p in base.rglob("*") if "jev-bench" not in str(p))


@app.local_entrypoint()
def main(model_id: str, run_id: str, sources: str = "", split: str = "test", limit: int = 0, mode: str = "index",
         chat: bool = True, permutations: int = 2, batch: int = 16, cand_chunk: int = 64, trust_remote_code: bool = False,
         out: str = "runs"):
    info = tier0.remote(model_id, run_id, sources, split, limit, mode, chat, permutations, batch, cand_chunk, trust_remote_code)
    local = Path(out) / run_id
    local.mkdir(parents=True, exist_ok=True)
    (local / f"{split}_predictions.jsonl").write_bytes(fetch.remote(run_id, f"{split}_predictions.jsonl"))
    (local / "run.json").write_text(json.dumps(info, indent=1))
    print(f"saved to {local}  (wall {info['wall_s']}s, est ${info['est_cost_usd']})")


@app.function(image=image, gpu=GPU, volumes={"/hf": hf_cache, "/runs": runs}, secrets=[modal.Secret.from_name("huggingface")], timeout=1800)
def diag(model_id: str, source: str = "clinc150", trust_remote_code: bool = False, fp32: bool = False) -> dict:
    """Compare tree / cache-expansion / naive scoring on real items, in the model's own dtype."""
    import numpy as np
    import torch

    from jevify.bench.record import read_jsonl
    from jevify.engine.readout import HFScorer, softmax
    from jevify.engine.template import render, to_chat

    root = _bench_root()
    recs = list(read_jsonl(root / "data" / source / "test.jsonl", limit=3))
    scorer = HFScorer(model_id, dtype=torch.float32 if fp32 else torch.bfloat16, trust_remote_code=trust_remote_code, hf_token=os.environ.get("HF_TOKEN"))
    report = {}
    for r in recs:
        rd = render(r.state, r.question)
        t = scorer.tokenize(to_chat(rd.prefix, scorer.tokenizer), rd.candidates)
        with torch.inference_mode():
            tree = np.array(scorer._score_tree(t)); multi = np.array(scorer._score_multi(t)); naive = np.array(scorer._score_naive(t))
        p = lambda v: np.array(softmax(list(v)))
        report[r.id] = {"K": len(t.cand_ids), "max|tree-naive| logp": float(np.abs(tree - naive).max()),
                        "max|multi-naive| logp": float(np.abs(multi - naive).max()),
                        "max|tree-naive| prob": float(np.abs(p(tree) - p(naive)).max()),
                        "max|multi-naive| prob": float(np.abs(p(multi) - p(naive)).max()),
                        "argmax agree tree/naive": bool(tree.argmax() == naive.argmax())}
    return report


@app.local_entrypoint()
def diagnose(model_id: str, source: str = "clinc150", trust_remote_code: bool = False, fp32: bool = False):
    print(json.dumps(diag.remote(model_id, source, trust_remote_code, fp32), indent=1))
