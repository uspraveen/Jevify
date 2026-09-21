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
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": "/hf", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("jevify")
)
hf_cache = modal.Volume.from_name("jevify-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("jevify-runs", create_if_missing=True)
BENCH = "Praveenrajus/jev-bench"


def _bench_root() -> Path:
    from huggingface_hub import snapshot_download

    root = Path("/runs/jev-bench")
    snapshot_download(BENCH, repo_type="dataset", local_dir=str(root), token=os.environ.get("HF_TOKEN"),
                      allow_patterns=["data/*/test.jsonl", "data/*/validation.jsonl", "data/*/train.jsonl", "manifest.json"])
    return root


def _tier0_impl(gpu_name: str, model_id: str, run_id: str, sources: str = "", split: str = "test", limit: int = 0, mode: str = "index",
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
    print(f"[{run_id}] {model_id} on {gpu_name}: {len(todo)} records ({len(skip)} already done)", flush=True)

    t0 = time.time()
    scorer = HFScorer(model_id, dtype=torch.bfloat16, batch_size=batch, cand_chunk=cand_chunk,
                      trust_remote_code=trust_remote_code, hf_token=os.environ.get("HF_TOKEN"))
    t_load = time.time() - t0
    engine = Tier0Engine(scorer, Recipe(mode=mode, chat=chat, permutations=permutations))
    n = write_predictions(out, engine.score_records(todo, batch=batch), append=bool(skip), progress_every=250)
    elapsed = time.time() - t0
    info = {"run_id": run_id, "model_id": model_id, "gpu": gpu_name, "split": split, "sources": want, "limit": limit,
            "recipe": engine.recipe.as_dict(), "chat_applied": engine.chat, "n_new": n, "n_total": len(recs),
            "load_s": round(t_load, 1), "wall_s": round(elapsed, 1),
            "est_cost_usd": round(elapsed / 3600 * RATE_PER_HOUR.get(gpu_name, 2.5), 3),
            "torch": torch.__version__, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "scorer_stats": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in scorer.stats.items()},
            "attn_impl": getattr(scorer.model.config, "_attn_implementation", None), "dtype": str(next(scorer.model.parameters()).dtype)}
    (out_dir / "run.json").write_text(json.dumps(info, indent=1))
    runs.commit()
    print(json.dumps(info), flush=True)
    return info


GPU_KW = dict(image=image, volumes={"/hf": hf_cache, "/runs": runs}, secrets=[modal.Secret.from_name("huggingface")],
              timeout=4 * 3600, memory=32768)


@app.function(gpu=GPU, **GPU_KW)
def tier0(model_id: str, run_id: str, sources: str = "", split: str = "test", limit: int = 0, mode: str = "index",
          chat: bool = True, permutations: int = 2, batch: int = 16, cand_chunk: int = 64, trust_remote_code: bool = False,
          resume: bool = True) -> dict:
    return _tier0_impl(GPU, model_id, run_id, sources, split, limit, mode, chat, permutations, batch, cand_chunk, trust_remote_code, resume)


@app.function(gpu="L4", **GPU_KW)
def tier0_l4(*args, **kwargs) -> dict:
    return _tier0_impl("L4", *args, **kwargs)


@app.function(gpu="A100-80GB", **GPU_KW)
def tier0_a100(*args, **kwargs) -> dict:
    return _tier0_impl("A100-80GB", *args, **kwargs)


@app.function(image=image, volumes={"/runs": runs}, timeout=24 * 3600)
def sweep(plan: list[dict], permutations: int = 2, val_limit: int = 200, batch: int = 32, smoke: bool = False) -> list[dict]:
    """Walk the plan server-side: validation then test for each model, resumable, with a ledger on the volume."""
    ledger_path = Path("/runs/ledger.jsonl")
    done: list[dict] = []
    for entry in plan:
        fn = tier0_a100 if entry["gpu"].startswith("A100") else tier0_l4
        jobs = [("test", 5)] if smoke else [("validation", val_limit), ("test", 0)]
        for split, limit in jobs:
            run_id = entry["run_id"] + ("-smoke" if smoke else "") + ("" if split == "test" else f"-{split}")
            runs.reload()
            t0 = time.time()
            try:
                info = fn.remote(entry["model_id"], run_id, "", split, limit, "index", entry.get("chat", True), permutations, batch, 64,
                                 bool(entry.get("trust_remote_code")), True)
                status = "ok"
            except Exception as e:  # keep going; the ledger records the failure
                info = {"run_id": run_id, "error": f"{type(e).__name__}: {str(e)[:300]}"}
                status = "FAILED"
            rec = {"run_id": run_id, "model_id": entry["model_id"], "gpu": entry["gpu"], "split": split, "status": status,
                   "wall_s": round(time.time() - t0, 1), "est_cost_usd": info.get("est_cost_usd"), "error": info.get("error")}
            with ledger_path.open("a") as f:
                f.write(json.dumps(rec) + chr(10))
            runs.commit()
            print(json.dumps(rec), flush=True)
            done.append(rec)
    return done


def _n_done(path: Path) -> int:
    return sum(1 for _ in path.open()) if path.exists() else 0


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


@app.local_entrypoint()
def run_sweep(plan: str = "scripts/sweep_plan.json", only: str = "", permutations: int = 2, val_limit: int = 200,
              batch: int = 32, smoke: bool = False):
    entries = json.loads(Path(plan).read_text())
    keep = {s for s in only.split(",") if s}
    entries = [e for e in entries if not keep or e["run_id"] in keep]
    call = sweep.spawn(entries, permutations, val_limit, batch, smoke)
    print(f"spawned sweep over {len(entries)} models: {[e['run_id'] for e in entries]}  call_id={call.object_id}")


@app.local_entrypoint()
def sweep_status():
    for line in ls.remote(""):
        if "ledger" in line or "run.json" in line or "predictions" in line:
            print(line)


# --------------------------------------------------------------------------- Tier 1
# Sources the heads never see in training. Chosen to span primitives, domains and K:
# large-K routing, knowledge MCQ, an ordinal rating, a soft-label safety scale, and two
# grounded yes/no tasks. chaosnli has no train split, so it is held out by construction.
HELDOUT_SOURCES = ["clinc150", "arc_challenge", "yelp5", "measuring_hate_speech", "fever_evidence", "strategyqa_grounded"]


def _tier1_impl(gpu_name: str, model_id: str, run_id: str, layer: int = -1, chat: bool = True,
                train_per_source: int = 600, val_per_source: int = 150, max_slots: int = 16, dim: int = 512,
                epochs: int = 20, lr: float = 3e-4, batch: int = 8, trust_remote_code: bool = False,
                heldout: str = "", test_per_source: int = 0, residual: bool = True) -> dict:
    import numpy as np
    import torch

    from jevify.bench.record import read_jsonl
    from jevify.engine.features import FeatureExtractor
    from jevify.engine.heads import HeadConfig, evaluate_loss, predict_rows, save_heads, train_heads
    from jevify.engine.readout import HFScorer
    from jevify.runners.base import Prediction, write_predictions

    held = [s for s in (heldout.split(",") if heldout else HELDOUT_SOURCES) if s]
    root = _bench_root()
    out_dir = Path(f"/runs/{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    scorer = HFScorer(model_id, dtype=torch.bfloat16, trust_remote_code=trust_remote_code, hf_token=os.environ.get("HF_TOKEN"))
    fx = FeatureExtractor(scorer, layer=layer, chat=chat)
    rng = np.random.default_rng(20260921)

    def load(split: str, limit: int, sources: list[str] | None = None, exclude: list[str] | None = None):
        recs = []
        for path in sorted((root / "data").glob(f"*/{split}.jsonl")):
            name = path.parent.name
            if sources is not None and name not in sources:
                continue
            if exclude and name in exclude:
                continue
            recs.extend(read_jsonl(path, limit=limit or None))
        return recs

    train_recs = load("train", train_per_source, exclude=held)
    val_recs = load("validation", val_per_source, exclude=held)
    test_recs = load("test", test_per_source)
    print(f"[{run_id}] {model_id} on {gpu_name}: {len(train_recs)} train / {len(val_recs)} val / {len(test_recs)} test; "
          f"heldout={held}", flush=True)

    t_feat = time.time()
    train_rows = fx.extract(train_recs, max_slots=max_slots, rng=rng, batch_size=batch)
    val_rows = fx.extract(val_recs, max_slots=max_slots, rng=rng, batch_size=batch)
    test_rows = fx.extract(test_recs, batch_size=batch)
    feat_s = time.time() - t_feat
    print(f"[{run_id}] features in {feat_s:.0f}s", flush=True)

    cfg = HeadConfig(hidden=fx.hidden, dim=dim, layer=layer, backbone=model_id, residual=residual)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_train = time.time()
    heads, info = train_heads(train_rows, val_rows, cfg, epochs=epochs, lr=lr, device=device, verbose=True)
    train_s = time.time() - t_train
    save_heads(heads, info, out_dir / "heads")

    probs = predict_rows(heads, test_rows, device=device)
    by_id = {r.id: r for r in test_recs}
    preds = []
    for rid, dist in probs.items():
        r = by_id[rid]
        if r.primitive == "noul":
            preds.append(Prediction(id=rid, primitive="noul", p_yes=dist.get("1", 0.0), answer=dist.get("1", 0.0), model=f"{model_id} (Tier 1)"))
        else:
            keys = r.option_keys()
            pm = {k: float(dist.get(k, 0.0)) for k in keys}
            total = sum(pm.values()) or 1.0
            pm = {k: v / total for k, v in pm.items()}
            best = max(pm, key=pm.get)
            from jevify.wire import choice_confidence, score_confidence, score_expectation
            vals = [pm[k] for k in keys]
            if r.primitive == "choice":
                preds.append(Prediction(id=rid, primitive="choice", probabilities=pm, answer=best,
                                        confidence=choice_confidence(vals), model=f"{model_id} (Tier 1)"))
            else:
                preds.append(Prediction(id=rid, primitive="score", probabilities=pm, answer=score_expectation(vals),
                                        confidence=score_confidence(vals), model=f"{model_id} (Tier 1)"))
    write_predictions(out_dir / "test_predictions.jsonl", preds, progress_every=0)
    elapsed = time.time() - t0
    meta = {"run_id": run_id, "model_id": model_id, "gpu": gpu_name, "tier": 1, "layer": layer, "chat_applied": fx.chat,
            "heldout_sources": held, "n_train": len(train_rows), "n_val": len(val_rows), "n_test": len(test_rows),
            "max_slots": max_slots, "dim": dim, "epochs": epochs, "lr": lr, "residual": residual,
            "lm_weight": [round(float(x), 3) for x in heads.lm_weight.detach().cpu()],
            "best_epoch": info["best_epoch"], "best_val_loss": round(info["best_val_loss"], 4),
            "feature_s": round(feat_s), "train_s": round(train_s), "wall_s": round(elapsed, 1),
            "est_cost_usd": round(elapsed / 3600 * RATE_PER_HOUR.get(gpu_name, 2.5), 3),
            "val_detail": {k: round(v, 4) for k, v in evaluate_loss(heads, val_rows, device=device).items()}}
    (out_dir / "run.json").write_text(json.dumps(meta, indent=1))
    runs.commit()
    print(json.dumps(meta), flush=True)
    return meta


@app.function(gpu="L4", **GPU_KW)
def tier1_l4(*args, **kwargs) -> dict:
    return _tier1_impl("L4", *args, **kwargs)


@app.function(gpu="A100-80GB", **GPU_KW)
def tier1_a100(*args, **kwargs) -> dict:
    return _tier1_impl("A100-80GB", *args, **kwargs)


@app.local_entrypoint()
def tier1(model_id: str, run_id: str, gpu: str = "A100-80GB", layer: int = -1, chat: bool = True,
          train_per_source: int = 600, val_per_source: int = 150, max_slots: int = 16, dim: int = 512,
          epochs: int = 20, lr: float = 3e-4, batch: int = 8, trust_remote_code: bool = False, heldout: str = "",
          test_per_source: int = 0, residual: bool = True, out: str = "runs"):
    fn = tier1_a100 if gpu.startswith("A100") else tier1_l4
    info = fn.remote(model_id, run_id, layer, chat, train_per_source, val_per_source, max_slots, dim, epochs, lr,
                     batch, trust_remote_code, heldout, test_per_source, residual)
    local = Path(out) / run_id
    local.mkdir(parents=True, exist_ok=True)
    (local / "test_predictions.jsonl").write_bytes(fetch.remote(run_id, "test_predictions.jsonl"))
    (local / "run.json").write_text(json.dumps(info, indent=1))
    print(f"saved to {local}  (wall {info['wall_s']}s, est ${info['est_cost_usd']})")
