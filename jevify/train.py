"""Train and score a Jevified model anywhere — Tier 1, Tier 2, or a vision Tier 0 pass.

This orchestration used to live only inside the Modal app, which made the project's
headline claim — Jevify a model, then deploy it wherever you like — true for inference
but not for training. Nothing here imports Modal: every entry point takes an explicit
bench root and output directory, so the same code runs on a laptop, a lab GPU box or a
rented A100, and ``modal_app.py`` calls these functions instead of keeping a second copy.

    python -m jevify.train tier0  --model-id Qwen/Qwen3.5-2B --run-id qwen35-2b --split test
    python -m jevify.train tier1  --model-id Qwen/Qwen3.5-2B --run-id qwen35-2b-t1r
    python -m jevify.train tier2  --model-id Qwen/Qwen3.5-2B --run-id qwen35-2b-t2
    python -m jevify.train vision --model-id Qwen/Qwen3-VL-2B-Instruct --run-id qwen3vl-2b
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

BENCH = "Praveenrajus/jev-bench"

# Six sources kept out of training entirely, so the reported generalization number is
# measured on families the heads have never seen — not just on unseen rows of seen tasks.
HELDOUT_SOURCES = ["clinc150", "arc_challenge", "yelp5", "measuring_hate_speech",
                   "fever_evidence", "strategyqa_grounded"]


# --------------------------------------------------------------------------- bench data
def bench_root(dest: Path | str, repo: str = BENCH, token: str | None = None) -> Path:
    """Materialize jev-bench locally and return its root (a no-op once cached)."""
    from huggingface_hub import snapshot_download

    root = Path(dest)
    snapshot_download(repo, repo_type="dataset", local_dir=str(root),
                      token=token or os.environ.get("HF_TOKEN"),
                      allow_patterns=["data/*/test.jsonl", "data/*/validation.jsonl",
                                      "data/*/train.jsonl", "manifest.json"])
    return root


def load_split(root: Path, split: str, limit: int = 0, *, sources: Sequence[str] | None = None,
               exclude: Sequence[str] | None = None) -> list:
    """Read one split across every source directory, optionally filtered."""
    from .bench.record import read_jsonl

    recs = []
    for path in sorted((Path(root) / "data").glob(f"*/{split}.jsonl")):
        name = path.parent.name
        if sources is not None and name not in sources:
            continue
        if exclude and name in exclude:
            continue
        recs.extend(read_jsonl(path, limit=limit or None))
    return recs


def predictions_from_probs(probs: dict[str, dict[str, float]], test_recs: Sequence, label: str) -> list:
    """Turn head outputs into wire-shaped Predictions, in each record's own key space."""
    from .runners.base import Prediction
    from .wire import choice_confidence, score_confidence, score_expectation

    by_id = {r.id: r for r in test_recs}
    preds = []
    for rid, dist in probs.items():
        r = by_id[rid]
        if r.primitive == "noul":
            p_yes = float(dist.get("1", 0.0))
            preds.append(Prediction(id=rid, primitive="noul", p_yes=p_yes, answer=p_yes, model=label))
            continue
        keys = r.option_keys()
        pm = {k: float(dist.get(k, 0.0)) for k in keys}
        total = sum(pm.values()) or 1.0
        pm = {k: v / total for k, v in pm.items()}
        vals = [pm[k] for k in keys]
        if r.primitive == "choice":
            preds.append(Prediction(id=rid, primitive="choice", probabilities=pm, answer=max(pm, key=pm.get),
                                    confidence=choice_confidence(vals), model=label))
        else:
            preds.append(Prediction(id=rid, primitive="score", probabilities=pm, answer=score_expectation(vals),
                                    confidence=score_confidence(vals), model=label))
    return preds


def _finish(out_dir: Path, meta: dict[str, Any]) -> dict[str, Any]:
    (out_dir / "run.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta), flush=True)
    return meta


# --------------------------------------------------------------------------- Tier 0
def run_tier0(model_id: str, run_id: str, out_dir: Path | str, root: Path | str, *, sources: Sequence[str] | None = None,
              split: str = "test", limit: int = 0, mode: str = "index", chat: bool = True, permutations: int = 2,
              batch: int = 16, cand_chunk: int = 64, trust_remote_code: bool = False, resume: bool = True,
              seed: int = 0, state_last: bool = False, engine: str = "hf") -> dict[str, Any]:
    """No training: prompt, logit readout over the answer set, raw log-scores kept for a recipe.

    Resumable: predictions already on disk are skipped, so a killed run continues where it
    stopped rather than starting over.
    """
    import torch

    from .engine.predict import Recipe, Tier0Engine
    from .engine.readout import HFScorer
    from .runners.base import done_ids, write_predictions

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{split}_predictions.jsonl"
    recs = load_split(Path(root), split, limit, sources=list(sources) if sources else None)
    skip = done_ids(out) if resume and out.exists() else set()
    todo = [r for r in recs if r.id not in skip]
    print(f"[{run_id}] {model_id}: {len(todo)} records ({len(skip)} already done)", flush=True)

    t0 = time.time()
    if engine == "vllm":
        from .engine.vllm_readout import VLLMScorer

        scorer = VLLMScorer(model_id, trust_remote_code=trust_remote_code, hf_token=os.environ.get("HF_TOKEN"))
    else:
        scorer = HFScorer(model_id, dtype=torch.bfloat16, batch_size=batch, cand_chunk=cand_chunk,
                          trust_remote_code=trust_remote_code, hf_token=os.environ.get("HF_TOKEN"))
    t_load = time.time() - t0
    eng = Tier0Engine(scorer, Recipe(mode=mode, chat=chat, permutations=permutations, state_last=state_last))
    n = write_predictions(out, eng.score_records(todo, batch=batch), append=bool(skip), progress_every=250)
    return _finish(out_dir, {
        "run_id": run_id, "model_id": model_id, "tier": 0, "split": split, "sources": list(sources) if sources else None,
        "limit": limit, "recipe": eng.recipe.as_dict(), "chat_applied": eng.chat, "engine": engine, "n_new": n, "n_total": len(recs),
        "load_s": round(t_load, 1), "wall_s": round(time.time() - t0, 1), "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "scorer_stats": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in scorer.stats.items()},
        "attn_impl": getattr(getattr(getattr(scorer, "model", None), "config", None), "_attn_implementation", None),
        "dtype": str(next(scorer.model.parameters()).dtype) if hasattr(scorer, "model") else "bfloat16"})


# --------------------------------------------------------------------------- Tier 1
def run_tier1(model_id: str, run_id: str, out_dir: Path | str, root: Path | str, *, layer: int = -1,
              chat: bool = True, train_per_source: int = 600, val_per_source: int = 150, max_slots: int = 16,
              dim: int = 512, epochs: int = 20, lr: float = 3e-4, batch: int = 8,
              trust_remote_code: bool = False, heldout: Sequence[str] | None = None,
              test_per_source: int = 0, residual: bool = True, device: str | None = None,
              soft_labels: bool = False, seed: int = 0, features_cache: Path | str | None = None) -> dict[str, Any]:
    """Frozen backbone, trained decision heads. Features are extracted once and reused.

    ``features_cache`` is a directory: features are saved there after extraction and loaded
    from it on later calls, so retraining the heads -- another seed, another learning rate --
    costs minutes instead of another pass over 34,000 records through the backbone.
    ``seed`` varies head initialization and data order only; the slot subsampling plan is
    fixed, so seeds measure the variance of training, not of the data.
    """
    import numpy as np
    import torch

    from .engine.features import FeatureExtractor
    from .engine.heads import HeadConfig, predict_rows, save_heads, train_heads
    from .engine.readout import HFScorer
    from .runners.base import write_predictions

    held = list(HELDOUT_SOURCES if heldout is None else heldout)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(root)
    t0 = time.time()
    scorer = HFScorer(model_id, dtype=torch.bfloat16, trust_remote_code=trust_remote_code,
                      hf_token=os.environ.get("HF_TOKEN"))
    fx = FeatureExtractor(scorer, layer=layer, chat=chat)
    rng = np.random.default_rng(20260921)

    train_recs = load_split(root, "train", train_per_source, exclude=held)
    val_recs = load_split(root, "validation", val_per_source, exclude=held)
    test_recs = load_split(root, "test", test_per_source)
    print(f"[{run_id}] {model_id}: {len(train_recs)} train / {len(val_recs)} val / {len(test_recs)} test; "
          f"heldout={held}", flush=True)

    from .engine.features import load_features, save_features

    t_feat = time.time()
    cache = Path(features_cache) if features_cache else None
    if cache and (cache / "test.npz").exists():
        train_rows, val_rows, test_rows = (load_features(cache / f"{n}.npz") for n in ("train", "validation", "test"))
        print(f"[{run_id}] features loaded from {cache}", flush=True)
    else:
        train_rows = fx.extract(train_recs, max_slots=max_slots, rng=rng, batch_size=batch)
        val_rows = fx.extract(val_recs, max_slots=max_slots, rng=rng, batch_size=batch)
        test_rows = fx.extract(test_recs, batch_size=batch)
        if cache:
            cache.mkdir(parents=True, exist_ok=True)
            for n, rows in (("train", train_rows), ("validation", val_rows), ("test", test_rows)):
                save_features(cache / f"{n}.npz", rows)
    feat_s = time.time() - t_feat
    print(f"[{run_id}] features in {feat_s:.0f}s", flush=True)

    cfg = HeadConfig(hidden=fx.hidden, dim=dim, layer=layer, backbone=model_id, residual=residual,
                     soft_labels=soft_labels)
    dev = device or scorer.device
    t_train = time.time()
    heads, info = train_heads(train_rows, val_rows, cfg, epochs=epochs, lr=lr, device=dev, seed=seed, verbose=True)
    train_s = time.time() - t_train
    save_heads(heads, info, out_dir / "heads")

    probs = predict_rows(heads, test_rows, device=dev)
    preds = predictions_from_probs(probs, test_recs, f"{model_id} (Tier 1)")
    write_predictions(out_dir / "test_predictions.jsonl", preds, progress_every=0)
    return _finish(out_dir, {
        "run_id": run_id, "model_id": model_id, "tier": 1, "layer": layer, "chat_applied": fx.chat,
        "heldout_sources": held, "n_train": len(train_recs), "n_val": len(val_recs), "n_test": len(test_recs),
        "max_slots": max_slots, "dim": dim, "epochs": epochs, "lr": lr, "residual": residual, "soft_labels": soft_labels, "seed": seed,
        "lm_weight": [round(float(x), 3) for x in heads.lm_weight.detach().cpu()],
        "best_epoch": info["best_epoch"], "best_val_loss": round(info["best_val_loss"], 4),
        "feat_s": round(feat_s, 1), "train_s": round(train_s, 1), "wall_s": round(time.time() - t0, 1),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"})


# --------------------------------------------------------------------------- Tier 2
def run_tier2(model_id: str, run_id: str, out_dir: Path | str, root: Path | str, *, layer: int = -1,
              chat: bool = True, train_per_source: int = 400, val_per_source: int = 100, max_slots: int = 16,
              dim: int = 512, epochs: int = 3, head_lr: float = 3e-4, lora_lr: float = 1e-4, lora_r: int = 16,
              batch: int = 4, grad_accum: int = 2, trust_remote_code: bool = False,
              heldout: Sequence[str] | None = None, test_per_source: int = 0,
              soft_labels: bool = False, seed: int = 0) -> dict[str, Any]:
    """LoRA on the backbone, trained jointly with the residual decision heads."""
    import torch

    from .engine.features import FeatureExtractor
    from .engine.heads import HeadConfig, save_heads
    from .engine.readout import HFScorer
    from .engine.tier2 import DifferentiableSlots, apply_lora, predict_tier2, train_tier2
    from .runners.base import write_predictions

    held = list(HELDOUT_SOURCES if heldout is None else heldout)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(root)
    t0 = time.time()
    scorer = HFScorer(model_id, dtype=torch.bfloat16, trust_remote_code=trust_remote_code,
                      hf_token=os.environ.get("HF_TOKEN"))
    fx = FeatureExtractor(scorer, layer=layer, chat=chat)
    peft_model, trainable = apply_lora(scorer.model, r=lora_r, alpha=2 * lora_r)
    scorer.model = peft_model
    fx.rebind()                      # the decoder stack moved under PEFT's wrapper

    train_recs = load_split(root, "train", train_per_source, exclude=held)
    val_recs = load_split(root, "validation", val_per_source, exclude=held)
    test_recs = load_split(root, "test", test_per_source)
    print(f"[{run_id}] {model_id}: {len(train_recs)} train / {len(val_recs)} val / {len(test_recs)} test; "
          f"heldout={held}", flush=True)

    cfg = HeadConfig(hidden=fx.hidden, dim=dim, layer=layer, backbone=model_id, residual=True,
                     soft_labels=soft_labels)
    heads, info = train_tier2(fx, train_recs, val_recs, cfg, epochs=epochs, batch_size=batch,
                              grad_accum=grad_accum, head_lr=head_lr, lora_lr=lora_lr, max_slots=max_slots,
                              checkpoint_dir=out_dir, seed=seed)
    save_heads(heads, info, out_dir / "heads")
    scorer.model.save_pretrained(str(out_dir / "lora"))

    slots = DifferentiableSlots(fx)
    probs = predict_tier2(slots, heads, test_recs, batch_size=max(batch, 8))
    preds = predictions_from_probs(probs, test_recs, f"{model_id} (Tier 2)")
    write_predictions(out_dir / "test_predictions.jsonl", preds, progress_every=0)
    return _finish(out_dir, {
        "run_id": run_id, "model_id": model_id, "tier": 2, "residual": True, "layer": layer,
        "chat_applied": fx.chat, "heldout_sources": held, "lora_r": lora_r, "lora_trainable": trainable,
        "n_train": len(train_recs), "n_val": len(val_recs), "n_test": len(test_recs), "max_slots": max_slots,
        "dim": dim, "epochs": epochs, "head_lr": head_lr, "lora_lr": lora_lr, "soft_labels": soft_labels, "seed": seed,
        "best_epoch": info["best_epoch"], "best_val_loss": round(info["best_val_loss"], 4),
        "lm_weight": [round(float(x), 3) for x in heads.lm_weight.detach().cpu()],
        "wall_s": round(time.time() - t0, 1),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"})


# --------------------------------------------------------------------------- vision Tier 0 / Tier 2
def run_vision(model_id: str, run_id: str, out_dir: Path | str, *, sources: Sequence[str] | None = None,
               split: str = "test", limit: int = 0, batch: int = 8, max_pixels: int = 0,
               trust_remote_code: bool = False, lora: str = "", train_sources: Sequence[str] | None = None,
               epochs: int = 3, lr: float = 1e-4, lora_r: int = 16, grad_accum: int = 2, seed: int = 0,
               soft_labels: bool = False) -> dict[str, Any]:
    """Score a vision-language model on the vision configs of jev-bench.

    Untrained by default (Tier 0). With ``lora`` set to ``vision``, ``decoder`` or ``both``,
    a LoRA confined to that half of the model is first trained on the readout over the
    ``train_sources`` train split (Tier 2), early-stopped on their validation split; the
    validation *and* test splits of every source are then scored, so the recipe can be
    fitted on validation and the held-out sources read on test.
    """
    import torch

    from .bench.adapters import VISION_REGISTRY
    from .engine.vision import VisionScorer, split_images
    from .engine.vision_backbone import describe, image_tokens, pixel_budget
    from .runners.base import write_predictions
    from .runners.vision_runner import VisionRunner, build_vision_records, write_vision_records

    want = list(sources) if sources else list(VISION_REGISTRY)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    recs = build_vision_records(want, split, limit)
    print(f"[{run_id}] {model_id}: {len(recs)} vision records from {want}", flush=True)
    scorer = VisionScorer(model_id, dtype=torch.bfloat16, batch_size=batch,
                          max_pixels=max_pixels or None, trust_remote_code=trust_remote_code,
                          hf_token=os.environ.get("HF_TOKEN"))
    t_load = time.time() - t0

    trained: dict[str, Any] = {}
    if lora:
        from .engine.vision_tier2 import attach_lora, train_vision_tier2

        tr_src = list(train_sources) if train_sources else [
            s for s in want if VISION_REGISTRY[s].spec.caps.get("train", 0) > 0]
        train_recs = build_vision_records(tr_src, "train", 0)
        stop_recs = build_vision_records(tr_src, "validation", 0)
        pattern, trainable = attach_lora(scorer, where=lora, r=lora_r)
        print(f"[{run_id}] LoRA({lora}) on {len(train_recs)} train records from {tr_src}, "
              f"early stopping on {len(stop_recs)}", flush=True)
        info = train_vision_tier2(scorer, train_recs, stop_recs, epochs=epochs, batch_size=batch,
                                  grad_accum=grad_accum, lr=lr, seed=seed, soft_labels=soft_labels,
                                  checkpoint_dir=out_dir)
        scorer.model.save_pretrained(str(out_dir / "lora"))
        trained = {"tier": 2, "lora_where": lora, "lora_pattern": pattern, "lora_r": lora_r,
                   "lora_trainable": trainable, "train_sources": tr_src, "n_train": len(train_recs),
                   "n_val": len(stop_recs), "epochs": epochs, "lr": lr, "grad_accum": grad_accum,
                   "seed": seed, "soft_labels": soft_labels, "heldout_sources": [s for s in want if s not in tr_src],
                   **info}
        # the recipe is fitted on validation predictions from the *adapted* model
        val_recs = build_vision_records(want, "validation", 0)
        n_val = write_predictions(out_dir / "validation_predictions.jsonl",
                                  VisionRunner(scorer).predict(val_recs, batch=batch), progress_every=200)
        write_vision_records(out_dir / "validation_records.jsonl", val_recs)
        trained["n_validation_scored"] = n_val

    # A pixel budget that was *set* is not necessarily a budget that *took*: processors
    # ignore keys they do not have. Record what applied, and the token count it actually
    # produced, so a sweep over budgets compares measured visual signal and not intent.
    budget = pixel_budget(scorer.processor, max_pixels=max_pixels or None)
    sample = [im for r in recs[:24] for im in split_images(r.state)[1]][:24]
    if sample:
        counts = sorted(image_tokens(scorer.processor, im) for im in sample)
        budget["image_tokens"] = {"median": counts[len(counts) // 2], "min": counts[0], "max": counts[-1],
                                  "n_sampled": len(counts)}

    n = write_predictions(out_dir / f"{split}_predictions.jsonl",
                          VisionRunner(scorer).predict(recs, batch=batch), progress_every=200)
    # the records carry PIL images; write_vision_records strips them before the row is built
    write_vision_records(out_dir / f"{split}_records.jsonl", recs)
    return _finish(out_dir, {
        "run_id": run_id, "model_id": model_id, "modality": "vision", "tier": 0, "sources": want,
        "split": split, "n": n, "max_pixels": max_pixels or None, "budget": budget,
        "vision_stage": describe(scorer.model), "load_s": round(t_load, 1),
        **trained,
        "wall_s": round(time.time() - t0, 1),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"})



def describe_vision(model_id: str, *, trust_remote_code: bool = False) -> dict[str, Any]:
    """Inventory a VLM's vision stage without scoring anything.

    Loaded on CPU: this answers a structural question, so it should not need a GPU or
    wait behind one.
    """
    import torch

    from .engine.vision_backbone import describe

    try:
        from transformers import AutoModelForImageTextToText as _AutoVLM
    except ImportError:                                  # pragma: no cover - older transformers
        from transformers import AutoModelForVision2Seq as _AutoVLM

    model = _AutoVLM.from_pretrained(model_id, dtype=torch.float32, trust_remote_code=trust_remote_code,
                                     token=os.environ.get("HF_TOKEN"))
    return {"model_id": model_id, **describe(model)}


# --------------------------------------------------------------------------- CLI
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="jevify.train",
                                 description="Jevify a model: Tier 1, Tier 2, or a vision Tier 0 pass.")
    ap.add_argument("tier", choices=["tier0", "tier1", "tier2", "vision"])
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", default="runs", help="parent directory for <run-id>/")
    ap.add_argument("--bench", default="", help="local jev-bench root; downloaded under --out if omitted")
    ap.add_argument("--bench-repo", default=BENCH)
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--no-chat", action="store_true", help="skip the model's chat template")
    ap.add_argument("--train-per-source", type=int, default=0)
    ap.add_argument("--val-per-source", type=int, default=0)
    ap.add_argument("--test-per-source", type=int, default=0)
    ap.add_argument("--max-slots", type=int, default=16)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4, help="tier1 head learning rate")
    ap.add_argument("--head-lr", type=float, default=3e-4)
    ap.add_argument("--lora-lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--replace", action="store_true", help="tier1: heads replace the LM score (default: residual)")
    ap.add_argument("--seeds", default="0", help="tier1: comma-separated seeds; features are extracted once and cached")
    ap.add_argument("--seed", type=int, default=0, help="tier2: training seed (init, data order, slot subsampling)")
    ap.add_argument("--soft-labels", action="store_true",
                    help="train against human label distributions where a source has them")
    ap.add_argument("--heldout", default="", help="comma-separated; defaults to the standard six")
    ap.add_argument("--sources", default="", help="tier0/vision: comma-separated source names")
    ap.add_argument("--mode", default="index", help="tier0: how options are presented (index|...)")
    ap.add_argument("--permutations", type=int, default=2, help="tier0: option-order permutations to average")
    ap.add_argument("--cand-chunk", type=int, default=64)
    ap.add_argument("--no-resume", action="store_true", help="tier0: ignore predictions already on disk")
    ap.add_argument("--state-last", action="store_true", help="tier0: question and options before the state (cacheable prefix)")
    ap.add_argument("--engine", default="hf", choices=["hf", "vllm"], help="tier0: research scorer or the vLLM serving path")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-pixels", type=int, default=0)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--lora", default="", choices=["", "vision", "decoder", "both"],
                    help="vision: train a LoRA confined to this half of the model on the readout (Tier 2)")
    ap.add_argument("--train-sources", default="", help="vision --lora: comma-separated; defaults to sources with a train split")
    ap.add_argument("--describe", action="store_true",
                    help="vision: print the model's vision-stage inventory and exit")
    a = ap.parse_args(argv)

    out_dir = Path(a.out) / a.run_id
    held = [s for s in a.heldout.split(",") if s] or None
    common = dict(model_id=a.model_id, run_id=a.run_id, out_dir=out_dir,
                  trust_remote_code=a.trust_remote_code)

    if a.tier == "vision" and a.describe:
        print(json.dumps(describe_vision(a.model_id, trust_remote_code=a.trust_remote_code), indent=1))
        return 0

    if a.tier == "vision":
        run_vision(**common, sources=[s for s in a.sources.split(",") if s] or None, split=a.split,
                   limit=a.limit, batch=a.batch or (4 if a.lora else 8), max_pixels=a.max_pixels,
                   lora=a.lora, train_sources=[s for s in a.train_sources.split(",") if s] or None,
                   epochs=a.epochs or 3, lr=a.lora_lr, lora_r=a.lora_r, grad_accum=a.grad_accum,
                   seed=a.seed, soft_labels=a.soft_labels)
        return 0

    root = Path(a.bench) if a.bench else bench_root(Path(a.out) / "jev-bench", a.bench_repo)
    if a.tier == "tier0":
        run_tier0(**common, root=root, sources=[s for s in a.sources.split(",") if s] or None, split=a.split,
                  limit=a.limit, mode=a.mode, chat=not a.no_chat, permutations=a.permutations,
                  batch=a.batch or 16, cand_chunk=a.cand_chunk, resume=not a.no_resume,
                  state_last=a.state_last, engine=a.engine)
        return 0
    if a.tier == "tier1":
        seeds = [int(x) for x in a.seeds.split(",") if x]
        for seed in seeds:
            rid = a.run_id if len(seeds) == 1 else f"{a.run_id}-s{seed}"
            run_tier1(model_id=a.model_id, run_id=rid, out_dir=Path(a.out) / rid, trust_remote_code=a.trust_remote_code,
                      root=root, layer=a.layer, chat=not a.no_chat,
                      train_per_source=a.train_per_source or 600, val_per_source=a.val_per_source or 150,
                      max_slots=a.max_slots, dim=a.dim, epochs=a.epochs or 20, lr=a.lr, batch=a.batch or 8,
                      heldout=held, test_per_source=a.test_per_source, residual=not a.replace,
                      soft_labels=a.soft_labels, seed=seed, features_cache=Path(a.out) / f"{a.run_id}-features")
    else:
        run_tier2(**common, root=root, layer=a.layer, chat=not a.no_chat,
                  train_per_source=a.train_per_source or 400, val_per_source=a.val_per_source or 100,
                  max_slots=a.max_slots, dim=a.dim, epochs=a.epochs or 3, head_lr=a.head_lr,
                  lora_lr=a.lora_lr, lora_r=a.lora_r, batch=a.batch or 4, grad_accum=a.grad_accum,
                  heldout=held, test_per_source=a.test_per_source, soft_labels=a.soft_labels, seed=a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
