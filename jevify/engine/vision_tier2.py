"""Tier 2 for vision: let part of the VLM move, with LoRA scoped to one half of it.

The text Tier 2 trains LoRA jointly with residual heads that read per-option slot
features. A VLM's prompt has no stable slot positions — image tokens expand inside it —
so the vision variant trains the *readout itself*: the restricted log-softmax over the
candidate tokens at the answer position, the very quantity Tier 0 reads. Nothing is added
to the model; the loss is the proper scoring rule of the primitive (log score for Choice
and Noul, ranked probability score for Score), the adapter merges into the backbone, and
the served path is unchanged. The calibration recipe is fitted afterwards on validation
predictions exactly as for Tier 0.

What the scope buys is the question this module exists to answer. ``where="vision"``
adapts only the encoder (and can only change *what the model sees*), ``"decoder"`` only
the language model, ``"both"`` the whole thing. Section 9 of the findings makes the tower
addressable; this is where it is exercised.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..bench.record import BenchRecord
from .features import _pad_soft, soft_targets
from .heads import ranked_probability_loss
from .tier2 import _load_lora_state, _lora_state, apply_lora
from .vision import VisionScorer
from .vision_backbone import lora_pattern

NEG = -1e4


@dataclass
class ReadoutBatch:
    """Log-scores over each row's candidates, still attached to the graph."""
    logscores: torch.Tensor      # (B, S) restricted log-softmax over the row's candidates
    mask: torch.Tensor           # (B, S) bool
    label: torch.Tensor          # (B,) gold index into the row's keys
    primitive: list[str]
    ids: list[str]
    keys: list[list[str]]
    soft: torch.Tensor           # (B, S) human distribution where the source has one
    has_soft: torch.Tensor       # (B,) bool


class VisionReadout:
    """Runs the VLM *with gradients* and returns the Tier 0 readout as tensors."""

    def __init__(self, scorer: VisionScorer, *, mode: str = "index") -> None:
        self.scorer = scorer
        self.mode = mode
        self._keep_kw: dict[str, Any] | None = None

    def _keep(self) -> dict[str, Any]:
        # logits at the answer position only: a batch x seq x vocab tensor, plus its autograd
        # graph, is an out-of-memory error dressed as a batch-size problem
        if self._keep_kw is None:
            import inspect

            try:
                params = inspect.signature(self.scorer.model.forward).parameters
            except (TypeError, ValueError):
                params = {}
            self._keep_kw = {"logits_to_keep": 1} if "logits_to_keep" in params or any(
                p.kind == p.VAR_KEYWORD for p in params.values()) else {}
        return self._keep_kw

    def batch(self, records: Sequence[BenchRecord]) -> ReadoutBatch:
        sc = self.scorer
        items, plans = [], []
        for r in records:
            it, rd = sc.item(r.state, r.question, mode=self.mode)
            keys = list(rd.keys)
            toks = [sc._candidate_token(it.prefix, c) for c in it.candidates]
            plans.append((keys, keys.index(str(r.label)), toks, soft_targets(r, keys)))
            items.append(it)
        texts = [it.prefix for it in items]
        images = [it.images for it in items]
        kwargs: dict[str, Any] = {"text": texts, "return_tensors": "pt", "padding": True, "padding_side": "left"}
        if any(images):
            kwargs["images"] = images
        enc = sc.processor(**kwargs)
        enc = {k: (v.to(sc.device) if hasattr(v, "to") else v) for k, v in enc.items()}
        try:
            out = sc.model(**enc, use_cache=False, **self._keep())
        except TypeError:                                   # a forward that rejects logits_to_keep
            self._keep_kw = {}
            out = sc.model(**enc, use_cache=False)
        logp = F.log_softmax(out.logits[:, -1].float(), dim=-1)        # (B, vocab)

        B, S = len(plans), max(len(p[0]) for p in plans)
        dev = logp.device
        ls = torch.full((B, S), NEG, dtype=torch.float32, device=dev)
        mask = torch.zeros(B, S, dtype=torch.bool, device=dev)
        label = torch.zeros(B, dtype=torch.long, device=dev)
        for row, (keys, gold, toks, _soft) in enumerate(plans):
            n = len(keys)
            picked = logp[row, torch.tensor(toks, device=dev)]
            ls[row, :n] = picked - torch.logsumexp(picked, dim=0)      # renormalize over the answer set
            mask[row, :n] = True
            label[row] = gold
        soft, has_soft = _pad_soft([p[3] for p in plans], S)
        return ReadoutBatch(ls, mask, label, [r.primitive for r in records], [r.id for r in records],
                            [p[0] for p in plans], soft.to(dev), has_soft.to(dev))


def readout_loss(batch: ReadoutBatch, *, soft_labels: bool = False) -> tuple[torch.Tensor, dict[str, float]]:
    """The primitive's proper scoring rule on the restricted readout: log score for Choice and
    Noul (a two-way softmax, so the same rule), RPS over the ordered levels for Score."""
    total = batch.logscores.new_zeros(())
    parts: dict[str, float] = {}
    counts: dict[str, int] = {}
    S = batch.logscores.shape[-1]
    for prim in ("choice", "score", "noul"):
        idx = [i for i, p in enumerate(batch.primitive) if p == prim]
        if not idx:
            continue
        sel = torch.tensor(idx, device=batch.logscores.device)
        target = F.one_hot(batch.label[sel], num_classes=S).float()
        if soft_labels:
            use = batch.has_soft[sel]
            if bool(use.any()):
                target = torch.where(use[:, None], batch.soft[sel], target)
        if prim == "score":
            loss = ranked_probability_loss(batch.logscores[sel].masked_fill(~batch.mask[sel], NEG), target, batch.mask[sel])
        else:
            loss = -(target * batch.logscores[sel] * batch.mask[sel]).sum(-1).mean()
        total = total + loss * len(idx)
        parts[prim] = float(loss.detach())
        counts[prim] = len(idx)
    n = sum(counts.values()) or 1
    return total / n, {**parts, **{f"n_{k}": v for k, v in counts.items()}}


def attach_lora(scorer: VisionScorer, *, where: str = "vision", r: int = 16, alpha: int = 32,
                dropout: float = 0.05) -> tuple[str, int]:
    """Wrap the scorer's model in LoRA confined to ``where``; returns (pattern, trainable params)."""
    pattern = lora_pattern(scorer.model, where)
    scorer.model, trainable = apply_lora(scorer.model, r=r, alpha=alpha, dropout=dropout, targets=pattern)
    return pattern, trainable


def train_vision_tier2(scorer: VisionScorer, train_recs: Sequence[BenchRecord], val_recs: Sequence[BenchRecord], *,
                       epochs: int = 3, batch_size: int = 4, grad_accum: int = 2, lr: float = 1e-4,
                       seed: int = 0, soft_labels: bool = False, mode: str = "index",
                       checkpoint_dir: Path | str | None = None) -> dict[str, Any]:
    """LoRA on the readout. The best epoch by validation loss is what remains in the model."""
    torch.manual_seed(seed)
    readout = VisionReadout(scorer, mode=mode)
    params = [p for p in scorer.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps = max(1, math.ceil(len(train_recs) / (batch_size * grad_accum))) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.2)
    history: list[dict[str, Any]] = []
    best: tuple[float, dict[str, torch.Tensor] | None, int] = (float("inf"), None, -1)

    for epoch in range(epochs):
        scorer.model.train()
        order = list(np.random.default_rng(seed + epoch).permutation(len(train_recs)))
        total, seen, t0 = 0.0, 0, time.time()
        opt.zero_grad(set_to_none=True)
        for i, start in enumerate(range(0, len(order), batch_size)):
            chunk = [train_recs[j] for j in order[start:start + batch_size]]
            loss, _ = readout_loss(readout.batch(chunk), soft_labels=soft_labels)
            (loss / grad_accum).backward()
            total += float(loss.detach()) * len(chunk)
            seen += len(chunk)
            if (i + 1) % grad_accum == 0:
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if seen % (batch_size * 50) == 0:
                print(f"  epoch {epoch} {seen}/{len(order)} loss {total/max(seen,1):.4f} "
                      f"({seen/(time.time()-t0):.1f} rec/s)", flush=True)
        val = evaluate_vision_tier2(readout, val_recs, batch_size=batch_size, soft_labels=soft_labels)
        history.append({"epoch": epoch, "train": round(total / max(seen, 1), 4), **{k: round(v, 4) for k, v in val.items()}})
        print(f"  epoch {epoch}: train {history[-1]['train']:.4f} val {val['loss']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if val["loss"] < best[0]:
            best = (val["loss"], {k: v.detach().cpu().clone() for k, v in _lora_state(scorer.model).items()}, epoch)
            if checkpoint_dir is not None:
                ck = Path(checkpoint_dir)
                scorer.model.save_pretrained(str(ck / "lora"))
                print(f"  checkpoint: epoch {epoch} written to {ck / 'lora'}", flush=True)
    if best[1] is not None:
        _load_lora_state(scorer.model, best[1])
    scorer.model.eval()
    return {"history": history, "best_epoch": best[2], "best_val_loss": best[0]}


@torch.inference_mode()
def evaluate_vision_tier2(readout: VisionReadout, recs: Sequence[BenchRecord], *, batch_size: int = 4,
                          soft_labels: bool = False) -> dict[str, float]:
    readout.scorer.model.eval()
    tot, n = 0.0, 0
    parts: dict[str, list[float]] = {}
    for start in range(0, len(recs), batch_size):
        chunk = recs[start:start + batch_size]
        loss, detail = readout_loss(readout.batch(chunk), soft_labels=soft_labels)
        tot += float(loss) * len(chunk)
        n += len(chunk)
        for k, v in detail.items():
            if not k.startswith("n_"):
                parts.setdefault(k, []).append(v)
    out = {"loss": tot / max(n, 1)}
    out.update({k: sum(v) / len(v) for k, v in parts.items()})
    return out
