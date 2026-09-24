"""Tier 2: let the backbone move, with LoRA, jointly with the decision heads.

Tier 1 reads slot features from a frozen backbone. That asks the backbone to already
*have* the judgment at the option's position — it was never trained to put it there. Tier 2
adds LoRA adapters so the backbone can, while the head keeps its residual on the model's own
log-score so the prior is still preserved (the Tier 1 finding).

Features cannot be cached here: the backbone changes every step, so each batch re-runs the
forward pass with gradients. That is the whole cost difference between Tier 1 and Tier 2.
"""
from __future__ import annotations

import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..bench.record import BenchRecord
from .features import FeatureExtractor, SlotBatch, _pad_soft
from .heads import DecisionHeads, HeadConfig, evaluate_loss, save_heads


def apply_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05,
               targets: Sequence[str] | str | None = None):
    """Wrap the backbone in LoRA adapters and freeze everything else.

    ``targets`` is either a list of module-name suffixes or a full-path regex. The regex
    form is what scopes adaptation to one half of a VLM: peft matches a suffix list
    against the end of a module path, and names like ``q_proj`` live on both the vision
    tower and the decoder, so a suffix list cannot express "the encoder only"
    (see ``jevify.engine.vision_backbone.lora_pattern``).
    """
    from peft import LoraConfig, get_peft_model

    if isinstance(targets, str):
        target_modules: Any = targets                   # peft reads a str as a regex
        shown = targets
    else:
        if targets is None:
            # the projections every decoder family shares; missing names are ignored by peft
            targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        names = {n.split(".")[-1] for n, _ in model.named_modules()}
        target_modules = [t for t in targets if t in names] or ["q_proj", "v_proj"]
        shown = sorted(target_modules)
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                     task_type="CAUSAL_LM", target_modules=target_modules)
    peft_model = get_peft_model(model, cfg)
    # storing activations for every layer's backward is what actually fills an 80 GB card
    # here; checkpointing trades ~30% compute for a large multiple of memory
    if hasattr(peft_model, "gradient_checkpointing_enable"):
        peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(peft_model, "enable_input_require_grads"):
        peft_model.enable_input_require_grads()      # PEFT needs this for checkpointing to pass grads
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"[tier2] LoRA on {shown}: {trainable:,} trainable of {total:,} ({100*trainable/total:.2f}%)", flush=True)
    return peft_model, trainable


class DifferentiableSlots:
    """Runs the backbone *with gradients* and returns a SlotBatch still in the graph.

    Mirrors FeatureExtractor exactly — same plan, same positions, same LM scores — but
    keeps tensors attached so LoRA can learn.
    """

    def __init__(self, extractor: FeatureExtractor) -> None:
        self.fx = extractor
        self.scorer = extractor.scorer

    def batch(self, records: Sequence[BenchRecord], *, max_slots: int = 0,
              rng: np.random.Generator | None = None) -> SlotBatch:
        """One forward pass, gradients intact.

        Left-padded so every sequence ends at the same index: that lets the model compute
        logits for the final position only (``logits_to_keep=1``). Asking for logits at every
        position would materialize batch x seq x vocab — 7 GB for 8 sequences on a 151k
        vocabulary, plus its autograd graph — which is an out-of-memory error, not a batch-size
        problem. Slot indices are shifted by each row's pad offset to compensate.
        """
        rng = rng or np.random.default_rng(0)
        ids_list = self.scorer.identifiers()
        plans = [self.fx._plan(r, max_slots, rng, ids_list) for r in records]
        pad = self.scorer.pad_id
        width = max(len(p[0]) for p in plans)
        dev = self.scorer.device
        B = len(plans)
        ids = torch.full((B, width), pad, dtype=torch.long)
        attn = torch.zeros((B, width), dtype=torch.long)
        offsets = []
        for row, (seq, *_rest) in enumerate(plans):
            off = width - len(seq)
            offsets.append(off)
            ids[row, off:] = torch.tensor(seq)
            attn[row, off:] = 1
        pos_ids = (attn.cumsum(-1) - 1).clamp(min=0)
        self.fx._captured = None
        res = self.scorer.model(input_ids=ids.to(dev), attention_mask=attn.to(dev),
                                position_ids=pos_ids.to(dev), use_cache=False,
                                **self.scorer._keep(1))
        hs = self.fx._captured
        assert hs is not None, "forward hook did not fire"

        S = max(len(p[2]) for p in plans)
        H = hs.shape[-1]
        # the backbone may run in bf16; the heads and losses are fp32, and casting here keeps
        # gradients flowing while avoiding a dtype mismatch inside the head
        dec = torch.zeros(B, H, dtype=torch.float32, device=dev)
        slots = torch.zeros(B, S, H, dtype=torch.float32, device=dev)
        lm = torch.full((B, S), -20.0, dtype=torch.float32, device=dev)
        mask = torch.zeros(B, S, dtype=torch.bool, device=dev)
        label = torch.zeros(B, dtype=torch.long, device=dev)
        last_logits = res.logits[:, -1].float()               # (B, vocab) only
        logp = torch.log_softmax(last_logits, dim=-1)
        for row, (seq, pos, keys, lab, cand, _soft) in enumerate(plans):
            off = offsets[row]
            n = len(keys)
            dec[row] = hs[row, -1].float()
            slots[row, :n] = hs[row, torch.tensor([p + off for p in pos], device=dev)].float()
            lm[row, :n] = logp[row, torch.tensor(cand, device=dev)]
            mask[row, :n] = True
            label[row] = lab
        soft, has_soft = _pad_soft([p[5] for p in plans], S)
        return SlotBatch(dec, slots, lm, mask, label, [r.primitive for r in records],
                         [r.id for r in records], [p[2] for p in plans], soft.to(dev), has_soft.to(dev))


def shuffle_choice_options(records: Sequence[BenchRecord], rng: np.random.Generator) -> list[BenchRecord]:
    """A copy of each Choice record with its options in a random order; Score levels keep their order
    (the order is the meaning) and Noul has none. Rendered in source order, a class sits at the same
    position in every example, and a model can learn the position instead of the option."""
    out = []
    for r in records:
        crit = r.question.get("criteria") if r.primitive == "choice" else None
        if isinstance(crit, dict) and len(crit) > 1:
            keys = list(crit)
            keys = [keys[i] for i in rng.permutation(len(keys))]
            r = replace(r, question={**r.question, "criteria": {k: crit[k] for k in keys}})
        out.append(r)
    return out


def train_tier2(extractor: FeatureExtractor, train_recs: Sequence[BenchRecord], val_recs: Sequence[BenchRecord],
                cfg: HeadConfig, *, epochs: int = 3, batch_size: int = 4, grad_accum: int = 2,
                head_lr: float = 3e-4, lora_lr: float = 1e-4, max_slots: int = 16, seed: int = 0,
                eval_every: int = 1, checkpoint_dir: Path | str | None = None,
                shuffle_options: bool = False) -> tuple[DecisionHeads, dict[str, Any]]:
    """Joint LoRA + head training. The head keeps its residual on the LM log-score.

    ``checkpoint_dir``: the best-so-far heads and adapter are written there at the end of
    every epoch that improves validation loss. A run that dies mid-way -- a host reboot took
    a six-hour 4B run with it once -- then loses at most the epoch in progress, not the run.

    ``shuffle_options``: each training batch sees its Choice options in a fresh random order
    (validation keeps the source order, so runs stay comparable).
    """
    torch.manual_seed(seed)
    dev = extractor.scorer.device
    heads = DecisionHeads(cfg).to(dev)
    slots = DifferentiableSlots(extractor)
    lora_params = [p for p in extractor.scorer.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": heads.parameters(), "lr": head_lr},
                             {"params": lora_params, "lr": lora_lr}], weight_decay=0.01)
    steps = max(1, math.ceil(len(train_recs) / (batch_size * grad_accum))) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[head_lr, lora_lr], total_steps=steps, pct_start=0.2)
    rng = np.random.default_rng(seed)
    # its own stream, so a shuffled run draws the same slot subsamples as an unshuffled one
    shuffle_rng = np.random.default_rng([seed, 1])
    history, best = [], (float("inf"), None, None, -1)

    for epoch in range(epochs):
        extractor.scorer.model.train()
        heads.train()
        order = list(np.random.default_rng(seed + epoch).permutation(len(train_recs)))
        total, seen, t0 = 0.0, 0, time.time()
        opt.zero_grad(set_to_none=True)
        for i, start in enumerate(range(0, len(order), batch_size)):
            chunk = [train_recs[j] for j in order[start:start + batch_size]]
            if shuffle_options:
                chunk = shuffle_choice_options(chunk, shuffle_rng)
            batch = slots.batch(chunk, max_slots=max_slots, rng=rng)
            loss, _ = heads.loss(batch)
            (loss / grad_accum).backward()
            total += float(loss.detach()) * len(chunk)
            seen += len(chunk)
            if (i + 1) % grad_accum == 0:
                nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if seen % (batch_size * 40) == 0:
                print(f"  epoch {epoch} {seen}/{len(order)} loss {total/max(seen,1):.4f} "
                      f"({seen/(time.time()-t0):.1f} rec/s)", flush=True)
        val = evaluate_tier2(slots, heads, val_recs, batch_size=batch_size)
        history.append({"epoch": epoch, "train": round(total / max(seen, 1), 4), **{k: round(v, 4) for k, v in val.items()}})
        print(f"  epoch {epoch}: train {history[-1]['train']:.4f} val {val['loss']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if val["loss"] < best[0]:
            from copy import deepcopy

            best = (val["loss"], {k: v.detach().clone() for k, v in heads.state_dict().items()},
                    deepcopy({k: v.detach().cpu() for k, v in _lora_state(extractor.scorer.model).items()}), epoch)
            if checkpoint_dir is not None:
                ck = Path(checkpoint_dir)
                save_heads(heads, {"history": history, "best_epoch": epoch, "best_val_loss": val["loss"],
                                   "config": cfg.as_dict(), "partial": True}, ck / "heads")
                extractor.scorer.model.save_pretrained(str(ck / "lora"))
                print(f"  checkpoint: epoch {epoch} written to {ck}", flush=True)
    if best[1] is not None:
        heads.load_state_dict(best[1])
        _load_lora_state(extractor.scorer.model, best[2])
    return heads, {"history": history, "best_epoch": best[3], "best_val_loss": best[0], "config": cfg.as_dict()}


@torch.inference_mode()
def evaluate_tier2(slots: DifferentiableSlots, heads: DecisionHeads, recs: Sequence[BenchRecord],
                   batch_size: int = 4) -> dict[str, float]:
    slots.scorer.model.eval()
    heads.eval()
    tot, n = 0.0, 0
    parts: dict[str, list[float]] = {}
    for start in range(0, len(recs), batch_size):
        chunk = recs[start:start + batch_size]
        batch = slots.batch(chunk)
        loss, detail = heads.loss(batch)
        tot += float(loss) * len(chunk)
        n += len(chunk)
        for k, v in detail.items():
            if not k.startswith("n_"):
                parts.setdefault(k, []).append(v)
    out = {"loss": tot / max(n, 1)}
    out.update({k: sum(v) / len(v) for k, v in parts.items()})
    return out


@torch.inference_mode()
def predict_tier2(slots: DifferentiableSlots, heads: DecisionHeads, recs: Sequence[BenchRecord],
                  batch_size: int = 8) -> dict[str, dict[str, float]]:
    """id -> {answer key: probability}, in each record's own key space."""
    slots.scorer.model.eval()
    heads.eval()
    out: dict[str, dict[str, float]] = {}
    for start in range(0, len(recs), batch_size):
        chunk = recs[start:start + batch_size]
        batch = slots.batch(chunk)
        for i, (row, dist) in enumerate(zip(chunk, heads.distributions(batch))):
            out[row.id] = {k: float(v) for k, v in zip(batch.keys[i], dist.cpu())}
    return out


def _lora_state(model) -> dict[str, torch.Tensor]:
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def _load_lora_state(model, state: dict[str, torch.Tensor] | None) -> None:
    if not state:
        return
    live = dict(model.named_parameters())
    for n, v in state.items():
        if n in live:
            live[n].data.copy_(v.to(live[n].device))
