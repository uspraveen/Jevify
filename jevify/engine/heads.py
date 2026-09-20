"""Tier 1 decision heads: read slot features, emit calibrated distributions.

One shared projection feeds two scorers, mirroring what the behavioral probes showed
about Jev — the primitives are different instruments, not one softmax with three skins:

- Choice and Score share a *relative* slot scorer: ``score_i = f([h_dec, h_i, h_dec ⊙ h_i])``
  then a softmax over the allowed answers. It scores each option independently of how many
  there are, so a head trained with 16 options applies unchanged at K=151.
- Noul gets its own *absolute* scorer: ``P(yes) = σ(g([h_dec, h_yes, h_dec ⊙ h_yes]))``,
  which is the semantics Jev's Noul has and which a softmax over {yes, no} does not.

Training optimizes strictly proper scoring rules directly — log score for Choice and Noul,
ranked probability score for the ordinal Score — so calibration is the objective rather
than a post-hoc repair. This is the same target TypeSafe calls RLCD; with a differentiable
head it needs no reinforcement learning.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import SlotBatch, collate

NEG = -1e4


@dataclass
class HeadConfig:
    hidden: int = 2048        # backbone hidden size
    dim: int = 512            # projection width
    dropout: float = 0.1
    layer: int = -1           # which backbone layer the features came from
    backbone: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class DecisionHeads(nn.Module):
    def __init__(self, cfg: HeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        self.proj = nn.Sequential(nn.LayerNorm(cfg.hidden), nn.Linear(cfg.hidden, d), nn.GELU())
        self.slot_scorer = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d, 1))
        self.noul_scorer = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d, 1))
        # per-primitive temperature, learned jointly (log-parameterized so it stays positive)
        self.log_temp = nn.Parameter(torch.zeros(3))
        self.prim_index = {"choice": 0, "score": 1, "noul": 2}

    def _pair(self, dec: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        d = self.proj(dec).unsqueeze(1).expand(-1, slots.shape[1], -1)
        s = self.proj(slots)
        return torch.cat([d, s, d * s], dim=-1)

    def forward(self, batch: SlotBatch) -> dict[str, torch.Tensor]:
        """Returns per-row logits over the row's slots (choice/score) or a yes-logit (noul)."""
        feat = self._pair(batch.decision, batch.slots)
        rel = self.slot_scorer(feat).squeeze(-1).masked_fill(~batch.mask, NEG)
        absolute = self.noul_scorer(feat).squeeze(-1)
        temps = self.log_temp.exp()
        out = {"relative": rel, "absolute": absolute, "temps": temps}
        return out

    def distributions(self, batch: SlotBatch) -> list[torch.Tensor]:
        """Probability vector per row, in that row's slot order."""
        o = self.forward(batch)
        dists = []
        for i, prim in enumerate(batch.primitive):
            n = int(batch.mask[i].sum())
            t = o["temps"][self.prim_index[prim]]
            if prim == "noul":
                keys = batch.keys[i]
                yes = keys.index("1") if "1" in keys else 0
                p_yes = torch.sigmoid(o["absolute"][i, yes] / t)
                p = torch.zeros(n, device=p_yes.device)
                p[yes] = p_yes
                p[1 - yes] = 1 - p_yes
            else:
                p = F.softmax(o["relative"][i, :n] / t, dim=-1)
            dists.append(p)
        return dists

    # ------------------------------------------------------------------ losses
    def loss(self, batch: SlotBatch) -> tuple[torch.Tensor, dict[str, float]]:
        o = self.forward(batch)
        temps = o["temps"]
        total = batch.decision.new_zeros(())
        parts: dict[str, float] = {}
        counts: dict[str, int] = {}
        for prim in ("choice", "score", "noul"):
            idx = [i for i, p in enumerate(batch.primitive) if p == prim]
            if not idx:
                continue
            sel = torch.tensor(idx, device=batch.decision.device)
            t = temps[self.prim_index[prim]]
            if prim == "noul":
                yes_pos = torch.tensor([batch.keys[i].index("1") if "1" in batch.keys[i] else 0 for i in idx],
                                       device=sel.device)
                logit = o["absolute"][sel, yes_pos] / t
                target = (batch.label[sel] == yes_pos).float()
                loss = F.binary_cross_entropy_with_logits(logit, target)
            elif prim == "choice":
                loss = F.cross_entropy(o["relative"][sel] / t, batch.label[sel])
            else:
                loss = ranked_probability_loss(o["relative"][sel] / t, batch.label[sel], batch.mask[sel])
            total = total + loss * len(idx)
            parts[prim] = float(loss.detach())
            counts[prim] = len(idx)
        n = sum(counts.values()) or 1
        return total / n, {**parts, **{f"n_{k}": v for k, v in counts.items()}}


def ranked_probability_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean RPS over a batch of ordinal rows: squared distance between the predicted and
    the one-hot cumulative distributions. Strictly proper, and unlike cross-entropy it
    charges less for a near miss than for a distant one."""
    p = F.softmax(logits, dim=-1) * mask
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    onehot = F.one_hot(labels, num_classes=logits.shape[-1]).float()
    cum_p, cum_y = p.cumsum(-1), onehot.cumsum(-1)
    k = mask.sum(-1).clamp_min(2) - 1
    return (((cum_p - cum_y) ** 2) * mask).sum(-1).div(k).mean()


# --------------------------------------------------------------------------- training

def train_heads(train_rows: Sequence[dict[str, Any]], val_rows: Sequence[dict[str, Any]], cfg: HeadConfig, *,
                epochs: int = 20, batch_size: int = 64, lr: float = 3e-4, weight_decay: float = 0.01,
                device: str = "cpu", seed: int = 0, verbose: bool = True) -> tuple[DecisionHeads, dict[str, Any]]:
    torch.manual_seed(seed)
    model = DecisionHeads(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    order = torch.randperm(len(train_rows), generator=torch.Generator().manual_seed(seed)).tolist()
    steps = max(1, math.ceil(len(order) / batch_size)) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.2)
    history, best = [], (float("inf"), None, -1)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_rows), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
        tot = 0.0
        for start in range(0, len(perm), batch_size):
            rows = [train_rows[i] for i in perm[start:start + batch_size]]
            batch = collate(rows).to(device)
            loss, _ = model.loss(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(rows)
        val = evaluate_loss(model, val_rows, batch_size, device)
        history.append({"epoch": epoch, "train": round(tot / max(len(perm), 1), 4), **{k: round(v, 4) for k, v in val.items()}})
        if verbose:
            print(f"  epoch {epoch:2d} train {history[-1]['train']:.4f} val {val['loss']:.4f}", flush=True)
        if val["loss"] < best[0]:
            best = (val["loss"], {k: v.detach().clone() for k, v in model.state_dict().items()}, epoch)
    if best[1] is not None:
        model.load_state_dict(best[1])
    return model, {"history": history, "best_epoch": best[2], "best_val_loss": best[0], "config": cfg.as_dict()}


@torch.inference_mode()
def evaluate_loss(model: DecisionHeads, rows: Sequence[dict[str, Any]], batch_size: int = 64, device: str = "cpu") -> dict[str, float]:
    model.eval()
    tot, n = 0.0, 0
    parts: dict[str, list[float]] = {}
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        batch = collate(chunk).to(device)
        loss, detail = model.loss(batch)
        tot += float(loss) * len(chunk)
        n += len(chunk)
        for k, v in detail.items():
            if not k.startswith("n_"):
                parts.setdefault(k, []).append(v)
    out = {"loss": tot / max(n, 1)}
    out.update({k: sum(v) / len(v) for k, v in parts.items()})
    return out


@torch.inference_mode()
def predict_rows(model: DecisionHeads, rows: Sequence[dict[str, Any]], batch_size: int = 64, device: str = "cpu") -> dict[str, dict[str, float]]:
    """id -> {answer key: probability}, in the record's own key space."""
    model.eval()
    out: dict[str, dict[str, float]] = {}
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        batch = collate(chunk).to(device)
        for row, p in zip(chunk, model.distributions(batch)):
            out[row["id"]] = {k: float(v) for k, v in zip(row["keys"], p.cpu())}
    return out


def save_heads(model: DecisionHeads, info: dict[str, Any], path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "heads.pt")
    (path / "heads.json").write_text(json.dumps(info, indent=1), encoding="utf-8")


def load_heads(path: Path, device: str = "cpu") -> DecisionHeads:
    info = json.loads((path / "heads.json").read_text(encoding="utf-8"))
    model = DecisionHeads(HeadConfig(**info["config"]))
    model.load_state_dict(torch.load(path / "heads.pt", map_location=device))
    return model.to(device).eval()
