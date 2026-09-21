"""Slot features: the hidden states a decision head reads.

For one rendered question we extract

- ``decision``: the hidden state at the final position (after the answer cue). It has
  attended to the state, the question and *every* option, so it carries the comparison.
- ``slots``: one hidden state per allowed answer, taken at the last token of that
  option's line in the prompt, so it carries what the option *means* rather than which
  identifier token names it. This is the difference from Tier 0, which can only use the
  LM head's logit for the identifier.
- ``lm``: that Tier 0 log-score itself, read from the same forward pass (every candidate
  is a single token by construction), so a head can be trained as a *residual* on the
  model's own prior instead of replacing it.

Everything is stored fp16 and option-subsampled at training time (``max_slots``), which
is what keeps the cache small and the head K-agnostic: it scores each option
independently, so a head trained with 16 options applies unchanged to 151.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from ..bench.record import BenchRecord
from .readout import HFScorer
from .template import CUE, Rendered, render, to_chat


@dataclass
class SlotBatch:
    decision: torch.Tensor          # (B, H)
    slots: torch.Tensor             # (B, S, H) padded
    lm: torch.Tensor                # (B, S) Tier 0 log-scores, padded
    mask: torch.Tensor              # (B, S) bool, True where a slot is real
    label: torch.Tensor             # (B,) index into the kept slots
    primitive: list[str]
    ids: list[str]
    keys: list[list[str]]           # answer key per kept slot
    # the human label distribution over the kept slots, where the source has one; rows
    # without one carry zeros and has_soft=False, so a loss can fall back to the hard label
    soft: torch.Tensor | None = None        # (B, S)
    has_soft: torch.Tensor | None = None    # (B,) bool

    def to(self, device: str) -> "SlotBatch":
        return SlotBatch(self.decision.to(device), self.slots.to(device), self.lm.to(device), self.mask.to(device),
                         self.label.to(device), self.primitive, self.ids, self.keys,
                         None if self.soft is None else self.soft.to(device),
                         None if self.has_soft is None else self.has_soft.to(device))


def soft_targets(r: BenchRecord, kept_keys: Sequence[str]) -> list[float] | None:
    """The human label distribution over the kept answer keys, or None if the record has none.

    jev-bench stores distributions in the shape of the primitive: a dict over option keys
    (Choice), a list over levels (Score), or P(yes) (Noul). Slot subsampling can drop
    options that carried mass, so the kept mass is renormalized; the gold option is always
    kept, and it is the argmax of the distribution, so nothing important is lost.
    """
    sl = r.soft_label
    if sl is None:
        return None
    keys = [str(k) for k in kept_keys]
    if r.primitive == "noul":
        p_yes = float(sl)
        vals = [p_yes if k == "1" else 1.0 - p_yes for k in keys]
    elif isinstance(sl, dict):
        vals = [float(sl.get(k, 0.0)) for k in keys]
    else:                                                # list over levels, index == key
        try:
            vals = [float(sl[int(k)]) for k in keys]
        except (ValueError, IndexError, TypeError):
            return None
    total = sum(vals)
    if total <= 0:
        return None
    return [v / total for v in vals]


def _pad_soft(softs: Sequence[list[float] | None], S: int) -> tuple[torch.Tensor, torch.Tensor]:
    soft = torch.zeros(len(softs), S)
    has = torch.zeros(len(softs), dtype=torch.bool)
    for i, q in enumerate(softs):
        if q is not None:
            soft[i, :len(q)] = torch.tensor(q)
            has[i] = True
    return soft, has


def _line_end_positions(scorer: HFScorer, prefix: str, needles: Sequence[str]) -> list[int]:
    """Token index of the last token of each option line, found by character offsets."""
    enc = scorer.tokenizer(prefix, add_special_tokens=True, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    out = []
    search_from = 0
    for text in needles:
        idx = prefix.find(text, search_from)
        if idx < 0:
            idx = prefix.find(text)
        end_char = (idx + len(text)) if idx >= 0 else len(prefix)
        search_from = max(search_from, end_char)
        pos = 0
        for t, (a, b) in enumerate(offsets):
            if b and b <= end_char:
                pos = t
        out.append(pos)
    return out


class FeatureExtractor:
    """Runs the backbone once per record (batched) and returns slot features.

    Only one layer's activations are kept, captured with a forward hook — asking
    ``output_hidden_states=True`` would materialize every layer and dominate memory.
    Right padding keeps every real token at its true position under a causal mask, so
    slot indices found by character offsets need no shift.
    """

    def __init__(self, scorer: HFScorer, layer: int = -1, chat: bool = True) -> None:
        self.scorer = scorer
        self.layer = layer
        self.chat = chat and getattr(scorer.tokenizer, "chat_template", None) is not None
        self.hidden = int(scorer.model.config.hidden_size)
        self._captured: torch.Tensor | None = None
        self._hook = self._target_module().register_forward_hook(self._capture)

    def _target_module(self):
        """The decoder block to read hidden states from.

        Found as the longest nn.ModuleList in the model, which is the decoder stack in every
        architecture we have met and, unlike attribute paths, survives PEFT/LoRA wrappers.
        """
        import torch.nn as nn

        stacks = [m for m in self.scorer.model.modules() if isinstance(m, nn.ModuleList) and len(m) > 1]
        if not stacks:
            raise TypeError("could not locate the decoder layers of this architecture")
        return max(stacks, key=len)[self.layer]

    def rebind(self) -> None:
        """Re-attach the hook after the backbone is wrapped or replaced (e.g. by LoRA)."""
        self._hook.remove()
        self._hook = self._target_module().register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output) -> None:
        self._captured = output[0] if isinstance(output, tuple) else output

    def close(self) -> None:
        self._hook.remove()

    def _prefix(self, rd: Rendered) -> str:
        return to_chat(rd.prefix, self.scorer.tokenizer) if self.chat else rd.prefix

    def _needles(self, rd: Rendered, question: dict[str, Any]) -> list[str]:
        """The text whose last token represents each option."""
        if rd.primitive == "noul":
            return ["yes", "no"]
        if rd.primitive == "score":
            return [str(c)[:60] for c in question["criteria"]]
        crit = question["criteria"]
        return [f"{k}" if not crit.get(k) else f"{k} — {crit[k]}"[:120] for k in rd.keys]

    def _plan(self, r: BenchRecord, max_slots: int, rng, identifiers):
        """Tokenize exactly as the Tier 0 scorer does, so the decision position and the
        candidate tokens are the same ones Tier 0 scores. Re-encoding the prompt text
        instead would read logits at the cue's trailing space and silently disagree."""
        rd = render(r.state, r.question, identifiers=identifiers)
        prefix = self._prefix(rd)
        ids = self.scorer.tokenizer(prefix, add_special_tokens=True)["input_ids"]
        if len(ids) > self.scorer.max_prefix_tokens:
            prefix = self.scorer.tokenizer.decode(ids[-self.scorer.max_prefix_tokens:], skip_special_tokens=True)
        tok = self.scorer.tokenize(prefix, rd.candidates)
        cand_tokens = [c[0] for c in tok.cand_ids]      # single-token by construction
        keys = list(rd.keys)
        keep = list(range(len(keys)))
        if max_slots and len(keys) > max_slots:
            gold = keys.index(str(r.label))
            others = [i for i in keep if i != gold]
            pick = rng.choice(others, size=max_slots - 1, replace=False)
            keep = sorted([gold] + [int(i) for i in pick])
        needles = self._needles(rd, r.question)
        last = len(tok.prefix_ids) - 1
        pos = [min(p, last) for p in _line_end_positions(self.scorer, prefix, [needles[i] for i in keep])]
        kept = [keys[i] for i in keep]
        return tok.prefix_ids, pos, kept, kept.index(str(r.label)), [cand_tokens[i] for i in keep], soft_targets(r, kept)

    @torch.inference_mode()
    def extract(self, records: Sequence[BenchRecord], *, max_slots: int = 0, rng: np.random.Generator | None = None,
                identifiers: Sequence[str] | None = None, batch_size: int = 8) -> list[dict[str, Any]]:
        rng = rng or np.random.default_rng(0)
        ids_list = identifiers or self.scorer.identifiers()
        plans = [self._plan(r, max_slots, rng, ids_list) for r in records]
        pad = self.scorer.pad_id
        out: list[dict[str, Any]] = []
        order = sorted(range(len(plans)), key=lambda i: len(plans[i][0]))   # group similar lengths
        for start in range(0, len(order), batch_size):
            chunk = [order[j] for j in range(start, min(start + batch_size, len(order)))]
            width = max(len(plans[i][0]) for i in chunk)
            ids = torch.full((len(chunk), width), pad, dtype=torch.long)
            attn = torch.zeros((len(chunk), width), dtype=torch.long)
            for row, i in enumerate(chunk):                    # right padding: real tokens keep their positions
                seq = plans[i][0]
                ids[row, : len(seq)] = torch.tensor(seq)
                attn[row, : len(seq)] = 1
            self._captured = None
            res = self.scorer.model(input_ids=ids.to(self.scorer.device), attention_mask=attn.to(self.scorer.device),
                                    use_cache=False)
            hs = self._captured
            assert hs is not None, "forward hook did not fire"
            for row, i in enumerate(chunk):
                seq, pos, keys, label, cand, soft = plans[i]
                last = len(seq) - 1
                logp = torch.log_softmax(res.logits[row, last].float(), dim=-1)
                out.append({"id": records[i].id, "source": records[i].source, "primitive": records[i].primitive,
                            "decision": hs[row, last].to(torch.float16).cpu().numpy(),
                            "slots": hs[row, torch.tensor(pos, device=hs.device)].to(torch.float16).cpu().numpy(),
                            "lm": logp[torch.tensor(cand, device=logp.device)].to(torch.float32).cpu().numpy(),
                            "keys": keys, "label": label, "soft": soft})
        by_id = {r["id"]: r for r in out}
        return [by_id[r.id] for r in records if r.id in by_id]


def save_features(path, rows: list[dict[str, Any]]) -> int:
    """One .npz per shard: ragged slots are stored flat with an index."""
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dec = np.stack([r["decision"] for r in rows])
    counts = np.array([len(r["keys"]) for r in rows], dtype=np.int32)
    slots = np.concatenate([r["slots"] for r in rows], axis=0)
    lm = np.concatenate([r["lm"] for r in rows], axis=0)
    labels = np.array([r["label"] for r in rows], dtype=np.int32)
    meta = [{"id": r["id"], "source": r["source"], "primitive": r["primitive"], "keys": r["keys"],
             "soft": r.get("soft")} for r in rows]
    np.savez_compressed(path, decision=dec, slots=slots, lm=lm, counts=counts, labels=labels, meta=json.dumps(meta))
    return len(rows)


def load_features(path) -> list[dict[str, Any]]:
    import json

    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    # Every array is read from the archive exactly once. Indexing the NpzFile inside the
    # loop (z["decision"][i]) re-reads and decompresses the whole array per row, and each
    # row's slice then pins its own private full copy: 22,773 rows x ~93 MB took a 1 TB host
    # to the OOM killer. Local names below are plain ndarrays; slices of them are views.
    counts, slots, lm, decision, labels = z["counts"], z["slots"], z["lm"], z["decision"], z["labels"]
    rows, off = [], 0
    for i, m in enumerate(meta):
        n = int(counts[i])
        rows.append({**m, "decision": decision[i], "slots": slots[off:off + n], "lm": lm[off:off + n],
                     "label": int(labels[i]), "soft": m.get("soft")})
        off += n
    return rows


def collate(rows: Sequence[dict[str, Any]]) -> SlotBatch:
    B = len(rows)
    S = max(len(r["keys"]) for r in rows)
    H = rows[0]["decision"].shape[-1]
    dec = torch.zeros(B, H)
    slots = torch.zeros(B, S, H)
    lm = torch.full((B, S), -20.0)
    mask = torch.zeros(B, S, dtype=torch.bool)
    label = torch.zeros(B, dtype=torch.long)
    for i, r in enumerate(rows):
        n = len(r["keys"])
        dec[i] = torch.from_numpy(r["decision"].astype(np.float32))
        slots[i, :n] = torch.from_numpy(r["slots"].astype(np.float32))
        lm[i, :n] = torch.from_numpy(np.asarray(r["lm"], dtype=np.float32))
        mask[i, :n] = True
        label[i] = r["label"]
    soft, has_soft = _pad_soft([r.get("soft") for r in rows], S)
    return SlotBatch(dec, slots, lm, mask, label, [r["primitive"] for r in rows], [r["id"] for r in rows],
                     [r["keys"] for r in rows], soft, has_soft)
