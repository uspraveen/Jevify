"""Teacher-forced candidate scoring on a causal LM: log P(candidate | prefix).

The prefix is run once; its KV cache is expanded across the candidates so a
151-option question costs one prefix pass plus a few short candidate passes.
Candidates are tokenized *jointly* with the prefix and scored on the tokens
past the common prefix, so boundary merges can never corrupt a score.

Two paths, chosen per question:
- single-token: every candidate is exactly one token past the prefix → one
  forward on the prefix and a gather on the last logits (batched across
  questions);
- multi-token: cache expansion + one forward per chunk of candidates.
``naive=True`` recomputes prefix+candidate for every candidate; it exists so
tests can pin the fast paths against the obviously-correct one.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F


@dataclass
class Tokenized:
    prefix_ids: list[int]
    cand_ids: list[list[int]]   # candidate tokens past the common prefix (>= 1 each)

    @property
    def single_token(self) -> bool:
        return all(len(c) == 1 for c in self.cand_ids)


class HFScorer:
    def __init__(self, model_id: str, *, device: str | None = None, dtype: torch.dtype | None = None,
                 batch_size: int = 16, cand_chunk: int = 64, max_prefix_tokens: int = 4096,
                 trust_remote_code: bool = False, hf_token: str | None = None) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if self.device == "cuda" else torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=self.dtype, trust_remote_code=trust_remote_code,
                                                          token=hf_token).to(self.device).eval()
        self.batch_size = batch_size
        self.cand_chunk = cand_chunk
        self.max_prefix_tokens = max_prefix_tokens
        self._bos = self.tokenizer.bos_token_id
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else (self.tokenizer.eos_token_id or 0)

    # ------------------------------------------------------------------ tokenization
    def tokenize(self, prefix: str, candidates: Sequence[str]) -> Tokenized:
        p_ids = self._encode(prefix)
        if len(p_ids) > self.max_prefix_tokens:   # keep the tail: the question and options live there
            prefix = self.tokenizer.decode(p_ids[-self.max_prefix_tokens:], skip_special_tokens=True)
            p_ids = self._encode(prefix)
        fulls = [self._encode(prefix + c) for c in candidates]
        n_common = min(len(p_ids), *(_common_prefix_len(p_ids, f) for f in fulls))
        if n_common == len(p_ids) and any(len(f) == n_common for f in fulls):
            n_common -= 1   # a candidate merged into the prefix's last token; score from there
        cands = [f[n_common:] for f in fulls]
        assert all(cands), "every candidate must contribute at least one token"
        return Tokenized(p_ids[:n_common], cands)

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=True)["input_ids"]

    # ------------------------------------------------------------------ scoring
    @torch.inference_mode()
    def score_many(self, items: Sequence[tuple[str, Sequence[str]]], *, naive: bool = False) -> list[list[float]]:
        """For each (prefix, candidates) return summed log-probs per candidate."""
        toks = [self.tokenize(p, c) for p, c in items]
        out: list[list[float] | None] = [None] * len(items)
        if naive:
            return [self._score_naive(t) for t in toks]
        single = [i for i, t in enumerate(toks) if t.single_token]
        for start in range(0, len(single), self.batch_size):
            idx = single[start:start + self.batch_size]
            res = self._score_single_batch([toks[i] for i in idx])
            for i, r in zip(idx, res):
                out[i] = r
        for i, t in enumerate(toks):
            if out[i] is None:
                out[i] = self._score_multi(t)
        return out  # type: ignore[return-value]

    def _score_single_batch(self, toks: list[Tokenized]) -> list[list[float]]:
        maxlen = max(len(t.prefix_ids) for t in toks)
        ids = torch.full((len(toks), maxlen), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(toks), maxlen), dtype=torch.long)
        for r, t in enumerate(toks):   # left-pad so the last position is the prefix end
            n = len(t.prefix_ids)
            ids[r, maxlen - n:] = torch.tensor(t.prefix_ids)
            mask[r, maxlen - n:] = 1
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        logits = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device), position_ids=pos.to(self.device)).logits[:, -1].float()
        logp = F.log_softmax(logits, dim=-1).cpu()
        return [[float(logp[r, c[0]]) for c in t.cand_ids] for r, t in enumerate(toks)]

    def _score_multi(self, t: Tokenized) -> list[float]:
        p = torch.tensor([t.prefix_ids], device=self.device)
        out = self.model(input_ids=p, use_cache=True)
        last = F.log_softmax(out.logits[0, -1].float(), dim=-1)
        cache = out.past_key_values
        scores: list[float] = []
        for start in range(0, len(t.cand_ids), self.cand_chunk):
            chunk = t.cand_ids[start:start + self.cand_chunk]
            scores.extend(self._score_chunk(chunk, cache, last, len(t.prefix_ids)))
        return scores

    def _score_chunk(self, chunk: list[list[int]], cache: Any, last_logp: torch.Tensor, plen: int) -> list[float]:
        C = len(chunk)
        L = max(len(c) for c in chunk)
        first = torch.tensor([float(last_logp[c[0]]) for c in chunk])
        if L == 1:
            return first.tolist()
        ids = torch.full((C, L - 1), self.pad_id, dtype=torch.long)
        cmask = torch.zeros((C, L - 1), dtype=torch.long)
        for r, c in enumerate(chunk):
            ids[r, : len(c) - 1] = torch.tensor(c[:-1])
            cmask[r, : len(c) - 1] = 1
        attn = torch.cat([torch.ones((C, plen), dtype=torch.long), cmask], dim=1).to(self.device)
        pos = (plen + torch.arange(L - 1)).unsqueeze(0).expand(C, -1).to(self.device)
        expanded = _expand_cache(cache, C)
        logits = self.model(input_ids=ids.to(self.device), attention_mask=attn, position_ids=pos,
                            past_key_values=expanded, use_cache=True).logits.float()
        logp = F.log_softmax(logits, dim=-1).cpu()
        total = first.clone()
        for r, c in enumerate(chunk):
            for j in range(1, len(c)):
                total[r] += float(logp[r, j - 1, c[j]])
        return total.tolist()

    def _score_naive(self, t: Tokenized) -> list[float]:
        scores = []
        for c in t.cand_ids:
            ids = torch.tensor([t.prefix_ids + c], device=self.device)
            logits = self.model(input_ids=ids).logits[0].float()
            logp = F.log_softmax(logits, dim=-1)
            n = len(t.prefix_ids)
            scores.append(float(sum(logp[n - 1 + j, c[j]] for j in range(len(c)))))
        return scores


def _common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _expand_cache(cache: Any, n: int) -> Any:
    """Return a copy of a KV cache with batch dimension repeated n times.
    Handles the modern layered DynamicCache, the legacy list-based one, and
    any cache exposing batch_repeat_interleave."""
    c = copy.deepcopy(cache)
    if hasattr(c, "batch_repeat_interleave"):
        c.batch_repeat_interleave(n)
        return c
    if hasattr(c, "layers"):
        for layer in c.layers:
            for attr in ("keys", "values"):
                t = getattr(layer, attr, None)
                if isinstance(t, torch.Tensor):
                    setattr(layer, attr, t.repeat_interleave(n, dim=0))
        return c
    if hasattr(c, "key_cache"):
        c.key_cache = [k.repeat_interleave(n, dim=0) for k in c.key_cache]
        c.value_cache = [v.repeat_interleave(n, dim=0) for v in c.value_cache]
        return c
    if isinstance(c, tuple):   # legacy tuple-of-tuples
        return tuple(tuple(x.repeat_interleave(n, dim=0) for x in layer) for layer in c)
    raise TypeError(f"don't know how to expand cache of type {type(cache)}")


def softmax(scores: Sequence[float], temperature: float = 1.0) -> list[float]:
    m = max(scores)
    ex = [math.exp((s - m) / temperature) for s in scores]
    z = sum(ex)
    return [e / z for e in ex]
