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
                 trust_remote_code: bool = False, hf_token: str | None = None, tree_attention: bool | None = None,
                 tree_max_tokens: int = 6144, revision: str | None = None) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if self.device == "cuda" else torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code, token=hf_token,
                                                       revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=self.dtype, trust_remote_code=trust_remote_code,
                                                          token=hf_token, revision=revision).to(self.device).eval()
        self.batch_size = batch_size
        self.cand_chunk = cand_chunk
        self.max_prefix_tokens = max_prefix_tokens
        self.tree_max_tokens = tree_max_tokens
        self._bos = self.tokenizer.bos_token_id
        self.stats = {"tokenize_s": 0.0, "single_s": 0.0, "multi_s": 0.0, "items": 0}
        self.tree_attention = tree_attention      # None = auto (verify once against naive), True/False = force
        self._tree_verified: bool | None = None
        self._recurrent: bool | None = None
        self._identifiers: list[str] | None = None
        self._identifiers_by_context: dict[str, list[str]] = {}
        import inspect
        params = inspect.signature(self.model.forward).parameters
        self._supports_keep = "logits_to_keep" in params or any(p.kind == p.VAR_KEYWORD for p in params.values())
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else (self.tokenizer.eos_token_id or 0)

    # ------------------------------------------------------------------ identifiers
    def identifiers(self, k: int = 255, *, context: str | None = None) -> list[str]:
        """Up to k option identifiers that are single tokens for THIS tokenizer after the
        answer cue: A–Z first, then two-letter combinations, so Choice with any K stays on
        the batched single-token path. Falls back to numbers if the vocabulary is too small.

        ``context`` is the text the answer follows when that is not our cue -- the Tev1 format's
        empty assistant turn, where a merge after the newline makes three of the first 151
        two-letter ids two tokens."""
        if context is not None:
            if context not in self._identifiers_by_context:
                import itertools
                import string

                base = self._encode_raw(context)
                ok: list[str] = []
                for cand in itertools.chain(string.ascii_uppercase, ("".join(p) for p in itertools.product(string.ascii_uppercase, repeat=2))):
                    ids = self._encode_raw(context + cand)
                    if len(ids) - _common_prefix_len(base, ids) == 1 and ids[: len(base)] == base:
                        ok.append(cand)
                    if len(ok) >= 255:
                        break
                self._identifiers_by_context[context] = ok
            return self._identifiers_by_context[context][:k]
        if self._identifiers is None:
            import itertools
            import string

            from .template import CUE, default_identifiers

            probe = "Allowed answers: A, B" + chr(10) + CUE
            base = self._encode_raw(probe)
            ok: list[str] = []
            for cand in itertools.chain(string.ascii_uppercase, ("".join(p) for p in itertools.product(string.ascii_uppercase, repeat=2))):
                ids = self._encode_raw(probe + cand)
                if len(ids) - _common_prefix_len(base, ids) == 1:   # exactly one token past the shared prefix
                    ok.append(cand)
                if len(ok) >= 255:
                    break
            self._identifiers = ok if len(ok) >= 26 else default_identifiers(255)
        return self._identifiers[:k]

    # ------------------------------------------------------------------ tokenization
    def tokenize(self, prefix: str, candidates: Sequence[str]) -> Tokenized:
        """Joint tokenization without re-encoding the whole prompt per candidate.

        The prefix is encoded once. Each candidate is encoded together with a
        short tail of the prefix (from the last newline, so the BPE boundary is
        reproduced exactly); if the tail's own tokens do not match the end of the
        full prefix encoding, fall back to full joint encoding for that candidate."""
        p_ids = self._encode(prefix)
        if len(p_ids) > self.max_prefix_tokens:   # keep the tail: the question and options live there
            prefix = self.tokenizer.decode(p_ids[-self.max_prefix_tokens:], skip_special_tokens=True)
            p_ids = self._encode(prefix)
        # the tail must re-encode to exactly the end of the prefix's own encoding. The last newline
        # fails that when the prompt ends in a merged "\n\n" (a chat template's empty assistant turn,
        # the Tev1 format), which sent every candidate down a full re-encode of the whole prompt:
        # 151 of them for a 151-option question, CPU-bound. Earlier newlines are tried before giving up.
        pos, tail, tail_ids = max(len(prefix) - 1, 0), prefix, []
        for _ in range(4):
            cut = prefix.rfind(chr(10), 0, pos) + 1
            tail = prefix[cut:]
            tail_ids = self._encode_raw(tail)
            if tail_ids and p_ids[-len(tail_ids):] == tail_ids:
                break
            if cut == 0:
                break
            pos = cut - 1
        fast = bool(tail_ids) and p_ids[-len(tail_ids):] == tail_ids
        fulls: list[list[int]] = []
        for c in candidates:
            if fast:
                ids = self._encode_raw(tail + c)
                if ids[: len(tail_ids)] == tail_ids:
                    fulls.append(p_ids + ids[len(tail_ids):])
                    continue
                if _common_prefix_len(tail_ids, ids) > 0:
                    n = _common_prefix_len(tail_ids, ids)
                    fulls.append(p_ids[: len(p_ids) - len(tail_ids) + n] + ids[n:])
                    continue
            fulls.append(self._encode(prefix + c))
        n_common = min(len(p_ids), *(_common_prefix_len(p_ids, f) for f in fulls))
        if n_common == len(p_ids) and any(len(f) == n_common for f in fulls):
            n_common -= 1   # a candidate merged into the prefix's last token; score from there
        cands = [f[n_common:] for f in fulls]
        assert all(cands), "every candidate must contribute at least one token"
        return Tokenized(p_ids[:n_common], cands)

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=True)["input_ids"]

    def _encode_raw(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    # ------------------------------------------------------------------ scoring
    @torch.inference_mode()
    def score_many(self, items: Sequence[tuple[str, Sequence[str]]], *, naive: bool = False) -> list[list[float]]:
        """For each (prefix, candidates) return summed log-probs per candidate."""
        import time as _t
        t0 = _t.perf_counter()
        toks = [self.tokenize(p, c) for p, c in items]
        self.stats["tokenize_s"] += _t.perf_counter() - t0
        out: list[list[float] | None] = [None] * len(items)
        if naive:
            return [self._score_naive(t) for t in toks]
        # batch by similar prefix length so left-padding wastes little compute
        single = sorted((i for i, t in enumerate(toks) if t.single_token), key=lambda i: len(toks[i].prefix_ids))
        t0 = _t.perf_counter()
        for start in range(0, len(single), self.batch_size):
            idx = single[start:start + self.batch_size]
            res = self._score_single_batch([toks[i] for i in idx])
            for i, r in zip(idx, res):
                out[i] = r
        self.stats["single_s"] += _t.perf_counter() - t0
        t0 = _t.perf_counter()
        for i, t in enumerate(toks):
            if out[i] is None:
                out[i] = self._score_tree(t) if self._tree_usable(t) else self._score_multi(t)
        self.stats["multi_s"] += _t.perf_counter() - t0
        self.stats["items"] += len(items)
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
        logits = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device), position_ids=pos.to(self.device),
                            **self._keep(1)).logits[:, -1].float()
        logp = F.log_softmax(logits, dim=-1)
        rr = [r for r, t in enumerate(toks) for _ in t.cand_ids]
        tt = [c[0] for t in toks for c in t.cand_ids]
        flat = logp[torch.tensor(rr, device=self.device), torch.tensor(tt, device=self.device)].tolist()
        out, k = [], 0
        for t in toks:
            out.append(flat[k:k + len(t.cand_ids)]); k += len(t.cand_ids)
        return out

    def _score_multi(self, t: Tokenized) -> list[float]:
        p = torch.tensor([t.prefix_ids], device=self.device)
        out = self.model(input_ids=p, use_cache=True, **self._keep(1))
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
        first = last_logp[torch.tensor([c[0] for c in chunk], device=last_logp.device)]
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
        logp = F.log_softmax(logits, dim=-1)
        rr, jj, tt, owner = [], [], [], []
        for r, c in enumerate(chunk):
            for j in range(1, len(c)):
                rr.append(r); jj.append(j - 1); tt.append(c[j]); owner.append(r)
        total = first.to(self.device)
        if rr:
            picked = logp[torch.tensor(rr, device=self.device), torch.tensor(jj, device=self.device), torch.tensor(tt, device=self.device)]
            total = total.index_add(0, torch.tensor(owner, device=self.device), picked)
        return total.tolist()

    # ---- tree attention: every candidate in one sequence, block-diagonal mask ------------
    def _tree_usable(self, t: Tokenized) -> bool:
        if self.tree_attention is False:
            return False
        if self._recurrent is None:
            self._recurrent = has_recurrent_layers(self.model)
        if self._recurrent:
            # a recurrence reads its inputs in sequence order, not through the attention mask, so a
            # block-diagonal mask cannot isolate the candidates -- and a confident model can pass the
            # self-check below anyway. Never use the tree on these layers (FINDINGS 8.11).
            return False
        total = len(t.prefix_ids) + sum(len(c) for c in t.cand_ids)
        if total > self.tree_max_tokens:
            return False
        if self.tree_attention is True:
            return True
        if self._tree_verified is None:
            # verify once on this very item: a custom 4D mask that the architecture ignores would
            # silently leak candidates into each other, so compare against naive recompute.
            try:
                tree = self._score_tree(t)
                ref = self._score_naive(t)
                # A mask that is ignored leaks candidates into each other and moves probabilities by
                # tenths; bf16 numerics move them by hundredths at most (measured: <= 0.045 on 151-way).
                p_tree, p_ref = softmax(tree), softmax(ref)
                ok = max(abs(a - b) for a, b in zip(p_tree, p_ref)) < 0.1 and (max(range(len(tree)), key=tree.__getitem__) == max(range(len(ref)), key=ref.__getitem__))
            except Exception as e:  # pragma: no cover - architecture specific
                print(f"[readout] tree attention unavailable ({type(e).__name__}: {e}); using cache expansion", flush=True)
                ok = False
            if not ok:
                print("[readout] tree attention failed verification; using cache expansion", flush=True)
            self._tree_verified = ok
        return self._tree_verified

    def _score_tree(self, t: Tokenized) -> list[float]:
        T = len(t.prefix_ids)
        ids = list(t.prefix_ids)
        pos = list(range(T))
        spans: list[tuple[int, int]] = []           # [start, end) of each candidate in the sequence
        for c in t.cand_ids:
            spans.append((len(ids), len(ids) + len(c)))
            ids.extend(c)
            pos.extend(range(T, T + len(c)))
        N = len(ids)
        mask = torch.zeros((N, N), dtype=torch.bool)
        mask[:T, :T] = torch.tril(torch.ones((T, T), dtype=torch.bool))
        for a, b in spans:
            mask[a:b, :T] = True
            mask[a:b, a:b] = torch.tril(torch.ones((b - a, b - a), dtype=torch.bool))
        dev = self.device
        out = self.model(input_ids=torch.tensor([ids], device=dev), attention_mask=mask[None, None].to(dev),
                         position_ids=torch.tensor([pos], device=dev), use_cache=False, **self._keep(N - (T - 1)))
        # logits are now only for positions T-1 .. N-1; re-index accordingly
        offset = T - 1
        # gather every needed (position, token) pair in one op: the prefix's last row predicts each
        # candidate's first token; row a+j-1 predicts token j of the candidate starting at a.
        rows, toks, owner = [], [], []
        for k, ((a, b), c) in enumerate(zip(spans, t.cand_ids)):
            rows.append(T - 1); toks.append(c[0]); owner.append(k)
            for j in range(1, len(c)):
                rows.append(a + j - 1); toks.append(c[j]); owner.append(k)
        need = sorted(set(rows))
        logp_rows = F.log_softmax(out.logits[0, [r - offset for r in need]].float(), dim=-1)
        row_index = {r: i for i, r in enumerate(need)}
        picked = logp_rows[torch.tensor([row_index[r] for r in rows], device=dev), torch.tensor(toks, device=dev)]
        totals = torch.zeros(len(spans), device=dev).index_add_(0, torch.tensor(owner, device=dev), picked)
        return totals.tolist()

    def _keep(self, n: int) -> dict[str, int]:
        """Only compute LM-head logits for the last n positions (the head over a 250k vocab
        for every position is the dominant memory and a large share of time otherwise)."""
        return {"logits_to_keep": n} if self._supports_keep else {}

    def _score_naive(self, t: Tokenized) -> list[float]:
        scores = []
        for c in t.cand_ids:
            ids = torch.tensor([t.prefix_ids + c], device=self.device)
            logits = self.model(input_ids=ids).logits[0].float()
            logp = F.log_softmax(logits, dim=-1)
            n = len(t.prefix_ids)
            scores.append(float(sum(logp[n - 1 + j, c[j]] for j in range(len(c)))))
        return scores


def has_recurrent_layers(model: Any) -> bool:
    """True when some layer carries state across positions instead of attending (Gated DeltaNet,
    Mamba, RWKV, linear attention): a custom attention mask does not reach those layers."""
    names = {type(m).__name__.lower() for m in model.modules()}
    return any(key in n for n in names for key in ("linearattention", "gateddelta", "mamba", "rwkv", "recurrent"))


def _common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _expand_cache(cache: Any, n: int) -> Any:
    """A copy of the cache with every per-layer tensor state repeated n times along the
    batch axis. Works for K/V layers and for recurrent (linear-attention) layers alike,
    because it expands whatever tensors the layer holds rather than assuming keys/values."""
    import copy

    def grow(x):
        if torch.is_tensor(x) and x.dim() >= 1:
            return x.expand(n, *x.shape[1:]).contiguous() if x.shape[0] == 1 else x.repeat_interleave(n, dim=0)
        if isinstance(x, (list, tuple)):
            return type(x)(grow(y) for y in x)
        if isinstance(x, dict):
            # Qwen3.5's LinearAttentionLayer keeps its conv and recurrent states in dicts; left at
            # batch 1 they made every multi-token candidate on that family fail (FINDINGS 8.11)
            return {k: grow(v) for k, v in x.items()}
        return x

    if hasattr(cache, "layers"):
        new = copy.copy(cache)
        new.layers = []
        for layer in cache.layers:
            L = copy.copy(layer)
            for name, val in vars(layer).items():
                setattr(L, name, grow(val))
            new.layers.append(L)
        return new
    if hasattr(cache, "key_cache"):
        new = copy.copy(cache)
        new.key_cache = [grow(k) for k in cache.key_cache]
        new.value_cache = [grow(v) for v in cache.value_cache]
        return new
    if isinstance(cache, tuple):
        return tuple(tuple(grow(x) for x in layer) for layer in cache)
    raise TypeError(f"don't know how to expand cache of type {type(cache)}")


def softmax(scores: Sequence[float], temperature: float = 1.0) -> list[float]:
    m = max(scores)
    ex = [math.exp((s - m) / temperature) for s in scores]
    z = sum(ex)
    return [e / z for e in ex]
