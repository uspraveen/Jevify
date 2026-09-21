"""The Tier 0 readout on a serving engine: vLLM, with prefix caching and continuous batching.

The research scorer (``HFScorer``) runs an eager Hugging Face forward per batch. It is
exact and inspectable, and it pays ~120 µs per prompt token on an A40 — most of it in
unfused kernels and per-request Python, not in the model. That is the whole reason its
latency climbs with the number of options while Jev's does not (Jev pays ~5.5 µs per
token; see ``results/jev-latency-probe``).

This scorer produces the *same numbers* through vLLM. Each option is a single token, so
the readout is: one forward pass, then the distribution over the K allowed identifiers
at the answer position. vLLM's ``allowed_token_ids`` masks every other token and its
processed log-probs are then exactly the log-softmax over those K -- which is what the
Hugging Face path computes after dropping the shared normaliser. With the state placed
last in the prompt (``Recipe.state_last``), the question and its options are a prefix
shared by every request that asks that question, and vLLM's prefix cache reuses their
KV blocks, so a repeated question costs only its state tokens.

Only the ``index`` mode is supported: the point of this path is single-token options.
"""
from __future__ import annotations

import itertools
import string
from typing import Any, Sequence

from .template import CUE, default_identifiers


def _common(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class VLLMScorer:
    """Duck-typed like ``HFScorer`` for everything ``Tier0Engine`` uses."""

    def __init__(self, model_id: str, *, gpu_memory_utilization: float = 0.6, max_model_len: int = 8192,
                 enable_prefix_caching: bool = True, trust_remote_code: bool = False, hf_token: str | None = None,
                 max_logprobs: int = 260, seed: int = 0) -> None:
        from vllm import LLM, SamplingParams  # noqa: F401  (import here so the HF path never needs vllm)

        self.model_id = model_id
        kw: dict[str, Any] = dict(model=model_id, dtype="bfloat16", enable_prefix_caching=enable_prefix_caching,
                                  gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
                                  max_logprobs=max_logprobs, trust_remote_code=trust_remote_code, seed=seed)
        try:
            self.llm = LLM(logprobs_mode="processed_logprobs", **kw)
            self._processed = True
        except TypeError:                                   # older vLLM without logprobs_mode
            self.llm = LLM(**kw)
            self._processed = False
        self.tokenizer = self.llm.get_tokenizer()
        self.device = "cuda"
        self.stats: dict[str, float] = {"prompts": 0, "generate_s": 0.0}
        self._identifiers: list[str] | None = None
        self._tok_cache: dict[str, int] = {}

    # ------------------------------------------------------------------ identifiers (same rule as HFScorer)
    def identifiers(self, k: int = 255) -> list[str]:
        if self._identifiers is None:
            probe = "Allowed answers: A, B" + chr(10) + CUE
            base = self._encode(probe)
            ok: list[str] = []
            for cand in itertools.chain(string.ascii_uppercase,
                                        ("".join(p) for p in itertools.product(string.ascii_uppercase, repeat=2))):
                ids = self._encode(probe + cand)
                if len(ids) - _common(base, ids) == 1:
                    ok.append(cand)
                if len(ok) >= 255:
                    break
            self._identifiers = ok if len(ok) >= 26 else default_identifiers(255)
        return self._identifiers[:k]

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _candidate_token(self, prefix: str, candidate: str) -> int:
        """The single token a candidate contributes after the cue (cached per candidate)."""
        key = candidate
        if key in self._tok_cache:
            return self._tok_cache[key]
        tail = prefix[prefix.rfind(chr(10)) + 1:]
        base = self._encode(tail)
        ids = self._encode(tail + candidate)
        n = _common(base, ids)
        tok = ids[n] if n < len(ids) else ids[-1]
        self._tok_cache[key] = tok
        return tok

    # ------------------------------------------------------------------ scoring
    def score_many(self, items: Sequence[tuple[str, list[str]]]) -> list[list[float]]:
        """Log-probability of each single-token candidate at the answer position, per item.

        Returned values are log-softmax over the item's own candidates (the normaliser over
        the rest of the vocabulary is a per-item constant that every downstream step --
        softmax, temperature, prior correction -- is invariant to).
        """
        import math
        import time

        from vllm import SamplingParams

        from vllm.inputs import TokensPrompt

        prompts, params, cand_ids = [], [], []
        for prefix, candidates in items:
            # The cue ends in a space, and "Answer: " + "A" tokenizes as [..., "Answer", ":", " A"]:
            # the space merges into the candidate. The HF scorer tokenizes prefix and candidate
            # jointly and so feeds the model "...Answer:" and scores " A". Feeding the prompt
            # *with* its trailing space instead makes the model predict what follows a
            # standalone space token -- a different distribution -- while the logit read is
            # still " A". So the prompt goes in as the exact token ids of the joint
            # tokenization up to the candidate, never as text.
            full = self.tokenizer(prefix, add_special_tokens=True)["input_ids"]
            joint = self.tokenizer(prefix + candidates[0], add_special_tokens=True)["input_ids"]
            n = _common(full, joint)
            toks = [self._candidate_token(prefix, c) for c in candidates]
            prompts.append(TokensPrompt(prompt_token_ids=joint[:n]))
            cand_ids.append(toks)
            params.append(SamplingParams(max_tokens=1, temperature=1.0, logprobs=len(toks),
                                         allowed_token_ids=list(dict.fromkeys(toks))))
        t0 = time.perf_counter()
        outs = self.llm.generate(prompts, params, use_tqdm=False)
        self.stats["generate_s"] += time.perf_counter() - t0
        self.stats["prompts"] += len(prompts)
        result: list[list[float]] = []
        for out, toks in zip(outs, cand_ids):
            lp = out.outputs[0].logprobs[0] if out.outputs[0].logprobs else {}
            raw = [float(lp[t].logprob) if t in lp else -1e4 for t in toks]
            if not self._processed:                         # raw mode: renormalise over the candidates ourselves
                m = max(raw)
                z = m + math.log(sum(math.exp(v - m) for v in raw))
                raw = [v - z for v in raw]
            result.append(raw)
        return result
