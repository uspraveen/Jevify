"""Render a (state, question) into a scoring prompt plus the allowed continuations.

A System One question is answered by *scoring* each allowed answer, never by
generating text. Rendering therefore produces a ``prefix`` (everything up to
the answer cue) and a list of ``candidates`` (the continuations to score).

Readout modes for Choice:
- ``index``: options are listed with an identifier (A, B, ... or 1, 2, ... when
  K > 26) and the identifier is scored. Cheap; the model must bind id→option.
- ``label``: options are listed by key and the *key itself* is scored. Carries
  meaning for large K; needs a length/prior correction (see calibrate.py).

Score levels are always scored by their index digit; Noul by " yes"/" no".
"""
from __future__ import annotations

import hashlib
import json
import random
import string
from dataclasses import dataclass, field
from typing import Any, Literal

from ..wire import ChoiceQuestion, NoulQuestion, ScoreQuestion, parse_question

ReadoutMode = Literal["index", "label"]

SYSTEM_PROMPT = (
    "You are a System One model: you read a state and answer one question about it "
    "with exactly one of the allowed answers. Do not explain."
)
CONTENT_FREE_STATE = "N/A"
MAX_STATE_CHARS = 12000


@dataclass
class Rendered:
    prefix: str
    candidates: list[str]          # continuations to score, in presentation order
    keys: list[str]                # answer key for each candidate (option key / level index / "1","0")
    primitive: str
    mode: str
    presentation: list[int] = field(default_factory=list)   # permutation applied to the original key order
    question_hash: str = ""

    def by_key(self, scores: list[float]) -> dict[str, float]:
        return dict(zip(self.keys, scores))


def render_state(state: Any) -> str:
    if isinstance(state, str):
        text = state
    else:
        text = json.dumps(state, indent=2, ensure_ascii=False)
    if len(text) > MAX_STATE_CHARS:
        text = text[:MAX_STATE_CHARS] + "\n…[truncated]"
    return text


def _describe(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False)


def _ids(k: int) -> list[str]:
    if k <= 26:
        return list(string.ascii_uppercase[:k])
    return [str(i + 1) for i in range(k)]


def question_hash(question: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(question, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def render(state: Any, question: dict[str, Any], *, mode: ReadoutMode = "index",
           permutation_seed: int | None = None, content_free: bool = False) -> Rendered:
    q = parse_question(question)
    body = render_state(CONTENT_FREE_STATE if content_free else state)
    qh = question_hash(question)
    instr = _describe(q.instructions).strip()

    if isinstance(q, NoulQuestion):
        lines = [f"State:\n{body}", "", f"Question: {instr or 'Is the statement true?'}"]
        if q.criteria:
            if q.criteria.true:
                lines.append(f"Answer yes if: {_describe(q.criteria.true)}")
            if q.criteria.false:
                lines.append(f"Answer no if: {_describe(q.criteria.false)}")
        lines += ["Allowed answers: yes, no", "Answer:"]
        return Rendered("\n".join(lines), [" yes", " no"], ["1", "0"], "noul", "yesno", [0, 1], qh)

    if isinstance(q, ScoreQuestion):
        levels = [_describe(c) for c in q.criteria]
        lines = [f"State:\n{body}", "", f"Question: {instr or 'Which level applies?'}", "Levels (ordered from lowest to highest):"]
        lines += [f"{i}. {d}" for i, d in enumerate(levels)]
        lines += [f"Allowed answers: {', '.join(str(i) for i in range(len(levels)))}", "Answer:"]
        keys = [str(i) for i in range(len(levels))]
        return Rendered("\n".join(lines), [f" {k}" for k in keys], keys, "score", "digit", list(range(len(keys))), qh)

    assert isinstance(q, ChoiceQuestion)
    keys = list(q.criteria.keys())
    order = list(range(len(keys)))
    if permutation_seed is not None:
        random.Random(permutation_seed).shuffle(order)
    shown = [keys[i] for i in order]
    lines = [f"State:\n{body}", "", f"Question: {instr or 'Which option applies?'}", "Options:"]
    if mode == "index":
        ids = _ids(len(shown))
        for ident, k in zip(ids, shown):
            desc = _describe(q.criteria[k])
            lines.append(f"{ident}. {k}" + (f" — {desc}" if desc and desc != k else ""))
        lines += [f"Allowed answers: {', '.join(ids)}", "Answer:"]
        return Rendered("\n".join(lines), [f" {i}" for i in ids], shown, "choice", "index", order, qh)
    for k in shown:
        desc = _describe(q.criteria[k])
        lines.append(f"- {k}" + (f": {desc}" if desc and desc != k else ""))
    lines += ["Answer with exactly one option name.", "Answer:"]
    return Rendered("\n".join(lines), [f" {k}" for k in shown], shown, "choice", "label", order, qh)


def to_chat(prefix: str, tokenizer, *, system: str = SYSTEM_PROMPT) -> str:
    """Wrap a rendered prefix in the model's chat template, leaving the assistant
    turn open so candidates continue it. The trailing "Answer:" cue moves into
    the assistant turn so the model is scored on *its* answer."""
    user, cue = prefix.rsplit("\nAnswer:", 1)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    try:
        text = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except (TypeError, ValueError):
        try:
            text = tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:
            messages = messages[1:]   # no system role support
            text = tokenizer.apply_chat_template(messages, **kwargs)
    return text + "Answer:"
