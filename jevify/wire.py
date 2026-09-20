"""The System One wire format: questions in, typed answers out.

This mirrors TypeSafe's ``POST /v1/systemone`` contract field-for-field so the
official ``typesafe-sdk`` (and anything built on it) works against a Jevified
model via ``TYPESAFE_BASE_URL``. Where TypeSafe's behaviour is undisclosed we
define our own and say so (see ``score_confidence``).
"""
from __future__ import annotations

import math
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

JSONContent = Union[str, dict[str, Any], list[Any]]

MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10
MIN_SCORE_LEVELS = 2


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- questions

class NoulCriteria(_Strict):
    true: JSONContent | None = None
    false: JSONContent | None = None


class NoulQuestion(_Strict):
    type: Literal["noul"] = "noul"
    instructions: JSONContent | None = None
    criteria: NoulCriteria | None = None


class ChoiceQuestion(_Strict):
    type: Literal["choice"] = "choice"
    instructions: JSONContent | None = None
    criteria: dict[str, JSONContent | None]

    @field_validator("criteria")
    @classmethod
    def _bounded(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not 2 <= len(v) <= MAX_CHOICE_OPTIONS:
            raise ValueError(f"choice needs 2..{MAX_CHOICE_OPTIONS} options, got {len(v)}")
        if any(not k for k in v):
            raise ValueError("choice option names must be non-empty")
        return v


class ScoreQuestion(_Strict):
    type: Literal["score"] = "score"
    instructions: JSONContent | None = None
    criteria: list[JSONContent]

    @field_validator("criteria")
    @classmethod
    def _bounded(cls, v: list[Any]) -> list[Any]:
        if not MIN_SCORE_LEVELS <= len(v) <= MAX_SCORE_LEVELS:
            raise ValueError(f"score needs {MIN_SCORE_LEVELS}..{MAX_SCORE_LEVELS} levels, got {len(v)}")
        return v


Question = Union[NoulQuestion, ChoiceQuestion, ScoreQuestion]


def parse_question(q: dict[str, Any] | Question) -> Question:
    if isinstance(q, (NoulQuestion, ChoiceQuestion, ScoreQuestion)):
        return q
    kind = q.get("type")
    if kind == "noul":
        return NoulQuestion.model_validate(q)
    if kind == "choice":
        return ChoiceQuestion.model_validate(q)
    if kind == "score":
        return ScoreQuestion.model_validate(q)
    raise ValueError(f"unknown question type: {kind!r}")


class SystemOneRequest(_Strict):
    state: JSONContent
    model: str = "jevify-latest"
    questions: dict[str, Question]

    @model_validator(mode="before")
    @classmethod
    def _parse_questions(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("questions"), dict):
            data = dict(data)
            data["questions"] = {k: parse_question(v) for k, v in data["questions"].items()}
            if not data["questions"]:
                raise ValueError("at least one question is required")
        return data


# --------------------------------------------------------------------------- answers

class NoulAnswer(_Strict):
    type: Literal["noul"] = "noul"
    noul: float = Field(ge=0.0, le=1.0)


class ChoiceAnswer(_Strict):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]


class ScoreAnswer(_Strict):
    type: Literal["score"] = "score"
    score: float
    confidence: float = Field(ge=0.0, le=1.0)
    legend: dict[str, JSONContent]
    probabilities: dict[str, float]


Answer = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]


class Usage(_Strict):
    input_tokens: int
    output_tokens: int


class SystemOneResponse(_Strict):
    model: str
    answers: dict[str, Answer]
    usage: Usage


# --------------------------------------------------------------------------- statistics

def choice_confidence(probabilities: dict[str, float] | list[float]) -> float:
    """TypeSafe's Choice confidence, verified empirically: ``(p_max - 1/K) / (1 - 1/K)``.

    Uniform -> 0, one-hot -> 1. This is the rescaled maximum probability, *not*
    normalized entropy (which does not fit their responses).
    """
    p = list(probabilities.values()) if isinstance(probabilities, dict) else list(probabilities)
    k = len(p)
    if k < 2:
        return 1.0
    return _clip01((max(p) - 1.0 / k) / (1.0 - 1.0 / k))


def score_expectation(probabilities: list[float]) -> float:
    """The probability-weighted level index (TypeSafe's ``score``)."""
    return float(sum(i * p for i, p in enumerate(probabilities)))


def score_confidence(probabilities: list[float]) -> float:
    """Jevify's Score confidence: ``1 - std / std_max``.

    TypeSafe's Score confidence is not reproducible from its published
    distributions by any standard statistic we tried, so this is *our*
    definition: one minus the standard deviation of the level index divided
    by the largest possible std for K levels (all mass split between the two
    extremes, ``(K-1)/2``). Concentrated on one level -> 1; bimodal at the
    extremes -> 0; mass on two adjacent levels scores higher than the same mass
    on distant levels, which is the ordinal behaviour a spread statistic should
    have and a plain max-probability does not.
    """
    k = len(probabilities)
    if k < 2:
        return 1.0
    mu = score_expectation(probabilities)
    var = sum(p * (i - mu) ** 2 for i, p in enumerate(probabilities))
    std_max = (k - 1) / 2.0
    return _clip01(1.0 - math.sqrt(max(var, 0.0)) / std_max)


def round_probabilities(values: list[float], places: int = 4) -> list[float]:
    """Round a distribution so it still sums to exactly 1 (largest-remainder)."""
    scale = 10 ** places
    raw = [max(v, 0.0) * scale for v in values]
    total = sum(raw) or 1.0
    raw = [v * scale / total for v in raw]
    floors = [math.floor(v) for v in raw]
    remainder = scale - sum(floors)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - floors[i], reverse=True)
    for i in order[:remainder]:
        floors[i] += 1
    return [f / scale for f in floors]


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))
