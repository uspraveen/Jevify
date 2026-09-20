import math

import pytest

from jevify.wire import (ChoiceQuestion, ScoreQuestion, SystemOneRequest, choice_confidence,
                         parse_question, round_probabilities, score_confidence, score_expectation)


def test_choice_confidence_matches_observed_jev_values():
    # (probabilities, confidence reported by jev-1.13.0) captured 2026-09-19
    cases = [
        ({"billing": 0.6, "technical": 0.4, "sales": 0.0}, 0.40),
        ({"other": 0.11, "shipping": 0.0, "billing": 0.41, "returns": 0.48}, 0.30),
        ({"other": 0.77, "support": 0.01, "price": 0.0, "camera": 0.22, "battery": 0.0, "screen": 0.0}, 0.72),
        ({"a": 0.65, "b": 0.35}, 0.31),
    ]
    for probs, want in cases:
        assert abs(choice_confidence(probs) - want) < 0.015


def test_score_statistics():
    assert score_expectation([0.0, 0.7, 0.3]) == pytest.approx(1.3)
    assert score_confidence([0, 1, 0]) == 1.0
    assert score_confidence([0.5, 0, 0.5]) == 0.0          # bimodal at the extremes
    adjacent = score_confidence([0, 0.5, 0.5, 0, 0])
    distant = score_confidence([0.5, 0, 0, 0, 0.5])
    assert adjacent > distant


def test_question_bounds():
    with pytest.raises(ValueError):
        ChoiceQuestion(criteria={"only": None})
    with pytest.raises(ValueError):
        ScoreQuestion(criteria=["one"])
    q = parse_question({"type": "noul", "instructions": "Is it urgent?"})
    assert q.type == "noul"
    req = SystemOneRequest(state="hi", questions={"q": {"type": "choice", "criteria": {"a": None, "b": "B"}}})
    assert req.questions["q"].type == "choice"


def test_round_probabilities_sums_to_one():
    vals = [1 / 3, 1 / 3, 1 / 3]
    r = round_probabilities(vals, 2)
    assert math.isclose(sum(r), 1.0)
    assert sorted(r) == [0.33, 0.33, 0.34]
