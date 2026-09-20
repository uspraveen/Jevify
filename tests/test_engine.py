"""Engine tests on a 135M model (CPU). The point is correctness of the scoring
machinery, not model quality."""
import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from jevify.bench.record import BenchRecord
from jevify.engine.calibrate import fit_temperature, prior_correct
from jevify.engine.predict import Recipe, Tier0Engine, finalize, refinalize
from jevify.engine.readout import HFScorer, softmax
from jevify.engine.template import render, to_chat

TINY = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.fixture(scope="module")
def scorer():
    return HFScorer(TINY, device="cpu", dtype=torch.float32, batch_size=4, cand_chunk=3)


CHOICE_Q = {"type": "choice", "instructions": "Which city is named in the state?",
            "criteria": {"paris": "The capital of France", "berlin": "The capital of Germany",
                         "tokyo": "The capital of Japan", "lima": "The capital of Peru"}}
SCORE_Q = {"type": "score", "instructions": "How positive is the review?", "criteria": ["negative", "neutral", "positive"]}
NOUL_Q = {"type": "noul", "instructions": "Is the statement in the state true?"}


def test_render_shapes():
    r = render("hello", CHOICE_Q, mode="index")
    assert r.candidates == ["A", "B", "C", "D"] and r.keys == ["paris", "berlin", "tokyo", "lima"]
    r2 = render("hello", CHOICE_Q, mode="index", permutation_seed=7)
    assert sorted(r2.keys) == sorted(r.keys) and r2.keys != r.keys
    big = {"type": "choice", "criteria": {f"k{i}": None for i in range(40)}}
    assert render("x", big).candidates[0] == "1" and render("x", big).candidates[-1] == "40"
    s = render("x", SCORE_Q)
    assert s.candidates == ["0", "1", "2"] and s.keys == ["0", "1", "2"]
    n = render("x", NOUL_Q)
    assert n.candidates == ["yes", "no"] and n.keys == ["1", "0"]
    lab = render("x", CHOICE_Q, mode="label")
    assert lab.candidates == ["paris", "berlin", "tokyo", "lima"]


def test_tokenization_paths(scorer):
    r = render("The city is Paris.", CHOICE_Q, mode="index")
    t = scorer.tokenize(r.prefix, r.candidates)
    assert t.single_token, [len(c) for c in t.cand_ids]
    lab = render("The city is Paris.", CHOICE_Q, mode="label")
    t2 = scorer.tokenize(lab.prefix, lab.candidates)
    assert all(len(c) >= 1 for c in t2.cand_ids)
    # joint tokenization: prefix tokens must be a prefix of every full sequence
    for c in lab.candidates:
        full = scorer._encode(lab.prefix + c)
        assert full[: len(t2.prefix_ids)] == t2.prefix_ids


def test_fast_paths_match_naive(scorer):
    items = []
    for mode in ("index", "label"):
        r = render("I visited Paris last summer and loved it.", CHOICE_Q, mode=mode)
        items.append((r.prefix, r.candidates))
    s = render("Absolutely wonderful, would buy again.", SCORE_Q)
    items.append((s.prefix, s.candidates))
    n = render("Paris is the capital of France.", NOUL_Q)
    items.append((to_chat(n.prefix, scorer.tokenizer), n.candidates))
    fast = scorer.score_many(items)
    naive = scorer.score_many(items, naive=True)
    for f, nv in zip(fast, naive):
        assert len(f) == len(nv)
        for a, b in zip(f, nv):
            assert abs(a - b) < 2e-3, (f, nv)


def test_semantics_and_distribution(scorer):
    engine = Tier0Engine(scorer, Recipe(mode="index", chat=True, permutations=2))
    recs = [
        BenchRecord("t/1", "t", "choice", "test", "I flew into Paris and walked along the Seine.", CHOICE_Q, "paris"),
        BenchRecord("t/2", "t", "noul", "test", "Paris is the capital of France.", NOUL_Q, 1),
        BenchRecord("t/3", "t", "noul", "test", "Berlin is the capital of France.", NOUL_Q, 0),
        BenchRecord("t/4", "t", "score", "test", "Absolutely wonderful, would buy again.", SCORE_Q, 2),
    ]
    preds = list(engine.score_records(recs, batch=8))
    assert [p.id for p in preds] == ["t/1", "t/2", "t/3", "t/4"]
    c = preds[0]
    assert abs(sum(c.probabilities.values()) - 1) < 1e-6 and set(c.probabilities) == set(CHOICE_Q["criteria"])
    assert len(c.extra["runs"]) == 2 and c.extra["prior"] is not None
    assert preds[1].p_yes > preds[2].p_yes, "true statement should get higher P(yes) than false one"
    sc = preds[3]
    assert abs(sum(sc.probabilities.values()) - 1) < 1e-6 and 0 <= sc.answer <= 2
    # recipes are offline: re-finalizing with prior correction and a temperature keeps valid distributions
    alt = refinalize(recs, preds, Recipe(mode="index", permutations=2, prior_weight=1.0, temperature={"choice": 2.0, "score": 0.5, "noul": 1.0}))
    assert abs(sum(alt[0].probabilities.values()) - 1) < 1e-6
    assert alt[0].probabilities != c.probabilities


def test_prior_alignment_by_mode():
    extra = {"mode": "index", "runs": [{"keys": ["b", "a"], "logscores": [0.0, 0.0]}],
             "prior": {"keys": ["a", "b"], "logscores": [math.log(0.9), math.log(0.1)]}}
    q = {"type": "choice", "criteria": {"a": None, "b": None}}
    p = finalize("choice", q, extra, Recipe(mode="index", prior_weight=1.0))
    # index mode: position 0 (identifier A) has prior 0.9 -> whatever sits there ("b") is corrected down
    assert p.probabilities["b"] < p.probabilities["a"]
    extra["mode"] = "label"
    p2 = finalize("choice", q, extra, Recipe(mode="label", prior_weight=1.0))
    # label mode: key "a" has prior 0.9 -> "a" is corrected down
    assert p2.probabilities["a"] < p2.probabilities["b"]


def test_temperature_recovery():
    import random
    rng = random.Random(0)
    logits, labels = [], []
    for _ in range(3000):
        z = [rng.gauss(0, 2) for _ in range(4)]
        p = softmax(z, 1.0)
        y = rng.choices(range(4), weights=p)[0]
        logits.append([v * 3.0 for v in z])   # model reports logits 3x too sharp -> true T = 3
        labels.append(y)
    T = fit_temperature(logits, labels)
    assert 2.5 < T < 3.5, T


def test_tree_attention_matches_naive(scorer):
    """The block-diagonal mask must isolate candidates exactly (Llama-style SDPA)."""
    from jevify.engine.readout import Tokenized

    r = render("I visited Paris last summer and loved it.", CHOICE_Q, mode="label")
    t = scorer.tokenize(r.prefix, r.candidates)
    tree = scorer._score_tree(t)
    ref = scorer._score_naive(t)
    for a, b in zip(tree, ref):
        assert abs(a - b) < 2e-3, (tree, ref)
    big = {"type": "choice", "instructions": "Pick the number of words in the state.",
           "criteria": {str(i): None for i in range(1, 41)}}
    r2 = render("one two three", big)
    t2 = scorer.tokenize(r2.prefix, r2.candidates)
    assert not t2.single_token
    tree2 = scorer._score_tree(t2)
    ref2 = scorer._score_naive(Tokenized(t2.prefix_ids, t2.cand_ids[:6]))
    for a, b in zip(tree2[:6], ref2):
        assert abs(a - b) < 2e-3


def test_identifiers_are_single_tokens(scorer):
    ids = scorer.identifiers(255)
    assert ids[:26] == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") and len(ids) >= 26
    big = {"type": "choice", "criteria": {f"k{i}": None for i in range(len(ids))}}
    r = render("x", big, identifiers=ids)
    t = scorer.tokenize(r.prefix, r.candidates)
    assert t.single_token, [c for c in t.cand_ids if len(c) != 1][:3]
    s = render("x", SCORE_Q)
    assert scorer.tokenize(s.prefix, s.candidates).single_token
    n = render("x", NOUL_Q)
    assert scorer.tokenize(to_chat(n.prefix, scorer.tokenizer), n.candidates).single_token
