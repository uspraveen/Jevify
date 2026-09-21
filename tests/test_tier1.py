"""Tier 1: slot features and decision heads, on CPU with a 135M model."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from jevify.bench.record import BenchRecord
from jevify.engine.features import FeatureExtractor, collate, load_features, save_features
from jevify.engine.heads import DecisionHeads, HeadConfig, predict_rows, ranked_probability_loss, train_heads
from jevify.engine.readout import HFScorer

TINY = "HuggingFaceTB/SmolLM2-135M-Instruct"
CHOICE_Q = {"type": "choice", "instructions": "Which city is named?",
            "criteria": {"paris": "The capital of France", "berlin": "The capital of Germany", "tokyo": "The capital of Japan"}}
SCORE_Q = {"type": "score", "instructions": "How positive?", "criteria": ["negative", "neutral", "positive"]}
NOUL_Q = {"type": "noul", "instructions": "Is the statement true?"}


@pytest.fixture(scope="module")
def extractor():
    return FeatureExtractor(HFScorer(TINY, device="cpu", dtype=torch.float32))


def _records(n=12):
    recs = []
    for i in range(n):
        city = ["paris", "berlin", "tokyo"][i % 3]
        recs.append(BenchRecord(f"c/{i}", "c", "choice", "train", f"I landed in {city} yesterday.", CHOICE_Q, city))
        recs.append(BenchRecord(f"s/{i}", "s", "score", "train", "Wonderful!" if i % 2 else "Awful.", SCORE_Q, 2 if i % 2 else 0))
        recs.append(BenchRecord(f"n/{i}", "n", "noul", "train", "Paris is in France." if i % 2 else "Paris is in Peru.", NOUL_Q, i % 2))
    return recs


def test_features_shapes_and_roundtrip(extractor, tmp_path):
    rows = extractor.extract(_records(2))
    assert len(rows) == 6
    H = extractor.hidden
    for r in rows:
        assert r["decision"].shape == (H,)
        assert r["slots"].shape == (len(r["keys"]), H)
        assert 0 <= r["label"] < len(r["keys"])
        # the slots must not be identical: they are different positions in the prompt
        assert not np.allclose(r["slots"][0], r["slots"][-1])
    p = tmp_path / "f.npz"
    save_features(p, rows)
    back = load_features(p)
    assert [b["id"] for b in back] == [r["id"] for r in rows]
    assert np.allclose(back[0]["slots"], rows[0]["slots"])


def test_option_subsampling_keeps_gold(extractor):
    q = {"type": "choice", "instructions": "Which?", "criteria": {f"k{i}": f"option {i}" for i in range(10)}}
    recs = [BenchRecord("x/1", "x", "choice", "train", "state", q, "k7")]
    rows = extractor.extract(recs, max_slots=4, rng=np.random.default_rng(0))
    assert len(rows[0]["keys"]) == 4 and "k7" in rows[0]["keys"]
    assert rows[0]["keys"][rows[0]["label"]] == "k7"


def test_rps_prefers_near_misses():
    logits = torch.tensor([[0.0, 5.0, 0.0, 0.0]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    near = ranked_probability_loss(logits, torch.tensor([2]), mask)
    far = ranked_probability_loss(logits, torch.tensor([3]), mask)
    assert near < far


def test_heads_learn_and_stay_normalized(extractor):
    rows = extractor.extract(_records(10))
    train, val = rows[:24], rows[24:]
    cfg = HeadConfig(hidden=extractor.hidden, dim=64, dropout=0.0)
    model, info = train_heads(train, val, cfg, epochs=12, batch_size=8, lr=3e-3, verbose=False)
    assert info["history"][-1]["train"] < info["history"][0]["train"], "training loss should fall"
    preds = predict_rows(model, val)
    for row in val:
        p = preds[row["id"]]
        assert set(p) == set(row["keys"])
        assert abs(sum(p.values()) - 1) < 1e-4
        assert all(0 <= v <= 1 for v in p.values())


def test_head_is_k_agnostic(extractor):
    """A head trained with 3 options must accept a 10-option question unchanged."""
    cfg = HeadConfig(hidden=extractor.hidden, dim=32, dropout=0.0)
    model = DecisionHeads(cfg)
    q10 = {"type": "choice", "instructions": "Which?", "criteria": {f"k{i}": f"option {i}" for i in range(10)}}
    rows = extractor.extract([BenchRecord("x/1", "x", "choice", "test", "state", q10, "k3")])
    p = predict_rows(model, rows)["x/1"]
    assert len(p) == 10 and abs(sum(p.values()) - 1) < 1e-4


def test_noul_is_absolute_not_softmax(extractor):
    """P(yes) must come from the sigmoid head, so two noul rows can both be high."""
    cfg = HeadConfig(hidden=extractor.hidden, dim=32, dropout=0.0)
    model = DecisionHeads(cfg)
    rows = extractor.extract([BenchRecord("n/1", "n", "noul", "test", "Paris is in France.", NOUL_Q, 1)])
    batch = collate(rows)
    out = model(batch)
    assert out["absolute"].shape[1] == 2
    p = predict_rows(model, rows)["n/1"]
    assert abs(p["1"] + p["0"] - 1) < 1e-5


def test_lm_scores_captured_and_match_tier0(extractor):
    """The LM log-score read from the feature pass must equal Tier 0's own scoring."""
    from jevify.engine.template import render, to_chat
    rec = BenchRecord("c/0", "c", "choice", "test", "I landed in berlin yesterday.", CHOICE_Q, "berlin")
    row = extractor.extract([rec])[0]
    rd = render(rec.state, rec.question, identifiers=extractor.scorer.identifiers())
    prefix = to_chat(rd.prefix, extractor.scorer.tokenizer) if extractor.chat else rd.prefix
    ref = extractor.scorer.score_many([(prefix, rd.candidates)])[0]
    assert len(row["lm"]) == len(ref)
    for a, b in zip(row["lm"], ref):
        assert abs(float(a) - b) < 2e-3, (row["lm"], ref)


def test_residual_head_starts_at_tier0(extractor):
    """With residual=True and a zero-initialized correction, an untrained head must
    reproduce the Tier 0 distribution exactly."""
    import torch
    from jevify.engine.features import collate
    rows = extractor.extract([BenchRecord("c/0", "c", "choice", "test", "I flew to tokyo.", CHOICE_Q, "tokyo")])
    model = DecisionHeads(HeadConfig(hidden=extractor.hidden, dim=32, dropout=0.0, residual=True))
    p = predict_rows(model, rows)["c/0"]
    tier0 = torch.softmax(torch.tensor(rows[0]["lm"], dtype=torch.float32), dim=-1)
    for k, ref in zip(rows[0]["keys"], tier0):
        assert abs(p[k] - float(ref)) < 1e-4, (p, tier0)


def test_non_residual_head_ignores_lm(extractor):
    rows = extractor.extract([BenchRecord("c/0", "c", "choice", "test", "I flew to tokyo.", CHOICE_Q, "tokyo")])
    model = DecisionHeads(HeadConfig(hidden=extractor.hidden, dim=32, dropout=0.0, residual=False))
    p = predict_rows(model, rows)["c/0"]
    assert max(p.values()) < 0.9, "a fresh non-residual head should be near-uniform"
