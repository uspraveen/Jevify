"""A published Jevified artifact must load and answer, Tier 0 and Tier 1."""
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from jevify import load_jevified
from jevify.bench.record import BenchRecord
from jevify.engine.features import FeatureExtractor
from jevify.engine.heads import HeadConfig, save_heads, train_heads
from jevify.engine.readout import HFScorer

TINY = "HuggingFaceTB/SmolLM2-135M-Instruct"
QUESTIONS = {
    "dept": {"type": "choice", "instructions": "Which team should handle `ticket`?",
             "criteria": {"billing": "Payments and refunds", "shipping": "Delivery problems", "other": None}},
    "refund": {"type": "noul", "instructions": "Does `ticket` ask for a refund?"},
    "anger": {"type": "score", "instructions": "How angry is the customer?", "criteria": ["calm", "annoyed", "furious"]},
}
STATE = {"ticket": "I was charged twice for order A-104, please refund the duplicate."}


def _check(answers):
    assert set(answers) == {"dept", "refund", "anger"}
    d = answers["dept"]
    assert d["type"] == "choice" and d["choice"] in QUESTIONS["dept"]["criteria"]
    assert abs(sum(d["probabilities"].values()) - 1) < 1e-3 and 0 <= d["confidence"] <= 1
    assert 0 <= answers["refund"]["noul"] <= 1 and answers["refund"]["type"] == "noul"
    a = answers["anger"]
    assert a["type"] == "score" and 0 <= a["score"] <= 2 and set(a["legend"]) == {"0", "1", "2"}
    assert abs(sum(a["probabilities"].values()) - 1) < 1e-3


def test_tier0_artifact_roundtrip(tmp_path):
    (tmp_path / "jevify_config.json").write_text(json.dumps({
        "jevify_version": 1, "tier": 0, "backbone": TINY, "chat": True,
        "recipe": {"mode": "index", "permutations": 1, "prior_weight": 0.0,
                   "temperature": {"choice": 1.4, "score": 1.2, "noul": 1.1}, "bias": {"noul": -0.3}},
    }))
    model = load_jevified(str(tmp_path), device="cpu")
    assert model.heads is None and "Tier 0" in repr(model)
    _check(model.ask(STATE, QUESTIONS))


def test_tier1_artifact_roundtrip(tmp_path):
    scorer = HFScorer(TINY, device="cpu", dtype=torch.float32)
    fx = FeatureExtractor(scorer)
    recs = []
    for i in range(8):
        recs.append(BenchRecord(f"c/{i}", "c", "choice", "train", "Charged twice, want money back." if i % 2 else "Where is my parcel?",
                                QUESTIONS["dept"], "billing" if i % 2 else "shipping"))
        recs.append(BenchRecord(f"n/{i}", "n", "noul", "train", "Please refund me." if i % 2 else "Just asking a question.",
                                QUESTIONS["refund"], i % 2))
    rows = fx.extract(recs)
    cfg = HeadConfig(hidden=fx.hidden, dim=32, dropout=0.0, residual=True)
    heads, info = train_heads(rows[:12], rows[12:], cfg, epochs=3, batch_size=4, lr=1e-3, verbose=False)
    save_heads(heads, info, tmp_path / "heads")
    (tmp_path / "jevify_config.json").write_text(json.dumps({
        "jevify_version": 1, "tier": 1, "backbone": TINY, "chat": True, "layer": -1, "residual": True,
        "head_config": cfg.as_dict(),
    }))
    model = load_jevified(str(tmp_path), device="cpu")
    assert model.heads is not None and "Tier 1" in repr(model)
    _check(model.ask(STATE, QUESTIONS))


def test_heads_and_features_share_a_device(tmp_path, monkeypatch):
    """predict_rows must be called on the scorer's device — a CPU default silently
    breaks any GPU deployment (heads on cuda, collated batch on cpu)."""
    import jevify.load as L
    seen = {}
    real = L.__dict__.get("predict_rows")

    scorer = HFScorer(TINY, device="cpu", dtype=torch.float32)
    fx = FeatureExtractor(scorer)
    rows = fx.extract([BenchRecord("n/0", "n", "noul", "test", "Refund please.", QUESTIONS["refund"], 1)])
    cfg = HeadConfig(hidden=fx.hidden, dim=16, dropout=0.0)
    heads, info = train_heads(rows, rows, cfg, epochs=1, batch_size=1, verbose=False)
    save_heads(heads, info, tmp_path / "heads")
    (tmp_path / "jevify_config.json").write_text(json.dumps(
        {"jevify_version": 1, "tier": 1, "backbone": TINY, "chat": True, "layer": -1, "head_config": cfg.as_dict()}))
    model = load_jevified(str(tmp_path), device="cpu")

    import jevify.engine.heads as H
    orig = H.predict_rows

    def spy(m, r, batch_size=64, device="cpu"):
        seen["device"] = device
        return orig(m, r, batch_size, device)

    monkeypatch.setattr("jevify.load.predict_rows", spy, raising=False)
    monkeypatch.setattr(H, "predict_rows", spy)
    model.ask("Refund please.", {"refund": QUESTIONS["refund"]})
    assert seen.get("device") == model.engine.scorer.device, seen
