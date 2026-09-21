"""Training against human label distributions: same proper scoring rules, a different target."""
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from jevify.bench.record import BenchRecord  # noqa: E402
from jevify.engine.features import SlotBatch, _pad_soft, collate, soft_targets  # noqa: E402
from jevify.engine.heads import DecisionHeads, HeadConfig, ranked_probability_loss  # noqa: E402


def _rec(prim, keys, label, soft):
    q = {"type": prim, "instructions": "?"}
    if prim == "choice":
        q["criteria"] = {k: k for k in keys}
    elif prim == "score":
        q["criteria"] = [f"level {k}" for k in keys]
    return BenchRecord(id=f"x/test/{prim}", source="x", primitive=prim, split="test", state="s", question=q,
                       label=label, soft_label=soft)


# --------------------------------------------------------------------------- the three shapes
def test_choice_dict_maps_onto_kept_keys_and_renormalizes():
    r = _rec("choice", ["a", "b", "c", "d"], "b", {"a": 0.1, "b": 0.6, "c": 0.2, "d": 0.1})
    assert soft_targets(r, ["a", "b", "c", "d"]) == pytest.approx([0.1, 0.6, 0.2, 0.1])
    # slot subsampling dropped `c`, which carried mass: renormalize over what is kept
    kept = soft_targets(r, ["a", "b", "d"])
    assert kept == pytest.approx([0.125, 0.75, 0.125])
    assert sum(kept) == pytest.approx(1.0)


def test_score_list_is_indexed_by_level():
    r = _rec("score", ["0", "1", "2"], 2, [0.0, 0.3, 0.7])
    assert soft_targets(r, ["0", "1", "2"]) == pytest.approx([0.0, 0.3, 0.7])


def test_noul_scalar_becomes_yes_no_over_the_keys():
    r = _rec("noul", ["1", "0"], 1, 0.8)
    assert soft_targets(r, ["1", "0"]) == pytest.approx([0.8, 0.2])
    assert soft_targets(r, ["0", "1"]) == pytest.approx([0.2, 0.8])   # follows the key order


def test_missing_or_empty_distribution_is_none():
    assert soft_targets(_rec("choice", ["a", "b"], "a", None), ["a", "b"]) is None
    assert soft_targets(_rec("choice", ["a", "b"], "a", {"a": 0.0, "b": 0.0}), ["a", "b"]) is None


def test_pad_soft_marks_rows_without_a_distribution():
    soft, has = _pad_soft([[0.2, 0.8], None, [1.0]], S=3)
    assert has.tolist() == [True, False, True]
    assert soft[0].tolist() == pytest.approx([0.2, 0.8, 0.0])
    assert soft[1].tolist() == [0.0, 0.0, 0.0]


# --------------------------------------------------------------------------- the loss
def _batch(prim, keys, label, soft, H=16):
    torch.manual_seed(0)
    n = len(keys)
    rows = [{"id": "r", "source": "x", "primitive": prim, "keys": keys, "label": label, "soft": soft,
             "decision": torch.randn(H).numpy(), "slots": torch.randn(n, H).numpy(), "lm": torch.randn(n).numpy()}]
    return collate(rows)


@pytest.mark.parametrize("prim,keys,label", [("choice", ["a", "b", "c"], 1), ("score", ["0", "1", "2", "3"], 2),
                                             ("noul", ["1", "0"], 0)])
def test_one_hot_distribution_reproduces_the_hard_label_loss(prim, keys, label):
    """The switch changes the target, not the rule: a one-hot 'distribution' must score
    exactly as the label index did."""
    onehot = [1.0 if i == label else 0.0 for i in range(len(keys))]
    hard = DecisionHeads(HeadConfig(hidden=16, dim=8, soft_labels=False)).eval()
    soft = DecisionHeads(HeadConfig(hidden=16, dim=8, soft_labels=True)).eval()
    soft.load_state_dict(hard.state_dict())
    l_hard, _ = hard.loss(_batch(prim, keys, label, None))
    l_soft, _ = soft.loss(_batch(prim, keys, label, onehot))
    assert float(l_hard) == pytest.approx(float(l_soft), abs=1e-6)


def test_soft_choice_loss_is_cross_entropy_against_the_distribution():
    heads = DecisionHeads(HeadConfig(hidden=16, dim=8, soft_labels=True)).eval()
    b = _batch("choice", ["a", "b", "c"], 1, [0.2, 0.5, 0.3])
    loss, _ = heads.loss(b)
    o = heads.forward(b)
    logp = F.log_softmax(o["relative"][0] / o["temps"][heads.prim_index["choice"]], dim=-1)
    expected = -(torch.tensor([0.2, 0.5, 0.3]) * logp).sum()
    assert float(loss) == pytest.approx(float(expected), abs=1e-5)


def test_soft_labels_off_ignores_the_distribution_even_when_present():
    off = DecisionHeads(HeadConfig(hidden=16, dim=8, soft_labels=False)).eval()
    l_with, _ = off.loss(_batch("choice", ["a", "b", "c"], 1, [0.2, 0.5, 0.3]))
    l_without, _ = off.loss(_batch("choice", ["a", "b", "c"], 1, None))
    assert float(l_with) == pytest.approx(float(l_without), abs=1e-6)


def test_rps_accepts_a_distribution_and_is_minimized_by_matching_it():
    logits = torch.log(torch.tensor([[0.1, 0.6, 0.3]]))
    mask = torch.ones(1, 3, dtype=torch.bool)
    exact = ranked_probability_loss(logits, torch.tensor([[0.1, 0.6, 0.3]]), mask)
    off = ranked_probability_loss(logits, torch.tensor([[0.6, 0.1, 0.3]]), mask)
    assert float(exact) == pytest.approx(0.0, abs=1e-6) and float(off) > float(exact)
    # the old label-index call still works and equals the one-hot form
    assert float(ranked_probability_loss(logits, torch.tensor([1]), mask)) == pytest.approx(
        float(ranked_probability_loss(logits, torch.tensor([[0.0, 1.0, 0.0]]), mask)), abs=1e-7)


def test_mixed_batch_uses_soft_where_present_and_hard_elsewhere():
    heads = DecisionHeads(HeadConfig(hidden=16, dim=8, soft_labels=True)).eval()
    torch.manual_seed(1)
    H = 16
    rows = [{"id": "a", "source": "x", "primitive": "choice", "keys": ["a", "b"], "label": 0, "soft": [0.7, 0.3],
             "decision": torch.randn(H).numpy(), "slots": torch.randn(2, H).numpy(), "lm": torch.randn(2).numpy()},
            {"id": "b", "source": "y", "primitive": "choice", "keys": ["a", "b"], "label": 1, "soft": None,
             "decision": torch.randn(H).numpy(), "slots": torch.randn(2, H).numpy(), "lm": torch.randn(2).numpy()}]
    b = collate(rows)
    loss, detail = heads.loss(b)
    o = heads.forward(b)
    t = o["temps"][heads.prim_index["choice"]]
    logp = F.log_softmax(o["relative"] / t, dim=-1)
    expected = (-(torch.tensor([0.7, 0.3]) * logp[0]).sum() + -logp[1, 1]) / 2
    assert float(loss) == pytest.approx(float(expected), abs=1e-5)
    assert detail["n_choice"] == 2
