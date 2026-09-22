"""Vision Tier 2: LoRA on the readout, confined to one half of a VLM."""
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
PIL = pytest.importorskip("PIL")

from PIL import Image

from jevify.bench.record import BenchRecord
from jevify.engine.vision import VisionScorer
from jevify.engine.vision_backbone import find_vision_tower
from jevify.engine.vision_tier2 import VisionReadout, attach_lora, readout_loss, train_vision_tier2

TINY_VLM = "HuggingFaceTB/SmolVLM-256M-Instruct"


def _img(color):
    return Image.new("RGB", (64, 64), color)


def _rec(i, color, label, prim="choice"):
    if prim == "choice":
        q = {"type": "choice", "instructions": "What colour fills `image`?", "criteria": {"red": None, "blue": None, "green": None}}
    elif prim == "noul":
        q = {"type": "noul", "instructions": "Is `image` red?"}
    else:
        q = {"type": "score", "instructions": "How red is `image`?", "criteria": ["not at all", "somewhat", "fully"]}
    return BenchRecord(id=f"toy/train/{i}", source="toy", primitive=prim, split="train",
                       state={"image": _img(color)}, question=q, label=label)


@pytest.fixture(scope="module")
def scorer():
    return VisionScorer(TINY_VLM, device="cpu", dtype=torch.float32, batch_size=2)


def test_readout_matches_tier0_at_step_zero(scorer):
    """Before any training the differentiable readout *is* the Tier 0 readout."""
    recs = [_rec(0, "red", "red"), _rec(1, "blue", "blue")]
    rb = VisionReadout(scorer).batch(recs)
    items = [scorer.item(r.state, r.question)[0] for r in recs]
    t0 = scorer.score_many(items)
    for row, sc in enumerate(t0):
        ref = torch.tensor(sc) - torch.logsumexp(torch.tensor(sc), 0)
        assert torch.allclose(rb.logscores[row, : len(sc)], ref, atol=1e-4)
        assert math.isclose(float(rb.logscores[row, : len(sc)].exp().sum()), 1.0, abs_tol=1e-4)


def test_loss_covers_every_primitive(scorer):
    recs = [_rec(0, "red", "red"), _rec(1, "red", 1, "noul"), _rec(2, "red", "2", "score")]
    rb = VisionReadout(scorer).batch(recs)
    loss, detail = readout_loss(rb)
    assert loss.item() > 0 and {"choice", "noul", "score"} <= set(detail)
    assert detail["n_choice"] == detail["n_noul"] == detail["n_score"] == 1


def test_vision_scoped_lora_touches_only_the_tower_and_restores_the_best_epoch():
    """The mechanism, not the outcome: a toy of six solid-colour images says nothing about whether
    a tower adapter *helps* (that is what the A-OKVQA run measures), but it can check that only the
    tower moves, that the base weights never do, and that what remains after training is the best
    epoch by validation loss rather than the last one."""
    sc = VisionScorer(TINY_VLM, device="cpu", dtype=torch.float32, batch_size=2)
    tower_name, _ = find_vision_tower(sc.model)
    assert tower_name
    base_before = {n: p.detach().clone() for n, p in sc.model.named_parameters()}
    pattern, trainable = attach_lora(sc, where="vision", r=4)
    assert trainable > 0
    grads = [n for n, p in sc.model.named_parameters() if p.requires_grad]
    assert grads and all("lora_" in n for n in grads)
    assert all(tower_name in n for n in grads), "a vision-scoped adapter reached the decoder"

    train = [_rec(i, c, c) for i, c in enumerate(["red", "blue", "green", "red", "blue", "green"])]
    val = [_rec(10, "red", "red"), _rec(11, "blue", "blue")]
    readout = VisionReadout(sc)
    info = train_vision_tier2(sc, train, val, epochs=2, batch_size=2, grad_accum=1, lr=1e-3, checkpoint_dir=None)
    assert info["best_epoch"] >= 0 and len(info["history"]) == 2
    after = readout_loss(readout.batch(val))[0].item()
    best = min(h["loss"] for h in info["history"])
    assert abs(after - best) < 1e-3, (after, best, info["history"])      # the best epoch is what remains
    live = dict(sc.model.named_parameters())
    for n, v in base_before.items():                                     # the base weights never moved
        wrapped = next((m for m in live if m.endswith(n)), None)
        assert wrapped is not None and torch.equal(live[wrapped], v), n


def test_decoder_scoped_lora_leaves_the_tower_alone():
    sc = VisionScorer(TINY_VLM, device="cpu", dtype=torch.float32, batch_size=2)
    tower_name, _ = find_vision_tower(sc.model)
    attach_lora(sc, where="decoder", r=4)
    grads = [n for n, p in sc.model.named_parameters() if p.requires_grad]
    assert grads and not any(f".{tower_name}." in f".{n}" or n.startswith(tower_name + ".") for n in grads)
