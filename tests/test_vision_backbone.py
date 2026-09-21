"""Control over the vision tower: find it, scope LoRA to it, budget what it sees."""
import re

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from jevify.engine import vision_backbone as vb  # noqa: E402


# --------------------------------------------------------------------------- synthetic VLM
class _Block(nn.Module):
    def __init__(self, dim, names):
        super().__init__()
        for n in names:
            setattr(self, n, nn.Linear(dim, dim))


class _Tower(nn.Module):
    def __init__(self, dim=32, layers=3):
        super().__init__()
        self.layers = nn.ModuleList([_Block(dim, ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"))
                                     for _ in range(layers)])


class _Decoder(nn.Module):
    def __init__(self, dim=64, layers=2):
        super().__init__()
        self.layers = nn.ModuleList([_Block(dim, ("q_proj", "k_proj", "v_proj", "o_proj",
                                                  "gate_proj", "up_proj", "down_proj"))
                                     for _ in range(layers)])


class _FakeVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_tower = _Tower()
        self.multi_modal_projector = nn.Linear(32, 64)
        self.language_model = _Decoder()


class _FakeLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _Decoder()


def test_describe_finds_tower_and_projector():
    info = vb.describe(_FakeVLM())
    assert info["is_vlm"] is True
    assert info["vision_tower"]["path"] == "vision_tower"
    assert info["vision_tower"]["layers"] == 3
    assert info["projector"]["path"] == "multi_modal_projector"
    assert 0 < info["vision_tower"]["share_of_model"] < 1
    assert info["decoder_params"] > 0


def test_describe_on_text_model_is_not_a_vlm():
    info = vb.describe(_FakeLM())
    assert info["is_vlm"] is False and info["vision_tower"] is None


def _matched(model, pattern):
    rx = re.compile(pattern)
    return [n for n, _ in model.named_modules() if rx.match(n)]


def test_lora_pattern_vision_touches_only_the_tower():
    """The bug this guards: `q_proj` is a suffix on both halves of a VLM.

    peft matches target_modules against the end of a module path, so a suffix list that
    means "adapt the encoder" silently adapts the decoder too. Only a full-path regex
    actually scopes it.
    """
    m = _FakeVLM()
    hit = _matched(m, vb.lora_pattern(m, "vision"))
    assert hit, "expected some vision linears"
    assert all(n.startswith("vision_tower.") for n in hit)
    assert any(n.endswith("q_proj") for n in hit)          # a name the decoder also has
    assert not any("language_model" in n for n in hit)


def test_lora_pattern_decoder_excludes_the_tower():
    m = _FakeVLM()
    hit = _matched(m, vb.lora_pattern(m, "decoder"))
    assert hit
    assert not any(n.startswith("vision_tower.") for n in hit)
    assert any(n.endswith("gate_proj") for n in hit)


def test_lora_pattern_both_covers_each_half():
    m = _FakeVLM()
    hit = _matched(m, vb.lora_pattern(m, "both"))
    assert any(n.startswith("vision_tower.") for n in hit)
    assert any(n.startswith("language_model.") for n in hit)


def test_lora_pattern_vision_rejects_a_text_model():
    with pytest.raises(ValueError):
        vb.lora_pattern(_FakeLM(), "vision")


def test_freeze_vision_only_touches_the_tower():
    m = _FakeVLM()
    changed = vb.freeze_vision(m, True)
    assert changed > 0
    assert all(not p.requires_grad for p in m.vision_tower.parameters())
    assert all(p.requires_grad for p in m.language_model.parameters())
    assert vb.describe(m)["vision_tower"]["frozen"] is True
    assert vb.freeze_vision(m, True) == 0                  # idempotent
    assert vb.freeze_vision(m, False) > 0


# --------------------------------------------------------------------------- real processor
TINY_VLM = "HuggingFaceTB/SmolVLM-256M-Instruct"


@pytest.fixture(scope="module")
def processor():
    pytest.importorskip("transformers")
    pytest.importorskip("torchvision")
    from transformers import AutoProcessor

    try:
        return AutoProcessor.from_pretrained(TINY_VLM)
    except Exception as exc:                                # offline / gated
        pytest.skip(f"processor unavailable: {exc}")


def test_pixel_budget_reports_what_actually_applied(processor):
    """A processor silently ignores a budget key it does not have, so the call reports back."""
    applied = vb.pixel_budget(processor, max_pixels=256 * 28 * 28)
    assert "effective" in applied
    if "max_pixels" in applied:                             # this family honours the key
        assert applied["effective"]["max_pixels"] == 256 * 28 * 28


def test_image_tokens_grows_with_the_image(processor):
    from PIL import Image

    small = vb.image_tokens(processor, Image.new("RGB", (64, 64), "red"))
    large = vb.image_tokens(processor, Image.new("RGB", (512, 512), "red"))
    assert small > 0 and large >= small


# --------------------------------------------------------------------------- budget plumbing
class _SizeDict:
    """Stand-in for transformers 5.x SizeDict: attributes, not dict keys."""

    def __init__(self):
        self.longest_edge = None
        self.shortest_edge = None
        self.max_pixels = None
        self.min_pixels = None


class _ModernIP:
    """transformers 5.x: the budget lives in `size`, the legacy attributes are gone."""

    def __init__(self):
        self.size = _SizeDict()


class _LegacyIP:
    """transformers 4.x: plain attributes."""

    def __init__(self):
        self.max_pixels = None
        self.min_pixels = None


class _Proc:
    def __init__(self, ip):
        self.image_processor = ip


def test_pixel_budget_writes_the_modern_size_object():
    """Regression: a real Qwen3-VL processor reported an empty budget.

    transformers 5.x folds `max_pixels` into `size.longest_edge` and drops the legacy
    attribute, so reading only `ip.max_pixels` reports None for a budget that is in fact
    applied -- which would make a sweep over budgets look like it changed nothing.
    """
    proc = _Proc(_ModernIP())
    applied = vb.pixel_budget(proc, max_pixels=64 * 28 * 28, min_pixels=16 * 28 * 28)
    assert applied["max_pixels"] == 64 * 28 * 28
    assert proc.image_processor.size.longest_edge == 64 * 28 * 28
    assert proc.image_processor.size.shortest_edge == 16 * 28 * 28
    assert applied["effective"]["max_pixels"] == 64 * 28 * 28
    assert applied["effective"]["min_pixels"] == 16 * 28 * 28


def test_pixel_budget_still_writes_legacy_attributes():
    proc = _Proc(_LegacyIP())
    applied = vb.pixel_budget(proc, max_pixels=1234)
    assert proc.image_processor.max_pixels == 1234
    assert applied["effective"]["max_pixels"] == 1234


def test_pixel_budget_reports_nothing_applied_when_nothing_matches():
    class _Blank:
        pass

    applied = vb.pixel_budget(_Proc(_Blank()), max_pixels=999)
    assert "max_pixels" not in applied            # honest: the budget did not take
    assert applied["effective"]["max_pixels"] is None
