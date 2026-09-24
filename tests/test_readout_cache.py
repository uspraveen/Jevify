"""The multi-token path expands a model's cache across candidates. Qwen3.5's linear-attention
layers keep their states in dicts, which the expansion used to skip -- every multi-token candidate
on that family then failed. CPU-only: a stand-in cache with the same shape of state."""
import pytest

torch = pytest.importorskip("torch")

from jevify.engine.readout import _expand_cache, has_recurrent_layers


class LinearAttentionLayer:          # the attribute layout transformers 5.x uses for Qwen3.5
    def __init__(self):
        self.conv_states = {0: torch.zeros(1, 8, 4)}
        self.recurrent_states = {0: torch.zeros(1, 2, 3, 3)}
        self.has_previous_state = {0: True}
        self.number_of_states = 1


class DynamicLayer:
    def __init__(self):
        self.keys = torch.zeros(1, 2, 5, 4)
        self.values = torch.zeros(1, 2, 5, 4)


class Cache:
    def __init__(self):
        self.layers = [LinearAttentionLayer(), DynamicLayer()]


def test_dict_states_are_expanded():
    c = _expand_cache(Cache(), 6)
    lin, att = c.layers
    assert lin.conv_states[0].shape[0] == 6 and lin.recurrent_states[0].shape[0] == 6
    assert lin.has_previous_state == {0: True} and lin.number_of_states == 1
    assert att.keys.shape[0] == 6 and att.values.shape[0] == 6


def test_recurrent_layers_are_detected():
    class GatedDeltaNet(torch.nn.Module):
        pass

    class Hybrid(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(2, 2)
            self.b = GatedDeltaNet()

    assert has_recurrent_layers(Hybrid()) and not has_recurrent_layers(torch.nn.Linear(2, 2))
