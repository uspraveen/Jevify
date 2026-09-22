"""A temperature that varies with the option count: it must recover a planted K-dependence,
stay off when the data does not ask for it, and leave old recipes byte-identical."""
import math
import random

import pytest

from jevify.bench.record import BenchRecord
from jevify.engine.predict import Recipe, finalize
from jevify.engine.readout import softmax
from jevify.runners.base import Prediction
from jevify.runners.recipe import K_SLOPE_MARGIN, fit_recipe, fit_temperature_k


def _synth(ks_and_sharpness, n_per=600, seed=0):
    """Records + predictions whose logits are `sharp` times too confident at each K.

    `sharp` is the temperature that would correct them, so a generator built from
    ``1 + slope*log2(K/2)`` plants exactly the affine dependence the fitter looks for."""
    rng = random.Random(seed)
    recs, preds = [], []
    for src, (k, sharp) in ks_and_sharpness.items():
        letters = [chr(65 + i) for i in range(k)]
        q = {"type": "choice", "instructions": "pick", "criteria": {c: None for c in letters}}
        for i in range(n_per):
            z = [rng.gauss(0, 1.5) for _ in range(k)]
            y = rng.choices(range(k), weights=softmax(z, 1.0))[0]
            rid = f"{src}/{'validation' if i % 2 else 'test'}/{i}"
            recs.append(BenchRecord(id=rid, source=src, primitive="choice",
                                    split="validation" if i % 2 else "test", state="s", question=q, label=letters[y]))
            preds.append(Prediction(id=rid, primitive="choice", probabilities=None, answer=None,
                                    extra={"mode": "index", "runs": [{"keys": letters, "logscores": [v * sharp for v in z]}], "prior": None}))
    return recs, preds


SLOPE = 0.75                                    # the dependence planted below: T(K) = 1 + 0.75*log2(K/2)


def test_recovers_a_planted_k_dependence():
    recs, preds = _synth({f"k{k}": (k, 1 + SLOPE * math.log2(k / 2)) for k in (2, 16, 128)})
    val = [r for r in recs if r.split == "validation"]
    vp = [p for p in preds if "/validation/" in p.id]
    scalar, log_s = fit_recipe(val, vp, Recipe(mode="index"), k_slope=False)
    withk, log_k = fit_recipe(val, vp, Recipe(mode="index"), k_slope=True)
    assert not scalar.temp_k_slope
    # one scalar has to compromise between the regimes; it lands between the extremes and fits none
    assert 1.0 < scalar.temperature["choice"] < 1 + SLOPE * math.log2(64)
    assert abs(withk.temp_k_slope.get("choice", 0) - SLOPE) < 0.3, withk.temp_k_slope
    assert log_k["chosen"]["choice"]["val_nll"] < log_s["chosen"]["choice"]["val_nll"] * (1 - K_SLOPE_MARGIN)
    for k in (2, 16, 128):
        planted = 1 + SLOPE * math.log2(k / 2)
        assert abs(withk.temp_for("choice", k) - planted) < 0.35 + 0.2 * planted, (k, withk.temp_for("choice", k), planted)


def test_stays_off_when_sharpness_is_constant():
    recs, preds = _synth({"small": (2, 3.0), "big": (64, 3.0)}, seed=1)
    val = [r for r in recs if r.split == "validation"]
    vp = [p for p in preds if "/validation/" in p.id]
    fitted, _ = fit_recipe(val, vp, Recipe(mode="index"), k_slope=True)
    assert not fitted.temp_k_slope, fitted.temp_k_slope          # nothing to gain -> no extra parameter
    assert 2.0 < fitted.temperature["choice"] < 4.0


def test_one_k_cannot_fit_a_slope():
    ls = [[0.0, 1.0, 2.0]] * 50
    T, slope, nll, base = fit_temperature_k(ls, [2] * 50, [3] * 50, 1.0)
    assert slope == 0.0 and nll == base


def test_slope_changes_answers_and_serializes_only_when_set():
    extra = {"mode": "index", "runs": [{"keys": ["A", "B", "C", "D"], "logscores": [3.0, 0.0, 0.0, 0.0]}], "prior": None}
    q = {"type": "choice", "instructions": "pick", "criteria": dict.fromkeys("ABCD")}
    flat = Recipe(mode="index", temperature={"choice": 1.0, "score": 1.0, "noul": 1.0})
    steep = Recipe(mode="index", temperature={"choice": 1.0, "score": 1.0, "noul": 1.0}, temp_k_slope={"choice": 1.0})
    # K=4 is two doublings above 2, so T = 1 + 1*log2(2) = 2: the answer gets less confident
    assert math.isclose(steep.temp_for("choice", 4), 2.0)
    assert finalize("choice", q, extra, steep).probabilities["A"] < finalize("choice", q, extra, flat).probabilities["A"]
    assert "temp_k_slope" not in flat.as_dict() and flat.as_dict() == Recipe(mode="index").as_dict()
    assert steep.as_dict()["temp_k_slope"] == {"choice": 1.0}
    assert steep.temp_for("choice", 2) == 1.0 and steep.temp_for("noul", 2) == 1.0    # K=2 and other prims untouched


def test_fitting_sees_unrounded_probabilities():
    """At K=151 a real probability of 1e-5 quantizes to 0 in the wire format; the fitter must not
    take the log of that. The answer path keeps the wire precision."""
    k = 151
    keys = [f"o{i}" for i in range(k)]
    q = {"type": "choice", "instructions": "pick", "criteria": dict.fromkeys(keys)}
    ls = [12.0] + [0.0] * (k - 1)                     # one option takes essentially all the mass
    extra = {"mode": "index", "runs": [{"keys": keys, "logscores": ls}], "prior": None}
    wire = finalize("choice", q, extra, Recipe(mode="index"))
    exact = finalize("choice", q, extra, Recipe(mode="index"), round_to=None)
    assert min(wire.probabilities.values()) == 0.0                     # quantized away
    assert min(exact.probabilities.values()) > 0.0                     # still a number
    assert abs(sum(wire.probabilities.values()) - 1.0) < 1e-9          # the wire form still sums to 1
    assert abs(sum(exact.probabilities.values()) - 1.0) < 1e-9
    assert wire.answer == exact.answer == "o0"
