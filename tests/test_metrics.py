import numpy as np

from jevify.bench.metrics import report_categorical, report_noul


def test_perfectly_calibrated_and_confident():
    probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    labels = np.array([0, 1, 0, 1])
    r = report_categorical(probs, labels)
    assert r.accuracy == 1.0
    assert r.ece < 0.2
    assert r.brier < 0.1


def test_ordinal_metrics_prefer_near_misses():
    labels = np.array([2, 2])
    near = np.array([[0, 0.1, 0.4, 0.5, 0], [0, 0, 0.5, 0.5, 0]])
    far = np.array([[0.5, 0, 0.4, 0, 0.1], [0.5, 0, 0.5, 0, 0]])
    rn = report_categorical(near, labels, ordinal=True)
    rf = report_categorical(far, labels, ordinal=True)
    assert rn.rps < rf.rps
    assert rn.mae < rf.mae


def test_noul_report_and_soft_labels():
    p = np.array([0.9, 0.2, 0.7, 0.1])
    y = np.array([1, 0, 1, 0])
    r = report_noul(p, y, soft=np.array([0.85, 0.25, 0.6, 0.05]))
    assert r.accuracy == 1.0
    assert r.auroc == 1.0
    assert r.tvd_to_human is not None and r.tvd_to_human < 0.1
