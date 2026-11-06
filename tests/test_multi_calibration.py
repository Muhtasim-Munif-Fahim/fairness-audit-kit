import numpy as np
import pytest

from fairness_audit_kit.multi_calibration import (
    compute_multi_calibration,
    multi_accuracy,
    render_multi_calibration_report,
)


def test_perfectly_calibrated_scores_have_small_error():
    rng = np.random.default_rng(0)
    n = 4000
    scores = rng.uniform(0, 1, size=n)
    y = (rng.uniform(0, 1, size=n) < scores).astype(float)
    groups = rng.choice(["a", "b"], size=n)
    res = compute_multi_calibration(y, scores, groups, n_bins=5, min_cell_size=50)
    assert res.multi_calibration_error < 0.08
    assert res.max_multi_accuracy < 0.05
    assert res.overall_ece < 0.05


def test_group_miscalibration_is_detected():
    rng = np.random.default_rng(1)
    n = 3000
    scores = rng.uniform(0.2, 0.8, size=n)
    groups = np.array(["ok"] * (n // 2) + ["bad"] * (n - n // 2))
    # "ok" is calibrated; "bad" always has label rate 0.9 regardless of score.
    y = np.empty(n)
    y[: n // 2] = (rng.uniform(0, 1, size=n // 2) < scores[: n // 2]).astype(float)
    y[n // 2 :] = (rng.uniform(0, 1, size=n - n // 2) < 0.9).astype(float)
    res = compute_multi_calibration(y, scores, groups, n_bins=4, min_cell_size=40)
    assert res.multi_accuracy_by_group["bad"] > res.multi_accuracy_by_group["ok"]
    assert res.multi_calibration_error > 0.1


def test_multi_accuracy_helper():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    s = np.array([0.9, 0.1, 0.2, 0.8])
    g = np.array(["a", "a", "b", "b"])
    ma = multi_accuracy(y, s, g)
    assert ma["a"] == pytest.approx(0.0)
    assert ma["b"] == pytest.approx(abs((1.0 - 0.2 + 0.0 - 0.8) / 2))


def test_report_and_rejects():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=200).astype(float)
    s = rng.uniform(0, 1, size=200)
    g = rng.choice(["x", "y"], size=200)
    res = compute_multi_calibration(y, s, g, n_bins=4, min_cell_size=10)
    text = render_multi_calibration_report(res)
    assert "Multi-calibration" in text
    assert "multi-calibration error" in text
    d = res.to_dict()
    assert "cells" in d and d["n_bins"] == 4
    with pytest.raises(ValueError):
        compute_multi_calibration(y, s, g, strategy="nope")
    with pytest.raises(ValueError):
        compute_multi_calibration(y, s * 2, g)
