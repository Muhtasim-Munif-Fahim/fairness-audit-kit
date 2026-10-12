"""
Tests for Kamiran-Calders reweighing.
"""

import argparse

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from fairness_audit_kit import (
    compute_demographic_parity,
    compute_reweighing_weights,
    fit_with_reweighing,
    render_reweighing_report,
)
from fairness_audit_kit.cli import cmd_reweigh


def _toy():
    """
    Group a: 3 positives, 1 negative. Group b: 1 positive, 3 negatives.
    n=8, n_a=n_b=4, n_pos=n_neg=4.
    w(a,1) = 4*4/(8*3) = 2/3, w(a,0) = 4*4/(8*1) = 2
    w(b,1) = 2,              w(b,0) = 2/3
    """
    y = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    g = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
    return y, g


def test_hand_computed_weights():
    y, g = _toy()
    res = compute_reweighing_weights(y, g)
    assert res.weight_table["a"]["1"] == pytest.approx(2 / 3)
    assert res.weight_table["a"]["0"] == pytest.approx(2.0)
    assert res.weight_table["b"]["1"] == pytest.approx(2.0)
    assert res.weight_table["b"]["0"] == pytest.approx(2 / 3)
    np.testing.assert_allclose(
        res.weights, [2 / 3, 2 / 3, 2 / 3, 2, 2, 2 / 3, 2 / 3, 2 / 3]
    )
    assert res.cell_count["a"] == {"0": 1, "1": 3}


def test_weights_sum_to_n_and_equalize_base_rates():
    y, g = _toy()
    res = compute_reweighing_weights(y, g)
    assert res.weights.sum() == pytest.approx(len(y))
    assert res.base_rates_before == {"a": 0.75, "b": 0.25}
    assert res.base_rate_gap_before == pytest.approx(0.5)
    assert res.base_rates_after["a"] == pytest.approx(0.5)
    assert res.base_rates_after["b"] == pytest.approx(0.5)
    assert res.base_rate_gap_after == pytest.approx(0.0, abs=1e-12)


def test_weighted_joint_factorizes_multiclass_multigroup():
    rng = np.random.default_rng(0)
    g = rng.choice(["x", "y", "z"], size=3000, p=[0.5, 0.3, 0.2])
    probs = {"x": [0.6, 0.3, 0.1], "y": [0.2, 0.5, 0.3], "z": [0.1, 0.2, 0.7]}
    y = np.array([rng.choice(3, p=probs[v]) for v in g])
    res = compute_reweighing_weights(y, g, positive_label=2)
    w = res.weights
    total = w.sum()
    for gv in ["x", "y", "z"]:
        for lab in range(3):
            joint = w[(g == gv) & (y == lab)].sum() / total
            marg = (w[g == gv].sum() / total) * (w[y == lab].sum() / total)
            assert joint == pytest.approx(marg, abs=1e-12)
    # Weighted group and label marginals are the unweighted ones.
    assert w[g == "x"].sum() / total == pytest.approx(np.mean(g == "x"))
    assert res.base_rate_gap_after == pytest.approx(0.0, abs=1e-12)


def test_already_independent_data_gets_unit_weights():
    y = np.array([1, 0, 1, 0])
    g = np.array([0, 0, 1, 1])
    res = compute_reweighing_weights(y, g)
    np.testing.assert_allclose(res.weights, 1.0)


def test_empty_cell_is_reported():
    y = np.array([1, 1, 0, 0])
    g = np.array([0, 0, 1, 1])
    res = compute_reweighing_weights(y, g)
    assert ("0", "0") in res.empty_cells and ("1", "1") in res.empty_cells
    assert np.isnan(res.weight_table["0"]["0"])
    assert "Empty cells" in render_reweighing_report(res)


def test_reweighed_training_reduces_demographic_parity_gap():
    # Historical label bias: 40% of group-1 positives were recorded as negatives.
    rng = np.random.default_rng(7)
    n = 4000
    g = rng.integers(0, 2, size=n)
    x = rng.normal(size=(n, 2))
    y = (x[:, 0] + 0.5 * x[:, 1] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    flip = (g == 1) & (y == 1) & (rng.random(n) < 0.4)
    y[flip] = 0
    X = np.column_stack([x, g])
    base = LogisticRegression(max_iter=1000).fit(X, y)
    fair, res = fit_with_reweighing(LogisticRegression(max_iter=1000), X, y, g)
    assert res.base_rate_gap_before > 0.1
    spd_base = compute_demographic_parity(base.predict(X), g).statistical_parity_difference
    spd_fair = compute_demographic_parity(fair.predict(X), g).statistical_parity_difference
    assert spd_fair < 0.5 * spd_base


def test_fit_with_reweighing_rejects_sample_weight():
    y, g = _toy()
    with pytest.raises(ValueError, match="sample_weight"):
        fit_with_reweighing(LogisticRegression(), np.zeros((8, 1)), y, g, sample_weight=np.ones(8))


def test_validation():
    with pytest.raises(ValueError, match="same length"):
        compute_reweighing_weights(np.array([0, 1]), np.array([0]))
    with pytest.raises(ValueError, match="at least one"):
        compute_reweighing_weights(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="positive_label"):
        compute_reweighing_weights(np.array([0, 1]), np.array([0, 1]), positive_label=5)


def test_report_and_to_dict():
    y, g = _toy()
    res = compute_reweighing_weights(y, g)
    md = render_reweighing_report(res)
    assert "# Reweighing Report" in md
    assert "| a | 2.0000 | 0.6667 | 0.7500 | 0.5000 |" in md
    d = res.to_dict()
    assert d["base_rate_gap_before"] == pytest.approx(0.5)
    assert "weights" not in d


def test_cli_reweigh(tmp_path, capsys):
    y, g = _toy()
    data = tmp_path / "train.csv"
    pd.DataFrame({"f": np.arange(8), "target": y, "sensitive_attr": g}).to_csv(data, index=False)
    out = tmp_path / "weighted.csv"
    report = tmp_path / "rw.md"
    args = argparse.Namespace(
        data=str(data), target_col="target", group_col="sensitive_attr",
        weight_col="sample_weight", output=str(out), report=str(report),
    )
    cmd_reweigh(args)
    df = pd.read_csv(out)
    assert df["sample_weight"].sum() == pytest.approx(8.0)
    assert report.read_text().startswith("# Reweighing Report")
    assert "0.5000 -> 0.0000" in capsys.readouterr().out
