"""
Tests for the Feldman et al. disparate impact remover.
"""

import argparse

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from fairness_audit_kit import (
    DisparateImpactRemover,
    ks_distance,
    max_pairwise_ks,
    render_disparate_impact_repair_report,
    repair_disparate_impact,
)
from fairness_audit_kit.cli import cmd_repair


def _shifted(n0=600, n1=400, seed=0):
    rng = np.random.default_rng(seed)
    g = np.r_[np.zeros(n0, int), np.ones(n1, int)]
    X = pd.DataFrame({
        "income": np.r_[rng.normal(50, 10, n0), rng.normal(70, 15, n1)],
        "score": np.r_[rng.exponential(1.0, n0), rng.exponential(2.0, n1)],
        "neutral": rng.normal(size=n0 + n1),
    })
    return X, g


def test_ks_distance_known_values():
    assert ks_distance([1, 2, 3], [1, 2, 3]) == 0.0
    assert ks_distance([1, 2], [3, 4]) == 1.0
    assert ks_distance([1, 2, 3, 4], [3, 4, 5, 6]) == pytest.approx(0.5)
    assert max_pairwise_ks([1, 2, 10, 11, 1, 2], ["a", "a", "b", "b", "c", "c"]) == 1.0
    with pytest.raises(ValueError):
        ks_distance([], [1])


def test_full_repair_equalizes_distributions_and_preserves_ranks():
    X, g = _shifted()
    repaired, res = repair_disparate_impact(X, g, repair_level=1.0)
    assert res.ks_before["income"] > 0.5
    for f in res.features:
        assert res.ks_after[f] < 0.03
        assert res.rank_preserved[f]
    # within each group the order of values is unchanged
    for gv in (0, 1):
        m = g == gv
        assert np.array_equal(np.argsort(X["income"][m].to_numpy()), np.argsort(repaired["income"][m].to_numpy()))


def test_two_group_full_repair_targets_the_median_quantiles():
    # two groups of equal size: median of two quantiles = their average
    a = np.array([0.0, 1.0, 2.0, 3.0])
    b = np.array([10.0, 12.0, 14.0, 16.0])
    X = pd.DataFrame({"x": np.r_[a, b]})
    g = np.r_[np.zeros(4, int), np.ones(4, int)]
    out = DisparateImpactRemover().fit_transform(X, g)["x"].to_numpy()
    assert np.allclose(out[:4], out[4:])
    u = (np.arange(4) + 0.5) / 4  # mid-rank CDF values
    expected = (np.quantile(a, u) + np.quantile(b, u)) / 2
    assert np.allclose(out[:4], expected)


def test_partial_repair_interpolates_linearly():
    X, g = _shifted(seed=1)
    rem = DisparateImpactRemover().fit(X, g)
    full = rem.transform(X, g)
    half = rem.transform(X, g, repair_level=0.5)
    none = rem.transform(X, g, repair_level=0.0)
    pd.testing.assert_frame_equal(none, X)
    assert np.allclose(half["income"], 0.5 * X["income"] + 0.5 * full["income"])
    ks_half = max_pairwise_ks(half["income"].to_numpy(), g)
    assert max_pairwise_ks(full["income"].to_numpy(), g) < ks_half < max_pairwise_ks(X["income"].to_numpy(), g)


def test_transform_new_rows_uses_training_cdfs_and_reduces_group_leakage():
    X, g = _shifted(seed=2)
    X_test, g_test = _shifted(n0=300, n1=200, seed=3)
    rem = DisparateImpactRemover(features=["income", "score"]).fit(X, g)
    test_rep = rem.transform(X_test, g_test)
    assert np.array_equal(test_rep["neutral"], X_test["neutral"])  # untouched column
    assert max_pairwise_ks(test_rep["income"].to_numpy(), g_test) < 0.12
    # a classifier predicting the group from the features gets much worse
    clf_raw = LogisticRegression().fit(X[["income", "score"]], g)
    clf_rep = LogisticRegression().fit(rem.transform(X, g)[["income", "score"]], g)
    acc_raw = clf_raw.score(X_test[["income", "score"]], g_test)
    acc_rep = clf_rep.score(test_rep[["income", "score"]], g_test)
    assert acc_raw > 0.8 and acc_rep < 0.65


def test_numpy_input_three_groups_and_report():
    rng = np.random.default_rng(4)
    g = np.repeat(["a", "b", "c"], 200)
    X = np.c_[np.r_[rng.normal(0, 1, 200), rng.normal(1, 1, 200), rng.normal(3, 2, 200)]]
    repaired, res = repair_disparate_impact(X, g)
    assert list(repaired.columns) == ["x0"]
    assert res.ks_after["x0"] < 0.03 and res.groups == ["a", "b", "c"]
    text = render_disparate_impact_repair_report(res)
    assert text.startswith("# Disparate Impact Repair Report")
    assert "| x0 |" in text
    assert res.to_dict()["group_sizes"] == {"a": 200, "b": 200, "c": 200}


def test_validation():
    X, g = _shifted(n0=20, n1=20)
    with pytest.raises(ValueError, match="repair_level"):
        DisparateImpactRemover(repair_level=1.5)
    with pytest.raises(ValueError, match="two groups"):
        DisparateImpactRemover().fit(X, np.zeros(40))
    with pytest.raises(ValueError, match="one entry per row"):
        DisparateImpactRemover().fit(X, g[:5])
    with pytest.raises(ValueError, match="not found"):
        DisparateImpactRemover(features=["nope"]).fit(X, g)
    Xn = X.copy()
    Xn.loc[0, "income"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        DisparateImpactRemover().fit(Xn, g)
    rem = DisparateImpactRemover().fit(X, g)
    with pytest.raises(ValueError, match="not seen"):
        rem.transform(X, np.full(40, 7))
    with pytest.raises(RuntimeError):
        DisparateImpactRemover().transform(X, g)


def test_cli_repair(tmp_path, capsys):
    X, g = _shifted(n0=200, n1=150)
    data = tmp_path / "train.csv"
    df = X.assign(sensitive_attr=g, target=(X["income"] > 60).astype(int))
    df.to_csv(data, index=False)
    out = tmp_path / "repaired.csv"
    report = tmp_path / "repair.md"
    args = argparse.Namespace(
        data=str(data), group_col="sensitive_attr", features=None,
        exclude=["target", "y_pred", "y_score"], repair_level=1.0,
        output=str(out), report=str(report),
    )
    cmd_repair(args)
    rep = pd.read_csv(out)
    assert rep["target"].equals(df["target"])  # excluded label untouched
    assert max_pairwise_ks(rep["income"].to_numpy(), g) < 0.05
    assert report.read_text().startswith("# Disparate Impact Repair Report")
    assert "Repaired 3 feature(s)" in capsys.readouterr().out
    bad = argparse.Namespace(**{**vars(args), "features": ["missing_col"]})
    with pytest.raises(SystemExit):
        cmd_repair(bad)
