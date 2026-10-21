"""Tests for reject option classification (Kamiran, Karim and Zhang, 2012)."""

import argparse

import numpy as np
import pandas as pd
import pytest

from fairness_audit_kit import (
    RejectOptionClassifier,
    fit_reject_option_classification,
    reject_option_predict,
    render_reject_option_report,
)
from fairness_audit_kit.cli import cmd_reject_option


def _data(seed=2026, n=300):
    r = np.random.default_rng(seed)
    a = (r.random(n) < 0.5).astype(int)
    y = (r.random(n) < 0.3 + 0.25 * a).astype(int)
    s = 1 / (1 + np.exp(-(1.5 * (y - 0.5) + 0.6 * (a - 0.5) + r.normal(size=n))))
    return y, a, s


# Computed offline with aif360 0.6.1 RejectOptionClassification(num_class_thresh=100,
# num_ROC_margin=50, metric_ub=0.05, metric_lb=-0.05) on _data(): (threshold, margin, #positives)
AIF360_REF = {
    "statistical_parity": (0.4851515151515151, 0.12871366728509584, 137),
    "average_odds": (0.5841414141414141, 0.033947639661925384, 105),
    "equal_opportunity": (0.5841414141414141, 0.033947639661925384, 105),
}


@pytest.mark.parametrize("metric", list(AIF360_REF))
def test_matches_aif360_reference(metric):
    y, a, s = _data()
    roc = RejectOptionClassifier(1, metric=metric).fit(y, a, s)
    tau, theta, n_pos = AIF360_REF[metric]
    assert roc.threshold_ == pytest.approx(tau, abs=1e-12)
    assert roc.margin_ == pytest.approx(theta, abs=1e-12)
    assert roc.predict(a, s).sum() == n_pos
    assert roc.feasible_


def test_predict_rule_by_hand():
    s = np.array([0.30, 0.45, 0.55, 0.70, 0.45, 0.55])
    priv = np.array([1, 1, 1, 1, 0, 0], dtype=bool)
    out = reject_option_predict(s, priv, threshold=0.5, margin=0.1)
    # critical region (0.4, 0.6]: privileged -> 0, unprivileged -> 1
    assert out.tolist() == [0, 0, 0, 1, 1, 1]
    assert reject_option_predict(s, priv, 0.5, 0.0).tolist() == [0, 0, 1, 1, 0, 1]


def test_constraint_satisfied_and_gap_reduced():
    y, a, s = _data(seed=3, n=2000)
    for metric in ("statistical_parity", "average_odds", "equal_opportunity"):
        res = fit_reject_option_classification(y, a, s, privileged=1, metric=metric, bound=0.03)
        assert res.feasible
        assert abs(res.after[metric]) <= 0.03 + 1e-12
        assert abs(res.after[metric]) < abs(res.before[metric])
        assert res.after["balanced_accuracy"] > 0.6


def test_chosen_point_is_best_feasible_on_grid():
    y, a, s = _data(seed=4, n=500)
    roc = RejectOptionClassifier(1, n_thresholds=15, n_margins=8, bound=0.05).fit(y, a, s)
    best = -1.0
    priv = a == 1
    for tau in np.linspace(0.01, 0.99, 15):
        for theta in np.linspace(0, min(tau, 1 - tau), 8):
            p = reject_option_predict(s, priv, tau, theta)
            spd = p[~priv].mean() - p[priv].mean()
            if abs(spd) <= 0.05:
                ba = 0.5 * (p[y == 1].mean() + 1 - p[y == 0].mean())
                best = max(best, ba)
    assert roc.result_.after["balanced_accuracy"] == pytest.approx(best)


def test_infeasible_bound_warns_and_picks_closest():
    y, a, s = _data(seed=5, n=400)
    with pytest.warns(RuntimeWarning):
        # a single grid point (tau = 0.5, theta = 0) cannot reach |SPD| <= 0.001
        roc = RejectOptionClassifier(1, bound=0.001, low=0.5, high=0.5, n_thresholds=1, n_margins=1).fit(y, a, s)
    assert not roc.feasible_ and roc.threshold_ == 0.5 and roc.margin_ == 0.0
    res = roc.result_
    assert res.after == res.before
    with pytest.warns(RuntimeWarning):
        roc = RejectOptionClassifier(1, bound=0.001, low=0.4, high=0.6, n_thresholds=3, n_margins=1).fit(y, a, s)
    gaps = [abs(reject_option_predict(s, a == 1, t, 0.0)[a == 0].mean() - reject_option_predict(s, a == 1, t, 0.0)[a == 1].mean())
            for t in (0.4, 0.5, 0.6)]
    assert abs(roc.result_.after["statistical_parity"]) == pytest.approx(min(gaps))


def test_string_groups_and_validation():
    y, a, s = _data(n=200)
    g = np.where(a == 1, "male", "female")
    r1 = RejectOptionClassifier("male", n_thresholds=20, n_margins=5).fit(y, g, s)
    r2 = RejectOptionClassifier(1, n_thresholds=20, n_margins=5).fit(y, a, s)
    assert r1.threshold_ == r2.threshold_ and r1.margin_ == r2.margin_
    with pytest.raises(ValueError, match="metric"):
        RejectOptionClassifier(1, metric="nope")
    with pytest.raises(ValueError, match="both"):
        RejectOptionClassifier("x").fit(y, g, s)
    with pytest.raises(ValueError, match="same length"):
        RejectOptionClassifier(1).fit(y, a, s[:-1])
    with pytest.raises(RuntimeError):
        RejectOptionClassifier(1).predict(a, s)


def test_report_and_cli(tmp_path, capsys):
    y, a, s = _data(n=400)
    res = fit_reject_option_classification(y, a, s, privileged=1, n_thresholds=30, n_margins=10)
    md = render_reject_option_report(res)
    assert md.startswith("# Reject Option Classification Report") and "critical region" in md
    assert res.to_dict()["threshold"] == res.threshold
    data = tmp_path / "scores.csv"
    pd.DataFrame({"target": y, "sensitive_attr": a, "y_score": s}).to_csv(data, index=False)
    out, report = tmp_path / "roc.csv", tmp_path / "roc.md"
    args = argparse.Namespace(
        data=str(data), target_col="target", score_col="y_score", group_col="sensitive_attr",
        privileged="1", metric="statistical_parity", bound=0.05, n_thresholds=30, n_margins=10,
        output_col="y_pred_roc", output=str(out), report=str(report),
    )
    cmd_reject_option(args)
    df = pd.read_csv(out)
    assert np.array_equal(df["y_pred_roc"].to_numpy(), reject_option_predict(s, a == 1, res.threshold, res.margin))
    assert "Balanced accuracy" in capsys.readouterr().out
    assert report.read_text().startswith("# Reject Option")
    with pytest.raises(SystemExit):
        cmd_reject_option(argparse.Namespace(**{**vars(args), "score_col": "missing"}))
