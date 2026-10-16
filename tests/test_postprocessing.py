"""Tests for Hardt et al. equalized-odds post-processing."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pytest

from fairness_audit_kit import (
    EqualizedOddsPostprocessor,
    fit_equalized_odds_postprocessing,
    render_equalized_odds_postprocessing_report,
)
from fairness_audit_kit.cli import cmd_postprocess


def _biased(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    groups = rng.choice(["a", "b"], n, p=[0.6, 0.4])
    y = rng.binomial(1, np.where(groups == "a", 0.4, 0.25))
    # group b gets systematically inflated scores -> higher TPR and FPR
    shift = np.where(groups == "b", 0.8, 0.0)
    scores = 1 / (1 + np.exp(-(1.6 * (2 * y - 1) + shift + rng.normal(0, 1.2, n))))
    return y, groups, scores


def test_constraints_hold_analytically_and_empirically():
    y, g, s = _biased()
    post = EqualizedOddsPostprocessor().fit(y, g, y_scores=s)
    res = post.result_
    assert res.tpr_gap_before > 0.05 and res.fpr_gap_before > 0.05
    assert res.tpr_gap_after == pytest.approx(0.0, abs=1e-8)
    assert res.fpr_gap_after == pytest.approx(0.0, abs=1e-8)
    decisions = post.predict(g, y_scores=s, seed=1)
    tpr = {k: decisions[(g == k) & (y == 1)].mean() for k in "ab"}
    fpr = {k: decisions[(g == k) & (y == 0)].mean() for k in "ab"}
    assert abs(tpr["a"] - tpr["b"]) < 0.03
    assert abs(fpr["a"] - fpr["b"]) < 0.03


def test_accuracy_cost_is_bounded_and_beats_trivial_fair_predictors():
    y, g, s = _biased(seed=2)
    res = fit_equalized_odds_postprocessing(y, g, y_scores=s)
    assert res.expected_accuracy_after <= res.accuracy_before + 1e-9
    # constant predictors are trivially fair; LP optimum must do at least as well
    assert res.expected_accuracy_after >= max(np.mean(y == 0), np.mean(y == 1)) - 1e-9
    probs = [r.p_keep_positive for r in res.groups.values()] + [
        r.p_flip_to_positive for r in res.groups.values()
    ]
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_equal_opportunity_only_equalizes_tpr():
    y, g, s = _biased(seed=3)
    eo = fit_equalized_odds_postprocessing(y, g, y_scores=s, constraint="equal_opportunity")
    odds = fit_equalized_odds_postprocessing(y, g, y_scores=s)
    assert eo.tpr_gap_after == pytest.approx(0.0, abs=1e-8)
    # the looser constraint can only be cheaper
    assert eo.expected_accuracy_after >= odds.expected_accuracy_after - 1e-9


def test_already_fair_predictor_is_left_unchanged():
    # identical confusion structure in both groups
    y = np.array([1, 1, 1, 0, 0, 0, 0, 1] * 50)
    pred = np.array([1, 1, 0, 0, 0, 1, 0, 1] * 50)
    g = np.array(["a"] * 200 + ["b"] * 200)
    post = EqualizedOddsPostprocessor().fit(y, g, y_pred=pred)
    for rates in post.mixing_rates_.values():
        assert rates[1] == pytest.approx(1.0)
        assert rates[0] == pytest.approx(0.0)
    assert post.result_.expected_accuracy_after == pytest.approx(post.result_.accuracy_before)


def test_costs_shift_the_operating_point():
    y, g, s = _biased(seed=4)
    fn_heavy = fit_equalized_odds_postprocessing(y, g, y_scores=s, fn_cost=5.0)
    fp_heavy = fit_equalized_odds_postprocessing(y, g, y_scores=s, fp_cost=5.0)
    tpr_fn = next(iter(fn_heavy.groups.values())).tpr_after
    tpr_fp = next(iter(fp_heavy.groups.values())).tpr_after
    assert tpr_fn > tpr_fp


def test_positive_probability_and_errors():
    y, g, s = _biased(n=2000, seed=5)
    post = EqualizedOddsPostprocessor()
    with pytest.raises(RuntimeError):
        post.positive_probability(g, y_scores=s)
    post.fit(y, g, y_scores=s)
    prob = post.positive_probability(g, y_scores=s)
    assert prob.shape == (2000,)
    assert set(np.round(np.unique(prob), 12)) <= {
        round(v, 12) for r in post.mixing_rates_.values() for v in r.values()
    }
    with pytest.raises(ValueError, match="unseen groups"):
        post.positive_probability(np.array(["z"]), y_pred=np.array([1]))
    with pytest.raises(ValueError):
        EqualizedOddsPostprocessor(constraint="parity")
    with pytest.raises(ValueError):
        EqualizedOddsPostprocessor().fit([1, 0, 1, 0], ["a"] * 4, y_pred=[1, 0, 1, 0])
    with pytest.raises(ValueError, match="positive and one negative"):
        EqualizedOddsPostprocessor().fit([1, 1, 1, 0], ["a", "a", "b", "b"], y_pred=[1, 0, 1, 0])


def test_report_renders():
    y, g, s = _biased(n=3000, seed=6)
    text = render_equalized_odds_postprocessing_report(fit_equalized_odds_postprocessing(y, g, y_scores=s))
    assert "Equalized Odds Post-processing Report" in text
    assert "| a |" in text and "| b |" in text
    assert "TPR gap" in text


def test_cli_postprocess(tmp_path, capsys):
    y, g, s = _biased(n=4000, seed=7)
    data = tmp_path / "scores.csv"
    pd.DataFrame({"target": y, "sensitive_attr": g, "y_score": s}).to_csv(data, index=False)
    out = tmp_path / "fair.csv"
    report = tmp_path / "report.md"
    args = argparse.Namespace(
        data=str(data), target_col="target", pred_col="y_pred", score_col="y_score",
        group_col="sensitive_attr", threshold=0.5, constraint="equalized_odds",
        fp_cost=1.0, fn_cost=1.0, seed=0, prob_col="fair_positive_prob",
        output_col="y_pred_fair", output=str(out), report=str(report),
    )
    cmd_postprocess(args)
    frame = pd.read_csv(out)
    assert set(frame["y_pred_fair"].unique()) <= {0, 1}
    assert frame["fair_positive_prob"].between(0, 1).all()
    assert report.read_text().startswith("# Equalized Odds Post-processing Report")
    assert "TPR gap" in capsys.readouterr().out
