"""Tests for calibrated equalized odds post-processing (Pleiss et al., 2017)."""

import argparse

import numpy as np
import pandas as pd
import pytest

from fairness_audit_kit import (
    CalibratedEqOddsPostprocessor,
    fit_calibrated_eq_odds,
    generalized_cost,
    generalized_rates,
    render_calibrated_eq_odds_report,
)
from fairness_audit_kit.cli import cmd_calibrated_eq_odds


def _calibrated(seed=7, n=20000, n_groups=2):
    """Scores calibrated by construction (y ~ Bernoulli(s)); groups differ in informativeness."""
    r = np.random.default_rng(seed)
    g = r.integers(0, n_groups, n)
    centre = 0.4 + 0.05 * g
    width = 0.1 + 0.35 * g / max(n_groups - 1, 1)
    s = np.clip(centre + r.uniform(-1, 1, n) * width, 0.0, 1.0)
    y = (r.random(n) < s).astype(int)
    return y, g, s


def _aif360_mix_rates(y, priv, s, cost_constraint):
    """Literal transcription of AIF360's CalibratedEqOddsPostprocessing.fit."""
    fp_rate, fn_rate = {"fpr": (1, 0), "fnr": (0, 1), "weighted": (1, 1)}[cost_constraint]
    norm = float(fp_rate + fn_rate) if (fp_rate != 0 and fn_rate != 0) else 1

    def cost(yy, ss):
        br = yy.mean()
        gfpr = ss[yy == 0].mean()
        gfnr = (1 - ss[yy == 1]).mean()
        return fp_rate / norm * gfpr * (1 - br) + fn_rate / norm * gfnr * br

    yp, sp, yu, su = y[priv], s[priv], y[~priv], s[~priv]
    priv_cost, unpriv_cost = cost(yp, sp), cost(yu, su)
    priv_triv = cost(yp, np.full(yp.size, yp.mean()))
    unpriv_triv = cost(yu, np.full(yu.size, yu.mean()))
    more = unpriv_cost > priv_cost
    priv_mix = (unpriv_cost - priv_cost) / (priv_triv - priv_cost) if more else 0
    unpriv_mix = 0 if more else (priv_cost - unpriv_cost) / (unpriv_triv - unpriv_cost)
    return priv_mix, unpriv_mix


@pytest.mark.parametrize("constraint", ["weighted", "fpr", "fnr"])
def test_matches_aif360_mix_rates(constraint):
    y, g, s = _calibrated()
    ceo = CalibratedEqOddsPostprocessor(cost_constraint=constraint).fit(y, g, s)
    priv_mix, unpriv_mix = _aif360_mix_rates(y, g == 1, s, constraint)
    assert ceo.mix_rates_["1"] == pytest.approx(priv_mix, abs=1e-12)
    assert ceo.mix_rates_["0"] == pytest.approx(unpriv_mix, abs=1e-12)
    # exactly one group is mixed
    assert min(ceo.mix_rates_.values()) == 0.0


def test_generalized_rates_and_cost_by_hand():
    y = np.array([0, 0, 1, 1, 1])
    s = np.array([0.2, 0.4, 0.9, 0.6, 0.3])
    gfpr, gfnr = generalized_rates(y, s)
    assert gfpr == pytest.approx(0.3)
    assert gfnr == pytest.approx((0.1 + 0.4 + 0.7) / 3)
    mu = 0.6
    assert generalized_cost(y, s, "fpr") == pytest.approx(0.3 * (1 - mu))
    assert generalized_cost(y, s, "fnr") == pytest.approx(0.4 * mu)
    assert generalized_cost(y, s, "weighted") == pytest.approx(0.5 * (0.3 * 0.4 + 0.4 * 0.6))


@pytest.mark.parametrize("constraint", ["weighted", "fpr", "fnr"])
def test_expected_costs_equalised(constraint):
    y, g, s = _calibrated(seed=1)
    ceo = CalibratedEqOddsPostprocessor(cost_constraint=constraint).fit(y, g, s)
    es = ceo.expected_scores(g, s)
    costs = [generalized_cost(y[g == k], es[g == k], constraint) for k in (0, 1)]
    assert costs[0] == pytest.approx(costs[1], abs=1e-12)
    assert costs[0] == pytest.approx(ceo.target_cost_, abs=1e-12)
    res = ceo.result_
    assert res.cost_gap_after == pytest.approx(0.0, abs=1e-12)
    assert res.cost_gap_before > 0.005
    for st in res.groups:
        m = g == int(st.group)
        gfpr, gfnr = generalized_rates(y[m], es[m])
        assert st.gfpr_after == pytest.approx(gfpr, abs=1e-12)
        assert st.gfnr_after == pytest.approx(gfnr, abs=1e-12)


def test_randomized_scores_concentrate_on_expectation():
    y, g, s = _calibrated(seed=2, n=60000)
    ceo = CalibratedEqOddsPostprocessor(random_state=0).fit(y, g, s)
    ps = ceo.predict_proba(g, s)
    costs = [generalized_cost(y[g == k], ps[g == k]) for k in (0, 1)]
    assert abs(costs[0] - costs[1]) < 0.004
    # only the mixed group changes, and only to its base rate
    for k in (0, 1):
        m = g == k
        changed = ps[m] != s[m]
        assert np.allclose(ps[m][changed], ceo.base_rates_[str(k)])
        assert changed.mean() == pytest.approx(ceo.mix_rates_[str(k)], abs=0.02)
    np.testing.assert_array_equal(ps, ceo.predict_proba(g, s))  # seeded
    np.testing.assert_array_equal(ceo.predict(g, s), (ps >= 0.5).astype(int))


def test_calibration_preserved_within_groups():
    y, g, s = _calibrated(seed=3, n=80000)
    ceo = CalibratedEqOddsPostprocessor(random_state=1).fit(y, g, s)
    ps = ceo.predict_proba(g, s)
    for k in (0, 1):
        m = g == k
        bins = np.minimum((ps[m] * 10).astype(int), 9)
        for b in np.unique(bins):
            sel = bins == b
            if sel.sum() < 500:
                continue
            assert abs(y[m][sel].mean() - ps[m][sel].mean()) < 0.03


def test_multigroup_raises_all_to_worst():
    y, g, s = _calibrated(seed=4, n=30000, n_groups=3)
    ceo = CalibratedEqOddsPostprocessor().fit(y, g, s)
    es = ceo.expected_scores(g, s)
    costs = [generalized_cost(y[g == k], es[g == k]) for k in range(3)]
    assert max(costs) - min(costs) < 1e-12
    worst = max(ceo.costs_before_, key=ceo.costs_before_.get)
    assert ceo.mix_rates_[worst] == 0.0
    assert ceo.feasible_


def test_infeasible_target_clips_mix_rate():
    # group 1 is already near-trivial (cost close to its trivial cost) but cheaper because of its low base rate,
    # group 0 has a high cost that group 1's trivial classifier cannot reach under the fnr cost
    r = np.random.default_rng(5)
    n = 4000
    g = np.repeat([0, 1], n)
    s0 = np.full(n, 0.5)
    y0 = (r.random(n) < s0).astype(int)
    s1 = np.full(n, 0.05)
    y1 = (r.random(n) < s1).astype(int)
    y, s = np.r_[y0, y1], np.r_[s0, s1]
    with np.errstate(all="ignore"):
        ceo = CalibratedEqOddsPostprocessor(cost_constraint="fnr").fit(y, g, s)
    assert ceo.mix_rates_["1"] == 1.0
    assert not ceo.feasible_
    assert "not reachable" in render_calibrated_eq_odds_report(ceo.result_)


def test_string_groups_and_wrapper():
    y, g, s = _calibrated(seed=6, n=5000)
    labels = np.where(g == 1, "male", "female")
    res = fit_calibrated_eq_odds(y, labels, s, cost_constraint="fpr")
    assert {st.group for st in res.groups} == {"male", "female"}
    d = res.to_dict()
    assert d["cost_constraint"] == "fpr" and len(d["groups"]) == 2


def test_validation():
    y, g, s = _calibrated(n=500)
    with pytest.raises(ValueError):
        CalibratedEqOddsPostprocessor(cost_constraint="bogus")
    with pytest.raises(ValueError):
        CalibratedEqOddsPostprocessor(threshold=1.5)
    with pytest.raises(ValueError, match="probabilities"):
        CalibratedEqOddsPostprocessor().fit(y, g, s * 3)
    with pytest.raises(ValueError, match="two groups"):
        CalibratedEqOddsPostprocessor().fit(y, np.zeros_like(g), s)
    with pytest.raises(RuntimeError):
        CalibratedEqOddsPostprocessor().predict(g, s)
    ceo = CalibratedEqOddsPostprocessor().fit(y, g, s)
    with pytest.raises(ValueError, match="unseen"):
        ceo.predict(np.full(3, 9), s[:3])


def test_report_and_cli(tmp_path, capsys):
    y, g, s = _calibrated(n=3000)
    ceo = CalibratedEqOddsPostprocessor().fit(y, g, s)
    md = render_calibrated_eq_odds_report(ceo.result_)
    assert "Pleiss" in md and "| Group |" in md
    data = tmp_path / "scores.csv"
    pd.DataFrame({"target": y, "sensitive_attr": g, "y_score": s}).to_csv(data, index=False)
    out, rep = tmp_path / "out.csv", tmp_path / "rep.md"
    args = argparse.Namespace(
        data=str(data), target_col="target", group_col="sensitive_attr", score_col="y_score",
        cost_constraint="weighted", threshold=0.5, seed=0, output_col="y_pred_ceo",
        output=str(out), report=str(rep),
    )
    cmd_calibrated_eq_odds(args)
    df = pd.read_csv(out)
    assert {"y_pred_ceo", "y_pred_ceo_score"} <= set(df.columns)
    assert set(df["y_pred_ceo"].unique()) <= {0, 1}
    assert rep.read_text().startswith("# Calibrated Equalized Odds Report")
    assert "Cost gap" in capsys.readouterr().out
