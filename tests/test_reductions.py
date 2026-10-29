import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from fairness_audit_kit import (
    ExponentiatedGradient,
    fit_exponentiated_gradient,
    generate_biased_dataset,
    render_exponentiated_gradient_report,
)
from fairness_audit_kit.reductions import _Moment


def _unfair_data(n=1500, seed=0):
    """Labels and a feature both shift with the group -> large DP gap."""
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 2, n)
    x1 = rng.normal(size=n) + 1.2 * g
    x2 = rng.normal(size=n)
    logits = 1.5 * x1 + 0.5 * x2 - 0.6
    y = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(int)
    return np.column_stack([x1, x2]), y, g


def _brute_gamma(h, y, g, constraint):
    out = []
    events = [None] if constraint == "demographic_parity" else [0, 1]
    for e in events:
        ev = np.ones_like(y, bool) if e is None else (y == e)
        for a in np.unique(g):
            out.append(h[ev & (g == a)].mean() - h[ev].mean())
    return np.array(out)


@pytest.mark.parametrize("constraint", ["demographic_parity", "equalized_odds"])
def test_moments_match_definition(constraint):
    X, y, g = _unfair_data(400)
    h = np.random.default_rng(1).random(len(y))
    m = _Moment(constraint, y, g)
    assert np.allclose(m.gamma(h), _brute_gamma(h, y, g, constraint))


@pytest.mark.parametrize("constraint", ["demographic_parity", "equalized_odds"])
def test_signed_weights_are_gradient(constraint):
    X, y, g = _unfair_data(300)
    m = _Moment(constraint, y, g)
    lam = np.random.default_rng(2).normal(size=m.k)
    h = np.random.default_rng(3).random(len(y))
    w = m.signed_weights(lam)
    # λᵀγ is linear in h, so its gradient is exact
    for i in [0, 7, 123]:
        e = np.zeros(len(y))
        e[i] = 1e-3
        num = (lam @ m.gamma(h + e) - lam @ m.gamma(h)) / 1e-3
        assert num == pytest.approx(w[i], rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("constraint", ["demographic_parity", "equalized_odds"])
def test_reduction_meets_constraint(constraint):
    X, y, g = _unfair_data()
    eg = ExponentiatedGradient(constraint=constraint, eps=0.02, max_iter=60).fit(X, y, g)
    r = eg.result_
    assert r.violation_before > 0.1
    assert r.violation_after <= 0.02 + 0.01
    assert r.error_after >= r.error_before - 1e-9  # fairness has a price
    assert r.error_after < 0.5
    assert abs(sum(r.weights) - 1) < 1e-9 and r.n_hypotheses >= 1
    p = eg.predict_proba(X)
    assert np.allclose(np.abs(_brute_gamma(p, y, g, constraint)).max(), r.violation_after)


def test_already_fair_problem_stops_immediately():
    d = generate_biased_dataset(n_samples=1500, random_state=0)
    r = fit_exponentiated_gradient(d.X, d.y, d.groups, constraint="demographic_parity", eps=0.02)
    assert r.converged and r.n_iter == 1 and r.n_hypotheses == 1
    assert r.error_after == pytest.approx(r.error_before)


def test_tighter_eps_costs_more_accuracy():
    X, y, g = _unfair_data()
    loose = fit_exponentiated_gradient(X, y, g, eps=0.1)
    tight = fit_exponentiated_gradient(X, y, g, eps=0.01)
    assert tight.violation_after < loose.violation_after
    assert tight.error_after >= loose.error_after - 1e-3


def test_predict_is_randomized_by_proba_and_custom_estimator():
    from sklearn.tree import DecisionTreeClassifier

    X, y, g = _unfair_data(800)
    eg = ExponentiatedGradient(DecisionTreeClassifier(max_depth=3, random_state=0), eps=0.03).fit(X, y, g)
    p = eg.predict_proba(X)
    draws = np.mean([eg.predict(X, random_state=s) for s in range(200)], axis=0)
    assert np.abs(draws - p).max() < 0.15
    assert set(np.unique(eg.predict(X, random_state=0))) <= {0, 1}


def test_validation():
    X, y, g = _unfair_data(100)
    with pytest.raises(ValueError):
        ExponentiatedGradient(constraint="nope")
    with pytest.raises(ValueError):
        ExponentiatedGradient(eps=0)
    with pytest.raises(ValueError):
        ExponentiatedGradient().fit(X, y + 1, g)
    with pytest.raises(ValueError):
        ExponentiatedGradient().fit(X, y, np.zeros_like(g))
    with pytest.raises(RuntimeError):
        ExponentiatedGradient().predict_proba(X)


def test_report_and_cli(tmp_path):
    X, y, g = _unfair_data(600)
    r = fit_exponentiated_gradient(X, y, g, constraint="equalized_odds", eps=0.05)
    md = render_exponentiated_gradient_report(r)
    assert "equalized odds" in md and "y=1,1" in md
    df = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "target": y, "sensitive_attr": g})
    data = tmp_path / "train.csv"
    df.to_csv(data, index=False)
    out, rep = tmp_path / "out.csv", tmp_path / "r.md"
    proc = subprocess.run(
        [sys.executable, "-m", "fairness_audit_kit.cli", "reduce", "--data", str(data),
         "--eps", "0.03", "--output", str(out), "--report", str(rep)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    res = pd.read_csv(out)
    assert {"y_pred_eg", "y_pred_eg_prob"} <= set(res.columns)
    assert "Max violation" in proc.stdout and rep.exists()
