"""Tests for Zafar et al. decision-boundary covariance constraints."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from fairness_audit_kit import (
    FairnessConstrainedClassifier,
    boundary_covariance,
    fit_fairness_constraints,
    p_percent_rule,
    render_fairness_constraints_report,
)


def _biased(n=2000, seed=0, n_groups=2):
    rng = np.random.default_rng(seed)
    s = rng.integers(0, n_groups, size=n)
    x1 = rng.normal(size=n) + 1.2 * s  # proxy for the group
    x2 = rng.normal(size=n)
    logits = 1.5 * x1 + 0.8 * x2 - 1.0
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logits))).astype(int)
    return np.column_stack([x1, x2]), y, np.array([f"g{v}" for v in s])


def test_p_percent_rule_and_covariance():
    g = np.array(["a", "a", "b", "b"])
    assert p_percent_rule(np.array([1, 1, 1, 0]), g) == pytest.approx(0.5)
    assert p_percent_rule(np.array([0, 0, 0, 0]), g) == 1.0
    cov = boundary_covariance(np.array([1.0, 1.0, -1.0, -1.0]), g)
    assert cov["a"] == pytest.approx(0.5) and cov["b"] == pytest.approx(-0.5)


def test_unconstrained_matches_sklearn():
    X, y, g = _biased(800)
    C = 1e-2
    m = FairnessConstrainedClassifier(c=1e12, C=C).fit(X, y, g)
    sk = LogisticRegression(C=1.0 / (len(y) * C), tol=1e-10, max_iter=5000).fit(X, y)
    np.testing.assert_allclose(m.coef_[:-1], sk.coef_.ravel(), atol=2e-3)
    assert m.coef_[-1] == pytest.approx(sk.intercept_[0], abs=2e-3)


def test_fairness_mode_respects_bound_and_improves_parity():
    X, y, g = _biased()
    free = FairnessConstrainedClassifier(c=1e12).fit(X, y, g)
    fair = FairnessConstrainedClassifier(c=0.0).fit(X, y, g)
    assert max(abs(v) for v in fair.covariance_.values()) <= 1e-6
    assert p_percent_rule(fair.predict(X), g) > 0.8
    assert p_percent_rule(free.predict(X), g) < 0.5
    acc_free = np.mean(free.predict(X) == y)
    acc_fair = np.mean(fair.predict(X) == y)
    assert acc_fair < acc_free
    mid = FairnessConstrainedClassifier(c=0.05).fit(X, y, g)
    assert max(abs(v) for v in mid.covariance_.values()) <= 0.05 + 1e-6


def test_accuracy_mode_respects_loss_budget():
    X, y, g = _biased(seed=1)
    m = FairnessConstrainedClassifier(mode="accuracy", gamma=0.1).fit(X, y, g)
    assert m.loss_ <= 1.1 * m.unconstrained_loss_ + 1e-6
    free = FairnessConstrainedClassifier(c=1e12).fit(X, y, g)
    assert max(abs(v) for v in m.covariance_.values()) < max(abs(v) for v in free.covariance_.values())
    tighter = FairnessConstrainedClassifier(mode="accuracy", gamma=0.5).fit(X, y, g)
    assert p_percent_rule(tighter.predict(X), g) >= p_percent_rule(m.predict(X), g) - 1e-9


def test_multigroup_constraints():
    X, y, g = _biased(n=2400, seed=2, n_groups=3)
    m = FairnessConstrainedClassifier(c=0.0).fit(X, y, g)
    assert set(m.covariance_) == {"g0", "g1", "g2"}
    assert max(abs(v) for v in m.covariance_.values()) <= 1e-6
    assert p_percent_rule(m.predict(X), g) > 0.75


def test_sweep_and_report():
    X, y, g = _biased(1000, seed=3)
    res = fit_fairness_constraints(X, y, g, levels=(None, 0.1, 0.0))
    assert res[0].mode == "unconstrained" and res[0].level == float("inf")
    assert res[-1].p_percent > res[0].p_percent
    md = render_fairness_constraints_report(res)
    assert "Zafar" in md and "p%-rule" in md
    assert res[1].to_dict()["mode"] == "fairness"


def test_validation():
    X, y, g = _biased(100)
    with pytest.raises(ValueError):
        FairnessConstrainedClassifier(mode="x").fit(X, y, g)
    with pytest.raises(ValueError):
        FairnessConstrainedClassifier(c=-1).fit(X, y, g)
    with pytest.raises(ValueError):
        FairnessConstrainedClassifier().fit(X, y, np.array(["a"] * 100))
    with pytest.raises(RuntimeError):
        FairnessConstrainedClassifier().predict(X)
