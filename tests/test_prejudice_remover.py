"""Tests for the Kamishima prejudice-remover regularizer."""

import numpy as np
import pytest
from scipy.optimize import check_grad
from sklearn.linear_model import LogisticRegression

from fairness_audit_kit import (
    PrejudiceRemover,
    fit_prejudice_remover,
    normalized_prejudice_index,
    render_prejudice_remover_report,
)


def _biased(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.integers(0, 2, size=n)
    # feature correlated with group -> proxy discrimination
    x1 = rng.normal(size=n) + 1.2 * s
    x2 = rng.normal(size=n)
    logits = 1.5 * x1 + 0.8 * x2 - 1.0
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logits))).astype(int)
    return np.column_stack([x1, x2]), y, np.where(s == 1, "m", "f")


def test_npi_bounds():
    s = np.array(["a", "a", "b", "b"])
    assert normalized_prejudice_index(np.array([1, 1, 0, 0]), s) == pytest.approx(1.0)
    assert normalized_prejudice_index(np.array([1, 0, 1, 0]), s) == pytest.approx(0.0)
    assert normalized_prejudice_index(np.ones(4), s) == 0.0
    with pytest.raises(ValueError):
        normalized_prejudice_index(np.ones(3), s)


@pytest.mark.parametrize("per_group", [True, False])
def test_gradient_matches_finite_differences(per_group):
    X, y, g = _biased(200)
    m = PrejudiceRemover(eta=3.0, C=0.5, per_group=per_group)
    m.groups_ = ["f", "m"]
    Z = m._design(X)
    gi = m._group_index(g, Z.shape[0])
    k = (2 if per_group else 1) * Z.shape[1]
    theta = np.random.default_rng(1).normal(scale=0.5, size=k)
    err = check_grad(
        lambda t: m._objective(t, Z, y.astype(float), gi, 2)[0],
        lambda t: m._objective(t, Z, y.astype(float), gi, 2)[1],
        theta,
    )
    assert err < 1e-4


def test_eta_zero_shared_equals_l2_logistic_regression():
    X, y, g = _biased()
    m = PrejudiceRemover(eta=0.0, C=1.0, per_group=False).fit(X, y, g)
    # sklearn: C_sklearn = 1 / C_ours (penalty 0.5 ||w||^2 / C_sklearn)
    ref = LogisticRegression(C=1.0, tol=1e-10, max_iter=1000).fit(X, y)
    assert np.allclose(m.coef_[0, :-1], ref.coef_[0], atol=1e-4)
    assert m.coef_[0, -1] == pytest.approx(ref.intercept_[0], abs=1e-4)


def test_eta_reduces_disparity_monotonically():
    X, y, g = _biased()
    res = fit_prejudice_remover(X, y, g, etas=(0.0, 10.0, 100.0))
    gaps = [r.demographic_parity_gap for r in res]
    npis = [r.normalized_prejudice_index for r in res]
    pis = [r.prejudice_index for r in res]
    assert gaps[0] > 0.2
    assert gaps[2] < gaps[1] < gaps[0]
    assert npis[2] < npis[0]
    assert pis[2] < pis[0]
    assert res[0].accuracy >= res[2].accuracy  # fairness costs accuracy here
    assert all(r.converged for r in res)


def test_predict_api_and_labels():
    X, y, g = _biased(400)
    labels = np.where(y == 1, "yes", "no")
    m = PrejudiceRemover(eta=5.0).fit(X, labels, g)
    proba = m.predict_proba(X, g)
    assert proba.shape == (400, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(m.predict(X, g)) <= {"yes", "no"}
    assert m.coef_.shape == (2, 3)


def test_eval_split_and_report():
    X, y, g = _biased(1200)
    res = fit_prejudice_remover(
        X[:800], y[:800], g[:800], etas=(0.0, 20.0), X_eval=X[800:], y_eval=y[800:], groups_eval=g[800:]
    )
    md = render_prejudice_remover_report(res)
    assert "| eta |" in md and "| 20 |" in md
    assert set(res[0].to_dict()) >= {"eta", "accuracy", "demographic_parity_gap", "positive_rates"}
    with pytest.raises(ValueError):
        fit_prejudice_remover(X, y, g, X_eval=X)


def test_validation():
    X, y, g = _biased(100)
    with pytest.raises(ValueError):
        PrejudiceRemover(eta=-1).fit(X, y, g)
    with pytest.raises(ValueError):
        PrejudiceRemover().fit(X, np.zeros(100), g)
    with pytest.raises(ValueError):
        PrejudiceRemover().fit(X, y, np.array(["a"] * 100))
    m = PrejudiceRemover().fit(X, y, g)
    with pytest.raises(ValueError):
        m.predict(X, np.array(["z"] * 100))
    with pytest.raises(ValueError):
        m.predict(X[:, :1], g)
    with pytest.raises(RuntimeError):
        PrejudiceRemover().predict(X, g)
