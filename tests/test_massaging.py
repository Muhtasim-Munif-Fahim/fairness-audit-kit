"""
Tests for Kamiran-Calders massaging.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from fairness_audit_kit import (
    compute_massaging,
    fit_with_massaging,
    render_massaging_report,
)


def _toy():
    # group a: 3/4 positive, group b: 1/4 positive -> disc = 0.5, M = 0.5*4*4/8 = 1
    y = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    g = np.array(["a"] * 4 + ["b"] * 4)
    s = np.array([0.9, 0.2, 0.6, 0.1, 0.8, 0.7, 0.3, 0.05])
    return y, g, s


def test_toy_swaps_borderline_samples():
    y, g, s = _toy()
    res = compute_massaging(y, g, s)
    assert res.privileged_group == "a"
    assert res.n_changes == 1
    assert res.demoted_indices.tolist() == [1]  # lowest-scored privileged positive
    assert res.promoted_indices.tolist() == [5]  # highest-scored unprivileged negative
    assert res.base_rates_after == {"a": 0.5, "b": 0.5}
    assert res.base_rate_gap_after == 0.0
    assert y.tolist() == [1, 1, 1, 0, 1, 0, 0, 0]  # input untouched
    assert res.labels.sum() == y.sum()  # overall positive rate preserved


def test_swap_count_formula_on_unequal_groups():
    rng = np.random.default_rng(0)
    n_a, n_b = 600, 300
    g = np.array(["a"] * n_a + ["b"] * n_b)
    y = np.concatenate([rng.random(n_a) < 0.7, rng.random(n_b) < 0.3]).astype(int)
    s = rng.random(n_a + n_b)
    res = compute_massaging(y, g, s, privileged_group="a")
    disc = y[:n_a].mean() - y[n_a:].mean()
    assert res.n_changes == round(disc * n_a * n_b / (n_a + n_b))
    assert res.base_rate_gap_after < 1.0 / n_b + 1e-12
    assert res.base_rate_gap_after < res.base_rate_gap_before


def test_reversed_privilege_is_handled():
    y, g, s = _toy()
    res = compute_massaging(y, g, s, privileged_group="b")
    assert res.notes
    assert res.base_rate_gap_after == 0.0


def test_extreme_gap_stays_within_candidate_pools():
    y = np.array([1, 1, 1, 1, 0, 0])
    g = np.array(["a", "a", "a", "a", "b", "b"])
    res = compute_massaging(y, g, np.linspace(0, 1, 6))
    assert res.n_changes == 1  # round(1.0 * 4 * 2 / 6)
    assert set(res.promoted_indices) <= {4, 5}
    assert set(res.demoted_indices) <= {0, 1, 2, 3}


def test_validation():
    y, g, s = _toy()
    with pytest.raises(ValueError):
        compute_massaging(np.array([0, 1, 2, 0, 1, 2, 0, 1]), g, s)
    with pytest.raises(ValueError):
        compute_massaging(y, np.array(list("aabbccdd")), s)
    with pytest.raises(ValueError):
        compute_massaging(y, g, s[:-1])
    with pytest.raises(ValueError):
        compute_massaging(y, g, s, privileged_group="z")


def test_fit_with_massaging_reduces_prediction_gap():
    rng = np.random.default_rng(1)
    n = 4000
    groups = rng.integers(0, 2, size=n)
    merit = rng.normal(size=n)
    # Historical labels favour group 1; the group is visible through a proxy.
    logits = 1.5 * merit + 1.2 * (groups - 0.5)
    y = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(int)
    X = np.column_stack([merit, groups + 0.3 * rng.normal(size=n)])
    baseline = LogisticRegression(max_iter=1000).fit(X, y)
    model, res = fit_with_massaging(LogisticRegression(max_iter=1000), X, y, groups)
    assert res.base_rate_gap_after < res.base_rate_gap_before

    def gap(m):
        pred = m.predict(X)
        rates = [pred[np.asarray(groups) == gv].mean() for gv in np.unique(groups)]
        return max(rates) - min(rates)

    assert gap(model) < 0.5 * gap(baseline)
    report = render_massaging_report(res)
    assert "Promotions / demotions" in report
    assert res.to_dict()["n_label_flips"] == 2 * res.n_changes
