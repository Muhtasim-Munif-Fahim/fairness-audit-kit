import numpy as np
import pytest
from scipy import stats

from fairness_audit_kit import (
    adjust_alpha,
    fair_mtable,
    fair_topk,
    mtable_failure_probability,
    ranked_group_fairness_test,
)


def test_mtable_matches_hand_computation():
    np.testing.assert_array_equal(fair_mtable(10, 0.5, 0.1), [0, 0, 0, 1, 1, 1, 2, 2, 3, 3])
    m = fair_mtable(30, 0.3, 0.05)
    i = np.arange(1, 31)
    assert np.all(stats.binom.cdf(m, i, 0.3) > 0.05)
    assert np.all((m == 0) | (stats.binom.cdf(m - 1, i, 0.3) <= 0.05))
    assert np.all(np.diff(m) >= 0) and np.all(np.diff(m) <= 1)
    np.testing.assert_array_equal(fair_mtable(5, 0.0, 0.1), 0)


def test_failure_probability_matches_monte_carlo():
    m = fair_mtable(20, 0.4, 0.1)
    exact = mtable_failure_probability(m, 0.4)
    rng = np.random.default_rng(0)
    sims = rng.random((40000, 20)) < 0.4
    mc = np.mean(np.any(np.cumsum(sims, axis=1) < m, axis=1))
    assert exact == pytest.approx(mc, abs=0.01)
    assert exact > 0.1  # multiple testing inflates the failure rate


def test_adjusted_alpha_controls_failure_rate():
    a_c = adjust_alpha(50, 0.5, 0.1)
    assert 0 < a_c < 0.1
    assert mtable_failure_probability(fair_mtable(50, 0.5, a_c), 0.5) <= 0.1 + 1e-12
    # Small k may need no adjustment at all.
    assert adjust_alpha(1, 0.5, 0.1) == 0.1


def test_fair_topk_satisfies_mtable_and_preserves_group_order():
    rng = np.random.default_rng(1)
    n = 200
    protected = rng.random(n) < 0.5
    scores = rng.normal(size=n) - 0.8 * protected  # protected group scored lower
    res = fair_topk(scores, protected, k=40, p=0.5, alpha=0.1)
    assert res.feasible
    assert np.all(np.cumsum(res.protected) >= res.mtable)
    for flag in (True, False):
        sub = res.scores[res.protected == flag]
        assert np.all(np.diff(sub) <= 0)
    blind = ranked_group_fairness_test(protected[res.colorblind_ranking], 0.5, 0.1)
    assert not blind.fair and blind.first_violation is not None
    check = ranked_group_fairness_test(res.protected, 0.5, 0.1)
    assert check.fair
    assert res.n_promoted > 0
    assert 0.8 < res.utility_ratio < 1.0


def test_fair_topk_equals_colorblind_when_already_fair():
    scores = np.arange(20, 0, -1, dtype=float)
    protected = np.arange(20) % 2 == 0
    res = fair_topk(scores, protected, k=10, p=0.5)
    np.testing.assert_array_equal(res.ranking, res.colorblind_ranking)
    assert res.n_promoted == 0 and res.utility_ratio == pytest.approx(1.0)
    zero = fair_topk(np.r_[np.ones(5), np.zeros(5)], np.r_[np.zeros(5, bool), np.ones(5, bool)], k=5, p=0.0)
    np.testing.assert_array_equal(zero.ranking, np.arange(5))


def test_hand_example_minimal_promotion():
    # Non-protected scores 10..5, protected 4..1; p=0.5, alpha=0.1, no adjustment.
    scores = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
    protected = np.array([False] * 6 + [True] * 4)
    res = fair_topk(scores, protected, k=10, p=0.5, alpha=0.1, adjust=False)
    # mtable [0,0,0,1,1,1,2,2,3,3]: protected forced in at positions 4, 7, 9.
    np.testing.assert_array_equal(res.ranking, [0, 1, 2, 6, 3, 4, 7, 5, 8, 9])
    assert res.alpha_adjusted == 0.1


def test_infeasible_and_validation():
    res = fair_topk([5, 4, 3, 2], [False, False, False, True], k=4, p=0.9, adjust=False)
    assert not res.feasible and res.notes
    with pytest.raises(ValueError):
        fair_topk([1, 2], [True], k=1, p=0.5)
    with pytest.raises(ValueError):
        fair_topk([1, 2], [True, False], k=3, p=0.5)
    with pytest.raises(ValueError):
        fair_topk([1, 2], [True, False], k=1, p=1.0)
    with pytest.raises(ValueError):
        fair_mtable(5, 0.5, 0.0)
    with pytest.raises(ValueError):
        ranked_group_fairness_test([], 0.5)
