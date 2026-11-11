import math

import numpy as np
import pandas as pd
import pytest

from fairness_audit_kit import compute_differential_fairness, render_differential_fairness_report


def _toy():
    # group a: 8/10 positive, group b: 2/10 positive
    y = np.array([1] * 8 + [0] * 2 + [1] * 2 + [0] * 8)
    s = np.array(["a"] * 10 + ["b"] * 10)
    return y, s


def test_raw_epsilon_closed_form():
    y, s = _toy()
    res = compute_differential_fairness(y, s, alpha=0.0)
    # outcome 1: log(0.8/0.2); outcome 0: log(0.8/0.2) as well
    assert res.epsilon == pytest.approx(math.log(4.0))
    assert res.disparate_impact_ratio == pytest.approx(0.25)
    assert res.per_outcome_epsilon == {0: pytest.approx(math.log(4)), 1: pytest.approx(math.log(4))}
    assert set(res.worst_pair) == {"a", "b"}
    assert res.group_probabilities["a"][1] == pytest.approx(0.8)
    assert res.group_size == {"a": 10, "b": 10}
    assert not res.satisfies(1.0) and res.satisfies(1.5)


def test_smoothing_shrinks_and_handles_zero_counts():
    y = np.array([1] * 5 + [0] * 5)
    s = np.array(["a"] * 5 + ["b"] * 5)
    raw = compute_differential_fairness(y, s, alpha=0.0)
    assert math.isinf(raw.epsilon)
    smooth = compute_differential_fairness(y, s, alpha=1.0)
    # (5+1)/(5+2) vs (0+1)/(5+2)
    assert smooth.epsilon == pytest.approx(math.log(6.0))
    assert smooth.group_probabilities["b"][1] == pytest.approx(1 / 7)


def test_identical_rates_are_zero_epsilon():
    y = np.tile([0, 1, 1, 0], 25)
    s = np.repeat(["x", "y"], 50)
    res = compute_differential_fairness(y, s)
    assert res.epsilon == pytest.approx(0.0, abs=1e-12)
    assert res.disparate_impact_ratio == pytest.approx(1.0)


def test_intersectional_subsets_never_exceed_full_intersection():
    rng = np.random.default_rng(0)
    n = 4000
    race = rng.choice(["r1", "r2", "r3"], size=n)
    gender = rng.choice(["f", "m"], size=n)
    p = 0.3 + 0.25 * (race == "r1") * (gender == "f") + 0.1 * (gender == "m")
    y = (rng.random(n) < p).astype(int)
    res = compute_differential_fairness(y, pd.DataFrame({"race": race, "gender": gender}), alpha=0.0)
    assert set(res.subset_epsilons) == {"race", "gender", "race x gender"}
    assert res.subset_epsilons["race x gender"] == pytest.approx(res.epsilon)
    for key in ("race", "gender"):
        assert res.subset_epsilons[key] <= res.epsilon + 1e-12
    assert len(res.group_probabilities) == 6
    assert "r1 & f" in res.worst_pair
    assert res.attributes == ("race", "gender")


def test_bias_amplification_and_bootstrap_ci():
    rng = np.random.default_rng(1)
    n = 3000
    s = rng.choice(["a", "b"], size=n)
    y_true = (rng.random(n) < np.where(s == "a", 0.5, 0.4)).astype(int)
    y_pred = (rng.random(n) < np.where(s == "a", 0.6, 0.3)).astype(int)  # amplifies the gap
    res = compute_differential_fairness(y_pred, s, y_true=y_true, n_bootstrap=100, seed=0)
    assert res.data_epsilon is not None and res.bias_amplification > 0.2
    assert res.ci_low <= res.epsilon <= res.ci_high
    fair = compute_differential_fairness(y_true, s, y_true=y_true)
    assert fair.bias_amplification == pytest.approx(0.0)


def test_multiclass_mapping_input_and_explicit_outcomes():
    y = np.array(["lo", "mid", "hi"] * 20 + ["lo"] * 30)
    s = {"g": np.array(["u"] * 60 + ["v"] * 30)}
    res = compute_differential_fairness(y, s, outcomes=["lo", "mid", "hi"], alpha=0.5)
    assert res.outcomes == ("lo", "mid", "hi")
    assert res.worst_outcome in ("mid", "hi")
    d = res.to_dict()
    assert d["attributes"] == ["g"] and "u" in d["group_probabilities"]
    with pytest.raises(ValueError, match="outside outcomes"):
        compute_differential_fairness(y, s, outcomes=["lo", "mid"])


def test_report_and_validation():
    y, s = _toy()
    res = compute_differential_fairness(y, s, y_true=y, n_bootstrap=10, seed=1)
    text = render_differential_fairness_report(res)
    assert "Differential Fairness" in text and "Bootstrap CI" in text and "| a | 10 |" in text
    with pytest.raises(ValueError, match="alpha"):
        compute_differential_fairness(y, s, alpha=-1)
    with pytest.raises(ValueError, match="two groups"):
        compute_differential_fairness(y, np.array(["a"] * 20))
    with pytest.raises(ValueError, match="one value per prediction"):
        compute_differential_fairness(y, s[:5])
    with pytest.raises(ValueError, match="same length"):
        compute_differential_fairness(y, s, y_true=y[:3])
