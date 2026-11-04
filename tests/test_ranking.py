import numpy as np
import pytest

from fairness_audit_kit import (
    compute_ranking_fairness,
    group_exposure,
    ndkl,
    position_weights,
    render_ranking_fairness_report,
    rkl,
    rnd,
)


def test_position_weights():
    assert position_weights(3).tolist() == pytest.approx([1.0, 1 / np.log2(3), 0.5])
    assert position_weights(3, "reciprocal").tolist() == pytest.approx([1, 0.5, 1 / 3])
    assert position_weights(3, "rbp", p=0.5).tolist() == pytest.approx([1, 0.5, 0.25])
    with pytest.raises(ValueError):
        position_weights(3, "zipf")
    with pytest.raises(ValueError):
        position_weights(0)


def test_group_exposure_hand_computed():
    exp = group_exposure(["a", "b", "a", "b"], kind="reciprocal")
    assert exp["a"] == pytest.approx((1 + 1 / 3) / 2)
    assert exp["b"] == pytest.approx((1 / 2 + 1 / 4) / 2)


def test_extremes_of_rnd_rkl():
    worst = ["m"] * 50 + ["f"] * 50
    assert rnd(worst, "f") == pytest.approx(1.0)
    assert rkl(worst, "f") == pytest.approx(1.0)
    alternating = ["m", "f"] * 50
    assert rnd(alternating, "f") == pytest.approx(0.0, abs=1e-12)
    assert rkl(alternating, "f") == pytest.approx(0.0, abs=1e-9)
    partial = ["m"] * 20 + ["m", "f"] * 30 + ["f"] * 20
    assert 0.0 < rnd(partial, "f") < 1.0
    assert 0.0 < rkl(partial, "f") < 1.0


def test_rnd_monotone_in_segregation():
    rng = np.random.default_rng(0)
    vals = []
    for shift in (0, 10, 20, 30):
        r = ["m"] * shift + ["m", "f"] * (40 - shift) + ["f"] * shift
        vals.append(rnd(r, "f"))
    assert vals == sorted(vals)
    # Random shuffles are much fairer than segregated rankings.
    shuffled = rng.permutation(["m"] * 50 + ["f"] * 50).tolist()
    assert rnd(shuffled, "f") < 0.3


def test_ndkl():
    expected_num = 0.0
    for i in range(1, 41):
        a = (i + 1) // 2 / i
        kl = sum(x * np.log(x / 0.5) for x in (a, 1 - a) if x > 0)
        expected_num += kl / np.log2(i + 1)
    expected = expected_num / np.sum(1 / np.log2(np.arange(2, 42)))
    assert ndkl(["a", "b"] * 20, target={"a": 0.5, "b": 0.5}) == pytest.approx(expected, rel=1e-9)
    seg = ["a"] * 20 + ["b"] * 20
    assert ndkl(seg) > ndkl(["a", "b"] * 20) > 0
    # Target containing a group absent from the ranking.
    assert ndkl(["a"] * 5, target={"a": 0.5, "b": 0.5}) == pytest.approx(np.log(2))
    with pytest.raises(ValueError):
        ndkl(["a"], target={"a": -1})


def test_compute_ranking_fairness_with_relevance():
    groups = ["m"] * 5 + ["f"] * 5
    rel_equal = [1.0] * 10
    res = compute_ranking_fairness(groups, rel_equal, protected="f", step=2)
    assert res.exposure["m"] > res.exposure["f"]
    assert res.exposure_ratio < 1
    assert sum(res.exposure_share.values()) == pytest.approx(1.0)
    assert res.disparate_treatment_ratio == pytest.approx(res.exposure_ratio)
    assert res.rnd == pytest.approx(1.0)
    # If 'm' really is more relevant in proportion to exposure, DTR -> 1.
    w = position_weights(10)
    rel_merit = [w[:5].mean()] * 5 + [w[5:].mean()] * 5
    res2 = compute_ranking_fairness(groups, rel_merit)
    assert res2.disparate_treatment_ratio == pytest.approx(1.0)
    assert res2.rnd is None
    report = render_ranking_fairness_report(res)
    assert "rND (f)" in report and "Disparate treatment ratio" in report


def test_validation_and_notes():
    with pytest.raises(ValueError):
        compute_ranking_fairness([])
    with pytest.raises(ValueError):
        rnd(["a", "a"], "b")
    with pytest.raises(ValueError):
        compute_ranking_fairness(["a", "b"], [1.0])
    with pytest.raises(ValueError):
        rnd(["a", "b"] * 2, "a", step=10)
    one = compute_ranking_fairness(["a"] * 3)
    assert one.exposure_ratio == 1.0 and one.notes
