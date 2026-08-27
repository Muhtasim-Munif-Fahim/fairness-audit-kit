"""
Tests for fairness metrics module.
"""

import numpy as np
import pytest
from fairness_audit_kit.metrics import (
    FairnessMetrics,
    compute_fairness_metrics,
    confusion_matrices_by_group,
    _safe_divide,
    _confusion_matrix,
    _group_rates,
)


class TestSafeDivide:
    def test_normal_division(self):
        assert _safe_divide(10, 2) == 5.0
        assert _safe_divide(7, 2) == 3.5

    def test_zero_denominator(self):
        assert _safe_divide(5, 0) == 0.0
        assert _safe_divide(5, 0, default=-1.0) == -1.0


class TestConfusionMatrix:
    def test_basic(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 1, 0])
        cm = _confusion_matrix(y_true, y_pred)
        assert cm == {"tp": 2, "tn": 1, "fp": 1, "fn": 1}

    def test_all_correct(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        cm = _confusion_matrix(y_true, y_pred)
        assert cm == {"tp": 2, "tn": 2, "fp": 0, "fn": 0}

    def test_all_wrong(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 1, 1])
        cm = _confusion_matrix(y_true, y_pred)
        assert cm == {"tp": 0, "tn": 0, "fp": 2, "fn": 2}


class TestGroupRates:
    def test_two_groups(self):
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 1, 0, 0])
        groups = np.array(["A", "A", "A", "B", "B", "B"])
        rates = _group_rates(y_true, y_pred, groups)
        assert "A" in rates and "B" in rates
        assert rates["A"]["total"] == 3
        assert rates["B"]["total"] == 3

    def test_group_rates_values(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 1])
        groups = np.array([0, 0, 1, 1])
        rates = _group_rates(y_true, y_pred, groups)
        # Group 0: y_true=[1,1], y_pred=[1,0] -> tp=1, fn=1, fp=0, tn=0
        assert rates["0"]["confusion_matrix"]["tp"] == 1
        assert rates["0"]["confusion_matrix"]["fn"] == 1
        assert rates["0"]["confusion_matrix"]["fp"] == 0
        assert rates["0"]["confusion_matrix"]["tn"] == 0
        assert rates["0"]["tpr"] == 0.5
        # Group 1: y_true=[0,0], y_pred=[0,1] -> tp=0, fn=0, fp=1, tn=1
        assert rates["1"]["confusion_matrix"]["tp"] == 0
        assert rates["1"]["confusion_matrix"]["fn"] == 0
        assert rates["1"]["confusion_matrix"]["fp"] == 1
        assert rates["1"]["confusion_matrix"]["tn"] == 1
        assert rates["1"]["tpr"] == 0.0


class TestComputeFairnessMetrics:
    def test_perfect_fairness(self):
        # Both groups have identical predictions
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        metrics = compute_fairness_metrics(y_true, y_pred, groups)
        assert metrics.demographic_parity_difference == 0.0
        assert metrics.equal_opportunity_difference == 0.0
        assert metrics.equalized_odds_difference == 0.0
        assert metrics.disparate_impact_ratio == 1.0

    def test_demographic_parity_difference(self):
        # Group 0: pos_rate=0.75 (3/4), Group 1: pos_rate=0.0 (0/4) -> DPD = 0.75
        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        metrics = compute_fairness_metrics(y_true, y_pred, groups)
        assert metrics.demographic_parity_difference == 0.75

    def test_equal_opportunity_difference(self):
        # Group 0: TPR=1.0, Group 1: TPR=0.5
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0])
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        metrics = compute_fairness_metrics(y_true, y_pred, groups)
        assert metrics.equal_opportunity_difference == 0.5

    def test_equalized_odds_difference(self):
        # Group 0: TPR=1.0, FPR=0.0; Group 1: TPR=0.5, FPR=0.5
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 1, 0, 1, 0])
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        metrics = compute_fairness_metrics(y_true, y_pred, groups)
        # max(TPR diff=0.5, FPR diff=0.5) = 0.5
        assert metrics.equalized_odds_difference == 0.5

    def test_disparate_impact_ratio(self):
        # Group 0: pos_rate=0.75, Group 1: pos_rate=0.0 -> DI = 0.0/0.75 = 0.0
        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        metrics = compute_fairness_metrics(y_true, y_pred, groups)
        assert metrics.disparate_impact_ratio == 0.0

    def test_disparate_impact_ratio_nonzero(self):
        # Group 0: pos_rate=0.75, Group 1: pos_rate=0.25 -> DI = 0.25/0.75 = 1/3
        y_true = np.array([1, 1, 1, 0, 0, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 1, 0, 0])
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        metrics = compute_fairness_metrics(y_true, y_pred, groups)
        assert abs(metrics.disparate_impact_ratio - 1/3) < 1e-6

    def test_calibration_by_group(self):
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0])
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        metrics = compute_fairness_metrics(y_true, y_pred, groups)
        # Group 0: sum(y_true)=2, sum(y_pred)=2 -> calibration=1.0
        # Group 1: sum(y_true)=2, sum(y_pred)=1 -> calibration=2.0
        assert abs(metrics.calibration_by_group["0"] - 1.0) < 1e-6
        assert abs(metrics.calibration_by_group["1"] - 2.0) < 1e-6

    def test_confusion_matrices_in_output(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1])
        groups = np.array(["A", "A", "B", "B"])
        metrics = compute_fairness_metrics(y_true, y_pred, groups)
        assert "A" in metrics.confusion_matrices
        assert "B" in metrics.confusion_matrices
        for cm in metrics.confusion_matrices.values():
            assert all(k in cm for k in ["tp", "tn", "fp", "fn"])

    def test_raises_on_single_group(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1])
        groups = np.array([0, 0, 0, 0])
        with pytest.raises(ValueError, match="At least two groups"):
            compute_fairness_metrics(y_true, y_pred, groups)

    def test_to_dict(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1])
        groups = np.array([0, 0, 1, 1])
        metrics = compute_fairness_metrics(y_true, y_pred, groups)
        d = metrics.to_dict()
        assert set(d.keys()) == {
            "demographic_parity_difference",
            "equal_opportunity_difference",
            "equalized_odds_difference",
            "disparate_impact_ratio",
            "calibration_by_group",
            "confusion_matrices",
        }


class TestConfusionMatricesByGroup:
    def test_returns_dict_of_matrices(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1])
        groups = np.array(["A", "A", "B", "B"])
        cms = confusion_matrices_by_group(y_true, y_pred, groups)
        assert set(cms.keys()) == {"A", "B"}
        for cm in cms.values():
            assert all(k in cm for k in ["tp", "tn", "fp", "fn"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
