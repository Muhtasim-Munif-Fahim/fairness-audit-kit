"""
Tests for intersectional subgroup fairness.
"""

import argparse
import numpy as np
import pandas as pd
import pytest

from fairness_audit_kit.metrics import compute_fairness_metrics
from fairness_audit_kit.intersectional import (
    make_intersectional_groups,
    compute_intersectional_metrics,
)
from fairness_audit_kit.report import render_intersectional_report
from fairness_audit_kit.cli import cmd_evaluate


def _hidden_bias_data():
    """
    Synthetic 2×2 data where one intersection is fully denied while
    marginal groups look only moderately unequal.

    gender: 0,0,0,0, 0,0,0,0, 1,1,1,1, 1,1,1,1
    race:   0,0,0,0, 1,1,1,1, 0,0,0,0, 1,1,1,1
    y_pred: 1,1,0,0, 0,0,0,0, 1,1,0,0, 1,1,0,0
    y_true: same as y_pred except two extra positives in the denied cell
            so TPR is defined everywhere.
    """
    gender = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    race = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
    return y_true, y_pred, gender, race


class TestMakeIntersectionalGroups:
    def test_label_format(self):
        a = np.array([0, 0, 1, 1])
        b = np.array(["X", "Y", "X", "Y"])
        labels = make_intersectional_groups(a, b)
        np.testing.assert_array_equal(labels, np.array(["0/X", "0/Y", "1/X", "1/Y"], dtype=object))

    def test_custom_separator(self):
        a = np.array(["f", "m"])
        b = np.array(["a", "b"])
        labels = make_intersectional_groups(a, b, separator="×")
        np.testing.assert_array_equal(labels, np.array(["f×a", "m×b"], dtype=object))

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            make_intersectional_groups(np.array([0, 1]), np.array([0]))


class TestComputeIntersectionalMetrics:
    def test_reuses_existing_group_metrics(self):
        y_true, y_pred, gender, race = _hidden_bias_data()
        result = compute_intersectional_metrics(y_true, y_pred, gender, race)
        crossed = make_intersectional_groups(gender, race)
        expected = compute_fairness_metrics(y_true, y_pred, crossed)
        assert result.metrics.demographic_parity_difference == expected.demographic_parity_difference
        assert result.metrics.equal_opportunity_difference == expected.equal_opportunity_difference
        assert result.metrics.equalized_odds_difference == expected.equalized_odds_difference
        assert result.metrics.disparate_impact_ratio == expected.disparate_impact_ratio

    def test_detects_hidden_intersection_bias(self):
        y_true, y_pred, gender, race = _hidden_bias_data()
        # Marginal gender DPD = 0.5 - 0.25 = 0.25
        gender_metrics = compute_fairness_metrics(y_true, y_pred, gender)
        # Marginal race DPD = 0.5 - 0.25 = 0.25
        race_metrics = compute_fairness_metrics(y_true, y_pred, race)
        # Intersectional DPD = 0.5 - 0.0 = 0.5, larger than either margin
        result = compute_intersectional_metrics(y_true, y_pred, gender, race)

        assert abs(gender_metrics.demographic_parity_difference - 0.25) < 1e-6
        assert abs(race_metrics.demographic_parity_difference - 0.25) < 1e-6
        assert abs(result.metrics.demographic_parity_difference - 0.5) < 1e-6
        assert result.metrics.disparate_impact_ratio == 0.0
        assert result.n_intersections == 4

    def test_worst_intersection_by_positive_rate(self):
        y_true, y_pred, gender, race = _hidden_bias_data()
        result = compute_intersectional_metrics(y_true, y_pred, gender, race)
        worst = result.worst_intersection
        assert worst.group == "0/1"
        assert worst.n_samples == 4
        assert worst.positive_rate == 0.0
        assert worst.tpr == 0.0
        assert worst.criterion == "positive_rate"
        assert abs(worst.gap_from_best - 0.5) < 1e-6

    def test_worst_intersection_by_tpr(self):
        y_true, y_pred, gender, race = _hidden_bias_data()
        result = compute_intersectional_metrics(
            y_true, y_pred, gender, race, worst_by="tpr"
        )
        assert result.worst_intersection.group == "0/1"
        assert result.worst_intersection.criterion == "tpr"
        assert result.worst_intersection.tpr == 0.0

    def test_perfect_fairness_across_intersections(self):
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        a = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        b = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        result = compute_intersectional_metrics(y_true, y_pred, a, b)
        assert result.metrics.demographic_parity_difference == 0.0
        assert result.metrics.equal_opportunity_difference == 0.0
        assert result.metrics.equalized_odds_difference == 0.0
        assert result.metrics.disparate_impact_ratio == 1.0
        assert result.worst_intersection.gap_from_best == 0.0

    def test_tie_breaks_lexicographically(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0, 0, 0, 0])
        a = np.array(["b", "b", "a", "a"])
        b = np.array([1, 1, 0, 0])
        result = compute_intersectional_metrics(y_true, y_pred, a, b)
        # Both intersections have positive_rate 0; pick lexicographically first
        assert result.worst_intersection.group == "a/0"

    def test_raises_on_single_intersection(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1])
        a = np.array([0, 0, 0, 0])
        b = np.array([1, 1, 1, 1])
        with pytest.raises(ValueError, match="At least two intersectional groups"):
            compute_intersectional_metrics(y_true, y_pred, a, b)

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            compute_intersectional_metrics(
                np.array([1, 0, 1]),
                np.array([1, 0]),
                np.array([0, 1, 0]),
                np.array([0, 1, 0]),
            )

    def test_invalid_worst_by(self):
        y_true, y_pred, gender, race = _hidden_bias_data()
        with pytest.raises(ValueError, match="worst_by"):
            compute_intersectional_metrics(y_true, y_pred, gender, race, worst_by="fpr")

    def test_to_dict(self):
        y_true, y_pred, gender, race = _hidden_bias_data()
        result = compute_intersectional_metrics(y_true, y_pred, gender, race)
        d = result.to_dict()
        assert set(d.keys()) == {
            "metrics",
            "n_intersections",
            "intersection_labels",
            "group_rates",
            "worst_intersection",
        }
        assert d["worst_intersection"]["group"] == "0/1"
        assert "demographic_parity_difference" in d["metrics"]


class TestIntersectionalReport:
    def test_includes_worst_intersection(self):
        y_true, y_pred, gender, race = _hidden_bias_data()
        result = compute_intersectional_metrics(y_true, y_pred, gender, race)
        report = render_intersectional_report(
            result, groups_a_name="gender", groups_b_name="race"
        )
        assert "## Worst Intersection" in report
        assert "`0/1`" in report
        assert "gender" in report and "race" in report
        assert "Demographic Parity Difference" in report
        assert "0.5000" in report
        assert "| 0/1 |" in report


class TestEvaluateCLIIntersectional:
    def test_evaluate_writes_worst_intersection(self, tmp_path):
        y_true, y_pred, gender, race = _hidden_bias_data()
        df = pd.DataFrame({
            "target": y_true,
            "y_pred": y_pred,
            "sensitive_attr": gender,
            "sensitive_attr_2": race,
        })
        data_path = tmp_path / "preds.csv"
        out_path = tmp_path / "report.md"
        df.to_csv(data_path, index=False)

        args = argparse.Namespace(
            data=str(data_path),
            target_col="target",
            pred_col="y_pred",
            group_col="sensitive_attr",
            group_col_2="sensitive_attr_2",
            title="Fairness Evaluation Report",
            output=str(out_path),
        )
        cmd_evaluate(args)

        text = out_path.read_text()
        assert "Worst Intersection" in text
        assert "0/1" in text
        assert "Intersectional Fairness Report" in text
        assert "Demographic Parity Difference" in text

    def test_evaluate_without_second_group_omits_intersection(self, tmp_path):
        y_true, y_pred, gender, race = _hidden_bias_data()
        df = pd.DataFrame({
            "target": y_true,
            "y_pred": y_pred,
            "sensitive_attr": gender,
        })
        data_path = tmp_path / "preds.csv"
        out_path = tmp_path / "report.md"
        df.to_csv(data_path, index=False)

        args = argparse.Namespace(
            data=str(data_path),
            target_col="target",
            pred_col="y_pred",
            group_col="sensitive_attr",
            group_col_2=None,
            title="Fairness Evaluation Report",
            output=str(out_path),
        )
        cmd_evaluate(args)

        text = out_path.read_text()
        assert "Worst Intersection" not in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
