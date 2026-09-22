"""
Tests for demographic parity (statistical parity difference) and disparate impact.
"""

import argparse

import numpy as np
import pandas as pd
import pytest

from fairness_audit_kit.demographic_parity import (
    DemographicParityResult,
    compute_demographic_parity,
)
from fairness_audit_kit.metrics import compute_fairness_metrics
from fairness_audit_kit.odds_parity import PairwiseGap
from fairness_audit_kit.report import render_demographic_parity_report
from fairness_audit_kit.cli import cmd_evaluate


def _known_gap():
    """
    Group 0: positive rate 0.75 (3/4)
    Group 1: positive rate 0.25 (1/4)
    SPD = 0.50, DI = 1/3
    """
    y_pred = np.array([1, 1, 1, 0, 0, 1, 0, 0])
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_true = np.array([1, 1, 1, 0, 0, 1, 0, 0])
    return y_pred, groups, y_true


def _perfect_parity():
    y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return y_pred, groups


class TestDemographicParity:
    def test_perfect_parity_is_zero_spd_and_unit_di(self):
        y_pred, groups = _perfect_parity()
        result = compute_demographic_parity(y_pred, groups)
        assert result.statistical_parity_difference == 0.0
        assert result.demographic_parity_difference == 0.0
        assert result.disparate_impact_ratio == 1.0
        assert result.positive_rates["0"] == 0.5
        assert result.positive_rates["1"] == 0.5
        assert result.positive_count["0"] == 2
        assert result.positive_count["1"] == 2
        assert result.group_size["0"] == 4
        assert result.n_groups == 2
        assert result.n_samples == 8
        assert result.threshold is None

    def test_known_spd_and_di(self):
        y_pred, groups, _y_true = _known_gap()
        result = compute_demographic_parity(y_pred, groups)
        assert abs(result.positive_rates["0"] - 0.75) < 1e-12
        assert abs(result.positive_rates["1"] - 0.25) < 1e-12
        assert result.positive_count["0"] == 3
        assert result.positive_count["1"] == 1
        assert abs(result.statistical_parity_difference - 0.5) < 1e-12
        assert abs(result.demographic_parity_difference - 0.5) < 1e-12
        assert abs(result.disparate_impact_ratio - (1 / 3)) < 1e-12
        assert result.positive_rate.rate_name == "positive_rate"
        assert abs(result.positive_rate.difference - 0.5) < 1e-12

    def test_matches_scalar_fairness_metrics(self):
        y_pred, groups, y_true = _known_gap()
        detailed = compute_demographic_parity(y_pred, groups)
        scalar = compute_fairness_metrics(y_true, y_pred, groups)
        assert abs(
            detailed.demographic_parity_difference
            - scalar.demographic_parity_difference
        ) < 1e-12
        assert abs(
            detailed.statistical_parity_difference
            - scalar.demographic_parity_difference
        ) < 1e-12
        assert abs(detailed.disparate_impact_ratio - scalar.disparate_impact_ratio) < 1e-12

    def test_zero_positive_rate_di_is_zero(self):
        # Group 0: 0.75, Group 1: 0.0 -> SPD = 0.75, DI = 0.0
        y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        result = compute_demographic_parity(y_pred, groups)
        assert abs(result.statistical_parity_difference - 0.75) < 1e-12
        assert result.disparate_impact_ratio == 0.0

    def test_all_negative_predictions_di_is_one(self):
        y_pred = np.array([0, 0, 0, 0, 0, 0])
        groups = np.array(["A", "A", "A", "B", "B", "B"])
        result = compute_demographic_parity(y_pred, groups)
        assert result.positive_rates["A"] == 0.0
        assert result.positive_rates["B"] == 0.0
        assert result.statistical_parity_difference == 0.0
        assert result.disparate_impact_ratio == 1.0

    def test_pairwise_gap_signed(self):
        y_pred, groups, _y_true = _known_gap()
        result = compute_demographic_parity(y_pred, groups)
        assert len(result.positive_rate.pairwise_gaps) == 1
        gap = result.positive_rate.pairwise_gaps[0]
        assert isinstance(gap, PairwiseGap)
        assert gap.group_a == "0"
        assert gap.group_b == "1"
        assert abs(gap.rate_a - 0.75) < 1e-12
        assert abs(gap.rate_b - 0.25) < 1e-12
        assert abs(gap.gap - 0.5) < 1e-12
        assert abs(gap.abs_gap - 0.5) < 1e-12

    def test_three_group_pairwise_gaps(self):
        # A=1.0, B=0.5, C=0.0
        y_pred = np.array([1, 1, 1, 0, 0, 0])
        groups = np.array(["A", "A", "B", "B", "C", "C"])
        result = compute_demographic_parity(y_pred, groups)
        assert abs(result.positive_rates["A"] - 1.0) < 1e-12
        assert abs(result.positive_rates["B"] - 0.5) < 1e-12
        assert abs(result.positive_rates["C"] - 0.0) < 1e-12
        assert abs(result.statistical_parity_difference - 1.0) < 1e-12
        assert result.disparate_impact_ratio == 0.0
        pairs = {
            (g.group_a, g.group_b): g.abs_gap
            for g in result.positive_rate.pairwise_gaps
        }
        assert set(pairs) == {("A", "B"), ("A", "C"), ("B", "C")}
        assert abs(pairs[("A", "B")] - 0.5) < 1e-12
        assert abs(pairs[("A", "C")] - 1.0) < 1e-12
        assert abs(pairs[("B", "C")] - 0.5) < 1e-12

    def test_string_groups_and_list_inputs(self):
        result = compute_demographic_parity(
            [1, 0, 1, 0],
            ["female", "female", "male", "male"],
        )
        assert result.positive_rates["female"] == 0.5
        assert result.positive_rates["male"] == 0.5
        assert result.statistical_parity_difference == 0.0

    def test_scores_only_matches_thresholded_pred(self):
        y_pred, groups, _y_true = _known_gap()
        y_scores = np.where(y_pred == 1, 0.9, 0.1).astype(float)
        from_pred = compute_demographic_parity(y_pred, groups)
        from_scores = compute_demographic_parity(
            None, groups, y_scores=y_scores, threshold=0.5
        )
        assert from_scores.threshold == 0.5
        assert from_pred.threshold is None
        assert from_scores.positive_rates == from_pred.positive_rates
        assert abs(
            from_scores.statistical_parity_difference
            - from_pred.statistical_parity_difference
        ) < 1e-12
        assert abs(
            from_scores.disparate_impact_ratio - from_pred.disparate_impact_ratio
        ) < 1e-12

    def test_custom_threshold(self):
        groups = np.array([0, 0, 1, 1])
        y_scores = np.array([0.6, 0.9, 0.6, 0.2])
        result = compute_demographic_parity(
            None, groups, y_scores=y_scores, threshold=0.7
        )
        # Group 0: [0, 1] rate 0.5; Group 1: [0, 0] rate 0.0
        assert abs(result.positive_rates["0"] - 0.5) < 1e-12
        assert result.positive_rates["1"] == 0.0
        assert abs(result.statistical_parity_difference - 0.5) < 1e-12
        assert result.disparate_impact_ratio == 0.0
        assert result.threshold == 0.7

    def test_hard_labels_win_when_both_given(self):
        y_pred, groups, _y_true = _known_gap()
        y_scores = np.where(y_pred == 1, 0.1, 0.9).astype(float)
        result = compute_demographic_parity(y_pred, groups, y_scores=y_scores)
        expected = compute_demographic_parity(y_pred, groups)
        assert result.threshold is None
        assert result.positive_rates == expected.positive_rates
        assert abs(
            result.statistical_parity_difference
            - expected.statistical_parity_difference
        ) < 1e-12

    def test_to_dict(self):
        y_pred, groups, _y_true = _known_gap()
        result = compute_demographic_parity(y_pred, groups)
        d = result.to_dict()
        assert set(d.keys()) == {
            "n_groups",
            "n_samples",
            "group_size",
            "positive_count",
            "positive_rates",
            "positive_rate",
            "statistical_parity_difference",
            "demographic_parity_difference",
            "disparate_impact_ratio",
            "threshold",
        }
        assert set(d["positive_rate"].keys()) == {
            "rate_name",
            "group_rates",
            "group_size",
            "group_support",
            "pairwise_gaps",
            "difference",
        }
        assert d["positive_rate"]["rate_name"] == "positive_rate"
        assert "gap" in d["positive_rate"]["pairwise_gaps"][0]
        assert isinstance(result, DemographicParityResult)

    def test_float_binary_labels(self):
        y_pred = np.array([1.0, 0.0, 1.0, 0.0])
        groups = np.array([0, 0, 1, 1])
        result = compute_demographic_parity(y_pred, groups)
        assert result.positive_rates["0"] == 0.5
        assert result.positive_rates["1"] == 0.5


class TestValidation:
    def test_requires_pred_or_scores(self):
        with pytest.raises(ValueError, match="y_pred or y_scores"):
            compute_demographic_parity(None, np.array([0, 0, 1, 1]))

    def test_rejects_single_group(self):
        with pytest.raises(ValueError, match="At least two groups"):
            compute_demographic_parity(
                np.array([1, 0, 1, 0]),
                np.array([0, 0, 0, 0]),
            )

    def test_rejects_non_binary_pred(self):
        with pytest.raises(ValueError, match="binary"):
            compute_demographic_parity(
                np.array([1, 0, 2, 1]),
                np.array([0, 0, 1, 1]),
            )

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            compute_demographic_parity(np.array([]), np.array([]))

    def test_rejects_length_mismatch_pred(self):
        with pytest.raises(ValueError, match="same length"):
            compute_demographic_parity(
                np.array([1, 0]),
                np.array([0, 1, 0]),
            )

    def test_rejects_length_mismatch_scores(self):
        with pytest.raises(ValueError, match="same length"):
            compute_demographic_parity(
                None,
                np.array([0, 1, 0]),
                y_scores=np.array([0.9, 0.1]),
            )

    def test_rejects_non_finite_scores(self):
        with pytest.raises(ValueError, match="finite"):
            compute_demographic_parity(
                None,
                np.array([0, 0, 1, 1]),
                y_scores=np.array([0.9, np.nan, 0.2, 0.1]),
            )


class TestDemographicParityReport:
    def test_includes_rates_spd_and_di(self):
        y_pred, groups, _y_true = _known_gap()
        result = compute_demographic_parity(y_pred, groups)
        report = render_demographic_parity_report(result)
        assert "# Demographic Parity / Disparate Impact Report" in report
        assert "Statistical Parity Difference" in report
        assert "Demographic Parity Difference" in report
        assert "Disparate Impact Ratio" in report
        assert "Positive Rates by Group" in report
        assert "Pairwise Gaps" in report
        assert "Positive Prediction Rate" in report
        assert "| 0 |" in report and "| 1 |" in report
        assert "0.7500" in report
        assert "0.2500" in report
        assert "0.5000" in report
        assert "0.3333" in report
        assert "Dwork" in report
        assert "four-fifths" in report


class TestEvaluateCLIDemographicParity:
    def test_evaluate_writes_demographic_parity_section(self, tmp_path):
        y_pred, groups, y_true = _known_gap()
        df = pd.DataFrame({
            "target": y_true,
            "y_pred": y_pred,
            "sensitive_attr": groups,
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
            entropy_alpha=1.0,
            title="Fairness Evaluation Report",
            output=str(out_path),
        )
        cmd_evaluate(args)

        text = out_path.read_text()
        assert "Demographic Parity / Disparate Impact Report" in text
        assert "Statistical Parity Difference" in text
        assert "Disparate Impact Ratio" in text
        assert "Positive Rates by Group" in text
        assert "0.7500" in text
        assert "Equalized Odds / Predictive Parity Report" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
