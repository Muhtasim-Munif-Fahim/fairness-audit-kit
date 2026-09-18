"""
Tests for Theil index / generalized entropy inequality metrics.
"""

import argparse
import math

import numpy as np
import pandas as pd
import pytest

from fairness_audit_kit.theil import (
    EntropyBreakdown,
    GeneralizedEntropyResult,
    classification_benefit,
    classification_error,
    generalized_entropy_index,
    theil_index,
    compute_generalized_entropy,
    compute_theil_metrics,
)
from fairness_audit_kit.report import render_theil_report
from fairness_audit_kit.cli import cmd_evaluate


def _two_group_split():
    """
    Group 0: all true positives (benefit=1, error=0).
    Group 1: all false negatives (benefit=0, error=1).
    """
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    groups = np.array([0, 0, 1, 1])
    return y_true, y_pred, groups


class TestClassificationOutcomes:
    def test_benefit_tp_tn_fp_fn(self):
        y_true = np.array([1, 0, 0, 1])
        y_pred = np.array([1, 0, 1, 0])
        b = classification_benefit(y_true, y_pred)
        np.testing.assert_array_equal(b, np.array([1.0, 1.0, 2.0, 0.0]))

    def test_error_indicator(self):
        y_true = np.array([1, 0, 0, 1])
        y_pred = np.array([1, 0, 1, 0])
        e = classification_error(y_true, y_pred)
        np.testing.assert_array_equal(e, np.array([0.0, 0.0, 1.0, 1.0]))


class TestGeneralizedEntropyIndex:
    def test_perfect_equality_is_zero(self):
        values = np.array([2.0, 2.0, 2.0, 2.0])
        assert generalized_entropy_index(values, alpha=1.0) == 0.0
        assert generalized_entropy_index(values, alpha=2.0) == 0.0
        assert generalized_entropy_index(values, alpha=0.0) == 0.0
        assert theil_index(values) == 0.0

    def test_all_zeros_is_zero(self):
        assert generalized_entropy_index(np.array([0.0, 0.0, 0.0]), alpha=1.0) == 0.0
        assert generalized_entropy_index(np.array([0.0, 0.0, 0.0]), alpha=2.0) == 0.0

    def test_theil_matches_alpha_one(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        assert theil_index(values) == generalized_entropy_index(values, alpha=1.0)

    def test_theil_known_half_half(self):
        # [1, 1, 0, 0], mean=0.5 -> Theil = ln(2)
        values = np.array([1.0, 1.0, 0.0, 0.0])
        assert abs(theil_index(values) - math.log(2.0)) < 1e-12

    def test_alpha_two_is_half_squared_cv(self):
        values = np.array([1.0, 1.0, 0.0, 0.0])
        mu = np.mean(values)
        cv2 = np.mean((values - mu) ** 2) / mu**2
        expected = 0.5 * cv2
        assert abs(generalized_entropy_index(values, alpha=2.0) - expected) < 1e-12
        assert abs(generalized_entropy_index(values, alpha=2.0) - 0.5) < 1e-12

    def test_alpha_zero_mean_log_deviation(self):
        values = np.array([1.0, 2.0, 3.0])
        mu = np.mean(values)
        expected = -np.mean(np.log(values / mu))
        assert abs(generalized_entropy_index(values, alpha=0.0) - expected) < 1e-12

    def test_theil_handles_zeros(self):
        values = np.array([0.0, 2.0])
        # mean=1, ratios 0 and 2 -> mean(0 + 2*ln(2)) / 2 = ln(2)
        assert abs(theil_index(values) - math.log(2.0)) < 1e-12

    def test_rejects_negative_values(self):
        with pytest.raises(ValueError, match="non-negative"):
            generalized_entropy_index(np.array([1.0, -0.1]))

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            generalized_entropy_index(np.array([]))

    def test_alpha_zero_rejects_zeros(self):
        with pytest.raises(ValueError, match="strictly positive"):
            generalized_entropy_index(np.array([0.0, 1.0]), alpha=0.0)


class TestComputeGeneralizedEntropy:
    def test_perfect_group_equality(self):
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        result = compute_theil_metrics(y_true, y_pred, groups)
        assert result.alpha == 1.0
        assert result.benefit.overall == 0.0
        assert result.benefit.between_group == 0.0
        assert result.benefit.within_group == 0.0
        assert result.error.overall == 0.0
        assert result.error.between_group == 0.0
        assert result.mean_error_rate == 0.0
        assert result.group_mean_benefit["0"] == result.group_mean_benefit["1"]

    def test_between_group_benefit_inequality(self):
        y_true, y_pred, groups = _two_group_split()
        result = compute_theil_metrics(y_true, y_pred, groups)
        # Individuals inside each group share the same benefit, so all
        # inequality is between groups and equals ln(2).
        assert abs(result.benefit.overall - math.log(2.0)) < 1e-12
        assert abs(result.benefit.between_group - math.log(2.0)) < 1e-12
        assert result.benefit.within_group == 0.0
        assert result.group_mean_benefit["0"] == 1.0
        assert result.group_mean_benefit["1"] == 0.0

    def test_between_group_error_inequality(self):
        y_true, y_pred, groups = _two_group_split()
        result = compute_theil_metrics(y_true, y_pred, groups)
        assert abs(result.error.overall - math.log(2.0)) < 1e-12
        assert abs(result.error.between_group - math.log(2.0)) < 1e-12
        assert result.error.within_group == 0.0
        assert result.group_error_rate["0"] == 0.0
        assert result.group_error_rate["1"] == 1.0
        assert result.mean_error_rate == 0.5

    def test_decomposition_overall_equals_between_plus_within(self):
        y_true = np.array([1, 1, 1, 0, 1, 0, 0, 0])
        y_pred = np.array([1, 0, 1, 1, 1, 0, 1, 0])
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        for alpha in (1.0, 2.0):
            result = compute_generalized_entropy(y_true, y_pred, groups, alpha=alpha)
            assert abs(
                result.benefit.overall
                - (result.benefit.between_group + result.benefit.within_group)
            ) < 1e-12
            assert abs(
                result.error.overall
                - (result.error.between_group + result.error.within_group)
            ) < 1e-12

    def test_within_group_positive_when_individuals_differ(self):
        # Group 0 mixed outcomes; group 1 identical true negatives.
        y_true = np.array([1, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        groups = np.array([0, 0, 1, 1])
        result = compute_theil_metrics(y_true, y_pred, groups)
        assert result.benefit.within_group > 0
        assert result.benefit.between_group >= 0
        assert result.benefit.overall >= result.benefit.between_group

    def test_alpha_two_on_benefits(self):
        y_true, y_pred, groups = _two_group_split()
        result = compute_generalized_entropy(y_true, y_pred, groups, alpha=2.0)
        assert result.alpha == 2.0
        assert abs(result.benefit.overall - 0.5) < 1e-12
        assert abs(result.benefit.between_group - 0.5) < 1e-12

    def test_compute_theil_metrics_is_alpha_one(self):
        y_true, y_pred, groups = _two_group_split()
        theil = compute_theil_metrics(y_true, y_pred, groups)
        gei = compute_generalized_entropy(y_true, y_pred, groups, alpha=1.0)
        assert theil.benefit.overall == gei.benefit.overall
        assert theil.error.between_group == gei.error.between_group

    def test_raises_on_single_group(self):
        with pytest.raises(ValueError, match="At least two groups"):
            compute_theil_metrics(
                np.array([1, 0, 1, 0]),
                np.array([1, 0, 0, 1]),
                np.array([0, 0, 0, 0]),
            )

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            compute_theil_metrics(
                np.array([1, 0, 1]),
                np.array([1, 0]),
                np.array([0, 1, 0]),
            )

    def test_to_dict(self):
        y_true, y_pred, groups = _two_group_split()
        result = compute_theil_metrics(y_true, y_pred, groups)
        d = result.to_dict()
        assert set(d.keys()) == {
            "alpha",
            "n_groups",
            "n_samples",
            "mean_benefit",
            "mean_error_rate",
            "benefit",
            "error",
            "group_mean_benefit",
            "group_error_rate",
            "group_size",
        }
        assert set(d["benefit"].keys()) == {"overall", "between_group", "within_group"}
        assert d["n_groups"] == 2
        assert d["n_samples"] == 4
        assert isinstance(result, GeneralizedEntropyResult)
        assert isinstance(result.benefit, EntropyBreakdown)


class TestTheilReport:
    def test_includes_benefit_and_error_sections(self):
        y_true, y_pred, groups = _two_group_split()
        result = compute_theil_metrics(y_true, y_pred, groups)
        report = render_theil_report(result)
        assert "# Theil / Generalized Entropy Report" in report
        assert "Theil index" in report
        assert "Benefit Inequality" in report
        assert "Error Inequality" in report
        assert "Between-Group" in report
        assert "Within-Group" in report
        assert "| 0 |" in report and "| 1 |" in report
        assert f"{math.log(2.0):.4f}" in report


class TestEvaluateCLITheil:
    def test_evaluate_writes_theil_section(self, tmp_path):
        y_true, y_pred, groups = _two_group_split()
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
        assert "Theil / Generalized Entropy Report" in text
        assert "Benefit Inequality" in text
        assert "Error Inequality" in text
        assert "Demographic Parity Difference" in text

    def test_evaluate_custom_alpha(self, tmp_path):
        y_true, y_pred, groups = _two_group_split()
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
            entropy_alpha=2.0,
            title="Fairness Evaluation Report",
            output=str(out_path),
        )
        cmd_evaluate(args)

        text = out_path.read_text()
        assert "alpha = 2" in text
        assert "0.5000" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
