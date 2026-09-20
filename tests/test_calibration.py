"""
Tests for reliability diagrams and expected calibration error.
"""

import argparse

import numpy as np
import pandas as pd
import pytest

from fairness_audit_kit.calibration import (
    CalibrationResult,
    ReliabilityBin,
    compute_reliability_bins,
    expected_calibration_error,
    maximum_calibration_error,
    compute_calibration_metrics,
)
from fairness_audit_kit.report import render_calibration_report
from fairness_audit_kit.cli import cmd_evaluate


def _two_group_constant_scores():
    """
    Group 0: scores = labels (perfectly calibrated).
    Group 1: all scores 0.8, half positive (ECE = 0.3).
    """
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_scores = np.array([1.0, 1.0, 0.0, 0.0, 0.8, 0.8, 0.8, 0.8])
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return y_true, y_scores, groups


class TestReliabilityBins:
    def test_perfect_calibration_is_zero(self):
        y_true = np.array([1, 1, 0, 0])
        y_scores = np.array([1.0, 1.0, 0.0, 0.0])
        assert expected_calibration_error(y_true, y_scores) == 0.0
        assert maximum_calibration_error(y_true, y_scores) == 0.0

    def test_constant_score_matches_base_rate(self):
        y_true = np.array([1, 1, 0, 0])
        y_scores = np.array([0.5, 0.5, 0.5, 0.5])
        assert expected_calibration_error(y_true, y_scores) == 0.0

    def test_known_overconfidence_ece(self):
        y_true = np.array([1, 1, 0, 0])
        y_scores = np.array([0.8, 0.8, 0.8, 0.8])
        ece = expected_calibration_error(y_true, y_scores, n_bins=10)
        mce = maximum_calibration_error(y_true, y_scores, n_bins=10)
        assert abs(ece - 0.3) < 1e-12
        assert abs(mce - 0.3) < 1e-12

    def test_ece_is_weighted_mean_of_gaps(self):
        y_true = np.array([1, 0, 1, 0, 1, 1, 0, 0])
        y_scores = np.array([0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8])
        bins = compute_reliability_bins(y_true, y_scores, n_bins=10)
        assert len(bins) == 2
        expected = sum(b.weight * b.gap for b in bins)
        assert abs(expected_calibration_error(y_true, y_scores) - expected) < 1e-12
        assert abs(sum(b.weight for b in bins) - 1.0) < 1e-12

    def test_uniform_bin_edges_and_assignment(self):
        y_true = np.array([0, 1, 0])
        y_scores = np.array([0.0, 0.49, 1.0])
        bins = compute_reliability_bins(y_true, y_scores, n_bins=2)
        by_index = {b.bin_index: b for b in bins}
        assert by_index[0].lower == 0.0
        assert by_index[0].upper == 0.5
        assert by_index[0].n_samples == 2
        assert by_index[1].lower == 0.5
        assert by_index[1].upper == 1.0
        assert by_index[1].n_samples == 1
        assert by_index[1].mean_confidence == 1.0
        assert by_index[1].observed_positive_rate == 0.0

    def test_empty_bins_omitted(self):
        y_true = np.array([1, 0])
        y_scores = np.array([0.05, 0.06])
        bins = compute_reliability_bins(y_true, y_scores, n_bins=10)
        assert len(bins) == 1
        assert bins[0].bin_index == 0
        assert bins[0].n_samples == 2

    def test_quantile_equal_mass_bins(self):
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        y_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        bins = compute_reliability_bins(
            y_true, y_scores, n_bins=2, strategy="quantile"
        )
        assert len(bins) == 2
        assert bins[0].n_samples == 4
        assert bins[1].n_samples == 4
        assert abs(bins[0].mean_confidence - 0.25) < 1e-12
        assert abs(bins[1].mean_confidence - 0.75) < 1e-12

    def test_quantile_identical_scores_single_bin(self):
        y_true = np.array([1, 0, 1, 0])
        y_scores = np.array([0.4, 0.4, 0.4, 0.4])
        bins = compute_reliability_bins(
            y_true, y_scores, n_bins=5, strategy="quantile"
        )
        assert len(bins) == 1
        assert bins[0].n_samples == 4
        assert abs(bins[0].mean_confidence - 0.4) < 1e-12
        assert abs(bins[0].observed_positive_rate - 0.5) < 1e-12
        assert abs(
            expected_calibration_error(
                y_true, y_scores, n_bins=5, strategy="quantile"
            )
            - 0.1
        ) < 1e-12

    def test_bin_to_dict(self):
        y_true = np.array([1, 0])
        y_scores = np.array([0.9, 0.9])
        row = compute_reliability_bins(y_true, y_scores, n_bins=10)[0]
        d = row.to_dict()
        assert set(d.keys()) == {
            "bin_index",
            "lower",
            "upper",
            "n_samples",
            "mean_confidence",
            "observed_positive_rate",
            "gap",
            "weight",
        }
        assert isinstance(row, ReliabilityBin)

    def test_rejects_scores_outside_unit_interval(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            expected_calibration_error(
                np.array([1, 0]),
                np.array([1.2, 0.1]),
            )

    def test_rejects_non_binary_labels(self):
        with pytest.raises(ValueError, match="binary"):
            expected_calibration_error(
                np.array([1, 2]),
                np.array([0.4, 0.6]),
            )

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            expected_calibration_error(np.array([]), np.array([]))

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            expected_calibration_error(np.array([1, 0]), np.array([0.5]))

    def test_rejects_bad_n_bins(self):
        with pytest.raises(ValueError, match="n_bins"):
            compute_reliability_bins(
                np.array([1, 0]),
                np.array([0.5, 0.5]),
                n_bins=0,
            )

    def test_rejects_bad_strategy(self):
        with pytest.raises(ValueError, match="strategy"):
            compute_reliability_bins(
                np.array([1, 0]),
                np.array([0.5, 0.5]),
                strategy="width",
            )


class TestComputeCalibrationMetrics:
    def test_overall_and_per_group_ece(self):
        y_true, y_scores, groups = _two_group_constant_scores()
        result = compute_calibration_metrics(y_true, y_scores, groups, n_bins=10)
        assert result.n_groups == 2
        assert result.n_samples == 8
        assert result.n_bins == 10
        assert result.strategy == "uniform"
        assert result.group_ece["0"] == 0.0
        assert abs(result.group_ece["1"] - 0.3) < 1e-12
        assert abs(result.ece_difference - 0.3) < 1e-12
        assert result.group_size["0"] == 4
        assert result.group_size["1"] == 4
        # Overall: four perfect points (gap 0) and four with gap 0.3
        assert abs(result.ece - 0.15) < 1e-12
        assert abs(result.mce - 0.3) < 1e-12

    def test_perfect_group_equality(self):
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_scores = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        result = compute_calibration_metrics(y_true, y_scores, groups)
        assert result.ece == 0.0
        assert result.group_ece["0"] == 0.0
        assert result.group_ece["1"] == 0.0
        assert result.ece_difference == 0.0

    def test_group_reliability_tables(self):
        y_true, y_scores, groups = _two_group_constant_scores()
        result = compute_calibration_metrics(y_true, y_scores, groups)
        assert set(result.group_reliability_bins.keys()) == {"0", "1"}
        assert all(b.gap == 0.0 for b in result.group_reliability_bins["0"])
        assert any(abs(b.gap - 0.3) < 1e-12 for b in result.group_reliability_bins["1"])
        assert sum(b.n_samples for b in result.reliability_bins) == 8

    def test_quantile_strategy_on_groups(self):
        y_true, y_scores, groups = _two_group_constant_scores()
        result = compute_calibration_metrics(
            y_true, y_scores, groups, n_bins=4, strategy="quantile"
        )
        assert result.strategy == "quantile"
        assert result.n_bins == 4
        assert result.group_ece["0"] == 0.0
        assert abs(result.group_ece["1"] - 0.3) < 1e-12

    def test_raises_on_single_group(self):
        with pytest.raises(ValueError, match="At least two groups"):
            compute_calibration_metrics(
                np.array([1, 0, 1, 0]),
                np.array([0.9, 0.1, 0.8, 0.2]),
                np.array([0, 0, 0, 0]),
            )

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            compute_calibration_metrics(
                np.array([1, 0, 1]),
                np.array([0.9, 0.1, 0.8]),
                np.array([0, 1]),
            )

    def test_to_dict(self):
        y_true, y_scores, groups = _two_group_constant_scores()
        result = compute_calibration_metrics(y_true, y_scores, groups)
        d = result.to_dict()
        assert set(d.keys()) == {
            "n_bins",
            "strategy",
            "n_groups",
            "n_samples",
            "ece",
            "mce",
            "ece_difference",
            "reliability_bins",
            "group_ece",
            "group_mce",
            "group_size",
            "group_reliability_bins",
        }
        assert isinstance(d["reliability_bins"], list)
        assert "mean_confidence" in d["reliability_bins"][0]
        assert set(d["group_reliability_bins"].keys()) == {"0", "1"}
        assert isinstance(result, CalibrationResult)


class TestCalibrationReport:
    def test_includes_ece_and_reliability_tables(self):
        y_true, y_scores, groups = _two_group_constant_scores()
        result = compute_calibration_metrics(y_true, y_scores, groups)
        report = render_calibration_report(result)
        assert "# Reliability / Calibration Report" in report
        assert "Overall ECE" in report
        assert "Overall MCE" in report
        assert "ECE Difference" in report
        assert "ECE by Group" in report
        assert "Overall Reliability Diagram" in report
        assert "Reliability by Group" in report
        assert "| 0 |" in report and "| 1 |" in report
        assert "0.3000" in report
        assert "0.1500" in report
        assert "Mean Confidence" in report
        assert "Observed Rate" in report


class TestEvaluateCLICalibration:
    def test_evaluate_writes_calibration_section(self, tmp_path):
        y_true, y_scores, groups = _two_group_constant_scores()
        df = pd.DataFrame({
            "target": y_true,
            "y_pred": (y_scores >= 0.5).astype(int),
            "y_score": y_scores,
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
            score_col="y_score",
            calibration_bins=10,
            calibration_strategy="uniform",
            title="Fairness Evaluation Report",
            output=str(out_path),
        )
        cmd_evaluate(args)

        text = out_path.read_text()
        assert "Reliability / Calibration Report" in text
        assert "Overall ECE" in text
        assert "Overall Reliability Diagram" in text
        assert "Theil / Generalized Entropy Report" in text
        assert "Demographic Parity Difference" in text

    def test_evaluate_without_scores_omits_calibration(self, tmp_path):
        y_true, y_scores, groups = _two_group_constant_scores()
        df = pd.DataFrame({
            "target": y_true,
            "y_pred": (y_scores >= 0.5).astype(int),
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
            score_col="y_score",
            calibration_bins=10,
            calibration_strategy="uniform",
            title="Fairness Evaluation Report",
            output=str(out_path),
        )
        cmd_evaluate(args)

        text = out_path.read_text()
        assert "Reliability / Calibration Report" not in text
        assert "Theil / Generalized Entropy Report" in text

    def test_evaluate_custom_bins_and_strategy(self, tmp_path):
        y_true, y_scores, groups = _two_group_constant_scores()
        df = pd.DataFrame({
            "target": y_true,
            "y_pred": (y_scores >= 0.5).astype(int),
            "y_score": y_scores,
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
            score_col="y_score",
            calibration_bins=4,
            calibration_strategy="quantile",
            title="Fairness Evaluation Report",
            output=str(out_path),
        )
        cmd_evaluate(args)

        text = out_path.read_text()
        assert "4 equal-mass (quantile) bins" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
