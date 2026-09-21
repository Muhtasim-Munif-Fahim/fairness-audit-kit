"""
Tests for equalized odds (TPR/FPR) and predictive parity (PPV) gaps.
"""

import argparse

import numpy as np
import pandas as pd
import pytest

from fairness_audit_kit.metrics import compute_fairness_metrics
from fairness_audit_kit.odds_parity import (
    EqualizedOddsResult,
    OddsParityResult,
    PairwiseGap,
    PredictiveParityResult,
    compute_equalized_odds,
    compute_odds_parity_metrics,
    compute_predictive_parity,
)
from fairness_audit_kit.report import render_odds_parity_report
from fairness_audit_kit.cli import cmd_evaluate


def _two_group_odds_split():
    """
    Group 0: TPR=1.0, FPR=0.0, PPV=1.0
    Group 1: TPR=0.5, FPR=0.5, PPV=0.5
    """
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return y_true, y_pred, groups


def _perfect_parity():
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return y_true, y_pred, groups


class TestEqualizedOdds:
    def test_perfect_parity_is_zero(self):
        y_true, y_pred, groups = _perfect_parity()
        result = compute_equalized_odds(y_true, y_pred, groups)
        assert result.tpr_difference == 0.0
        assert result.fpr_difference == 0.0
        assert result.equalized_odds_difference == 0.0
        assert result.tpr.group_rates["0"] == 1.0
        assert result.tpr.group_rates["1"] == 1.0
        assert result.fpr.group_rates["0"] == 0.0
        assert result.fpr.group_rates["1"] == 0.0

    def test_known_tpr_fpr_gaps(self):
        y_true, y_pred, groups = _two_group_odds_split()
        result = compute_equalized_odds(y_true, y_pred, groups)
        assert abs(result.tpr.group_rates["0"] - 1.0) < 1e-12
        assert abs(result.tpr.group_rates["1"] - 0.5) < 1e-12
        assert abs(result.fpr.group_rates["0"] - 0.0) < 1e-12
        assert abs(result.fpr.group_rates["1"] - 0.5) < 1e-12
        assert abs(result.tpr_difference - 0.5) < 1e-12
        assert abs(result.fpr_difference - 0.5) < 1e-12
        assert abs(result.equalized_odds_difference - 0.5) < 1e-12
        assert result.n_groups == 2
        assert result.n_samples == 8
        assert result.group_size["0"] == 4
        assert result.group_size["1"] == 4
        assert result.tpr.group_support["0"] == 2
        assert result.fpr.group_support["0"] == 2

    def test_pairwise_tpr_gap_signed(self):
        y_true, y_pred, groups = _two_group_odds_split()
        result = compute_equalized_odds(y_true, y_pred, groups)
        assert len(result.tpr.pairwise_gaps) == 1
        gap = result.tpr.pairwise_gaps[0]
        assert gap.group_a == "0"
        assert gap.group_b == "1"
        assert abs(gap.gap - 0.5) < 1e-12
        assert abs(gap.abs_gap - 0.5) < 1e-12
        assert abs(gap.rate_a - 1.0) < 1e-12
        assert abs(gap.rate_b - 0.5) < 1e-12
        assert isinstance(gap, PairwiseGap)

    def test_matches_scalar_fairness_metrics(self):
        y_true, y_pred, groups = _two_group_odds_split()
        detailed = compute_equalized_odds(y_true, y_pred, groups)
        scalar = compute_fairness_metrics(y_true, y_pred, groups)
        assert abs(detailed.tpr_difference - scalar.equal_opportunity_difference) < 1e-12
        assert abs(detailed.equalized_odds_difference - scalar.equalized_odds_difference) < 1e-12

    def test_three_group_pairwise_gaps(self):
        # All positives: TPR A=1, B=0.5, C=0. Include negatives so FPR is defined.
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"])
        result = compute_equalized_odds(y_true, y_pred, groups)
        assert abs(result.tpr.group_rates["A"] - 1.0) < 1e-12
        assert abs(result.tpr.group_rates["B"] - 0.5) < 1e-12
        assert abs(result.tpr.group_rates["C"] - 0.0) < 1e-12
        assert abs(result.tpr_difference - 1.0) < 1e-12
        pairs = {(g.group_a, g.group_b): g.abs_gap for g in result.tpr.pairwise_gaps}
        assert set(pairs) == {("A", "B"), ("A", "C"), ("B", "C")}
        assert abs(pairs[("A", "B")] - 0.5) < 1e-12
        assert abs(pairs[("A", "C")] - 1.0) < 1e-12
        assert abs(pairs[("B", "C")] - 0.5) < 1e-12

    def test_scores_only_matches_thresholded_pred(self):
        y_true, y_pred, groups = _two_group_odds_split()
        y_scores = np.where(y_pred == 1, 0.9, 0.1).astype(float)
        from_pred = compute_equalized_odds(y_true, y_pred, groups)
        from_scores = compute_equalized_odds(
            y_true, None, groups, y_scores=y_scores, threshold=0.5
        )
        assert from_scores.threshold == 0.5
        assert from_pred.threshold is None
        assert abs(from_scores.equalized_odds_difference - from_pred.equalized_odds_difference) < 1e-12
        assert from_scores.tpr.group_rates == from_pred.tpr.group_rates
        assert from_scores.fpr.group_rates == from_pred.fpr.group_rates

    def test_hard_labels_win_when_both_given(self):
        y_true, y_pred, groups = _two_group_odds_split()
        # Scores that would invert every prediction at 0.5
        y_scores = np.where(y_pred == 1, 0.1, 0.9).astype(float)
        result = compute_equalized_odds(
            y_true, y_pred, groups, y_scores=y_scores
        )
        expected = compute_equalized_odds(y_true, y_pred, groups)
        assert result.threshold is None
        assert abs(result.equalized_odds_difference - expected.equalized_odds_difference) < 1e-12

    def test_to_dict(self):
        y_true, y_pred, groups = _two_group_odds_split()
        result = compute_equalized_odds(y_true, y_pred, groups)
        d = result.to_dict()
        assert set(d.keys()) == {
            "n_groups",
            "n_samples",
            "group_size",
            "tpr",
            "fpr",
            "tpr_difference",
            "fpr_difference",
            "equalized_odds_difference",
            "confusion_matrices",
            "threshold",
        }
        assert set(d["tpr"].keys()) == {
            "rate_name",
            "group_rates",
            "group_size",
            "group_support",
            "pairwise_gaps",
            "difference",
        }
        assert d["tpr"]["rate_name"] == "tpr"
        assert "gap" in d["tpr"]["pairwise_gaps"][0]
        assert isinstance(result, EqualizedOddsResult)


class TestPredictiveParity:
    def test_perfect_parity_is_zero(self):
        y_true, y_pred, groups = _perfect_parity()
        result = compute_predictive_parity(y_true, y_pred, groups)
        assert result.ppv_difference == 0.0
        assert result.ppv.group_rates["0"] == 1.0
        assert result.ppv.group_rates["1"] == 1.0

    def test_known_ppv_gap(self):
        y_true, y_pred, groups = _two_group_odds_split()
        result = compute_predictive_parity(y_true, y_pred, groups)
        assert abs(result.ppv.group_rates["0"] - 1.0) < 1e-12
        assert abs(result.ppv.group_rates["1"] - 0.5) < 1e-12
        assert abs(result.ppv_difference - 0.5) < 1e-12
        assert result.ppv.group_support["0"] == 2
        assert result.ppv.group_support["1"] == 2
        gap = result.ppv.pairwise_gaps[0]
        assert abs(gap.abs_gap - 0.5) < 1e-12

    def test_ppv_undefined_defaults_to_zero(self):
        # Group 1 never predicts positive -> PPV denominator 0 -> 0.0
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 0])
        groups = np.array([0, 0, 1, 1])
        result = compute_predictive_parity(y_true, y_pred, groups)
        assert result.ppv.group_rates["1"] == 0.0
        assert result.ppv.group_support["1"] == 0
        assert abs(result.ppv.group_rates["0"] - 1.0) < 1e-12

    def test_scores_only(self):
        y_true, y_pred, groups = _two_group_odds_split()
        y_scores = np.where(y_pred == 1, 0.8, 0.2).astype(float)
        from_scores = compute_predictive_parity(
            y_true, None, groups, y_scores=y_scores, threshold=0.5
        )
        from_pred = compute_predictive_parity(y_true, y_pred, groups)
        assert abs(from_scores.ppv_difference - from_pred.ppv_difference) < 1e-12
        assert from_scores.threshold == 0.5

    def test_to_dict(self):
        y_true, y_pred, groups = _two_group_odds_split()
        result = compute_predictive_parity(y_true, y_pred, groups)
        d = result.to_dict()
        assert set(d.keys()) == {
            "n_groups",
            "n_samples",
            "group_size",
            "ppv",
            "ppv_difference",
            "confusion_matrices",
            "threshold",
        }
        assert d["ppv"]["rate_name"] == "ppv"
        assert isinstance(result, PredictiveParityResult)


class TestOddsParityCombined:
    def test_combined_matches_individual(self):
        y_true, y_pred, groups = _two_group_odds_split()
        combined = compute_odds_parity_metrics(y_true, y_pred, groups)
        eo = compute_equalized_odds(y_true, y_pred, groups)
        pp = compute_predictive_parity(y_true, y_pred, groups)
        assert abs(combined.equalized_odds_difference - eo.equalized_odds_difference) < 1e-12
        assert abs(combined.ppv_difference - pp.ppv_difference) < 1e-12
        assert combined.tpr_difference == eo.tpr_difference
        assert combined.fpr_difference == eo.fpr_difference
        assert isinstance(combined, OddsParityResult)
        assert isinstance(combined.equalized_odds, EqualizedOddsResult)
        assert isinstance(combined.predictive_parity, PredictiveParityResult)

    def test_to_dict(self):
        y_true, y_pred, groups = _two_group_odds_split()
        d = compute_odds_parity_metrics(y_true, y_pred, groups).to_dict()
        assert set(d.keys()) == {
            "n_groups",
            "n_samples",
            "group_size",
            "equalized_odds",
            "predictive_parity",
            "tpr_difference",
            "fpr_difference",
            "equalized_odds_difference",
            "ppv_difference",
            "confusion_matrices",
            "threshold",
        }
        assert "tpr" in d["equalized_odds"]
        assert "ppv" in d["predictive_parity"]


class TestValidation:
    def test_requires_pred_or_scores(self):
        with pytest.raises(ValueError, match="y_pred or y_scores"):
            compute_equalized_odds(
                np.array([1, 0, 1, 0]),
                None,
                np.array([0, 0, 1, 1]),
            )

    def test_rejects_single_group(self):
        with pytest.raises(ValueError, match="At least two groups"):
            compute_equalized_odds(
                np.array([1, 0, 1, 0]),
                np.array([1, 0, 0, 1]),
                np.array([0, 0, 0, 0]),
            )

    def test_rejects_non_binary_labels(self):
        with pytest.raises(ValueError, match="binary"):
            compute_equalized_odds(
                np.array([1, 2, 0, 1]),
                np.array([1, 0, 0, 1]),
                np.array([0, 0, 1, 1]),
            )

    def test_rejects_non_binary_pred(self):
        with pytest.raises(ValueError, match="binary"):
            compute_predictive_parity(
                np.array([1, 0, 1, 0]),
                np.array([1, 0, 2, 1]),
                np.array([0, 0, 1, 1]),
            )

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            compute_equalized_odds(np.array([]), np.array([]), np.array([]))

    def test_rejects_length_mismatch_pred(self):
        with pytest.raises(ValueError, match="same length"):
            compute_equalized_odds(
                np.array([1, 0, 1]),
                np.array([1, 0]),
                np.array([0, 1, 0]),
            )

    def test_rejects_length_mismatch_scores(self):
        with pytest.raises(ValueError, match="same length"):
            compute_equalized_odds(
                np.array([1, 0, 1]),
                None,
                np.array([0, 1, 0]),
                y_scores=np.array([0.9, 0.1]),
            )

    def test_rejects_length_mismatch_groups(self):
        with pytest.raises(ValueError, match="same length"):
            compute_equalized_odds(
                np.array([1, 0, 1, 0]),
                np.array([1, 0, 1, 0]),
                np.array([0, 1]),
            )

    def test_rejects_non_finite_scores(self):
        with pytest.raises(ValueError, match="finite"):
            compute_equalized_odds(
                np.array([1, 0, 1, 0]),
                None,
                np.array([0, 0, 1, 1]),
                y_scores=np.array([0.9, np.nan, 0.2, 0.1]),
            )


class TestOddsParityReport:
    def test_includes_rates_and_pairwise_tables(self):
        y_true, y_pred, groups = _two_group_odds_split()
        result = compute_odds_parity_metrics(y_true, y_pred, groups)
        report = render_odds_parity_report(result)
        assert "# Equalized Odds / Predictive Parity Report" in report
        assert "TPR Difference" in report
        assert "FPR Difference" in report
        assert "Equalized Odds Difference" in report
        assert "PPV Difference" in report
        assert "Rates by Group" in report
        assert "Pairwise Gaps" in report
        assert "True Positive Rate" in report
        assert "False Positive Rate" in report
        assert "Positive Predictive Value" in report
        assert "| 0 |" in report and "| 1 |" in report
        assert "0.5000" in report
        assert "Hardt" in report
        assert "Chouldechova" in report


class TestEvaluateCLIOddsParity:
    def test_evaluate_writes_odds_parity_section(self, tmp_path):
        y_true, y_pred, groups = _two_group_odds_split()
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
        assert "Equalized Odds / Predictive Parity Report" in text
        assert "TPR Difference" in text
        assert "PPV Difference" in text
        assert "Pairwise Gaps" in text
        assert "Theil / Generalized Entropy Report" in text
        assert "Demographic Parity Difference" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
