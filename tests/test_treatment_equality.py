"""Tests for treatment equality and accuracy equality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fairness_audit_kit.cli import cmd_treatment
from fairness_audit_kit.report import render_treatment_equality_report
from fairness_audit_kit.treatment_equality import compute_treatment_equality


def _known_gap():
    """
    Group 0: FN=2, FP=1 -> ratio 2.0; accuracy 5/8=0.625
      y_true: 1,1,1,1, 0,0,0,0
      y_pred: 1,1,0,0, 1,0,0,0  -> TP2 FN2 FP1 TN3
    Group 1: FN=1, FP=2 -> ratio 0.5; accuracy 5/8=0.625
      y_true: 1,1,1,1, 0,0,0,0
      y_pred: 1,1,1,0, 1,1,0,0  -> TP3 FN1 FP2 TN2
    TE diff = 2.0 - 0.5 = 1.5; accuracy gap = 0
    """
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0])
    groups = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    return y_true, y_pred, groups


def test_known_treatment_and_accuracy_gaps():
    y_true, y_pred, groups = _known_gap()
    result = compute_treatment_equality(y_true, groups, y_pred=y_pred)
    assert result.fn_fp_ratios["0"] == pytest.approx(2.0)
    assert result.fn_fp_ratios["1"] == pytest.approx(0.5)
    assert result.treatment_equality_difference == pytest.approx(1.5)
    assert result.accuracies["0"] == pytest.approx(0.625)
    assert result.accuracies["1"] == pytest.approx(0.625)
    assert result.accuracy_equality_difference == pytest.approx(0.0)
    assert result.infinite_ratio_groups == ()


def test_perfect_equality():
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    result = compute_treatment_equality(y_true, groups, y_pred=y_pred)
    assert result.treatment_equality_difference == 0.0
    assert result.accuracy_equality_difference == 0.0
    assert result.fn_fp_ratios["0"] == 0.0
    assert result.fn_fp_ratios["1"] == 0.0


def test_infinite_ratio_excluded_from_gap():
    # Group 0: FP=0, FN=2 -> inf; Group 1: FN=2, FP=2 -> 1.0
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([0, 0, 0, 0, 0, 0, 1, 1])
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    result = compute_treatment_equality(y_true, groups, y_pred=y_pred)
    assert "0" in result.infinite_ratio_groups
    assert result.fn_fp_ratios["0"] == float("inf")
    assert result.fn_fp_ratios["1"] == pytest.approx(1.0)
    # Only one finite ratio -> difference 0
    assert result.treatment_equality_difference == 0.0


def test_score_thresholding():
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1, 0.9, 0.8, 0.2, 0.1])
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    result = compute_treatment_equality(
        y_true, groups, y_scores=scores, threshold=0.5
    )
    assert result.threshold == 0.5
    assert result.treatment_equality_difference == 0.0


def test_report_contains_metrics():
    y_true, y_pred, groups = _known_gap()
    result = compute_treatment_equality(y_true, groups, y_pred=y_pred)
    md = render_treatment_equality_report(result)
    assert "Treatment equality difference" in md
    assert "1.5000" in md
    assert "FN/FP" in md


def test_cli_treatment(tmp_path):
    y_true, y_pred, groups = _known_gap()
    df = pd.DataFrame({"target": y_true, "y_pred": y_pred, "sensitive_attr": groups})
    data = tmp_path / "preds.csv"
    out = tmp_path / "report.md"
    df.to_csv(data, index=False)

    class Args:
        pass

    args = Args()
    args.data = str(data)
    args.target_col = "target"
    args.pred_col = "y_pred"
    args.score_col = "y_score"
    args.group_col = "sensitive_attr"
    args.threshold = 0.5
    args.title = "TE Report"
    args.output = str(out)
    cmd_treatment(args)
    text = out.read_text()
    assert "Treatment Equality" in text or "TE Report" in text
