"""Tests for the counterfactual fairness proxy (sensitive-attribute flip)."""

from __future__ import annotations

import numpy as np
import pytest

from fairness_audit_kit.counterfactual import (
    compute_counterfactual_fairness_proxy,
    flip_sensitive_attribute,
    remap_sensitive_column,
    score_with_sensitive_flip,
)
from fairness_audit_kit.report import render_counterfactual_report


def test_flip_binary_swaps_levels():
    groups = np.array(["a", "b", "a", "b"])
    flipped = flip_sensitive_attribute(groups)
    assert flipped.tolist() == ["b", "a", "b", "a"]


def test_flip_requires_mapping_for_multilevel():
    with pytest.raises(ValueError, match="exactly two"):
        flip_sensitive_attribute(["a", "b", "c"])


def test_remap_sensitive_column():
    X = np.array([[0.1, 0], [0.2, 1], [0.3, 0]], dtype=float)
    out = remap_sensitive_column(X, 1)
    assert out[:, 0].tolist() == [0.1, 0.2, 0.3]
    assert out[:, 1].tolist() == [1.0, 0.0, 1.0]


def test_identical_scores_are_perfectly_fair():
    scores = np.array([0.1, 0.4, 0.7, 0.9])
    groups = np.array([0, 0, 1, 1])
    result = compute_counterfactual_fairness_proxy(scores, scores, groups)
    assert result.mean_abs_delta == 0.0
    assert result.max_abs_delta == 0.0
    assert result.decision_flip_rate == 0.0
    assert result.group_mean_abs_delta["0"] == 0.0
    assert result.group_mean_abs_delta["1"] == 0.0


def test_known_deltas_and_decision_flips():
    factual = np.array([0.2, 0.6, 0.4, 0.8])
    counterfactual = np.array([0.7, 0.6, 0.4, 0.3])  # deltas 0.5, 0, 0, 0.5
    groups = np.array(["g0", "g0", "g1", "g1"])
    result = compute_counterfactual_fairness_proxy(
        factual, counterfactual, groups, threshold=0.5
    )
    assert result.mean_abs_delta == pytest.approx(0.25)
    assert result.max_abs_delta == pytest.approx(0.5)
    assert result.median_abs_delta == pytest.approx(0.25)
    # flips at indices 0 (0.2->0.7) and 3 (0.8->0.3)
    assert result.decision_flip_rate == pytest.approx(0.5)
    assert result.group_mean_abs_delta["g0"] == pytest.approx(0.25)
    assert result.group_mean_abs_delta["g1"] == pytest.approx(0.25)
    assert result.group_decision_flip_rate["g0"] == pytest.approx(0.5)
    assert result.group_decision_flip_rate["g1"] == pytest.approx(0.5)


def test_threshold_none_skips_decision_flips():
    result = compute_counterfactual_fairness_proxy(
        [0.2, 0.8], [0.9, 0.1], [0, 1], threshold=None
    )
    assert result.decision_flip_rate == 0.0
    assert result.threshold is None


def test_rejects_length_mismatch_and_nonfinite():
    with pytest.raises(ValueError, match="same length"):
        compute_counterfactual_fairness_proxy([0.1, 0.2], [0.1], [0, 1])
    with pytest.raises(ValueError, match="finite"):
        compute_counterfactual_fairness_proxy([0.1, np.nan], [0.1, 0.2], [0, 1])


def test_score_with_sensitive_flip_linear_model():
    # Score = 0.1 * x0 + 0.8 * sensitive. Flipping sensitive changes score by 0.8.
    X = np.array(
        [
            [1.0, 0.0],
            [2.0, 1.0],
            [0.5, 0.0],
            [1.5, 1.0],
        ]
    )

    def score_fn(mat):
        return 0.1 * mat[:, 0] + 0.8 * mat[:, 1]

    result = score_with_sensitive_flip(X, sensitive_index=1, score_fn=score_fn, threshold=0.5)
    assert result.mean_abs_delta == pytest.approx(0.8)
    assert result.max_abs_delta == pytest.approx(0.8)
    assert result.decision_flip_rate == 1.0


def test_score_fn_invariant_to_sensitive_is_fair():
    X = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0]])

    def score_fn(mat):
        return mat[:, 0]  # ignores sensitive column

    result = score_with_sensitive_flip(X, 1, score_fn)
    assert result.mean_abs_delta == 0.0
    assert result.decision_flip_rate == 0.0


def test_render_counterfactual_report_contains_table():
    result = compute_counterfactual_fairness_proxy(
        [0.2, 0.8], [0.3, 0.7], ["a", "b"]
    )
    md = render_counterfactual_report(result)
    assert "Counterfactual Fairness Proxy" in md
    assert "mean_abs_delta" in md
    assert "| a |" in md
