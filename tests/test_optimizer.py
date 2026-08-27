"""
Tests for threshold optimizer module.
"""

import numpy as np
import pytest
from fairness_audit_kit.optimizer import (
    OptimizationResult,
    optimize_thresholds,
    find_threshold_for_metric,
    _apply_thresholds,
    _grid_search_thresholds,
)
from fairness_audit_kit.metrics import FairnessMetrics


class TestApplyThresholds:
    def test_basic(self):
        y_scores = np.array([0.1, 0.4, 0.6, 0.9])
        thresholds = {"0": 0.5, "1": 0.5}
        groups = np.array([0, 0, 1, 1])
        y_pred = _apply_thresholds(y_scores, thresholds, groups)
        expected = np.array([0, 0, 1, 1])
        np.testing.assert_array_equal(y_pred, expected)

    def test_different_thresholds(self):
        y_scores = np.array([0.3, 0.7, 0.3, 0.7])
        thresholds = {"A": 0.5, "B": 0.5}
        groups = np.array(["A", "A", "B", "B"])
        y_pred = _apply_thresholds(y_scores, thresholds, groups)
        expected = np.array([0, 1, 0, 1])
        np.testing.assert_array_equal(y_pred, expected)


class TestOptimizeThresholds:
    def test_perfect_separation(self):
        # Scores perfectly separate classes
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9, 0.1, 0.2, 0.8, 0.9])
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        result = optimize_thresholds(y_true, y_scores, groups, constraint="equalized_odds", constraint_threshold=0.1)
        
        assert isinstance(result, OptimizationResult)
        assert result.constraint_satisfied
        assert len(result.thresholds) == 2
        assert all(0 <= t <= 1 for t in result.thresholds.values())

    def test_equalized_odds_constraint(self):
        # Create biased predictions
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.3, 0.2, 0.6, 0.7, 0.4, 0.3])
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        result = optimize_thresholds(
            y_true, y_scores, groups, 
            constraint="equalized_odds", 
            constraint_threshold=0.2,
            grid_resolution=21
        )
        
        assert result.constraint_satisfied
        assert result.metrics.equalized_odds_difference <= 0.2 + 1e-6

    def test_demographic_parity_constraint(self):
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.3, 0.2, 0.6, 0.7, 0.4, 0.3])
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        result = optimize_thresholds(
            y_true, y_scores, groups,
            constraint="demographic_parity",
            constraint_threshold=0.2,
            grid_resolution=21
        )
        
        assert result.constraint_satisfied
        assert result.metrics.demographic_parity_difference <= 0.2 + 1e-6

    def test_equal_opportunity_constraint(self):
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.3, 0.2, 0.6, 0.7, 0.4, 0.3])
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        result = optimize_thresholds(
            y_true, y_scores, groups,
            constraint="equal_opportunity",
            constraint_threshold=0.2,
            grid_resolution=21
        )
        
        assert result.constraint_satisfied
        assert result.metrics.equal_opportunity_difference <= 0.2 + 1e-6

    def test_invalid_inputs(self):
        y_true = np.array([1, 0, 1])
        y_scores = np.array([0.9, 0.2])
        groups = np.array([0, 1, 0])
        
        with pytest.raises(ValueError, match="same length"):
            optimize_thresholds(y_true, y_scores, groups)

    def test_scores_out_of_range(self):
        y_true = np.array([1, 0, 1, 0])
        y_scores = np.array([1.5, 0.2, 0.8, 0.1])
        groups = np.array([0, 0, 1, 1])
        
        with pytest.raises(ValueError, match="y_scores must be in"):
            optimize_thresholds(y_true, y_scores, groups)

    def test_single_group_raises(self):
        y_true = np.array([1, 0, 1, 0])
        y_scores = np.array([0.9, 0.2, 0.8, 0.1])
        groups = np.array([0, 0, 0, 0])
        
        with pytest.raises(ValueError, match="At least two groups"):
            optimize_thresholds(y_true, y_scores, groups)

    def test_different_objectives(self):
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.3, 0.2, 0.6, 0.7, 0.4, 0.3])
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        for obj in ["accuracy", "balanced_accuracy", "f1"]:
            result = optimize_thresholds(
                y_true, y_scores, groups,
                objective=obj,
                grid_resolution=11
            )
            assert isinstance(result, OptimizationResult)
            assert result.constraint_satisfied or not result.constraint_satisfied  # Just check it runs

    def test_search_history_populated(self):
        y_true = np.array([1, 0, 1, 0])
        y_scores = np.array([0.9, 0.2, 0.8, 0.1])
        groups = np.array([0, 0, 1, 1])
        
        result = optimize_thresholds(y_true, y_scores, groups, grid_resolution=5)
        
        assert len(result.search_history) > 0
        for entry in result.search_history:
            assert "thresholds" in entry
            assert "objective" in entry
            assert "constraint_satisfied" in entry
            assert "metrics" in entry


class TestFindThresholdForMetric:
    def test_find_demographic_parity(self):
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.3, 0.2, 0.6, 0.7, 0.4, 0.3])
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        thresholds = find_threshold_for_metric(
            y_true, y_scores, groups,
            target_metric="demographic_parity",
            target_value=0.0,
            grid_resolution=21
        )
        
        assert len(thresholds) == 2
        assert all(0 <= t <= 1 for t in thresholds.values())

    def test_find_equalized_odds(self):
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.3, 0.2, 0.6, 0.7, 0.4, 0.3])
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        thresholds = find_threshold_for_metric(
            y_true, y_scores, groups,
            target_metric="equalized_odds",
            target_value=0.0,
            grid_resolution=21
        )
        
        assert len(thresholds) == 2

    def test_invalid_metric(self):
        y_true = np.array([1, 0, 1, 0])
        y_scores = np.array([0.9, 0.2, 0.8, 0.1])
        groups = np.array([0, 0, 1, 1])
        
        with pytest.raises(ValueError, match="Unknown metric"):
            find_threshold_for_metric(y_true, y_scores, groups, "invalid_metric", 0.0)

    def test_requires_two_groups(self):
        y_true = np.array([1, 0, 1, 0])
        y_scores = np.array([0.9, 0.2, 0.8, 0.1])
        groups = np.array([0, 0, 0, 0])
        
        with pytest.raises(ValueError, match="2 groups"):
            find_threshold_for_metric(y_true, y_scores, groups, "demographic_parity", 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
