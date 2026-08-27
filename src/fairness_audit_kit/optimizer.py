"""
Threshold optimization for fairness constraints.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from fairness_audit_kit.metrics import compute_fairness_metrics, FairnessMetrics


@dataclass
class OptimizationResult:
    """Result of threshold optimization."""
    thresholds: Dict[str, float]
    metrics: FairnessMetrics
    objective_value: float
    constraint_satisfied: bool
    search_history: List[Dict]


def _apply_thresholds(y_scores: np.ndarray, thresholds: Dict[str, float], groups: np.ndarray) -> np.ndarray:
    """Apply per-group thresholds to scores."""
    y_pred = np.zeros_like(y_scores, dtype=int)
    unique_groups = np.unique(groups)
    for g in unique_groups:
        mask = groups == g
        thresh = thresholds.get(str(g), 0.5)
        y_pred[mask] = (y_scores[mask] >= thresh).astype(int)
    return y_pred

def _grid_search_thresholds(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    groups: np.ndarray,
    grid: np.ndarray,
    objective_fn: Callable[[FairnessMetrics], float],
    constraint_fn: Optional[Callable[[FairnessMetrics], bool]] = None,
) -> Tuple[Dict[str, float], FairnessMetrics, float, bool, List[Dict]]:
    """Grid search over per-group thresholds."""
    unique_groups = sorted([str(g) for g in np.unique(groups)])
    n_groups = len(unique_groups)
    
    best_thresholds = {g: 0.5 for g in unique_groups}
    best_metrics = None
    best_objective = float("inf")
    best_constraint_satisfied = False
    history = []
    
    if n_groups == 2:
        g0, g1 = unique_groups[0], unique_groups[1]
        for t0 in grid:
            for t1 in grid:
                thresholds = {g0: t0, g1: t1}
                y_pred = _apply_thresholds(y_scores, thresholds, groups)
                metrics = compute_fairness_metrics(y_true, y_pred, groups)
                obj_val = objective_fn(metrics)
                constraint_ok = constraint_fn(metrics) if constraint_fn else True
                
                history.append({
                    "thresholds": thresholds.copy(),
                    "objective": obj_val,
                    "constraint_satisfied": constraint_ok,
                    "metrics": metrics.to_dict(),
                })
                
                if constraint_ok and obj_val < best_objective:
                    best_objective = obj_val
                    best_thresholds = thresholds.copy()
                    best_metrics = metrics
                    best_constraint_satisfied = True
                elif not best_constraint_satisfied and obj_val < best_objective:
                    best_objective = obj_val
                    best_thresholds = thresholds.copy()
                    best_metrics = metrics
                    best_constraint_satisfied = False
    
    else:
        thresholds = {g: 0.5 for g in unique_groups}
        for _ in range(10):
            improved = False
            for g in unique_groups:
                best_t = thresholds[g]
                best_obj = objective_fn(compute_fairness_metrics(y_true, _apply_thresholds(y_scores, thresholds, groups), groups))
                for t in grid:
                    thresholds[g] = t
                    y_pred = _apply_thresholds(y_scores, thresholds, groups)
                    metrics = compute_fairness_metrics(y_true, y_pred, groups)
                    obj_val = objective_fn(metrics)
                    constraint_ok = constraint_fn(metrics) if constraint_fn else True
                    
                    history.append({
                        "thresholds": thresholds.copy(),
                        "objective": obj_val,
                        "constraint_satisfied": constraint_ok,
                        "metrics": metrics.to_dict(),
                    })
                    
                    if constraint_ok and obj_val < best_obj:
                        best_obj = obj_val
                        best_t = t
                        improved = True
                thresholds[g] = best_t
            if not improved:
                break
        
        best_thresholds = thresholds
        best_metrics = compute_fairness_metrics(y_true, _apply_thresholds(y_scores, thresholds, groups), groups)
        best_objective = objective_fn(best_metrics)
        best_constraint_satisfied = constraint_fn(best_metrics) if constraint_fn else True
    
    return best_thresholds, best_metrics, best_objective, best_constraint_satisfied, history

def optimize_thresholds(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    groups: np.ndarray,
    constraint: str = "equalized_odds",
    constraint_threshold: float = 0.1,
    grid_resolution: int = 51,
    objective: str = "accuracy",
) -> OptimizationResult:
    """
    Optimize per-group decision thresholds under fairness constraints.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_scores: Predicted probabilities/scores in [0, 1]
        groups: Group membership array
        constraint: Fairness constraint type ("demographic_parity", "equal_opportunity", "equalized_odds")
        constraint_threshold: Maximum allowed fairness metric value
        grid_resolution: Number of threshold values to test per group
        objective: Objective to optimize ("accuracy", "balanced_accuracy", "f1")

    Returns:
        OptimizationResult with optimal thresholds and metrics
    """
    if len(y_true) != len(y_scores) or len(y_true) != len(groups):
        raise ValueError("y_true, y_scores, and groups must have same length")
    if not (np.all(y_scores >= 0) and np.all(y_scores <= 1)):
        raise ValueError("y_scores must be in [0, 1]")
    if len(np.unique(groups)) < 2:
        raise ValueError("At least two groups required")
    
    grid = np.linspace(0.0, 1.0, grid_resolution)
    
    def accuracy_obj(metrics: FairnessMetrics) -> float:
        return 1.0 - (metrics.demographic_parity_difference + metrics.equalized_odds_difference) / 2
    
    def balanced_accuracy_obj(metrics: FairnessMetrics) -> float:
        return 1.0 - metrics.equalized_odds_difference
    
    def f1_obj(metrics: FairnessMetrics) -> float:
        return metrics.equalized_odds_difference
    
    obj_fns = {
        "accuracy": accuracy_obj,
        "balanced_accuracy": balanced_accuracy_obj,
        "f1": f1_obj,
    }
    objective_fn = obj_fns.get(objective, accuracy_obj)
    
    constraint_map = {
        "demographic_parity": lambda m: m.demographic_parity_difference <= constraint_threshold,
        "equal_opportunity": lambda m: m.equal_opportunity_difference <= constraint_threshold,
        "equalized_odds": lambda m: m.equalized_odds_difference <= constraint_threshold,
    }
    constraint_fn = constraint_map.get(constraint)
    
    thresholds, metrics, obj_val, constraint_ok, history = _grid_search_thresholds(
        y_true, y_scores, groups, grid, objective_fn, constraint_fn
    )
    
    return OptimizationResult(
        thresholds=thresholds,
        metrics=metrics,
        objective_value=obj_val,
        constraint_satisfied=constraint_ok,
        search_history=history,
    )

def find_threshold_for_metric(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    groups: np.ndarray,
    target_metric: str,
    target_value: float,
    grid_resolution: int = 101,
) -> Dict[str, float]:
    """
    Find thresholds that achieve a target fairness metric value.
    
    Args:
        y_true: Ground truth labels
        y_scores: Predicted scores
        groups: Group membership
        target_metric: Metric to target ("demographic_parity", "equal_opportunity", "equalized_odds")
        target_value: Target value for the metric
        grid_resolution: Grid resolution for search
        
    Returns:
        Dictionary of per-group thresholds
    """
    unique_groups = sorted([str(g) for g in np.unique(groups)])
    n_groups = len(unique_groups)
    grid = np.linspace(0.0, 1.0, grid_resolution)
    
    if n_groups != 2:
        raise ValueError("Currently only supports 2 groups")
    
    g0, g1 = unique_groups[0], unique_groups[1]
    best_thresholds = {g0: 0.5, g1: 0.5}
    best_diff = float("inf")
    
    for t0 in grid:
        for t1 in grid:
            thresholds = {g0: t0, g1: t1}
            y_pred = _apply_thresholds(y_scores, thresholds, groups)
            metrics = compute_fairness_metrics(y_true, y_pred, groups)
            
            if target_metric == "demographic_parity":
                val = metrics.demographic_parity_difference
            elif target_metric == "equal_opportunity":
                val = metrics.equal_opportunity_difference
            elif target_metric == "equalized_odds":
                val = metrics.equalized_odds_difference
            else:
                raise ValueError(f"Unknown metric: {target_metric}")
            
            diff = abs(val - target_value)
            if diff < best_diff:
                best_diff = diff
                best_thresholds = thresholds.copy()
    
    return best_thresholds


__all__ = [
    "OptimizationResult",
    "optimize_thresholds",
    "find_threshold_for_metric",
]
