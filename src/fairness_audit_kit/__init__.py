"""
Fairness Audit Kit - Model fairness and bias evaluation toolkit.
"""

__version__ = "0.1.0"
__author__ = "Muhtasim Munif Fahim"

from fairness_audit_kit.metrics import (
    FairnessMetrics,
    compute_fairness_metrics,
    confusion_matrices_by_group,
)

from fairness_audit_kit.optimizer import (
    OptimizationResult,
    optimize_thresholds,
    find_threshold_for_metric,
)

__all__ = [
    "FairnessMetrics",
    "compute_fairness_metrics",
    "confusion_matrices_by_group",
    "OptimizationResult",
    "optimize_thresholds",
    "find_threshold_for_metric",
]
