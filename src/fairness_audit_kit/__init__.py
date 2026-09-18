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

from fairness_audit_kit.intersectional import (
    IntersectionalFairnessResult,
    WorstIntersection,
    make_intersectional_groups,
    compute_intersectional_metrics,
)

from fairness_audit_kit.optimizer import (
    OptimizationResult,
    optimize_thresholds,
    find_threshold_for_metric,
)

from fairness_audit_kit.generator import (
    BiasedDataset,
    generate_biased_dataset,
    generate_multiple_biased_datasets,
)

from fairness_audit_kit.report import (
    render_metrics_report,
    render_intersectional_report,
    render_optimization_report,
    render_comparison_report,
)

__all__ = [
    "FairnessMetrics",
    "compute_fairness_metrics",
    "confusion_matrices_by_group",
    "IntersectionalFairnessResult",
    "WorstIntersection",
    "make_intersectional_groups",
    "compute_intersectional_metrics",
    "OptimizationResult",
    "optimize_thresholds",
    "find_threshold_for_metric",
    "BiasedDataset",
    "generate_biased_dataset",
    "generate_multiple_biased_datasets",
    "render_metrics_report",
    "render_intersectional_report",
    "render_optimization_report",
    "render_comparison_report",
]
