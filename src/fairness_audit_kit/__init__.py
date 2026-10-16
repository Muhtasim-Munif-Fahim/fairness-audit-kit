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

from fairness_audit_kit.calibration import (
    ReliabilityBin,
    CalibrationResult,
    compute_reliability_bins,
    expected_calibration_error,
    maximum_calibration_error,
    compute_calibration_metrics,
)

from fairness_audit_kit.odds_parity import (
    PairwiseGap,
    GroupRateTable,
    EqualizedOddsResult,
    PredictiveParityResult,
    OddsParityResult,
    compute_equalized_odds,
    compute_predictive_parity,
    compute_odds_parity_metrics,
)

from fairness_audit_kit.demographic_parity import (
    DemographicParityResult,
    compute_demographic_parity,
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
    render_theil_report,
    render_calibration_report,
    render_odds_parity_report,
    render_demographic_parity_report,
    render_counterfactual_report,
    render_optimization_report,
    render_comparison_report,
    render_abroca_report,
    render_reweighing_report,
    render_treatment_equality_report,
    render_equalized_odds_postprocessing_report,
)

from fairness_audit_kit.reweighing import (
    ReweighingResult,
    compute_reweighing_weights,
    fit_with_reweighing,
)

from fairness_audit_kit.treatment_equality import (
    TreatmentEqualityResult,
    compute_treatment_equality,
)

from fairness_audit_kit.postprocessing import (
    EqualizedOddsPostprocessingResult,
    EqualizedOddsPostprocessor,
    GroupRates,
    fit_equalized_odds_postprocessing,
)

__all__ = [
    "EqualizedOddsPostprocessingResult",
    "EqualizedOddsPostprocessor",
    "GroupRates",
    "fit_equalized_odds_postprocessing",
    "render_equalized_odds_postprocessing_report",
    "render_counterfactual_report",
    "score_with_sensitive_flip",
    "remap_sensitive_column",
    "flip_sensitive_attribute",
    "compute_counterfactual_fairness_proxy",
    "CounterfactualFairnessResult",
    "FairnessMetrics",
    "compute_fairness_metrics",
    "confusion_matrices_by_group",
    "IntersectionalFairnessResult",
    "WorstIntersection",
    "make_intersectional_groups",
    "compute_intersectional_metrics",
    "EntropyBreakdown",
    "GeneralizedEntropyResult",
    "classification_benefit",
    "classification_error",
    "generalized_entropy_index",
    "theil_index",
    "compute_generalized_entropy",
    "compute_theil_metrics",
    "ReliabilityBin",
    "CalibrationResult",
    "compute_reliability_bins",
    "expected_calibration_error",
    "maximum_calibration_error",
    "compute_calibration_metrics",
    "PairwiseGap",
    "GroupRateTable",
    "EqualizedOddsResult",
    "PredictiveParityResult",
    "OddsParityResult",
    "compute_equalized_odds",
    "compute_predictive_parity",
    "compute_odds_parity_metrics",
    "DemographicParityResult",
    "compute_demographic_parity",
    "OptimizationResult",
    "optimize_thresholds",
    "find_threshold_for_metric",
    "BiasedDataset",
    "generate_biased_dataset",
    "generate_multiple_biased_datasets",
    "render_metrics_report",
    "render_intersectional_report",
    "render_theil_report",
    "render_calibration_report",
    "render_odds_parity_report",
    "render_demographic_parity_report",
    "render_optimization_report",
    "render_comparison_report",
    "GroupROC",
    "AbrocaPair",
    "AbrocaResult",
    "compute_abroca",
    "render_abroca_report",
    "ReweighingResult",
    "compute_reweighing_weights",
    "fit_with_reweighing",
    "render_reweighing_report",
    "TreatmentEqualityResult",
    "compute_treatment_equality",
    "render_treatment_equality_report",
]

from fairness_audit_kit.counterfactual import (
    CounterfactualFairnessResult,
    compute_counterfactual_fairness_proxy,
    flip_sensitive_attribute,
    remap_sensitive_column,
    score_with_sensitive_flip,
)

from fairness_audit_kit.abroca import (
    GroupROC,
    AbrocaPair,
    AbrocaResult,
    compute_abroca,
)
