"""
Demographic parity (statistical parity difference) and disparate impact.

Demographic parity (Dwork et al., ITCS 2012), also called statistical
parity, requires the positive prediction rate to be equal across groups
defined by a sensitive attribute. It does not use ground-truth labels.

``compute_fairness_metrics`` already reports scalar max-minus-min
demographic parity and a min/max disparate impact ratio, but only as
part of a label-based audit. This module accepts binary predictions (or
scores) and a sensitive attribute, and returns the per-group positive
rates, the statistical parity difference, and the disparate impact ratio.

Statistical parity difference (SPD) is max positive rate minus min
positive rate, matching ``FairnessMetrics.demographic_parity_difference``.

Disparate impact ratio (Feldman et al., KDD 2015) is min positive rate
divided by max positive rate, matching
``FairnessMetrics.disparate_impact_ratio``. 1.0 means the rates match.
The four-fifths rule treats a ratio below 0.8 as evidence of adverse
impact.
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np

from fairness_audit_kit.metrics import _safe_divide
from fairness_audit_kit.odds_parity import (
    GroupRateTable,
    _as_binary_labels,
    _rate_table,
)


@dataclass
class DemographicParityResult:
    """Per-group positive rates, statistical parity difference, and disparate impact."""
    n_groups: int
    n_samples: int
    group_size: Dict[str, int]
    positive_count: Dict[str, int]
    positive_rates: Dict[str, float]
    positive_rate: GroupRateTable
    statistical_parity_difference: float
    demographic_parity_difference: float
    disparate_impact_ratio: float
    threshold: Optional[float]

    def to_dict(self) -> Dict:
        return {
            "n_groups": self.n_groups,
            "n_samples": self.n_samples,
            "group_size": self.group_size,
            "positive_count": self.positive_count,
            "positive_rates": self.positive_rates,
            "positive_rate": self.positive_rate.to_dict(),
            "statistical_parity_difference": self.statistical_parity_difference,
            "demographic_parity_difference": self.demographic_parity_difference,
            "disparate_impact_ratio": self.disparate_impact_ratio,
            "threshold": self.threshold,
        }


def _resolve_predictions(
    y_pred: Optional[np.ndarray],
    y_scores: Optional[np.ndarray],
    threshold: float,
) -> Tuple[np.ndarray, Optional[float]]:
    """
    Return binary predictions.

    Hard labels win when both ``y_pred`` and ``y_scores`` are given.
    If only scores are provided, they are thresholded (default 0.5).
    """
    used_threshold: Optional[float] = None

    if y_pred is None:
        if y_scores is None:
            raise ValueError("Provide y_pred or y_scores")
        y_scores = np.asarray(y_scores, dtype=float).ravel()
        if y_scores.size == 0:
            raise ValueError("y_scores must be non-empty")
        if np.any(~np.isfinite(y_scores)):
            raise ValueError("y_scores must be finite")
        y_pred_arr = (y_scores >= threshold).astype(int)
        used_threshold = float(threshold)
        return y_pred_arr, used_threshold

    return _as_binary_labels("y_pred", y_pred), used_threshold


def _selection_rates(
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, int]]:
    """Positive prediction rate, group size, and positive count per group."""
    groups = np.asarray(groups).ravel()
    if groups.size != y_pred.shape[0]:
        raise ValueError("y_pred and groups must have the same length")

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("At least two groups required for fairness computation")

    positive_rates: Dict[str, float] = {}
    group_size: Dict[str, int] = {}
    positive_count: Dict[str, int] = {}
    for g in unique_groups:
        mask = groups == g
        n = int(np.sum(mask))
        if n == 0:
            raise ValueError("each group must contain at least one sample")
        n_pos = int(np.sum(y_pred[mask]))
        key = str(g)
        positive_count[key] = n_pos
        group_size[key] = n
        positive_rates[key] = float(n_pos) / float(n)
    return positive_rates, group_size, positive_count


def compute_demographic_parity(
    y_pred: Optional[np.ndarray],
    groups: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> DemographicParityResult:
    """
    Compute demographic parity and disparate impact from predictions.

    Accepts binary ``y_pred`` and a sensitive attribute (``groups``).
    Ground-truth labels are not required. Pass ``y_pred=None`` and
    ``y_scores`` to threshold scores instead (default cutoff 0.5).

    Per-group positive rate is ``P(y_pred = 1 | group)``.
    Statistical parity difference is max rate minus min rate (0 = parity).
    ``demographic_parity_difference`` is the same value, matching
    ``FairnessMetrics``. Disparate impact ratio is min rate / max rate
    (1.0 = no disparate impact). When every group has a zero positive
    rate, the ratio is defined as 1.0.

    Args:
        y_pred: Binary predictions (0 or 1). Pass ``None`` if ``y_scores`` is given
        groups: Sensitive-attribute membership
        y_scores: Optional scores; thresholded when ``y_pred`` is omitted
        threshold: Cutoff applied to ``y_scores`` (default 0.5)

    Returns:
        DemographicParityResult with per-group positive rates, SPD, and DI ratio
    """
    y_pred_arr, used_threshold = _resolve_predictions(y_pred, y_scores, threshold)
    if y_scores is not None and y_pred is None:
        if np.asarray(groups).ravel().shape[0] != y_pred_arr.shape[0]:
            raise ValueError("y_scores and groups must have the same length")

    positive_rates, group_size, positive_count = _selection_rates(y_pred_arr, groups)
    rate_table = _rate_table(
        "positive_rate",
        positive_rates,
        group_size,
        dict(group_size),
    )
    rates = list(positive_rates.values())
    spd = float(max(rates) - min(rates)) if rates else 0.0
    disparate_impact = float(_safe_divide(min(rates), max(rates), default=1.0))

    return DemographicParityResult(
        n_groups=len(group_size),
        n_samples=int(y_pred_arr.shape[0]),
        group_size=group_size,
        positive_count=positive_count,
        positive_rates=positive_rates,
        positive_rate=rate_table,
        statistical_parity_difference=spd,
        demographic_parity_difference=spd,
        disparate_impact_ratio=disparate_impact,
        threshold=used_threshold,
    )


__all__ = [
    "DemographicParityResult",
    "compute_demographic_parity",
]
