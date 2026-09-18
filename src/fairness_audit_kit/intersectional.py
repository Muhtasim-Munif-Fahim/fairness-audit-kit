"""
Intersectional subgroup fairness.

Evaluates existing group metrics on the Cartesian product of two
sensitive attributes, and identifies the worst-off intersection.
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass

from fairness_audit_kit.metrics import (
    FairnessMetrics,
    compute_fairness_metrics,
    _group_rates,
)


DEFAULT_SEPARATOR = "/"
WORST_BY_CHOICES = ("positive_rate", "tpr")


def make_intersectional_groups(
    groups_a: np.ndarray,
    groups_b: np.ndarray,
    separator: str = DEFAULT_SEPARATOR,
) -> np.ndarray:
    """
    Build intersection labels from two sensitive attributes.

    Labels are ``"{a}{separator}{b}"`` (default ``"0/1"``).
    """
    groups_a = np.asarray(groups_a)
    groups_b = np.asarray(groups_b)
    if groups_a.shape[0] != groups_b.shape[0]:
        raise ValueError("groups_a and groups_b must have the same length")
    return np.array(
        [f"{a}{separator}{b}" for a, b in zip(groups_a, groups_b)],
        dtype=object,
    )


@dataclass
class WorstIntersection:
    """Most disadvantaged intersectional subgroup."""
    group: str
    n_samples: int
    positive_rate: float
    tpr: float
    fpr: float
    criterion: str
    gap_from_best: float

    def to_dict(self) -> Dict:
        return {
            "group": self.group,
            "n_samples": self.n_samples,
            "positive_rate": self.positive_rate,
            "tpr": self.tpr,
            "fpr": self.fpr,
            "criterion": self.criterion,
            "gap_from_best": self.gap_from_best,
        }


@dataclass
class IntersectionalFairnessResult:
    """Fairness metrics computed over intersectional subgroups."""
    metrics: FairnessMetrics
    n_intersections: int
    intersection_labels: List[str]
    group_rates: Dict[str, Dict]
    worst_intersection: WorstIntersection

    def to_dict(self) -> Dict:
        return {
            "metrics": self.metrics.to_dict(),
            "n_intersections": self.n_intersections,
            "intersection_labels": self.intersection_labels,
            "group_rates": {
                g: {
                    "total": rates["total"],
                    "positive_rate": rates["positive_rate"],
                    "tpr": rates["tpr"],
                    "fpr": rates["fpr"],
                    "precision": rates["precision"],
                    "calibration": rates["calibration"],
                    "confusion_matrix": rates["confusion_matrix"],
                }
                for g, rates in self.group_rates.items()
            },
            "worst_intersection": self.worst_intersection.to_dict(),
        }


def _pick_worst_intersection(
    rates: Dict[str, Dict],
    worst_by: str,
) -> WorstIntersection:
    """Select the intersection with the lowest rate on ``worst_by``."""
    if worst_by not in WORST_BY_CHOICES:
        raise ValueError(
            f"worst_by must be one of {WORST_BY_CHOICES}, got {worst_by!r}"
        )

    min_val = min(r[worst_by] for r in rates.values())
    max_val = max(r[worst_by] for r in rates.values())
    candidates = [g for g, r in rates.items() if r[worst_by] == min_val]
    worst_group = sorted(candidates)[0]
    worst_rates = rates[worst_group]

    return WorstIntersection(
        group=worst_group,
        n_samples=int(worst_rates["total"]),
        positive_rate=float(worst_rates["positive_rate"]),
        tpr=float(worst_rates["tpr"]),
        fpr=float(worst_rates["fpr"]),
        criterion=worst_by,
        gap_from_best=float(max_val - min_val),
    )


def compute_intersectional_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups_a: np.ndarray,
    groups_b: np.ndarray,
    separator: str = DEFAULT_SEPARATOR,
    worst_by: str = "positive_rate",
) -> IntersectionalFairnessResult:
    """
    Compute group fairness metrics on the cross of two sensitive attributes.

    Reuses demographic parity, equal opportunity, equalized odds, and
    disparate impact from ``compute_fairness_metrics``, treating each
    ``(groups_a, groups_b)`` pair as its own subgroup.

    The worst intersection is the subgroup with the lowest value of
    ``worst_by`` (``"positive_rate"`` or ``"tpr"``). Ties break
    lexicographically.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        groups_a: First sensitive attribute
        groups_b: Second sensitive attribute
        separator: Label joiner for intersection names
        worst_by: Criterion used to pick the worst-off intersection

    Returns:
        IntersectionalFairnessResult with metrics, per-intersection rates,
        and the worst intersection summary
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups_a = np.asarray(groups_a)
    groups_b = np.asarray(groups_b)

    n = len(y_true)
    if not (len(y_pred) == n and len(groups_a) == n and len(groups_b) == n):
        raise ValueError("y_true, y_pred, groups_a, and groups_b must have the same length")

    intersections = make_intersectional_groups(groups_a, groups_b, separator=separator)
    unique = [str(g) for g in np.unique(intersections)]
    if len(unique) < 2:
        raise ValueError("At least two intersectional groups required for fairness computation")

    metrics = compute_fairness_metrics(y_true, y_pred, intersections)
    rates = _group_rates(y_true, y_pred, intersections)
    worst = _pick_worst_intersection(rates, worst_by)

    return IntersectionalFairnessResult(
        metrics=metrics,
        n_intersections=len(unique),
        intersection_labels=sorted(unique),
        group_rates=rates,
        worst_intersection=worst,
    )


__all__ = [
    "WorstIntersection",
    "IntersectionalFairnessResult",
    "make_intersectional_groups",
    "compute_intersectional_metrics",
]
