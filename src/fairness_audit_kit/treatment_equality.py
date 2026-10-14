"""
Treatment equality and accuracy equality gaps across sensitive groups.

Treatment equality (Berk et al., 2018) requires the ratio of false negatives
to false positives to be the same across groups:

    FN_g / FP_g  ≈  FN_h / FP_h

A large gap means one group bears relatively more false negatives (missed
positives) per false positive than another. Groups with zero false positives
and a positive FN count are reported with an infinite ratio and excluded from
the max−min gap so a single empty cell does not dominate the audit; they are
still listed in the per-group table.

Accuracy equality requires overall accuracy to match across groups. The
accuracy gap is max accuracy minus min accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from fairness_audit_kit.metrics import _confusion_matrix
from fairness_audit_kit.odds_parity import (
    GroupRateTable,
    PairwiseGap,
    _pairwise_gaps,
    _rate_table,
    _resolve_predictions,
)


@dataclass
class TreatmentEqualityResult:
    """Per-group FN/FP ratios, accuracy, and the corresponding gaps."""

    n_groups: int
    n_samples: int
    group_size: Dict[str, int]
    false_negatives: Dict[str, int]
    false_positives: Dict[str, int]
    fn_fp_ratios: Dict[str, float]
    accuracies: Dict[str, float]
    treatment_equality_difference: float
    accuracy_equality_difference: float
    fn_fp_ratio: GroupRateTable
    accuracy: GroupRateTable
    infinite_ratio_groups: Tuple[str, ...]
    threshold: Optional[float]

    def to_dict(self) -> Dict:
        return {
            "n_groups": self.n_groups,
            "n_samples": self.n_samples,
            "group_size": self.group_size,
            "false_negatives": self.false_negatives,
            "false_positives": self.false_positives,
            "fn_fp_ratios": self.fn_fp_ratios,
            "accuracies": self.accuracies,
            "treatment_equality_difference": self.treatment_equality_difference,
            "accuracy_equality_difference": self.accuracy_equality_difference,
            "fn_fp_ratio": self.fn_fp_ratio.to_dict(),
            "accuracy": self.accuracy.to_dict(),
            "infinite_ratio_groups": list(self.infinite_ratio_groups),
            "threshold": self.threshold,
        }


def _per_group_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> Tuple[
    Dict[str, int],
    Dict[str, int],
    Dict[str, int],
    Dict[str, float],
    Dict[str, float],
    List[str],
]:
    groups = np.asarray(groups).ravel()
    if groups.size != y_true.shape[0]:
        raise ValueError("y_true/y_pred and groups must have the same length")
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("At least two groups required for fairness computation")

    group_size: Dict[str, int] = {}
    fns: Dict[str, int] = {}
    fps: Dict[str, int] = {}
    ratios: Dict[str, float] = {}
    accuracies: Dict[str, float] = {}
    infinite: List[str] = []

    for g in unique:
        mask = groups == g
        n = int(np.sum(mask))
        if n == 0:
            raise ValueError("each group must contain at least one sample")
        cm = _confusion_matrix(y_true[mask], y_pred[mask])
        key = str(g)
        group_size[key] = n
        fns[key] = cm["fn"]
        fps[key] = cm["fp"]
        correct = cm["tp"] + cm["tn"]
        accuracies[key] = float(correct) / float(n)
        if cm["fp"] == 0:
            ratios[key] = float("inf") if cm["fn"] > 0 else 0.0
            if cm["fn"] > 0:
                infinite.append(key)
        else:
            ratios[key] = float(cm["fn"]) / float(cm["fp"])
    return group_size, fns, fps, ratios, accuracies, infinite


def _finite_rate_table(
    rate_name: str,
    ratios: Dict[str, float],
    group_size: Dict[str, int],
) -> Tuple[GroupRateTable, float]:
    """Build a rate table using only finite ratios for the gap."""
    finite = {k: v for k, v in ratios.items() if np.isfinite(v)}
    if len(finite) >= 2:
        difference = float(max(finite.values()) - min(finite.values()))
        gaps = _pairwise_gaps(finite)
        gaps.sort(key=lambda g: (-g.abs_gap, g.group_a, g.group_b))
    elif len(finite) == 1:
        difference = 0.0
        gaps = []
    else:
        difference = float("nan")
        gaps = []
    # Expose every group's ratio (including inf) in group_rates.
    support = {k: group_size[k] for k in ratios}
    table = GroupRateTable(
        rate_name=rate_name,
        group_rates=dict(ratios),
        group_size=dict(group_size),
        group_support=support,
        pairwise_gaps=gaps,
        difference=difference,
    )
    return table, difference


def compute_treatment_equality(
    y_true: np.ndarray,
    groups: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    *,
    y_scores: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> TreatmentEqualityResult:
    """Compute treatment equality (FN/FP) and accuracy equality gaps.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    groups:
        Sensitive-group membership.
    y_pred:
        Hard binary predictions. Optional when ``y_scores`` is provided.
    y_scores:
        Optional scores thresholded when ``y_pred`` is omitted.
    threshold:
        Decision threshold applied to ``y_scores`` (default 0.5).
    """
    y_true_arr, y_pred_arr, used_threshold = _resolve_predictions(
        y_true, y_pred, y_scores, threshold
    )
    group_size, fns, fps, ratios, accuracies, infinite = _per_group_counts(
        y_true_arr, y_pred_arr, groups
    )
    te_table, te_diff = _finite_rate_table("fn_fp_ratio", ratios, group_size)
    acc_table = _rate_table(
        "accuracy",
        accuracies,
        group_size,
        {k: group_size[k] for k in accuracies},
    )

    return TreatmentEqualityResult(
        n_groups=len(group_size),
        n_samples=int(y_true_arr.shape[0]),
        group_size=group_size,
        false_negatives=fns,
        false_positives=fps,
        fn_fp_ratios=ratios,
        accuracies=accuracies,
        treatment_equality_difference=te_diff,
        accuracy_equality_difference=acc_table.difference,
        fn_fp_ratio=te_table,
        accuracy=acc_table,
        infinite_ratio_groups=tuple(infinite),
        threshold=used_threshold,
    )


__all__ = [
    "TreatmentEqualityResult",
    "compute_treatment_equality",
]
