"""
Equalized odds (TPR/FPR gaps) and predictive parity (PPV gaps).

Equalized odds (Hardt, Price, and Srebro, NIPS 2016) requires the true
positive rate and false positive rate to be equal across groups.
Predictive parity (Chouldechova, 2017) requires positive predictive
value (precision) to be equal across groups.

``compute_fairness_metrics`` already reports scalar max-minus-min
summaries for TPR (equal opportunity) and equalized odds. This module
returns the per-group rates and every pairwise gap, and adds PPV which
was not previously exposed.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

from fairness_audit_kit.metrics import _group_rates


@dataclass
class PairwiseGap:
    """Gap between two groups on a single rate."""
    group_a: str
    group_b: str
    rate_a: float
    rate_b: float
    gap: float
    abs_gap: float

    def to_dict(self) -> Dict:
        return {
            "group_a": self.group_a,
            "group_b": self.group_b,
            "rate_a": self.rate_a,
            "rate_b": self.rate_b,
            "gap": self.gap,
            "abs_gap": self.abs_gap,
        }


@dataclass
class GroupRateTable:
    """Per-group rates and pairwise gaps for one classification rate."""
    rate_name: str
    group_rates: Dict[str, float]
    group_size: Dict[str, int]
    group_support: Dict[str, int]
    pairwise_gaps: List[PairwiseGap]
    difference: float

    def to_dict(self) -> Dict:
        return {
            "rate_name": self.rate_name,
            "group_rates": self.group_rates,
            "group_size": self.group_size,
            "group_support": self.group_support,
            "pairwise_gaps": [g.to_dict() for g in self.pairwise_gaps],
            "difference": self.difference,
        }


@dataclass
class EqualizedOddsResult:
    """Per-group TPR/FPR and pairwise equalized-odds gaps."""
    n_groups: int
    n_samples: int
    group_size: Dict[str, int]
    tpr: GroupRateTable
    fpr: GroupRateTable
    tpr_difference: float
    fpr_difference: float
    equalized_odds_difference: float
    confusion_matrices: Dict[str, Dict[str, int]]
    threshold: Optional[float]

    def to_dict(self) -> Dict:
        return {
            "n_groups": self.n_groups,
            "n_samples": self.n_samples,
            "group_size": self.group_size,
            "tpr": self.tpr.to_dict(),
            "fpr": self.fpr.to_dict(),
            "tpr_difference": self.tpr_difference,
            "fpr_difference": self.fpr_difference,
            "equalized_odds_difference": self.equalized_odds_difference,
            "confusion_matrices": self.confusion_matrices,
            "threshold": self.threshold,
        }


@dataclass
class PredictiveParityResult:
    """Per-group PPV (precision) and pairwise predictive-parity gaps."""
    n_groups: int
    n_samples: int
    group_size: Dict[str, int]
    ppv: GroupRateTable
    ppv_difference: float
    confusion_matrices: Dict[str, Dict[str, int]]
    threshold: Optional[float]

    def to_dict(self) -> Dict:
        return {
            "n_groups": self.n_groups,
            "n_samples": self.n_samples,
            "group_size": self.group_size,
            "ppv": self.ppv.to_dict(),
            "ppv_difference": self.ppv_difference,
            "confusion_matrices": self.confusion_matrices,
            "threshold": self.threshold,
        }


@dataclass
class OddsParityResult:
    """Equalized odds and predictive parity for the same predictions."""
    n_groups: int
    n_samples: int
    group_size: Dict[str, int]
    equalized_odds: EqualizedOddsResult
    predictive_parity: PredictiveParityResult
    tpr_difference: float
    fpr_difference: float
    equalized_odds_difference: float
    ppv_difference: float
    confusion_matrices: Dict[str, Dict[str, int]]
    threshold: Optional[float]

    def to_dict(self) -> Dict:
        return {
            "n_groups": self.n_groups,
            "n_samples": self.n_samples,
            "group_size": self.group_size,
            "equalized_odds": self.equalized_odds.to_dict(),
            "predictive_parity": self.predictive_parity.to_dict(),
            "tpr_difference": self.tpr_difference,
            "fpr_difference": self.fpr_difference,
            "equalized_odds_difference": self.equalized_odds_difference,
            "ppv_difference": self.ppv_difference,
            "confusion_matrices": self.confusion_matrices,
            "threshold": self.threshold,
        }


def _as_binary_labels(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    uniques = set(np.unique(arr).tolist())
    if not uniques.issubset({0, 1, 0.0, 1.0, False, True}):
        raise ValueError(f"{name} must be binary labels in {{0, 1}}")
    return arr.astype(int)


def _resolve_predictions(
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray],
    y_scores: Optional[np.ndarray],
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    """
    Return binary ``(y_true, y_pred)``.

    Hard labels win when both ``y_pred`` and ``y_scores`` are given.
    If only scores are provided, they are thresholded (default 0.5).
    """
    y_true = _as_binary_labels("y_true", y_true)
    used_threshold: Optional[float] = None

    if y_pred is None:
        if y_scores is None:
            raise ValueError("Provide y_pred or y_scores")
        y_scores = np.asarray(y_scores, dtype=float).ravel()
        if y_scores.size == 0:
            raise ValueError("y_scores must be non-empty")
        if y_true.shape[0] != y_scores.shape[0]:
            raise ValueError("y_true and y_scores must have the same length")
        if np.any(~np.isfinite(y_scores)):
            raise ValueError("y_scores must be finite")
        y_pred = (y_scores >= threshold).astype(int)
        used_threshold = float(threshold)
    else:
        y_pred = _as_binary_labels("y_pred", y_pred)
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("y_true and y_pred must have the same length")

    return y_true, y_pred, used_threshold


def _pairwise_gaps(group_rates: Dict[str, float]) -> List[PairwiseGap]:
    """All unordered pairs, labels sorted lexicographically."""
    keys = sorted(group_rates.keys(), key=str)
    gaps: List[PairwiseGap] = []
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            rate_a = float(group_rates[a])
            rate_b = float(group_rates[b])
            gap = rate_a - rate_b
            gaps.append(
                PairwiseGap(
                    group_a=a,
                    group_b=b,
                    rate_a=rate_a,
                    rate_b=rate_b,
                    gap=float(gap),
                    abs_gap=float(abs(gap)),
                )
            )
    return gaps


def _rate_table(
    rate_name: str,
    group_rates: Dict[str, float],
    group_size: Dict[str, int],
    group_support: Dict[str, int],
) -> GroupRateTable:
    values = list(group_rates.values())
    difference = float(max(values) - min(values)) if values else 0.0
    return GroupRateTable(
        rate_name=rate_name,
        group_rates=group_rates,
        group_size=group_size,
        group_support=group_support,
        pairwise_gaps=_pairwise_gaps(group_rates),
        difference=difference,
    )


def _group_tables(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]], GroupRateTable, GroupRateTable, GroupRateTable]:
    groups = np.asarray(groups)
    if len(groups) != len(y_true):
        raise ValueError("y_true, y_pred, and groups must have the same length")

    unique_groups = [str(g) for g in np.unique(groups)]
    if len(unique_groups) < 2:
        raise ValueError("At least two groups required for fairness computation")

    rates = _group_rates(y_true, y_pred, groups)
    group_size = {g: int(rates[g]["total"]) for g in rates}
    confusion_matrices = {g: rates[g]["confusion_matrix"] for g in rates}

    tpr = _rate_table(
        "tpr",
        {g: float(rates[g]["tpr"]) for g in rates},
        group_size,
        {g: int(rates[g]["confusion_matrix"]["tp"] + rates[g]["confusion_matrix"]["fn"]) for g in rates},
    )
    fpr = _rate_table(
        "fpr",
        {g: float(rates[g]["fpr"]) for g in rates},
        group_size,
        {g: int(rates[g]["confusion_matrix"]["fp"] + rates[g]["confusion_matrix"]["tn"]) for g in rates},
    )
    ppv = _rate_table(
        "ppv",
        {g: float(rates[g]["precision"]) for g in rates},
        group_size,
        {g: int(rates[g]["confusion_matrix"]["tp"] + rates[g]["confusion_matrix"]["fp"]) for g in rates},
    )
    return group_size, confusion_matrices, tpr, fpr, ppv


def compute_equalized_odds(
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray],
    groups: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> EqualizedOddsResult:
    """
    Compute equalized odds as per-group TPR/FPR and pairwise gaps.

    Equalized odds holds when TPR and FPR are the same for every group.
    ``equalized_odds_difference`` is ``max(TPR difference, FPR difference)``,
    matching ``FairnessMetrics.equalized_odds_difference``.
    ``tpr_difference`` is the equal-opportunity gap.

    Args:
        y_true: Binary ground-truth labels (0 or 1)
        y_pred: Binary predictions (0 or 1). Pass ``None`` if ``y_scores`` is given
        groups: Sensitive-attribute membership
        y_scores: Optional scores; thresholded when ``y_pred`` is omitted
        threshold: Cutoff applied to ``y_scores`` (default 0.5)

    Returns:
        EqualizedOddsResult with per-group rates and pairwise TPR/FPR gaps
    """
    y_true, y_pred, used_threshold = _resolve_predictions(
        y_true, y_pred, y_scores, threshold
    )
    group_size, confusion_matrices, tpr, fpr, _ppv = _group_tables(
        y_true, y_pred, groups
    )
    return EqualizedOddsResult(
        n_groups=len(group_size),
        n_samples=int(len(y_true)),
        group_size=group_size,
        tpr=tpr,
        fpr=fpr,
        tpr_difference=tpr.difference,
        fpr_difference=fpr.difference,
        equalized_odds_difference=float(max(tpr.difference, fpr.difference)),
        confusion_matrices=confusion_matrices,
        threshold=used_threshold,
    )


def compute_predictive_parity(
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray],
    groups: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> PredictiveParityResult:
    """
    Compute predictive parity as per-group PPV and pairwise gaps.

    Predictive parity holds when positive predictive value (precision)
    is the same for every group. ``ppv_difference`` is max PPV minus min PPV.

    Args:
        y_true: Binary ground-truth labels (0 or 1)
        y_pred: Binary predictions (0 or 1). Pass ``None`` if ``y_scores`` is given
        groups: Sensitive-attribute membership
        y_scores: Optional scores; thresholded when ``y_pred`` is omitted
        threshold: Cutoff applied to ``y_scores`` (default 0.5)

    Returns:
        PredictiveParityResult with per-group PPV and pairwise gaps
    """
    y_true, y_pred, used_threshold = _resolve_predictions(
        y_true, y_pred, y_scores, threshold
    )
    group_size, confusion_matrices, _tpr, _fpr, ppv = _group_tables(
        y_true, y_pred, groups
    )
    return PredictiveParityResult(
        n_groups=len(group_size),
        n_samples=int(len(y_true)),
        group_size=group_size,
        ppv=ppv,
        ppv_difference=ppv.difference,
        confusion_matrices=confusion_matrices,
        threshold=used_threshold,
    )


def compute_odds_parity_metrics(
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray],
    groups: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> OddsParityResult:
    """
    Compute equalized odds and predictive parity for the same predictions.

    Signature matches ``compute_fairness_metrics(y_true, y_pred, groups)``.
    Pass ``y_pred=None`` and ``y_scores`` to threshold scores instead
    (default cutoff 0.5).
    """
    y_true, y_pred, used_threshold = _resolve_predictions(
        y_true, y_pred, y_scores, threshold
    )
    group_size, confusion_matrices, tpr, fpr, ppv = _group_tables(
        y_true, y_pred, groups
    )
    equalized_odds = EqualizedOddsResult(
        n_groups=len(group_size),
        n_samples=int(len(y_true)),
        group_size=group_size,
        tpr=tpr,
        fpr=fpr,
        tpr_difference=tpr.difference,
        fpr_difference=fpr.difference,
        equalized_odds_difference=float(max(tpr.difference, fpr.difference)),
        confusion_matrices=confusion_matrices,
        threshold=used_threshold,
    )
    predictive_parity = PredictiveParityResult(
        n_groups=len(group_size),
        n_samples=int(len(y_true)),
        group_size=group_size,
        ppv=ppv,
        ppv_difference=ppv.difference,
        confusion_matrices=confusion_matrices,
        threshold=used_threshold,
    )
    return OddsParityResult(
        n_groups=len(group_size),
        n_samples=int(len(y_true)),
        group_size=group_size,
        equalized_odds=equalized_odds,
        predictive_parity=predictive_parity,
        tpr_difference=tpr.difference,
        fpr_difference=fpr.difference,
        equalized_odds_difference=equalized_odds.equalized_odds_difference,
        ppv_difference=ppv.difference,
        confusion_matrices=confusion_matrices,
        threshold=used_threshold,
    )


__all__ = [
    "PairwiseGap",
    "GroupRateTable",
    "EqualizedOddsResult",
    "PredictiveParityResult",
    "OddsParityResult",
    "compute_equalized_odds",
    "compute_predictive_parity",
    "compute_odds_parity_metrics",
]
