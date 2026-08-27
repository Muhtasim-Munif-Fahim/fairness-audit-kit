"""
Group fairness metrics for binary classification.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FairnessMetrics:
    """Container for group fairness metrics."""
    demographic_parity_difference: float
    equal_opportunity_difference: float
    equalized_odds_difference: float
    disparate_impact_ratio: float
    calibration_by_group: Dict[str, float]
    confusion_matrices: Dict[str, Dict[str, int]]

    def to_dict(self) -> Dict:
        return {
            "demographic_parity_difference": self.demographic_parity_difference,
            "equal_opportunity_difference": self.equal_opportunity_difference,
            "equalized_odds_difference": self.equalized_odds_difference,
            "disparate_impact_ratio": self.disparate_impact_ratio,
            "calibration_by_group": self.calibration_by_group,
            "confusion_matrices": self.confusion_matrices,
        }


def _safe_divide(num: float, den: float, default: float = 0.0) -> float:
    """Safe division with default for zero denominator."""
    return num / den if den != 0 else default


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """Compute confusion matrix counts."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}


def _group_rates(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> Dict[str, Dict]:
    """Compute per-group prediction rates and confusion matrices."""
    unique_groups = np.unique(groups)
    rates = {}
    for g in unique_groups:
        mask = groups == g
        y_t = y_true[mask]
        y_p = y_pred[mask]
        cm = _confusion_matrix(y_t, y_p)
        total = len(y_t)
        pos_rate = _safe_divide(np.sum(y_p), total)
        tpr = _safe_divide(cm["tp"], cm["tp"] + cm["fn"])
        fpr = _safe_divide(cm["fp"], cm["fp"] + cm["tn"])
        precision = _safe_divide(cm["tp"], cm["tp"] + cm["fp"])
        calibration = _safe_divide(np.sum(y_t), np.sum(y_p)) if np.sum(y_p) > 0 else 0.0
        rates[str(g)] = {
            "total": total,
            "positive_rate": pos_rate,
            "tpr": tpr,
            "fpr": fpr,
            "precision": precision,
            "calibration": calibration,
            "confusion_matrix": cm,
        }
    return rates


def compute_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    favorable_label: int = 1,
) -> FairnessMetrics:
    """
    Compute group fairness metrics for binary classification.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        groups: Group membership array (e.g., sensitive attributes)
        favorable_label: Label considered favorable (default 1)

    Returns:
        FairnessMetrics object with all computed metrics
    """
    rates = _group_rates(y_true, y_pred, groups)
    group_ids = list(rates.keys())

    if len(group_ids) < 2:
        raise ValueError("At least two groups required for fairness computation")

    pos_rates = [rates[g]["positive_rate"] for g in group_ids]
    tprs = [rates[g]["tpr"] for g in group_ids]
    fprs = [rates[g]["fpr"] for g in group_ids]
    calibrations = {g: rates[g]["calibration"] for g in group_ids}
    confusion_matrices = {g: rates[g]["confusion_matrix"] for g in group_ids}

    dpd = max(pos_rates) - min(pos_rates)
    eod = max(tprs) - min(tprs)
    eodds = max(eod, max(fprs) - min(fprs))

    dir_num = min(pos_rates)
    dir_den = max(pos_rates)
    disparate_impact = _safe_divide(dir_num, dir_den, default=1.0)

    return FairnessMetrics(
        demographic_parity_difference=dpd,
        equal_opportunity_difference=eod,
        equalized_odds_difference=eodds,
        disparate_impact_ratio=disparate_impact,
        calibration_by_group=calibrations,
        confusion_matrices=confusion_matrices,
    )


def confusion_matrices_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> Dict[str, Dict[str, int]]:
    """Return per-group confusion matrices."""
    rates = _group_rates(y_true, y_pred, groups)
    return {g: rates[g]["confusion_matrix"] for g in rates}


__all__ = [
    "FairnessMetrics",
    "compute_fairness_metrics",
    "confusion_matrices_by_group",
]
