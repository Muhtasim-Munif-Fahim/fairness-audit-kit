"""
Theil index and generalized entropy inequality metrics.

Measures inequality of classification benefit or error across individuals
and groups, following Speicher et al. (KDD 2018) and the AIF360
generalized entropy index. The Theil index is the special case alpha=1.
"""

import numpy as np
from typing import Dict
from dataclasses import dataclass


def classification_benefit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    favorable_label: int = 1,
) -> np.ndarray:
    """
    Per-individual classification benefit used by Speicher / AIF360.

    ``b_i = 1 + 1[y_pred = favorable] - 1[y_true = favorable]``

    This yields 1 for a correct prediction, 2 for a false positive, and
    0 for a false negative.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return (
        1.0
        + (y_pred == favorable_label).astype(float)
        - (y_true == favorable_label).astype(float)
    )


def classification_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-individual 0/1 misclassification indicator."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return (y_true != y_pred).astype(float)


def generalized_entropy_index(values: np.ndarray, alpha: float = 2.0) -> float:
    """
    Generalized entropy index of non-negative values.

    ``alpha=0`` is the mean log deviation, ``alpha=1`` is the Theil index,
    and ``alpha=2`` is half the squared coefficient of variation.

    Perfect equality (including the all-zero vector) returns 0. Values
    must be non-negative. ``alpha=0`` additionally requires strictly
    positive values because of the logarithm.
    """
    b = np.asarray(values, dtype=float).ravel()
    if b.size == 0:
        raise ValueError("values must be non-empty")
    if np.any(b < 0):
        raise ValueError("Generalized entropy requires non-negative values")

    mu = float(np.mean(b))
    if mu == 0.0:
        return 0.0

    if abs(alpha - 1.0) < 1e-12:
        ratio = b / mu
        with np.errstate(divide="ignore", invalid="ignore"):
            contrib = ratio * np.log(ratio)
        contrib = np.where(ratio > 0, contrib, 0.0)
        return float(np.mean(contrib))

    if abs(alpha) < 1e-12:
        if np.any(b == 0):
            raise ValueError(
                "Generalized entropy with alpha=0 requires strictly positive values"
            )
        return float(-np.mean(np.log(b / mu)))

    return float(np.mean((b / mu) ** alpha - 1.0) / (alpha * (alpha - 1.0)))


def theil_index(values: np.ndarray) -> float:
    """Theil index: generalized entropy with ``alpha=1``."""
    return generalized_entropy_index(values, alpha=1.0)


def _expand_group_means(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Replace each value with its group's mean (between-group projection)."""
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    out = np.empty(values.shape[0], dtype=float)
    for g in np.unique(groups):
        mask = groups == g
        out[mask] = float(np.mean(values[mask]))
    return out


def _entropy_breakdown(
    values: np.ndarray,
    groups: np.ndarray,
    alpha: float,
) -> "EntropyBreakdown":
    overall = generalized_entropy_index(values, alpha=alpha)
    between = generalized_entropy_index(_expand_group_means(values, groups), alpha=alpha)
    within = overall - between
    if abs(within) < 1e-12:
        within = 0.0
    return EntropyBreakdown(
        overall=overall,
        between_group=between,
        within_group=within,
    )


@dataclass
class EntropyBreakdown:
    """Overall generalized entropy and its between/within group split."""
    overall: float
    between_group: float
    within_group: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "overall": self.overall,
            "between_group": self.between_group,
            "within_group": self.within_group,
        }


@dataclass
class GeneralizedEntropyResult:
    """Theil / generalized entropy inequality of benefit and error."""
    alpha: float
    n_groups: int
    n_samples: int
    mean_benefit: float
    mean_error_rate: float
    benefit: EntropyBreakdown
    error: EntropyBreakdown
    group_mean_benefit: Dict[str, float]
    group_error_rate: Dict[str, float]
    group_size: Dict[str, int]

    def to_dict(self) -> Dict:
        return {
            "alpha": self.alpha,
            "n_groups": self.n_groups,
            "n_samples": self.n_samples,
            "mean_benefit": self.mean_benefit,
            "mean_error_rate": self.mean_error_rate,
            "benefit": self.benefit.to_dict(),
            "error": self.error.to_dict(),
            "group_mean_benefit": self.group_mean_benefit,
            "group_error_rate": self.group_error_rate,
            "group_size": self.group_size,
        }


def compute_generalized_entropy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    alpha: float = 1.0,
    favorable_label: int = 1,
) -> GeneralizedEntropyResult:
    """
    Compute generalized entropy inequality of benefit and error rates.

    Individual benefit follows Speicher et al. / AIF360:
    ``b_i = 1 + 1[y_pred = favorable] - 1[y_true = favorable]``.
    Individual error is the 0/1 misclassification indicator.

    The index is computed on the individual outcomes (overall), then
    decomposed by replacing each person with their group mean
    (between-group) and taking the residual (within-group). With the
    default ``alpha=1`` this is the Theil index.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        groups: Group membership array
        alpha: Entropy parameter. 1 = Theil, 0 = mean log deviation,
            2 = half the squared coefficient of variation
        favorable_label: Label treated as the favorable class for benefit

    Returns:
        GeneralizedEntropyResult with overall / between / within indices
        for both benefit and error, plus per-group means
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    n = len(y_true)
    if not (len(y_pred) == n and len(groups) == n):
        raise ValueError("y_true, y_pred, and groups must have the same length")
    if n == 0:
        raise ValueError("y_true, y_pred, and groups must be non-empty")

    unique_groups = [str(g) for g in np.unique(groups)]
    if len(unique_groups) < 2:
        raise ValueError("At least two groups required for fairness computation")

    benefit = classification_benefit(y_true, y_pred, favorable_label=favorable_label)
    error = classification_error(y_true, y_pred)

    group_mean_benefit: Dict[str, float] = {}
    group_error_rate: Dict[str, float] = {}
    group_size: Dict[str, int] = {}
    for g in np.unique(groups):
        mask = groups == g
        key = str(g)
        group_mean_benefit[key] = float(np.mean(benefit[mask]))
        group_error_rate[key] = float(np.mean(error[mask]))
        group_size[key] = int(np.sum(mask))

    return GeneralizedEntropyResult(
        alpha=float(alpha),
        n_groups=len(unique_groups),
        n_samples=n,
        mean_benefit=float(np.mean(benefit)),
        mean_error_rate=float(np.mean(error)),
        benefit=_entropy_breakdown(benefit, groups, alpha),
        error=_entropy_breakdown(error, groups, alpha),
        group_mean_benefit=group_mean_benefit,
        group_error_rate=group_error_rate,
        group_size=group_size,
    )


def compute_theil_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    favorable_label: int = 1,
) -> GeneralizedEntropyResult:
    """Theil index (generalized entropy with ``alpha=1``) of benefit and error."""
    return compute_generalized_entropy(
        y_true,
        y_pred,
        groups,
        alpha=1.0,
        favorable_label=favorable_label,
    )


__all__ = [
    "EntropyBreakdown",
    "GeneralizedEntropyResult",
    "classification_benefit",
    "classification_error",
    "generalized_entropy_index",
    "theil_index",
    "compute_generalized_entropy",
    "compute_theil_metrics",
]
