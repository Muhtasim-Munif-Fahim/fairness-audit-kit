"""
Reliability diagrams and expected calibration error (ECE).

Bins predicted probabilities and compares mean confidence to the
observed positive rate, overall and per sensitive group. ECE is the
sample-weighted average of those per-bin gaps (Naeini et al.;
Guo et al., ICML 2017).
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


STRATEGIES = ("uniform", "quantile")


@dataclass
class ReliabilityBin:
    """One bin of a reliability diagram."""
    bin_index: int
    lower: float
    upper: float
    n_samples: int
    mean_confidence: float
    observed_positive_rate: float
    gap: float
    weight: float

    def to_dict(self) -> Dict:
        return {
            "bin_index": self.bin_index,
            "lower": self.lower,
            "upper": self.upper,
            "n_samples": self.n_samples,
            "mean_confidence": self.mean_confidence,
            "observed_positive_rate": self.observed_positive_rate,
            "gap": self.gap,
            "weight": self.weight,
        }


@dataclass
class CalibrationResult:
    """Overall and per-group ECE with reliability diagram tables."""
    n_bins: int
    strategy: str
    n_groups: int
    n_samples: int
    ece: float
    mce: float
    ece_difference: float
    reliability_bins: List[ReliabilityBin]
    group_ece: Dict[str, float]
    group_mce: Dict[str, float]
    group_size: Dict[str, int]
    group_reliability_bins: Dict[str, List[ReliabilityBin]]

    def to_dict(self) -> Dict:
        return {
            "n_bins": self.n_bins,
            "strategy": self.strategy,
            "n_groups": self.n_groups,
            "n_samples": self.n_samples,
            "ece": self.ece,
            "mce": self.mce,
            "ece_difference": self.ece_difference,
            "reliability_bins": [b.to_dict() for b in self.reliability_bins],
            "group_ece": self.group_ece,
            "group_mce": self.group_mce,
            "group_size": self.group_size,
            "group_reliability_bins": {
                g: [b.to_dict() for b in bins]
                for g, bins in self.group_reliability_bins.items()
            },
        }


def _validate_labels_and_scores(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_scores = np.asarray(y_scores, dtype=float).ravel()
    if y_true.size == 0 or y_scores.size == 0:
        raise ValueError("y_true and y_scores must be non-empty")
    if y_true.shape[0] != y_scores.shape[0]:
        raise ValueError("y_true and y_scores must have the same length")
    if np.any(~np.isfinite(y_scores)):
        raise ValueError("y_scores must be finite")
    if np.any((y_scores < 0.0) | (y_scores > 1.0)):
        raise ValueError("y_scores must be in [0, 1]")
    uniques = set(np.unique(y_true).tolist())
    if not uniques.issubset({0.0, 1.0}):
        raise ValueError("y_true must be binary labels in {0, 1}")
    return y_true, y_scores


def _bin_edges(y_scores: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(y_scores, quantiles)
    edges = np.unique(edges)
    if edges.size < 2:
        return np.array([0.0, 1.0])
    return edges


def _assign_bins(
    y_scores: np.ndarray,
    n_bins: int,
    strategy: str,
) -> Tuple[np.ndarray, np.ndarray]:
    edges = _bin_edges(y_scores, n_bins, strategy)
    if strategy == "uniform":
        indices = np.minimum((y_scores * n_bins).astype(int), n_bins - 1)
        return edges, indices
    indices = np.digitize(y_scores, edges[1:-1], right=False)
    return edges, indices


def compute_reliability_bins(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> List[ReliabilityBin]:
    """
    Build the occupied bins of a reliability diagram.

    Empty bins are omitted. ``strategy="uniform"`` uses equal-width bins
    on ``[0, 1]`` (standard ECE). ``strategy="quantile"`` uses equal-mass
    bins on the observed scores.

    Args:
        y_true: Binary ground-truth labels (0 or 1)
        y_scores: Predicted probabilities of the positive class in [0, 1]
        n_bins: Number of bins
        strategy: ``"uniform"`` or ``"quantile"``

    Returns:
        Occupied ``ReliabilityBin`` rows, ordered by increasing score
    """
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")

    y_true, y_scores = _validate_labels_and_scores(y_true, y_scores)
    edges, indices = _assign_bins(y_scores, n_bins, strategy)
    n = y_true.shape[0]
    n_actual = len(edges) - 1

    bins: List[ReliabilityBin] = []
    for m in range(n_actual):
        mask = indices == m
        count = int(np.sum(mask))
        if count == 0:
            continue
        conf = float(np.mean(y_scores[mask]))
        acc = float(np.mean(y_true[mask]))
        bins.append(
            ReliabilityBin(
                bin_index=m,
                lower=float(edges[m]),
                upper=float(edges[m + 1]),
                n_samples=count,
                mean_confidence=conf,
                observed_positive_rate=acc,
                gap=abs(acc - conf),
                weight=count / n,
            )
        )
    return bins


def _ece_from_bins(bins: List[ReliabilityBin]) -> float:
    return float(sum(b.weight * b.gap for b in bins))


def _mce_from_bins(bins: List[ReliabilityBin]) -> float:
    if not bins:
        return 0.0
    return float(max(b.gap for b in bins))


def expected_calibration_error(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """
    Expected calibration error: sample-weighted mean of |acc − conf| per bin.

    Perfect calibration (including a constant score equal to the base
    rate) returns 0.
    """
    return _ece_from_bins(
        compute_reliability_bins(y_true, y_scores, n_bins=n_bins, strategy=strategy)
    )


def maximum_calibration_error(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Maximum |acc − conf| across occupied bins (MCE)."""
    return _mce_from_bins(
        compute_reliability_bins(y_true, y_scores, n_bins=n_bins, strategy=strategy)
    )


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    groups: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> CalibrationResult:
    """
    Compute ECE and reliability diagram data overall and per sensitive group.

    Group ECE uses the same binning strategy on that group's scores.
    Uniform bins share the ``[0, 1]`` edges, so per-group reliability
    tables are comparable. ``ece_difference`` is max group ECE minus min
    group ECE.

    Args:
        y_true: Binary ground-truth labels (0 or 1)
        y_scores: Predicted probabilities of the positive class in [0, 1]
        groups: Group membership array
        n_bins: Number of bins (default 10)
        strategy: ``"uniform"`` (equal-width) or ``"quantile"`` (equal-mass)

    Returns:
        CalibrationResult with overall / per-group ECE, MCE, and bins
    """
    y_true, y_scores = _validate_labels_and_scores(y_true, y_scores)
    groups = np.asarray(groups)
    n = y_true.shape[0]
    if len(groups) != n:
        raise ValueError("y_true, y_scores, and groups must have the same length")

    unique_groups = [str(g) for g in np.unique(groups)]
    if len(unique_groups) < 2:
        raise ValueError("At least two groups required for fairness computation")

    overall_bins = compute_reliability_bins(
        y_true, y_scores, n_bins=n_bins, strategy=strategy
    )

    group_ece: Dict[str, float] = {}
    group_mce: Dict[str, float] = {}
    group_size: Dict[str, int] = {}
    group_reliability_bins: Dict[str, List[ReliabilityBin]] = {}
    for g in np.unique(groups):
        mask = groups == g
        key = str(g)
        g_bins = compute_reliability_bins(
            y_true[mask], y_scores[mask], n_bins=n_bins, strategy=strategy
        )
        group_ece[key] = _ece_from_bins(g_bins)
        group_mce[key] = _mce_from_bins(g_bins)
        group_size[key] = int(np.sum(mask))
        group_reliability_bins[key] = g_bins

    return CalibrationResult(
        n_bins=int(n_bins),
        strategy=strategy,
        n_groups=len(unique_groups),
        n_samples=n,
        ece=_ece_from_bins(overall_bins),
        mce=_mce_from_bins(overall_bins),
        ece_difference=float(max(group_ece.values()) - min(group_ece.values())),
        reliability_bins=overall_bins,
        group_ece=group_ece,
        group_mce=group_mce,
        group_size=group_size,
        group_reliability_bins=group_reliability_bins,
    )


__all__ = [
    "ReliabilityBin",
    "CalibrationResult",
    "compute_reliability_bins",
    "expected_calibration_error",
    "maximum_calibration_error",
    "compute_calibration_metrics",
]
