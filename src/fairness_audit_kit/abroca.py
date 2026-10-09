"""
ABROCA: Absolute Between-ROC Area (Gardner et al., LAK 2019).

For each pair of groups, integrate the absolute vertical gap between
their ROC curves over false-positive rate:

    ABROCA(a, b) = ∫_0^1 |TPR_a(f) - TPR_b(f)| df

Unlike a single equalized-odds snapshot at one threshold, ABROCA
summarises disparity across the entire score ranking. Zero means the
group ROC curves coincide; larger values mean larger slice-wise gaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np



@dataclass
class GroupROC:
    """ROC curve for one sensitive group."""

    group: str
    n_samples: int
    n_positives: int
    n_negatives: int
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auc: float

    def to_dict(self) -> Dict:
        return {
            "group": self.group,
            "n_samples": self.n_samples,
            "n_positives": self.n_positives,
            "n_negatives": self.n_negatives,
            "fpr": self.fpr.tolist(),
            "tpr": self.tpr.tolist(),
            "thresholds": self.thresholds.tolist(),
            "auc": self.auc,
        }


@dataclass
class AbrocaPair:
    """ABROCA between two groups."""

    group_a: str
    group_b: str
    abroca: float
    auc_a: float
    auc_b: float
    auc_gap: float

    def to_dict(self) -> Dict:
        return {
            "group_a": self.group_a,
            "group_b": self.group_b,
            "abroca": self.abroca,
            "auc_a": self.auc_a,
            "auc_b": self.auc_b,
            "auc_gap": self.auc_gap,
        }


@dataclass
class AbrocaResult:
    """Per-group ROC curves and pairwise Absolute Between-ROC Areas."""

    n_groups: int
    n_samples: int
    group_size: Dict[str, int]
    group_roc: Dict[str, GroupROC]
    pairwise: List[AbrocaPair]
    max_abroca: float
    mean_abroca: float

    def to_dict(self) -> Dict:
        return {
            "n_groups": self.n_groups,
            "n_samples": self.n_samples,
            "group_size": self.group_size,
            "group_roc": {g: r.to_dict() for g, r in self.group_roc.items()},
            "pairwise": [p.to_dict() for p in self.pairwise],
            "max_abroca": self.max_abroca,
            "mean_abroca": self.mean_abroca,
        }


def _validate_inputs(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    groups: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_scores = np.asarray(y_scores, dtype=float).ravel()
    groups = np.asarray(groups).ravel()
    if y_true.size == 0:
        raise ValueError("y_true, y_scores, and groups must be non-empty")
    if not (y_true.shape == y_scores.shape == groups.shape):
        raise ValueError("y_true, y_scores, and groups must have the same length")
    if np.any(~np.isfinite(y_scores)):
        raise ValueError("y_scores must be finite")
    uniques = set(np.unique(y_true).tolist())
    if not uniques.issubset({0.0, 1.0}):
        raise ValueError("y_true must be binary labels in {0, 1}")
    if len(np.unique(groups)) < 2:
        raise ValueError("At least two groups required for ABROCA")
    return y_true, y_scores, groups


def _roc_curve(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute ROC (FPR, TPR, thresholds) and AUC via rank trapezoid."""
    n_pos = int(np.sum(y_true == 1.0))
    n_neg = int(np.sum(y_true == 0.0))
    if n_pos == 0 or n_neg == 0:
        # Degenerate: constant TPR/FPR.
        fpr = np.array([0.0, 1.0])
        tpr = np.array([0.0, 1.0]) if n_pos > 0 else np.array([0.0, 0.0])
        thr = np.array([np.inf, -np.inf])
        return fpr, tpr, thr, float("nan")

    # Sort scores descending; stable so ties keep input order.
    order = np.argsort(-y_scores, kind="mergesort")
    y_true = y_true[order]
    y_scores = y_scores[order]

    # Thresholds at unique score changes (sklearn-style).
    distinct = np.where(np.diff(y_scores))[0]
    threshold_idxs = np.r_[distinct, y_true.size - 1]

    tps = np.cumsum(y_true == 1.0)[threshold_idxs]
    fps = np.cumsum(y_true == 0.0)[threshold_idxs]
    thresholds = y_scores[threshold_idxs]

    tpr = np.r_[0.0, tps / n_pos]
    fpr = np.r_[0.0, fps / n_neg]
    thresholds = np.r_[np.inf, thresholds]

    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, thresholds, auc


def _interp_tpr(fpr: np.ndarray, tpr: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Left-constant / linear interpolation of TPR onto an FPR grid."""
    # ROC steps are non-decreasing in FPR; use np.interp (linear).
    return np.interp(grid, fpr, tpr)


def _pairwise_abroca(
    fpr_a: np.ndarray,
    tpr_a: np.ndarray,
    fpr_b: np.ndarray,
    tpr_b: np.ndarray,
    n_grid: int,
) -> float:
    grid = np.linspace(0.0, 1.0, n_grid)
    ta = _interp_tpr(fpr_a, tpr_a, grid)
    tb = _interp_tpr(fpr_b, tpr_b, grid)
    return float(np.trapezoid(np.abs(ta - tb), grid))


def compute_abroca(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    groups: np.ndarray,
    *,
    n_grid: int = 501,
) -> AbrocaResult:
    """
    Compute Absolute Between-ROC Area for every pair of groups.

    Args:
        y_true: Binary labels in {0, 1}.
        y_scores: Real-valued decision scores (higher = more positive).
            Need not lie in [0, 1].
        groups: Sensitive group membership aligned with ``y_true``.
        n_grid: Number of FPR grid points used for the trapezoidal
            integral of ``|TPR_a - TPR_b|``.

    Returns:
        AbrocaResult with per-group ROC curves and pairwise ABROCA.
    """
    if not isinstance(n_grid, int) or n_grid < 2:
        raise ValueError("n_grid must be an integer >= 2")

    y_true, y_scores, groups = _validate_inputs(y_true, y_scores, groups)
    unique_groups = [str(g) for g in np.unique(groups)]
    group_roc: Dict[str, GroupROC] = {}
    group_size: Dict[str, int] = {}

    for g in unique_groups:
        mask = groups.astype(str) == g
        yt = y_true[mask]
        ys = y_scores[mask]
        fpr, tpr, thr, auc = _roc_curve(yt, ys)
        group_size[g] = int(mask.sum())
        group_roc[g] = GroupROC(
            group=g,
            n_samples=int(mask.sum()),
            n_positives=int(np.sum(yt == 1.0)),
            n_negatives=int(np.sum(yt == 0.0)),
            fpr=fpr,
            tpr=tpr,
            thresholds=thr,
            auc=auc,
        )

    pairwise: List[AbrocaPair] = []
    for a, b in combinations(unique_groups, 2):
        ra, rb = group_roc[a], group_roc[b]
        value = _pairwise_abroca(ra.fpr, ra.tpr, rb.fpr, rb.tpr, n_grid)
        auc_a = ra.auc if np.isfinite(ra.auc) else 0.0
        auc_b = rb.auc if np.isfinite(rb.auc) else 0.0
        pairwise.append(
            AbrocaPair(
                group_a=a,
                group_b=b,
                abroca=value,
                auc_a=ra.auc,
                auc_b=rb.auc,
                auc_gap=abs(auc_a - auc_b),
            )
        )

    values = [p.abroca for p in pairwise]
    return AbrocaResult(
        n_groups=len(unique_groups),
        n_samples=int(y_true.size),
        group_size=group_size,
        group_roc=group_roc,
        pairwise=pairwise,
        max_abroca=float(max(values)) if values else 0.0,
        mean_abroca=float(np.mean(values)) if values else 0.0,
    )
