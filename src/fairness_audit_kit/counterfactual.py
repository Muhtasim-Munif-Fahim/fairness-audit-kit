"""
Counterfactual fairness proxy via sensitive-attribute score flips.

Kusner et al. (NIPS 2017) define counterfactual fairness as invariance of the
prediction when the sensitive attribute is set to a different value in a
causal model. Without a structural causal model, a practical *proxy* is to
compare the model's score on the factual features against the score on a
counterfactual input where only the sensitive attribute is flipped (or
otherwise remapped), holding other covariates fixed.

This module does not fit a causal graph. Callers supply factual scores and
counterfactual scores (for example by scoring ``X`` and ``X`` with the
sensitive column remapped). The proxy reports how much predictions move
under that intervention, overall and per group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np


ArrayLike = Union[np.ndarray, Sequence]


def _as_1d_float(values: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _as_groups(groups: ArrayLike) -> np.ndarray:
    arr = np.asarray(groups).reshape(-1)
    if arr.size == 0:
        raise ValueError("groups must be non-empty")
    return arr


def flip_sensitive_attribute(
    groups: ArrayLike,
    mapping: Optional[Mapping[object, object]] = None,
) -> np.ndarray:
    """Return counterfactual group labels by flipping (or remapping) values.

    For binary attributes the default mapping swaps the two unique values.
    For more than two levels, pass an explicit ``mapping`` covering every
    observed value.
    """
    groups_arr = _as_groups(groups)
    unique = list(dict.fromkeys(groups_arr.tolist()))
    if mapping is None:
        if len(unique) != 2:
            raise ValueError(
                "default flip requires exactly two sensitive levels; "
                "pass mapping= for multi-level attributes"
            )
        mapping = {unique[0]: unique[1], unique[1]: unique[0]}
    missing = [value for value in unique if value not in mapping]
    if missing:
        raise ValueError(f"mapping missing keys for levels: {missing!r}")
    return np.asarray([mapping[value] for value in groups_arr.tolist()])


def remap_sensitive_column(
    X: np.ndarray,
    sensitive_index: int,
    *,
    mapping: Optional[Mapping[object, object]] = None,
) -> np.ndarray:
    """Copy ``X`` with column ``sensitive_index`` flipped / remapped."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2-d feature matrix")
    if not 0 <= sensitive_index < X.shape[1]:
        raise ValueError("sensitive_index out of range")
    out = np.array(X, copy=True)
    out[:, sensitive_index] = flip_sensitive_attribute(
        out[:, sensitive_index], mapping=mapping
    )
    return out


@dataclass
class CounterfactualFairnessResult:
    """Proxy counterfactual fairness summary for a score flip."""

    n_samples: int
    n_groups: int
    group_size: Dict[str, int]
    mean_abs_delta: float
    max_abs_delta: float
    median_abs_delta: float
    decision_flip_rate: float
    group_mean_abs_delta: Dict[str, float]
    group_decision_flip_rate: Dict[str, float]
    threshold: Optional[float]

    def to_dict(self) -> Dict:
        return {
            "n_samples": self.n_samples,
            "n_groups": self.n_groups,
            "group_size": self.group_size,
            "mean_abs_delta": self.mean_abs_delta,
            "max_abs_delta": self.max_abs_delta,
            "median_abs_delta": self.median_abs_delta,
            "decision_flip_rate": self.decision_flip_rate,
            "group_mean_abs_delta": self.group_mean_abs_delta,
            "group_decision_flip_rate": self.group_decision_flip_rate,
            "threshold": self.threshold,
        }


def compute_counterfactual_fairness_proxy(
    y_score: ArrayLike,
    y_score_cf: ArrayLike,
    groups: ArrayLike,
    *,
    threshold: Optional[float] = 0.5,
) -> CounterfactualFairnessResult:
    """Compare factual and counterfactual scores under a sensitive flip.

    ``y_score`` is the model output on factual inputs; ``y_score_cf`` is the
    output after intervening on the sensitive attribute (same length). When
    ``threshold`` is not ``None``, hard decisions are compared and
    ``decision_flip_rate`` is the fraction of rows whose predicted label
    changes. A perfectly counterfactually fair scorer yields zero deltas.
    """
    factual = _as_1d_float(y_score, "y_score")
    counterfactual = _as_1d_float(y_score_cf, "y_score_cf")
    groups_arr = _as_groups(groups)
    if not (factual.size == counterfactual.size == groups_arr.size):
        raise ValueError("y_score, y_score_cf, and groups must have the same length")
    if threshold is not None and not np.isfinite(threshold):
        raise ValueError("threshold must be finite or None")

    delta = np.abs(factual - counterfactual)
    if threshold is None:
        flips = np.zeros(factual.size, dtype=bool)
    else:
        pred = factual >= float(threshold)
        pred_cf = counterfactual >= float(threshold)
        flips = pred != pred_cf

    labels = [str(g) for g in groups_arr.tolist()]
    unique = sorted(set(labels))
    group_size = {g: int(labels.count(g)) for g in unique}
    group_mean: Dict[str, float] = {}
    group_flip: Dict[str, float] = {}
    for g in unique:
        mask = np.asarray([lab == g for lab in labels], dtype=bool)
        group_mean[g] = float(np.mean(delta[mask])) if mask.any() else 0.0
        group_flip[g] = float(np.mean(flips[mask])) if mask.any() else 0.0

    return CounterfactualFairnessResult(
        n_samples=int(factual.size),
        n_groups=len(unique),
        group_size=group_size,
        mean_abs_delta=float(np.mean(delta)),
        max_abs_delta=float(np.max(delta)),
        median_abs_delta=float(np.median(delta)),
        decision_flip_rate=float(np.mean(flips)),
        group_mean_abs_delta=group_mean,
        group_decision_flip_rate=group_flip,
        threshold=None if threshold is None else float(threshold),
    )


def score_with_sensitive_flip(
    X: np.ndarray,
    sensitive_index: int,
    score_fn: Callable[[np.ndarray], np.ndarray],
    *,
    mapping: Optional[Mapping[object, object]] = None,
    groups: Optional[ArrayLike] = None,
    threshold: Optional[float] = 0.5,
) -> CounterfactualFairnessResult:
    """Score ``X`` and a sensitive-column flip, then compute the proxy.

    ``score_fn`` must accept a 2-d array and return a 1-d score vector.
    ``groups`` defaults to the factual sensitive column.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2-d feature matrix")
    factual_groups = X[:, sensitive_index] if groups is None else groups
    X_cf = remap_sensitive_column(X, sensitive_index, mapping=mapping)
    y_score = np.asarray(score_fn(X), dtype=float).reshape(-1)
    y_score_cf = np.asarray(score_fn(X_cf), dtype=float).reshape(-1)
    return compute_counterfactual_fairness_proxy(
        y_score, y_score_cf, factual_groups, threshold=threshold
    )


__all__ = [
    "CounterfactualFairnessResult",
    "compute_counterfactual_fairness_proxy",
    "flip_sensitive_attribute",
    "remap_sensitive_column",
    "score_with_sensitive_flip",
]
