"""
Reweighing: pre-processing bias mitigation (Kamiran & Calders, KAIS 2012).

The audit modules measure disparity. Reweighing is a way to reduce it before
training. Each ``(group, label)`` cell gets the weight

    w(g, y) = P(G = g) * P(Y = y) / P(G = g, Y = y)
            = (n_g * n_y) / (n * n_{g,y})

Under these weights the sensitive attribute and the label are statistically
independent: the weighted base rate ``P_w(Y = 1 | G = g)`` is the same for
every group, and the weights sum to ``n``. Training a classifier with
``sample_weight=weights`` removes the incentive to learn the group/label
association that is present in the training data. Features and labels are
not changed.

Cells that never occur (``n_{g,y} = 0``) cannot be rebalanced by weighting;
they are reported in ``empty_cells``.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ReweighingResult:
    """Per-sample weights and the per-cell weight table."""

    n_samples: int
    n_groups: int
    labels: List[Any]
    weights: np.ndarray
    weight_table: Dict[str, Dict[str, float]]
    cell_count: Dict[str, Dict[str, int]]
    base_rates_before: Dict[str, float]
    base_rates_after: Dict[str, float]
    base_rate_gap_before: float
    base_rate_gap_after: float
    positive_label: Any
    empty_cells: List[Tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "n_samples": self.n_samples,
            "n_groups": self.n_groups,
            "labels": [str(v) for v in self.labels],
            "weight_table": self.weight_table,
            "cell_count": self.cell_count,
            "base_rates_before": self.base_rates_before,
            "base_rates_after": self.base_rates_after,
            "base_rate_gap_before": self.base_rate_gap_before,
            "base_rate_gap_after": self.base_rate_gap_after,
            "positive_label": str(self.positive_label),
            "empty_cells": [list(c) for c in self.empty_cells],
        }


def compute_reweighing_weights(
    y_true: np.ndarray,
    groups: np.ndarray,
    positive_label: Optional[Any] = None,
) -> ReweighingResult:
    """
    Compute Kamiran-Calders reweighing weights.

    Args:
        y_true: Training labels (binary or multiclass).
        groups: Sensitive attribute per sample.
        positive_label: Label whose per-group rate is reported in
            ``base_rates_*``. Defaults to ``1`` when present, otherwise the
            largest label.

    Returns:
        ReweighingResult with per-sample ``weights`` (mean 1) and diagnostics.
    """
    y = np.asarray(y_true)
    g = np.asarray(groups)
    if y.ndim != 1 or g.ndim != 1:
        raise ValueError("y_true and groups must be 1-D")
    if len(y) != len(g):
        raise ValueError("y_true and groups must have the same length")
    if len(y) == 0:
        raise ValueError("need at least one sample")

    n = len(y)
    labels = sorted(np.unique(y).tolist(), key=lambda v: (str(type(v)), v))
    group_values = sorted(np.unique(g).tolist(), key=str)
    if positive_label is None:
        positive_label = 1 if 1 in labels else labels[-1]
    elif positive_label not in labels:
        raise ValueError(f"positive_label {positive_label!r} not found in y_true")

    weights = np.zeros(n, dtype=float)
    weight_table: Dict[str, Dict[str, float]] = {}
    cell_count: Dict[str, Dict[str, int]] = {}
    empty: List[Tuple[str, str]] = []
    label_counts = {lab: int(np.sum(y == lab)) for lab in labels}

    for gv in group_values:
        g_mask = g == gv
        n_g = int(g_mask.sum())
        key = str(gv)
        weight_table[key] = {}
        cell_count[key] = {}
        for lab in labels:
            cell = g_mask & (y == lab)
            n_gy = int(cell.sum())
            cell_count[key][str(lab)] = n_gy
            if n_gy == 0:
                empty.append((key, str(lab)))
                weight_table[key][str(lab)] = float("nan")
                continue
            w = (n_g * label_counts[lab]) / (n * n_gy)
            weight_table[key][str(lab)] = float(w)
            weights[cell] = w

    pos = y == positive_label
    before: Dict[str, float] = {}
    after: Dict[str, float] = {}
    for gv in group_values:
        g_mask = g == gv
        before[str(gv)] = float(pos[g_mask].mean())
        w_g = weights[g_mask]
        total = float(w_g.sum())
        after[str(gv)] = float(w_g[pos[g_mask]].sum() / total) if total > 0 else float("nan")

    def _gap(rates: Dict[str, float]) -> float:
        vals = [v for v in rates.values() if v == v]
        return float(max(vals) - min(vals)) if vals else 0.0

    return ReweighingResult(
        n_samples=n,
        n_groups=len(group_values),
        labels=labels,
        weights=weights,
        weight_table=weight_table,
        cell_count=cell_count,
        base_rates_before=before,
        base_rates_after=after,
        base_rate_gap_before=_gap(before),
        base_rate_gap_after=_gap(after),
        positive_label=positive_label,
        empty_cells=empty,
    )


def fit_with_reweighing(estimator, X, y_true, groups, **fit_params):
    """
    Fit ``estimator`` with Kamiran-Calders sample weights.

    The estimator's ``fit`` must accept ``sample_weight`` (most scikit-learn
    classifiers do). Returns ``(fitted_estimator, ReweighingResult)``.
    """
    if "sample_weight" in fit_params:
        raise ValueError("sample_weight is computed by reweighing; do not pass it")
    result = compute_reweighing_weights(y_true, groups)
    estimator.fit(X, y_true, sample_weight=result.weights, **fit_params)
    return estimator, result
