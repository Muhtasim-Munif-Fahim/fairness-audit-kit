"""Multi-calibration audit for scored classifiers.

Ordinary calibration (ECE) averages over the whole population. A model can
look calibrated overall while being badly miscalibrated inside a sensitive
group, or inside a group×score slice — exactly the failure mode that
multi-calibration (Hébert-Johnson, Kim, Reingold & Rothblum, COLT 2018)
and multi-accuracy (Kim, Ghorbani & Zou, AIES 2019) target.

This module bins scores (uniform or quantile), splits each bin by sensitive
group, and reports:

* per-cell calibration gap ``|mean(score) - mean(y)|``
* the worst cell gap (multi-calibration residual)
* multi-accuracy: ``|E[y - score | group]|`` per group
* a structured result suitable for reports and thresholds

No model is retrained; this is an audit of an already-scored classifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .calibration import STRATEGIES, compute_reliability_bins

__all__ = [
    "MultiCalibrationCell",
    "MultiCalibrationResult",
    "compute_multi_calibration",
    "multi_accuracy",
    "render_multi_calibration_report",
]


@dataclass
class MultiCalibrationCell:
    """One (group, score-bin) cell of a multi-calibration table."""

    group: str
    bin_index: int
    lower: float
    upper: float
    n_samples: int
    mean_score: float
    mean_label: float
    gap: float

    def to_dict(self) -> Dict:
        return {
            "group": self.group,
            "bin_index": self.bin_index,
            "lower": self.lower,
            "upper": self.upper,
            "n_samples": self.n_samples,
            "mean_score": self.mean_score,
            "mean_label": self.mean_label,
            "gap": self.gap,
        }


@dataclass
class MultiCalibrationResult:
    """Multi-calibration / multi-accuracy audit summary."""

    n_samples: int
    n_bins: int
    strategy: str
    min_cell_size: int
    cells: List[MultiCalibrationCell] = field(default_factory=list)
    multi_calibration_error: float = 0.0  # max |gap| over cells meeting min size
    multi_accuracy_by_group: Dict[str, float] = field(default_factory=dict)
    max_multi_accuracy: float = 0.0
    overall_ece: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "n_samples": self.n_samples,
            "n_bins": self.n_bins,
            "strategy": self.strategy,
            "min_cell_size": self.min_cell_size,
            "multi_calibration_error": self.multi_calibration_error,
            "multi_accuracy_by_group": dict(self.multi_accuracy_by_group),
            "max_multi_accuracy": self.max_multi_accuracy,
            "overall_ece": self.overall_ece,
            "cells": [c.to_dict() for c in self.cells],
        }


def multi_accuracy(y_true, y_scores, groups) -> Dict[str, float]:
    """Return ``|E[y - score | g]|`` for each sensitive group ``g``."""
    y = np.asarray(y_true, dtype=float).ravel()
    s = np.asarray(y_scores, dtype=float).ravel()
    g = np.asarray(groups)
    if y.size != s.size or y.size != g.size:
        raise ValueError("y_true, y_scores, and groups must have the same length")
    if y.size == 0:
        raise ValueError("inputs must not be empty")
    out: Dict[str, float] = {}
    for label in sorted(set(map(str, g.tolist())), key=str):
        mask = g.astype(str) == label
        out[label] = float(abs(np.mean(y[mask] - s[mask])))
    return out


def compute_multi_calibration(
    y_true,
    y_scores,
    groups,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
    min_cell_size: int = 30,
) -> MultiCalibrationResult:
    """Audit multi-calibration of scores conditioned on group × score bin.

    Parameters
    ----------
    y_true, y_scores, groups:
        Parallel arrays of labels in ``{0,1}``, scores in ``[0,1]``, and
        sensitive-group labels.
    n_bins:
        Number of score bins (same semantics as :func:`compute_reliability_bins`).
    strategy:
        ``"uniform"`` or ``"quantile"``.
    min_cell_size:
        Cells with fewer samples are reported but ignored when taking the
        worst-gap multi-calibration residual (sparse cells are noisy).
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    if min_cell_size < 1:
        raise ValueError("min_cell_size must be at least 1")

    y = np.asarray(y_true, dtype=float).ravel()
    s = np.asarray(y_scores, dtype=float).ravel()
    g = np.asarray(groups)
    if y.size != s.size or y.size != g.size:
        raise ValueError("y_true, y_scores, and groups must have the same length")
    if y.size == 0:
        raise ValueError("inputs must not be empty")
    if np.any((s < 0) | (s > 1)) or not np.all(np.isfinite(s)):
        raise ValueError("y_scores must be finite and in [0, 1]")
    if not set(np.unique(y)).issubset({0.0, 1.0}):
        raise ValueError("y_true must be binary labels in {0, 1}")

    overall_bins = compute_reliability_bins(y, s, n_bins=n_bins, strategy=strategy)
    # Recover bin edges from the overall bins (contiguous).
    edges = [overall_bins[0].lower] + [b.upper for b in overall_bins]

    cells: List[MultiCalibrationCell] = []
    worst = 0.0
    g_str = g.astype(str)
    for label in sorted(set(g_str.tolist()), key=str):
        mask_g = g_str == label
        yg, sg = y[mask_g], s[mask_g]
        # Digitize with the global edges so groups share a score partition.
        idx = np.digitize(sg, edges[1:-1], right=False)
        for b in range(n_bins):
            m = idx == b
            n = int(m.sum())
            if n == 0:
                continue
            mean_s = float(sg[m].mean())
            mean_y = float(yg[m].mean())
            gap = abs(mean_s - mean_y)
            cells.append(
                MultiCalibrationCell(
                    group=label,
                    bin_index=b,
                    lower=float(edges[b]),
                    upper=float(edges[b + 1]),
                    n_samples=n,
                    mean_score=mean_s,
                    mean_label=mean_y,
                    gap=gap,
                )
            )
            if n >= min_cell_size:
                worst = max(worst, gap)

    ma = multi_accuracy(y, s, g_str)
    ece = float(sum(b.gap * b.weight for b in overall_bins))
    return MultiCalibrationResult(
        n_samples=int(y.size),
        n_bins=n_bins,
        strategy=strategy,
        min_cell_size=min_cell_size,
        cells=cells,
        multi_calibration_error=float(worst),
        multi_accuracy_by_group=ma,
        max_multi_accuracy=float(max(ma.values()) if ma else 0.0),
        overall_ece=ece,
    )


def render_multi_calibration_report(result: MultiCalibrationResult) -> str:
    """Plain-text summary of a multi-calibration audit."""
    lines = [
        "Multi-calibration audit",
        f"  n={result.n_samples}  bins={result.n_bins} ({result.strategy})  "
        f"min_cell_size={result.min_cell_size}",
        f"  overall ECE                 = {result.overall_ece:.4f}",
        f"  multi-calibration error     = {result.multi_calibration_error:.4f}",
        f"  max multi-accuracy          = {result.max_multi_accuracy:.4f}",
        "  multi-accuracy by group:",
    ]
    for g, v in sorted(result.multi_accuracy_by_group.items()):
        lines.append(f"    {g}: {v:.4f}")
    eligible = [c for c in result.cells if c.n_samples >= result.min_cell_size]
    if eligible:
        top = sorted(eligible, key=lambda c: c.gap, reverse=True)[:5]
        lines.append("  worst cells:")
        for c in top:
            lines.append(
                f"    group={c.group} bin={c.bin_index} "
                f"[{c.lower:.2f},{c.upper:.2f}] n={c.n_samples} gap={c.gap:.4f}"
            )
    return "\n".join(lines) + "\n"
