"""
Differential fairness (ε-DF) for intersectional groups (Foulds et al., 2020).

A mechanism ``M(x)`` is ε-differentially fair with respect to the
intersectional groups ``s`` formed by one or more protected attributes when,
for every outcome ``y`` and every pair of groups ``s_i, s_j`` that occur,

    e^{-ε} ≤ P(M(x) = y | s_i) / P(M(x) = y | s_j) ≤ e^{ε}

so ε is the largest absolute log-ratio of outcome probabilities between any
two groups. ``e^{-ε}`` is a worst-case disparate-impact ratio (the "80% rule"
holds when ``ε ≤ -log 0.8 ≈ 0.223``), and the definition covers *every*
outcome, including negative ones, and every intersection at once.

Small intersections make raw frequencies noisy (and a zero count makes ε
infinite), so probabilities are estimated with Dirichlet smoothing,

    P(y | s) = (N_{y,s} + α) / (N_s + |Y| α),

as in the paper's smoothed empirical estimator (``α = 0`` gives raw
frequencies).

Foulds et al. prove that ε-DF on the full intersection implies ε-DF for any
subset of the protected attributes; ``subset_epsilons`` reports ε for every
non-empty subset (e.g. race alone, gender alone, race×gender) so the
intersectional disparity can be compared with the marginal ones.

When the observed labels ``y_true`` are given, the same ε is computed for
the data and ``bias_amplification = ε(model) − ε(data)`` measures how much the
classifier increases (positive) or reduces (negative) the disparity already
present in the labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

SensitiveInput = Union[np.ndarray, Sequence, Mapping[str, Sequence]]

SEPARATOR = " & "


@dataclass
class DifferentialFairnessResult:
    """ε-differential fairness of predictions across intersectional groups."""

    epsilon: float
    disparate_impact_ratio: float
    worst_outcome: object
    worst_pair: Tuple[str, str]
    per_outcome_epsilon: Dict[object, float]
    group_probabilities: Dict[str, Dict[object, float]]
    group_size: Dict[str, int]
    subset_epsilons: Dict[str, float]
    alpha: float
    outcomes: Tuple[object, ...]
    data_epsilon: Optional[float] = None
    bias_amplification: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    attributes: Tuple[str, ...] = field(default_factory=tuple)

    def satisfies(self, epsilon: float) -> bool:
        """Whether the predictions are ``epsilon``-differentially fair."""
        return bool(self.epsilon <= float(epsilon))

    def to_dict(self) -> Dict:
        return {
            "epsilon": self.epsilon,
            "disparate_impact_ratio": self.disparate_impact_ratio,
            "worst_outcome": _py(self.worst_outcome),
            "worst_pair": list(self.worst_pair),
            "per_outcome_epsilon": {str(_py(k)): v for k, v in self.per_outcome_epsilon.items()},
            "group_probabilities": {
                g: {str(_py(k)): v for k, v in probs.items()} for g, probs in self.group_probabilities.items()
            },
            "group_size": self.group_size,
            "subset_epsilons": self.subset_epsilons,
            "alpha": self.alpha,
            "outcomes": [_py(o) for o in self.outcomes],
            "data_epsilon": self.data_epsilon,
            "bias_amplification": self.bias_amplification,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "attributes": list(self.attributes),
        }


def _py(value):
    return value.item() if hasattr(value, "item") else value


def _attributes(sensitive: SensitiveInput, n: int) -> Tuple[Tuple[str, ...], List[np.ndarray]]:
    if hasattr(sensitive, "columns"):  # pandas DataFrame
        names = tuple(str(c) for c in sensitive.columns)
        cols = [np.asarray(sensitive[c]) for c in sensitive.columns]
    elif isinstance(sensitive, Mapping):
        names = tuple(str(k) for k in sensitive)
        cols = [np.asarray(v) for v in sensitive.values()]
    else:
        arr = np.asarray(sensitive)
        if arr.ndim == 1:
            names, cols = ("group",), [arr]
        elif arr.ndim == 2:
            names = tuple("attr%d" % j for j in range(arr.shape[1]))
            cols = [arr[:, j] for j in range(arr.shape[1])]
        else:
            raise ValueError("sensitive must be 1-D, 2-D, a mapping or a DataFrame")
    if not cols:
        raise ValueError("at least one sensitive attribute is required")
    for c in cols:
        if c.shape != (n,):
            raise ValueError("each sensitive attribute must have one value per prediction")
    return names, cols


def _labels(cols: List[np.ndarray], idx: Sequence[int]) -> np.ndarray:
    if len(idx) == 1:
        return np.array([str(v) for v in cols[idx[0]]], dtype=object)
    return np.array(
        [SEPARATOR.join(str(cols[j][i]) for j in idx) for i in range(cols[0].shape[0])],
        dtype=object,
    )


def _probabilities(y: np.ndarray, groups: np.ndarray, outcomes: Sequence, alpha: float):
    levels = sorted(set(groups.tolist()), key=str)
    k = len(outcomes)
    probs = np.empty((len(levels), k))
    sizes = {}
    for gi, g in enumerate(levels):
        mask = groups == g
        n_g = int(mask.sum())
        sizes[g] = n_g
        y_g = y[mask]
        for oi, o in enumerate(outcomes):
            probs[gi, oi] = (np.sum(y_g == o) + alpha) / (n_g + k * alpha)
    return levels, probs, sizes


def _epsilon(probs: np.ndarray) -> Tuple[float, int, int, int, np.ndarray]:
    """Return ε, the worst outcome index and the worst (high, low) group pair."""
    with np.errstate(divide="ignore"):
        logp = np.log(probs)
    per_outcome = np.empty(probs.shape[1])
    best = (-1.0, 0, 0, 0)
    for oi in range(probs.shape[1]):
        col = logp[:, oi]
        hi, lo = int(np.argmax(col)), int(np.argmin(col))
        if np.isneginf(col[lo]) and np.isneginf(col[hi]):
            eps = 0.0
        else:
            eps = float(col[hi] - col[lo])
        per_outcome[oi] = eps
        if eps > best[0]:
            best = (eps, oi, hi, lo)
    return best[0], best[1], best[2], best[3], per_outcome


def _eps_only(y, groups, outcomes, alpha) -> float:
    _, probs, _ = _probabilities(y, groups, outcomes, alpha)
    return _epsilon(probs)[0] if probs.shape[0] > 1 else 0.0


def compute_differential_fairness(
    y_pred,
    sensitive: SensitiveInput,
    *,
    y_true=None,
    alpha: float = 1.0,
    outcomes: Optional[Sequence] = None,
    n_bootstrap: int = 0,
    confidence: float = 0.95,
    seed: Optional[int] = None,
) -> DifferentialFairnessResult:
    """
    Smoothed empirical ε-differential fairness of ``y_pred``.

    Parameters
    ----------
    y_pred : array-like (n,)
        Discrete predictions (binary or multi-class labels).
    sensitive : array-like (n,), (n, m), mapping of name -> array, or DataFrame
        One or more protected attributes; groups are their intersections.
    y_true : array-like (n,), optional
        Observed labels; adds ``data_epsilon`` and ``bias_amplification``.
    alpha : float
        Dirichlet smoothing per outcome (``0`` = raw frequencies).
    outcomes : sequence, optional
        Outcome space; defaults to the sorted union of observed labels.
    n_bootstrap : int
        Row-bootstrap replicates for a percentile CI of ε.
    """
    y_pred = np.asarray(y_pred)
    if y_pred.ndim != 1 or y_pred.size == 0:
        raise ValueError("y_pred must be a non-empty 1-D array")
    n = y_pred.shape[0]
    alpha = float(alpha)
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    names, cols = _attributes(sensitive, n)
    if y_true is not None:
        y_true = np.asarray(y_true)
        if y_true.shape != (n,):
            raise ValueError("y_true must have the same length as y_pred")
    if outcomes is None:
        pool = set(y_pred.tolist()) | (set(y_true.tolist()) if y_true is not None else set())
        outcomes = tuple(sorted(pool, key=lambda v: (str(type(v)), v)))
    else:
        outcomes = tuple(outcomes)
        if len(outcomes) < 2:
            raise ValueError("outcomes must contain at least two values")
        unknown = set(y_pred.tolist()) - set(outcomes)
        if unknown:
            raise ValueError("y_pred contains values outside outcomes: %r" % sorted(unknown, key=str))
    if len(outcomes) < 2:
        outcomes = tuple(outcomes) + ("__other__",)

    groups = _labels(cols, list(range(len(cols))))
    levels, probs, sizes = _probabilities(y_pred, groups, outcomes, alpha)
    if len(levels) < 2:
        raise ValueError("need at least two groups")
    eps, oi, hi, lo, per_outcome = _epsilon(probs)

    subset_eps: Dict[str, float] = {}
    for r in range(1, len(cols) + 1):
        for idx in combinations(range(len(cols)), r):
            key = " x ".join(names[j] for j in idx)
            subset_eps[key] = _eps_only(y_pred, _labels(cols, idx), outcomes, alpha)

    data_eps = amp = None
    if y_true is not None:
        data_eps = _eps_only(y_true, groups, outcomes, alpha)
        amp = eps - data_eps

    ci_low = ci_high = None
    if n_bootstrap:
        rng = np.random.default_rng(seed)
        boots = np.empty(int(n_bootstrap))
        for b in range(int(n_bootstrap)):
            idx = rng.integers(0, n, n)
            boots[b] = _eps_only(y_pred[idx], groups[idx], outcomes, alpha)
        tail = (1.0 - float(confidence)) / 2.0 * 100.0
        ci_low = float(np.percentile(boots, tail))
        ci_high = float(np.percentile(boots, 100.0 - tail))

    return DifferentialFairnessResult(
        epsilon=eps,
        disparate_impact_ratio=float(np.exp(-eps)),
        worst_outcome=_py(outcomes[oi]),
        worst_pair=(str(levels[hi]), str(levels[lo])),
        per_outcome_epsilon={_py(o): float(e) for o, e in zip(outcomes, per_outcome)},
        group_probabilities={
            str(g): {_py(o): float(p) for o, p in zip(outcomes, probs[gi])} for gi, g in enumerate(levels)
        },
        group_size={str(g): int(s) for g, s in sizes.items()},
        subset_epsilons=subset_eps,
        alpha=alpha,
        outcomes=tuple(_py(o) for o in outcomes),
        data_epsilon=data_eps,
        bias_amplification=amp,
        ci_low=ci_low,
        ci_high=ci_high,
        attributes=names,
    )


def render_differential_fairness_report(result: DifferentialFairnessResult) -> str:
    """Markdown summary of a :class:`DifferentialFairnessResult`."""
    lines = [
        "# Differential Fairness (ε-DF)",
        "",
        f"- **ε**: {result.epsilon:.4f} (worst-case ratio e^-ε = {result.disparate_impact_ratio:.3f})",
        f"- **Worst outcome**: {result.worst_outcome!r} between `{result.worst_pair[0]}` and `{result.worst_pair[1]}`",
        f"- **Smoothing α**: {result.alpha:g}",
    ]
    if result.ci_low is not None:
        lines.append(f"- **Bootstrap CI**: [{result.ci_low:.4f}, {result.ci_high:.4f}]")
    if result.data_epsilon is not None:
        lines.append(
            f"- **ε of labels**: {result.data_epsilon:.4f}; bias amplification {result.bias_amplification:+.4f}"
        )
    lines += ["", "| Attributes | ε |", "|---|---|"]
    for key, value in result.subset_epsilons.items():
        lines.append(f"| {key} | {value:.4f} |")
    outcomes = list(result.outcomes)
    lines += ["", "| Group | n | " + " | ".join(f"P({o})" for o in outcomes) + " |"]
    lines.append("|---|---|" + "---|" * len(outcomes))
    for g, probs in result.group_probabilities.items():
        lines.append(
            f"| {g} | {result.group_size[g]} | " + " | ".join(f"{probs[o]:.3f}" for o in outcomes) + " |"
        )
    return "\n".join(lines) + "\n"
