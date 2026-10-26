"""
Individual fairness: kNN consistency and Lipschitz violations.

Group metrics compare averages across sensitive groups. *Individual*
fairness (Dwork et al., ITCS 2012) asks that **similar individuals are
treated similarly**. Two standard audits:

* **Consistency** (Zemel et al., ICML 2013):

      consistency = 1 - (1 / n) * sum_i | yhat_i - mean_{j in kNN(x_i)} yhat_j |

  computed in the non-sensitive feature space. 1 means every individual
  gets the same decision as their nearest neighbours. ``include_self=True``
  reproduces AIF360's ``consistency_score``, whose neighbour set includes
  the point itself.

* **Lipschitz violations**: for every k-nearest-neighbour pair, flag
  ``|f(x_i) - f(x_j)| > L * d(x_i, x_j)``. The violation rate and the
  worst ratio ``|f(x_i) - f(x_j)| / d(x_i, x_j)`` show how far the model is
  from being ``L``-Lipschitz on the data.

Both are reported overall and per sensitive group (using each
individual's neighbourhood in the full population).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class LipschitzViolation:
    """One neighbour pair that breaks the Lipschitz condition."""

    i: int
    j: int
    distance: float
    output_gap: float
    ratio: float

    def to_dict(self) -> Dict:
        return {
            "i": self.i,
            "j": self.j,
            "distance": self.distance,
            "output_gap": self.output_gap,
            "ratio": self.ratio,
        }


@dataclass
class IndividualFairnessResult:
    """Consistency and Lipschitz audit of a model's outputs."""

    n_samples: int
    n_neighbors: int
    include_self: bool
    standardized: bool
    consistency: float
    group_consistency: Dict[str, float]
    group_size: Dict[str, int]
    lipschitz_constant: Optional[float] = None
    n_pairs: int = 0
    violation_rate: Optional[float] = None
    group_violation_rate: Dict[str, float] = field(default_factory=dict)
    max_ratio: Optional[float] = None
    worst_violations: List[LipschitzViolation] = field(default_factory=list)

    @property
    def consistency_gap(self) -> float:
        """max - min group consistency (0 when there is one group)."""
        if not self.group_consistency:
            return 0.0
        values = list(self.group_consistency.values())
        return float(max(values) - min(values))

    def to_dict(self) -> Dict:
        return {
            "n_samples": self.n_samples,
            "n_neighbors": self.n_neighbors,
            "include_self": self.include_self,
            "standardized": self.standardized,
            "consistency": self.consistency,
            "group_consistency": dict(self.group_consistency),
            "group_size": dict(self.group_size),
            "consistency_gap": self.consistency_gap,
            "lipschitz_constant": self.lipschitz_constant,
            "n_pairs": self.n_pairs,
            "violation_rate": self.violation_rate,
            "group_violation_rate": dict(self.group_violation_rate),
            "max_ratio": self.max_ratio,
            "worst_violations": [v.to_dict() for v in self.worst_violations],
        }


def _prepare_features(X, standardize: bool) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X must be a non-empty 2-D array")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must not contain NaN or inf")
    if standardize:
        std = X.std(axis=0)
        std[std == 0] = 1.0
        X = (X - X.mean(axis=0)) / std
    return X


def _neighbours(X: np.ndarray, n_neighbors: int, include_self: bool):
    n = X.shape[0]
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be >= 1")
    k = n_neighbors if include_self else n_neighbors + 1
    if k > n:
        raise ValueError(
            f"n_neighbors={n_neighbors} is too large for {n} samples"
        )
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    dist, idx = nn.kneighbors(X)
    if include_self:
        return dist, idx
    # Drop each point itself. With duplicate rows the self index may not be
    # in column 0, so remove it explicitly (falling back to the last column).
    out_idx = np.empty((n, n_neighbors), dtype=int)
    out_dist = np.empty((n, n_neighbors))
    for i in range(n):
        row = idx[i]
        pos = np.flatnonzero(row == i)
        drop = pos[0] if pos.size else k - 1
        keep = np.delete(np.arange(k), drop)
        out_idx[i] = row[keep]
        out_dist[i] = dist[i][keep]
    return out_dist, out_idx


def consistency_score(
    X,
    y_pred,
    n_neighbors: int = 5,
    *,
    include_self: bool = False,
    standardize: bool = True,
) -> float:
    """Zemel et al. kNN consistency of predictions ``y_pred`` in feature space ``X``."""
    Xp = _prepare_features(X, standardize)
    y = np.asarray(y_pred, dtype=float).ravel()
    if y.size != Xp.shape[0]:
        raise ValueError("X and y_pred must have the same number of rows")
    _, idx = _neighbours(Xp, n_neighbors, include_self)
    return float(1.0 - np.mean(np.abs(y - y[idx].mean(axis=1))))


def compute_individual_fairness(
    X,
    y_pred,
    sensitive_attr=None,
    *,
    n_neighbors: int = 5,
    include_self: bool = False,
    standardize: bool = True,
    lipschitz_constant: Optional[float] = None,
    top_violations: int = 10,
) -> IndividualFairnessResult:
    """Consistency (overall and per group) plus an optional Lipschitz audit.

    ``y_pred`` may be hard labels or scores. ``X`` should hold the
    *non-sensitive* features the similarity metric is defined on.
    """
    Xp = _prepare_features(X, standardize)
    y = np.asarray(y_pred, dtype=float).ravel()
    n = Xp.shape[0]
    if y.size != n:
        raise ValueError("X and y_pred must have the same number of rows")
    if not np.all(np.isfinite(y)):
        raise ValueError("y_pred must be finite")
    groups = (
        np.full(n, "all", dtype=object)
        if sensitive_attr is None
        else np.asarray(sensitive_attr, dtype=object).ravel()
    )
    if groups.size != n:
        raise ValueError("sensitive_attr must have the same length as y_pred")

    dist, idx = _neighbours(Xp, n_neighbors, include_self)
    per_point = np.abs(y - y[idx].mean(axis=1))
    group_consistency: Dict[str, float] = {}
    group_size: Dict[str, int] = {}
    for g in sorted(set(groups.tolist()), key=str):
        mask = groups == g
        group_consistency[str(g)] = float(1.0 - per_point[mask].mean())
        group_size[str(g)] = int(mask.sum())

    result = IndividualFairnessResult(
        n_samples=n,
        n_neighbors=n_neighbors,
        include_self=include_self,
        standardized=standardize,
        consistency=float(1.0 - per_point.mean()),
        group_consistency=group_consistency,
        group_size=group_size,
    )
    if lipschitz_constant is None:
        return result
    if lipschitz_constant < 0:
        raise ValueError("lipschitz_constant must be non-negative")

    # Unique unordered neighbour pairs (i, j), i != j.
    pairs: Dict[tuple, float] = {}
    for i in range(n):
        for d, j in zip(dist[i], idx[i]):
            if i == j:
                continue
            key = (min(i, int(j)), max(i, int(j)))
            pairs.setdefault(key, float(d))
    if not pairs:
        result.lipschitz_constant = float(lipschitz_constant)
        result.violation_rate = 0.0
        return result
    keys = np.array(list(pairs.keys()), dtype=int)
    d = np.array(list(pairs.values()))
    gap = np.abs(y[keys[:, 0]] - y[keys[:, 1]])
    tol = 1e-12
    violated = gap > lipschitz_constant * d + tol
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(d > 0, gap / np.where(d > 0, d, 1.0), np.where(gap > tol, np.inf, 0.0))

    group_rate: Dict[str, float] = {}
    for g in group_consistency:
        involved = (groups[keys[:, 0]].astype(str) == g) | (groups[keys[:, 1]].astype(str) == g)
        group_rate[g] = float(violated[involved].mean()) if involved.any() else 0.0

    order = np.argsort(-ratio, kind="stable")
    worst = [
        LipschitzViolation(
            i=int(keys[o, 0]),
            j=int(keys[o, 1]),
            distance=float(d[o]),
            output_gap=float(gap[o]),
            ratio=float(ratio[o]),
        )
        for o in order
        if violated[o]
    ][: max(int(top_violations), 0)]

    result.lipschitz_constant = float(lipschitz_constant)
    result.n_pairs = int(keys.shape[0])
    result.violation_rate = float(violated.mean())
    result.group_violation_rate = group_rate
    result.max_ratio = float(ratio.max())
    result.worst_violations = worst
    return result


def render_individual_fairness_report(
    result: IndividualFairnessResult,
    title: str = "Individual Fairness Report",
) -> str:
    """Markdown report for consistency and Lipschitz violations."""
    lines = [f"# {title}", ""]
    lines.append(f"- Samples: **{result.n_samples}**")
    lines.append(
        f"- Neighbours: **{result.n_neighbors}**"
        + (" (including self, AIF360 convention)" if result.include_self else "")
    )
    lines.append(f"- Features standardized: **{'yes' if result.standardized else 'no'}**")
    lines.append(f"- Consistency (1 = similar individuals treated alike): **{result.consistency:.4f}**")
    lines.append(f"- Group consistency gap (max−min): **{result.consistency_gap:.4f}**")
    if result.lipschitz_constant is not None:
        lines.append(f"- Lipschitz constant L: **{result.lipschitz_constant:g}**")
        lines.append(
            f"- Neighbour pairs violating |f(x)−f(x')| ≤ L·d(x, x'): "
            f"**{result.violation_rate:.4f}** of {result.n_pairs}"
        )
        if result.max_ratio is not None:
            lines.append(f"- Worst observed ratio |Δf| / d: **{result.max_ratio:.4g}**")
    lines += ["", "## Per-group", ""]
    if result.lipschitz_constant is not None:
        lines.append("| Group | Size | Consistency | Violation rate |")
        lines.append("|---|---:|---:|---:|")
        for g, c in result.group_consistency.items():
            lines.append(
                f"| {g} | {result.group_size[g]} | {c:.4f} | "
                f"{result.group_violation_rate.get(g, 0.0):.4f} |"
            )
    else:
        lines.append("| Group | Size | Consistency |")
        lines.append("|---|---:|---:|")
        for g, c in result.group_consistency.items():
            lines.append(f"| {g} | {result.group_size[g]} | {c:.4f} |")
    if result.worst_violations:
        lines += ["", "## Worst Lipschitz violations", ""]
        lines.append("| i | j | Distance | Output gap | Ratio |")
        lines.append("|---:|---:|---:|---:|---:|")
        for v in result.worst_violations:
            lines.append(
                f"| {v.i} | {v.j} | {v.distance:.4g} | {v.output_gap:.4g} | {v.ratio:.4g} |"
            )
    return "\n".join(lines) + "\n"
