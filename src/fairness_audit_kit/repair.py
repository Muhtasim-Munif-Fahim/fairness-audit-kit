"""
Disparate impact remover: feature repair (Feldman et al., KDD 2015).

Reweighing (:mod:`fairness_audit_kit.reweighing`) leaves the data alone and
changes sample weights. The disparate impact remover instead *edits numeric
features* so that their distribution no longer reveals the sensitive group,
while keeping the order of values within each group.

For feature ``x`` and group ``g`` with empirical CDF ``F_g``, the target
("median") distribution has quantile function

    F_A^{-1}(u) = median_g F_g^{-1}(u)

and a value ``x`` in group ``g`` is repaired to

    x' = (1 - lambda) * x + lambda * F_A^{-1}(F_g(x))

``lambda = 1`` (full repair) maps every group onto the same distribution, so
a downstream model cannot recover the group from that feature alone;
``lambda = 0`` returns the data unchanged, and values in between interpolate
(Feldman et al.'s *geometric* partial repair). Ranks within a group are
preserved because ``F_A^{-1}`` and ``F_g`` are monotone, so the feature keeps
its predictive ordering.

The repair is fit on training data (per-group sorted values) and can be
applied to new rows, whose ``F_g(x)`` is evaluated against the training
values of their group. The two-sample Kolmogorov-Smirnov distance between
groups, before and after repair, measures how much group information each
feature still carries.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic ``sup_x |F_a(x) - F_b(x)|``."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    if a.size == 0 or b.size == 0:
        raise ValueError("both samples must be non-empty")
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / a.size
    fb = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(fa - fb)))


def max_pairwise_ks(values: np.ndarray, groups: np.ndarray) -> float:
    """Largest KS distance between any two groups' distributions of ``values``."""
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    levels = sorted(np.unique(groups).tolist(), key=str)
    worst = 0.0
    for i, gi in enumerate(levels):
        for gj in levels[i + 1:]:
            worst = max(worst, ks_distance(values[groups == gi], values[groups == gj]))
    return worst


@dataclass
class DisparateImpactRepairResult:
    """Diagnostics from fitting / applying :class:`DisparateImpactRemover`."""

    repair_level: float
    features: List[str]
    groups: List[str]
    group_sizes: Dict[str, int]
    ks_before: Dict[str, float]
    ks_after: Dict[str, float]
    mean_abs_change: Dict[str, float]
    rank_preserved: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "repair_level": self.repair_level,
            "features": list(self.features),
            "groups": list(self.groups),
            "group_sizes": dict(self.group_sizes),
            "ks_before": dict(self.ks_before),
            "ks_after": dict(self.ks_after),
            "mean_abs_change": dict(self.mean_abs_change),
            "rank_preserved": dict(self.rank_preserved),
        }


class DisparateImpactRemover:
    """
    Feldman et al. (2015) quantile repair of numeric features toward the
    per-quantile median of the group distributions.

    Args:
        repair_level: ``lambda`` in ``[0, 1]``; 1 is full repair.
        features: Columns to repair (default: every numeric column of the
            DataFrame passed to :meth:`fit`).
    """

    def __init__(self, repair_level: float = 1.0, features: Optional[Sequence[str]] = None):
        repair_level = float(repair_level)
        if not 0.0 <= repair_level <= 1.0:
            raise ValueError("repair_level must be in [0, 1]")
        self.repair_level = repair_level
        self.features = list(features) if features is not None else None

    @staticmethod
    def _as_frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        arr = np.asarray(X, dtype=float)
        if arr.ndim != 2:
            raise ValueError("X must be 2-D")
        return pd.DataFrame(arr, columns=[f"x{j}" for j in range(arr.shape[1])])

    def fit(self, X, groups) -> "DisparateImpactRemover":
        """Store each group's sorted training values for every repaired feature."""
        df = self._as_frame(X)
        g = np.asarray(groups)
        if g.ndim != 1 or len(g) != len(df):
            raise ValueError("groups must be 1-D with one entry per row of X")
        if len(df) == 0:
            raise ValueError("need at least one row")
        if self.features is None:
            feats = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        else:
            missing = [c for c in self.features if c not in df.columns]
            if missing:
                raise ValueError(f"features not found in X: {missing}")
            feats = list(self.features)
        if not feats:
            raise ValueError("no numeric features to repair")
        for c in feats:
            if df[c].isna().any():
                raise ValueError(f"feature {c!r} contains missing values; impute first")
        self.features_ = feats
        self.groups_ = sorted(np.unique(g).tolist(), key=str)
        if len(self.groups_) < 2:
            raise ValueError("need at least two groups")
        self.sorted_values_: Dict[str, Dict[Any, np.ndarray]] = {}
        for c in feats:
            col = df[c].to_numpy(dtype=float)
            self.sorted_values_[c] = {gv: np.sort(col[g == gv]) for gv in self.groups_}
        self.group_sizes_ = {str(gv): int(np.sum(g == gv)) for gv in self.groups_}
        return self

    def _check_fitted(self):
        if not hasattr(self, "features_"):
            raise RuntimeError("DisparateImpactRemover is not fitted; call fit() first")

    def _repair_column(self, c: str, col: np.ndarray, g: np.ndarray, lam: float) -> np.ndarray:
        out = col.copy()
        ref = self.sorted_values_[c]
        for gv in np.unique(g):
            if gv not in ref:
                raise ValueError(f"group {gv!r} was not seen during fit")
            mask = g == gv
            train = ref[gv]
            x = col[mask]
            # mid-rank empirical CDF against the group's training values
            u = (np.searchsorted(train, x, side="left") + np.searchsorted(train, x, side="right")) / (
                2.0 * train.size
            )
            u = np.clip(u, 0.0, 1.0)
            targets = np.median(
                np.vstack([np.quantile(ref[h], u) for h in self.groups_]), axis=0
            )
            out[mask] = (1.0 - lam) * x + lam * targets
        return out

    def transform(self, X, groups, repair_level: Optional[float] = None) -> pd.DataFrame:
        """Return a repaired copy of ``X`` (non-repaired columns unchanged).

        ``repair_level`` overrides the constructor value for this call.
        """
        self._check_fitted()
        lam = self.repair_level if repair_level is None else float(repair_level)
        if not 0.0 <= lam <= 1.0:
            raise ValueError("repair_level must be in [0, 1]")
        df = self._as_frame(X)
        g = np.asarray(groups)
        if g.ndim != 1 or len(g) != len(df):
            raise ValueError("groups must be 1-D with one entry per row of X")
        out = df.copy()
        for c in self.features_:
            if c not in df.columns:
                raise ValueError(f"feature {c!r} missing from X")
            out[c] = self._repair_column(c, df[c].to_numpy(dtype=float), g, lam)
        return out

    def fit_transform(self, X, groups) -> pd.DataFrame:
        return self.fit(X, groups).transform(X, groups)

    def diagnose(self, X, groups, repaired: Optional[pd.DataFrame] = None) -> DisparateImpactRepairResult:
        """KS distances between groups before / after repair, plus mean change."""
        self._check_fitted()
        df = self._as_frame(X)
        g = np.asarray(groups)
        rep = self.transform(df, g) if repaired is None else repaired
        ks_before, ks_after, change, ranks = {}, {}, {}, {}
        for c in self.features_:
            a = df[c].to_numpy(dtype=float)
            b = rep[c].to_numpy(dtype=float)
            ks_before[c] = max_pairwise_ks(a, g)
            ks_after[c] = max_pairwise_ks(b, g)
            change[c] = float(np.mean(np.abs(b - a)))
            ok = True
            for gv in np.unique(g):
                m = g == gv
                order = np.argsort(a[m], kind="stable")
                ok &= bool(np.all(np.diff(b[m][order]) >= -1e-9))
            ranks[c] = ok
        return DisparateImpactRepairResult(
            repair_level=self.repair_level,
            features=list(self.features_),
            groups=[str(v) for v in self.groups_],
            group_sizes=dict(self.group_sizes_),
            ks_before=ks_before,
            ks_after=ks_after,
            mean_abs_change=change,
            rank_preserved=ranks,
        )


def repair_disparate_impact(X, groups, repair_level: float = 1.0, features=None):
    """Fit and apply the repair in one call; returns ``(repaired, result)``."""
    remover = DisparateImpactRemover(repair_level=repair_level, features=features)
    repaired = remover.fit_transform(X, groups)
    return repaired, remover.diagnose(X, groups, repaired=repaired)
