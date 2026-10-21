"""
Reject option classification (Kamiran, Karim and Zhang, ICDM 2012).

A post-processing method for a *scored* binary classifier. Decisions far
from the classification threshold are left alone. Inside a band of width
``2θ`` around the threshold ``τ`` (the "critical region", where the model is
least certain), the unprivileged group gets the favourable label and the
privileged group gets the unfavourable label::

    ŷ = 1[s > τ]                       outside   τ - θ < s ≤ τ + θ
    ŷ = 1 (unprivileged), 0 (privileged)  inside

``τ`` and ``θ`` are found by a grid search. Following AIF360's
``RejectOptionClassification``, ``τ`` runs over ``n_thresholds`` values in
``[low, high]`` and, for each ``τ``, ``θ`` runs over ``n_margins`` values in
``[0, min(τ, 1 - τ)]``. The pair with the highest balanced accuracy whose
fairness metric lies in ``[-bound, bound]`` is kept.

Supported metrics (all are unprivileged minus privileged):

* ``statistical_parity``: ``P(ŷ=1 | U) - P(ŷ=1 | P)``
* ``average_odds``: ``½[(FPR_U - FPR_P) + (TPR_U - TPR_P)]``
* ``equal_opportunity``: ``TPR_U - TPR_P``

The method is deterministic and needs only the scores and the group, not
the features or the model. It does use the group at decision time, which
is disparate treatment and may be legally restricted in some domains.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from fairness_audit_kit.odds_parity import _as_binary_labels

ROC_METRICS = ("statistical_parity", "average_odds", "equal_opportunity")


def _rate(num: np.ndarray, den: float) -> np.ndarray:
    return num / den if den > 0 else np.zeros_like(num, dtype=float)


def _metrics(y: np.ndarray, priv: np.ndarray, pred: np.ndarray) -> Dict[str, np.ndarray]:
    """Vectorised metrics; ``pred`` has shape (n,) or (n, m)."""
    P = pred if pred.ndim == 2 else pred[:, None]
    pos, neg = y == 1, y == 0
    unpriv = ~priv

    def tpr(mask):
        return _rate(P[pos & mask].sum(axis=0), float((pos & mask).sum()))

    def fpr(mask):
        return _rate(P[neg & mask].sum(axis=0), float((neg & mask).sum()))

    tpr_all = _rate(P[pos].sum(axis=0), float(pos.sum()))
    tnr_all = 1.0 - _rate(P[neg].sum(axis=0), float(neg.sum()))
    sel_u = _rate(P[unpriv].sum(axis=0), float(unpriv.sum()))
    sel_p = _rate(P[priv].sum(axis=0), float(priv.sum()))
    tpr_u, tpr_p = tpr(unpriv), tpr(priv)
    fpr_u, fpr_p = fpr(unpriv), fpr(priv)
    out = {
        "balanced_accuracy": 0.5 * (tpr_all + tnr_all),
        "accuracy": (P == y[:, None]).mean(axis=0),
        "statistical_parity": sel_u - sel_p,
        "average_odds": 0.5 * ((fpr_u - fpr_p) + (tpr_u - tpr_p)),
        "equal_opportunity": tpr_u - tpr_p,
        "selection_rate_unprivileged": sel_u,
        "selection_rate_privileged": sel_p,
    }
    if pred.ndim == 1:
        out = {k: float(v[0]) for k, v in out.items()}
    return out


def reject_option_predict(scores, privileged_mask, threshold: float, margin: float) -> np.ndarray:
    """Apply a fitted ROC rule: flip decisions inside ``(τ-θ, τ+θ]`` by group."""
    s = np.asarray(scores, dtype=float).ravel()
    priv = np.asarray(privileged_mask, dtype=bool).ravel()
    if s.shape != priv.shape:
        raise ValueError("scores and groups must have the same length")
    pred = (s > threshold).astype(int)
    crit = (s <= threshold + margin) & (s > threshold - margin)
    pred[crit & priv] = 0
    pred[crit & ~priv] = 1
    return pred


@dataclass
class RejectOptionResult:
    """Chosen threshold / margin and fairness-accuracy before and after."""

    metric: str
    bound: float
    threshold: float
    margin: float
    feasible: bool
    n_samples: int
    n_critical: int
    privileged: object
    before: Dict[str, float]
    after: Dict[str, float]
    baseline_threshold: float = 0.5
    grid_size: int = 0
    frontier: list = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "metric": self.metric,
            "bound": self.bound,
            "threshold": self.threshold,
            "margin": self.margin,
            "feasible": self.feasible,
            "n_samples": self.n_samples,
            "n_critical": self.n_critical,
            "privileged": self.privileged,
            "baseline_threshold": self.baseline_threshold,
            "grid_size": self.grid_size,
            "before": self.before,
            "after": self.after,
        }


class RejectOptionClassifier:
    """Grid-searched reject option classification for a binary sensitive attribute.

    Parameters
    ----------
    privileged:
        Value of the group column that marks the privileged group; every
        other value is treated as unprivileged.
    metric:
        One of ``"statistical_parity"``, ``"average_odds"``, ``"equal_opportunity"``.
    bound:
        Feasible pairs satisfy ``|metric| <= bound`` (AIF360 default 0.05).
    low, high, n_thresholds, n_margins:
        Threshold grid ``linspace(low, high, n_thresholds)``; for each τ the
        margins are ``linspace(0, min(τ, 1-τ), n_margins)``.
    baseline_threshold:
        Threshold used for the "before" metrics (plain ``s > τ0``).

    If no pair is feasible a ``RuntimeWarning`` is issued and the pair with
    the smallest ``|metric|`` (ties: higher balanced accuracy) is used.
    """

    def __init__(
        self,
        privileged,
        metric: str = "statistical_parity",
        bound: float = 0.05,
        low: float = 0.01,
        high: float = 0.99,
        n_thresholds: int = 100,
        n_margins: int = 50,
        baseline_threshold: float = 0.5,
    ):
        if metric not in ROC_METRICS:
            raise ValueError(f"metric must be one of {ROC_METRICS}")
        if not (bound >= 0):
            raise ValueError("bound must be non-negative")
        if not (0.0 <= low <= high <= 1.0):
            raise ValueError("need 0 <= low <= high <= 1")
        if int(n_thresholds) < 1 or int(n_margins) < 1:
            raise ValueError("n_thresholds and n_margins must be positive")
        self.privileged = privileged
        self.metric = metric
        self.bound = float(bound)
        self.low = float(low)
        self.high = float(high)
        self.n_thresholds = int(n_thresholds)
        self.n_margins = int(n_margins)
        self.baseline_threshold = float(baseline_threshold)

    def _priv_mask(self, groups) -> np.ndarray:
        g = np.asarray(groups).ravel()
        return np.array([str(v) == str(self.privileged) for v in g], dtype=bool)

    def fit(self, y_true, groups, y_scores) -> "RejectOptionClassifier":
        y = _as_binary_labels("y_true", y_true)
        s = np.asarray(y_scores, dtype=float).ravel()
        if s.shape != y.shape:
            raise ValueError("y_true and y_scores must have the same length")
        if not np.all(np.isfinite(s)):
            raise ValueError("y_scores must be finite")
        priv = self._priv_mask(groups)
        if priv.shape != y.shape:
            raise ValueError("groups must have one value per sample")
        if priv.all() or not priv.any():
            raise ValueError("need samples from both the privileged and unprivileged groups")

        thr_list, mar_list, ba_list, fm_list = [], [], [], []
        for tau in np.linspace(self.low, self.high, self.n_thresholds):
            hi = tau if tau <= 0.5 else 1.0 - tau
            margins = np.linspace(0.0, hi, self.n_margins)
            base = s[:, None] > tau
            crit = (s[:, None] <= tau + margins[None, :]) & (s[:, None] > tau - margins[None, :])
            pred = np.where(crit, ~priv[:, None], base).astype(int)
            m = _metrics(y, priv, pred)
            thr_list.append(np.full(margins.size, tau))
            mar_list.append(margins)
            ba_list.append(m["balanced_accuracy"])
            fm_list.append(m[self.metric])
        thr = np.concatenate(thr_list)
        mar = np.concatenate(mar_list)
        ba = np.concatenate(ba_list)
        fm = np.concatenate(fm_list)

        feasible = (fm >= -self.bound) & (fm <= self.bound)
        if feasible.any():
            idx = np.flatnonzero(feasible)
            best = idx[np.argmax(ba[idx])]  # first maximum, as in AIF360
        else:
            warnings.warn(
                "Unable to satisfy the fairness bound; using the smallest |metric|",
                RuntimeWarning,
                stacklevel=2,
            )
            best = np.lexsort((-ba, np.abs(fm)))[0]
        self.threshold_ = float(thr[best])
        self.margin_ = float(mar[best])
        self.feasible_ = bool(feasible.any())

        before = _metrics(y, priv, (s > self.baseline_threshold).astype(int))
        pred = reject_option_predict(s, priv, self.threshold_, self.margin_)
        after = _metrics(y, priv, pred)
        # balanced-accuracy / |metric| Pareto frontier over the grid (for plots)
        order = np.lexsort((-ba, np.abs(fm)))
        frontier, best_ba = [], -np.inf
        for i in order:
            if ba[i] > best_ba + 1e-12:
                frontier.append(
                    {"threshold": float(thr[i]), "margin": float(mar[i]),
                     "abs_metric": float(abs(fm[i])), "balanced_accuracy": float(ba[i])}
                )
                best_ba = ba[i]
        crit = (s <= self.threshold_ + self.margin_) & (s > self.threshold_ - self.margin_)
        self.result_ = RejectOptionResult(
            metric=self.metric,
            bound=self.bound,
            threshold=self.threshold_,
            margin=self.margin_,
            feasible=self.feasible_,
            n_samples=int(y.size),
            n_critical=int(crit.sum()),
            privileged=self.privileged,
            before=before,
            after=after,
            baseline_threshold=self.baseline_threshold,
            grid_size=int(thr.size),
            frontier=frontier,
        )
        return self

    def predict(self, groups, y_scores) -> np.ndarray:
        if not hasattr(self, "threshold_"):
            raise RuntimeError("call fit() first")
        return reject_option_predict(y_scores, self._priv_mask(groups), self.threshold_, self.margin_)


def fit_reject_option_classification(
    y_true, groups, y_scores, privileged, **kwargs
) -> RejectOptionResult:
    """Convenience wrapper returning :class:`RejectOptionResult`."""
    return RejectOptionClassifier(privileged, **kwargs).fit(y_true, groups, y_scores).result_
