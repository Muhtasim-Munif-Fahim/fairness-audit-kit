"""
Calibrated equalized odds post-processing (Pleiss, Raghavan, Wu, Kleinberg
and Weinberger, NIPS 2017).

Hardt et al.'s equalized odds and calibration usually can't both hold.
Pleiss et al. keep each group's scores *calibrated* and equalise a single
**generalized cost** instead:

    GFPR_g = E[s | y = 0, g]            (generalized false positive rate)
    GFNR_g = E[1 - s | y = 1, g]        (generalized false negative rate)
    c_g    = a · GFPR_g · (1 - μ_g) + b · GFNR_g · μ_g

where ``μ_g`` is the group's base rate.  ``cost_constraint`` sets the weights:
``"fpr"`` gives ``(a, b) = (1, 0)``, ``"fnr"`` gives ``(0, 1)`` and
``"weighted"`` gives ``(½, ½)``, as in AIF360.

The trivial classifier ``s ≡ μ_g`` is also calibrated.  Replacing a random
fraction ``α_g`` of a group's scores with ``μ_g`` keeps that group calibrated,
and moves its expected cost linearly from ``c_g`` toward the trivial cost
``c̃_g``.  Groups that cost less than the worst group are mixed with

    α_g = (c_max - c_g) / (c̃_g - c_g)

so every group ends at the same expected cost ``c_max``.  For two groups
this is exactly AIF360's ``CalibratedEqOddsPostprocessing``.  For more than
two groups, every group is raised to the worst group's cost.  If a group's
trivial cost is below ``c_max``, it can't reach the target; ``α_g`` is clipped to 1
and ``feasible`` is False.

The method is randomized, uses the group at decision time, and only ever
*withholds* information.  It trades accuracy for parity on the chosen
cost.  It is impossible to also match both GFPR and GFNR unless the base rates
are equal (Pleiss et al., Thm. 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from fairness_audit_kit.odds_parity import _as_binary_labels

COST_CONSTRAINTS = ("weighted", "fpr", "fnr")


def _cost_weights(cost_constraint: str):
    if cost_constraint == "fpr":
        return 1.0, 0.0
    if cost_constraint == "fnr":
        return 0.0, 1.0
    if cost_constraint == "weighted":
        return 0.5, 0.5
    raise ValueError(f"cost_constraint must be one of {COST_CONSTRAINTS}")


def generalized_rates(y_true, y_scores) -> tuple:
    """``(GFPR, GFNR)``: mean score on negatives and mean ``1 - score`` on positives.

    A rate is ``nan`` if that class is absent.
    """
    y = _as_binary_labels("y_true", y_true)
    s = np.asarray(y_scores, dtype=float).ravel()
    if s.shape != y.shape:
        raise ValueError("y_true and y_scores must have the same length")
    neg, pos = y == 0, y == 1
    gfpr = float(s[neg].mean()) if neg.any() else float("nan")
    gfnr = float((1.0 - s[pos]).mean()) if pos.any() else float("nan")
    return gfpr, gfnr


def generalized_cost(y_true, y_scores, cost_constraint: str = "weighted") -> float:
    """Pleiss et al.'s cost ``a·GFPR·(1-μ) + b·GFNR·μ`` (AIF360's normalisation)."""
    a, b = _cost_weights(cost_constraint)
    y = _as_binary_labels("y_true", y_true)
    mu = float(y.mean())
    gfpr, gfnr = generalized_rates(y, y_scores)
    cost = 0.0
    if a:
        cost += a * (0.0 if np.isnan(gfpr) else gfpr) * (1.0 - mu)
    if b:
        cost += b * (0.0 if np.isnan(gfnr) else gfnr) * mu
    return float(cost)


@dataclass
class CalibratedGroupStats:
    """Per-group statistics before and after mixing (expected values)."""

    group: object
    size: int
    base_rate: float
    mix_rate: float
    cost_before: float
    cost_after: float
    trivial_cost: float
    gfpr_before: float
    gfpr_after: float
    gfnr_before: float
    gfnr_after: float

    def to_dict(self) -> Dict:
        return dict(self.__dict__)


@dataclass
class CalibratedEqOddsResult:
    """Summary of a fitted :class:`CalibratedEqOddsPostprocessor`."""

    cost_constraint: str
    target_cost: float
    feasible: bool
    n_samples: int
    groups: List[CalibratedGroupStats] = field(default_factory=list)
    cost_gap_before: float = 0.0
    cost_gap_after: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "cost_constraint": self.cost_constraint,
            "target_cost": self.target_cost,
            "feasible": self.feasible,
            "n_samples": self.n_samples,
            "cost_gap_before": self.cost_gap_before,
            "cost_gap_after": self.cost_gap_after,
            "groups": [g.to_dict() for g in self.groups],
        }


class CalibratedEqOddsPostprocessor:
    """Calibration-preserving equalized-odds post-processing.

    Parameters
    ----------
    cost_constraint:
        ``"weighted"`` (default), ``"fpr"`` or ``"fnr"``.
    threshold:
        Hard decisions are ``score >= threshold``, as in AIF360.
    random_state:
        Seed for the randomized replacement in :meth:`predict_proba`.
    """

    def __init__(self, cost_constraint: str = "weighted", threshold: float = 0.5,
                 random_state: Optional[int] = None):
        _cost_weights(cost_constraint)
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("threshold must lie in [0, 1]")
        self.cost_constraint = cost_constraint
        self.threshold = float(threshold)
        self.random_state = random_state

    @staticmethod
    def _keys(groups) -> np.ndarray:
        return np.array([str(v) for v in np.asarray(groups, dtype=object).ravel()], dtype=object)

    def fit(self, y_true, groups, y_scores) -> "CalibratedEqOddsPostprocessor":
        y = _as_binary_labels("y_true", y_true)
        s = np.asarray(y_scores, dtype=float).ravel()
        if s.shape != y.shape:
            raise ValueError("y_true and y_scores must have the same length")
        if not np.all(np.isfinite(s)) or s.min() < 0 or s.max() > 1:
            raise ValueError("y_scores must be finite probabilities in [0, 1]")
        keys = self._keys(groups)
        if keys.shape != y.shape:
            raise ValueError("groups must have one value per sample")
        labels = sorted(set(keys.tolist()))
        if len(labels) < 2:
            raise ValueError("need at least two groups")

        base, cost, trivial, rates = {}, {}, {}, {}
        for g in labels:
            m = keys == g
            mu = float(y[m].mean())
            base[g] = mu
            cost[g] = generalized_cost(y[m], s[m], self.cost_constraint)
            trivial[g] = generalized_cost(y[m], np.full(m.sum(), mu), self.cost_constraint)
            rates[g] = generalized_rates(y[m], s[m])
        target = max(cost.values())
        mix, feasible = {}, True
        for g in labels:
            gap = target - cost[g]
            if gap <= 0:
                mix[g] = 0.0
                continue
            room = trivial[g] - cost[g]
            if room <= gap:
                feasible = feasible and room >= gap - 1e-12
                mix[g] = 1.0
            else:
                mix[g] = gap / room
        self.groups_ = labels
        self.base_rates_ = base
        self.mix_rates_ = mix
        self.costs_before_ = cost
        self.trivial_costs_ = trivial
        self.target_cost_ = float(target)
        self.feasible_ = bool(feasible)

        stats = []
        for g in labels:
            a = mix[g]
            mu = base[g]
            gfpr, gfnr = rates[g]
            after = (1 - a) * cost[g] + a * trivial[g]
            stats.append(CalibratedGroupStats(
                group=g, size=int((keys == g).sum()), base_rate=mu, mix_rate=a,
                cost_before=cost[g], cost_after=float(after), trivial_cost=trivial[g],
                gfpr_before=gfpr, gfpr_after=float((1 - a) * gfpr + a * mu),
                gfnr_before=gfnr, gfnr_after=float((1 - a) * gfnr + a * (1 - mu)),
            ))
        afters = [st.cost_after for st in stats]
        self.result_ = CalibratedEqOddsResult(
            cost_constraint=self.cost_constraint,
            target_cost=self.target_cost_,
            feasible=self.feasible_,
            n_samples=int(y.size),
            groups=stats,
            cost_gap_before=float(max(cost.values()) - min(cost.values())),
            cost_gap_after=float(max(afters) - min(afters)),
        )
        return self

    def _check(self, groups, y_scores):
        if not hasattr(self, "mix_rates_"):
            raise RuntimeError("call fit() first")
        s = np.asarray(y_scores, dtype=float).ravel()
        keys = self._keys(groups)
        if keys.shape != s.shape:
            raise ValueError("groups and y_scores must have the same length")
        unknown = set(keys.tolist()) - set(self.groups_)
        if unknown:
            raise ValueError(f"unseen group(s): {sorted(unknown)}")
        return keys, s

    def expected_scores(self, groups, y_scores) -> np.ndarray:
        """Deterministic expectation ``(1 - α_g)·s + α_g·μ_g``."""
        keys, s = self._check(groups, y_scores)
        alpha = np.array([self.mix_rates_[k] for k in keys])
        mu = np.array([self.base_rates_[k] for k in keys])
        return (1 - alpha) * s + alpha * mu

    def predict_proba(self, groups, y_scores, random_state: Optional[int] = None) -> np.ndarray:
        """Randomized scores: each score is replaced by ``μ_g`` with probability ``α_g``."""
        keys, s = self._check(groups, y_scores)
        seed = self.random_state if random_state is None else random_state
        rng = np.random.default_rng(seed)
        alpha = np.array([self.mix_rates_[k] for k in keys])
        mu = np.array([self.base_rates_[k] for k in keys])
        swap = rng.random(s.size) < alpha
        return np.where(swap, mu, s)

    def predict(self, groups, y_scores, random_state: Optional[int] = None) -> np.ndarray:
        return (self.predict_proba(groups, y_scores, random_state) >= self.threshold).astype(int)


def fit_calibrated_eq_odds(y_true, groups, y_scores, **kwargs) -> CalibratedEqOddsResult:
    """Convenience wrapper returning :class:`CalibratedEqOddsResult`."""
    return CalibratedEqOddsPostprocessor(**kwargs).fit(y_true, groups, y_scores).result_
