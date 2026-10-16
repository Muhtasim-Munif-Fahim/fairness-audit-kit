"""
Equalized-odds post-processing (Hardt, Price and Srebro, NIPS 2016).

Given a binary predictor ``Ŷ`` (hard labels, or scores thresholded at a cut)
and a sensitive attribute ``A``, Hardt et al. derive a new randomized
predictor ``Ỹ`` that only looks at ``(Ŷ, A)``:

    p[g, ŷ] = P(Ỹ = 1 | Ŷ = ŷ, A = g)

so each group flips some of its positive and/or negative predictions with a
group-specific probability. The derived rates are linear in ``p``:

    TPR_g(Ỹ) = p[g,1] · TPR_g + p[g,0] · (1 − TPR_g)
    FPR_g(Ỹ) = p[g,1] · FPR_g + p[g,0] · (1 − FPR_g)

Equalized odds requires TPR and FPR to be equal across groups; equal
opportunity requires only TPR. Among all mixing rates satisfying the
constraint, the one with the smallest expected (cost-weighted) error is
found with a linear program (``scipy.optimize.linprog``). The optimum is the
point in the intersection of the groups' ROC-achievable regions closest to
perfect classification.

This is a post-processing method: it needs no access to features or model
internals, and it can be applied to any frozen classifier. Because it is
randomized, deployed decisions differ run to run; use a seed for
reproducibility and ``positive_probability`` for the deterministic rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.optimize import linprog

from fairness_audit_kit.odds_parity import _resolve_predictions

_CONSTRAINTS = ("equalized_odds", "equal_opportunity")


@dataclass
class GroupRates:
    """Analytic rates of one group before and after post-processing."""

    size: int
    base_rate: float
    tpr_before: float
    fpr_before: float
    tpr_after: float
    fpr_after: float
    p_flip_to_positive: float  # p[g, 0]
    p_keep_positive: float  # p[g, 1]

    def to_dict(self) -> Dict:
        return dict(self.__dict__)


@dataclass
class EqualizedOddsPostprocessingResult:
    """Fitted mixing rates, rate tables, and expected accuracy before/after."""

    constraint: str
    threshold: Optional[float]
    n_samples: int
    groups: Dict[str, GroupRates]
    tpr_gap_before: float
    fpr_gap_before: float
    tpr_gap_after: float
    fpr_gap_after: float
    accuracy_before: float
    expected_accuracy_after: float
    expected_cost_after: float
    fp_cost: float
    fn_cost: float

    def to_dict(self) -> Dict:
        out = dict(self.__dict__)
        out["groups"] = {g: r.to_dict() for g, r in self.groups.items()}
        return out


def _rates(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    pos = y_true == 1
    neg = ~pos
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("every group needs at least one positive and one negative label")
    tpr = float(np.mean(y_pred[pos] == 1))
    fpr = float(np.mean(y_pred[neg] == 1))
    return tpr, fpr, n_pos, n_neg


class EqualizedOddsPostprocessor:
    """Hardt et al. derived predictor via a linear program over mixing rates.

    Parameters
    ----------
    constraint:
        ``"equalized_odds"`` (equal TPR and FPR) or ``"equal_opportunity"``
        (equal TPR only).
    threshold:
        Score cut used when fitting/predicting from ``y_scores``.
    fp_cost, fn_cost:
        Relative cost of a false positive / false negative in the objective.
    """

    def __init__(
        self,
        constraint: str = "equalized_odds",
        threshold: float = 0.5,
        fp_cost: float = 1.0,
        fn_cost: float = 1.0,
    ) -> None:
        if constraint not in _CONSTRAINTS:
            raise ValueError(f"constraint must be one of {_CONSTRAINTS}")
        if fp_cost < 0 or fn_cost < 0 or fp_cost + fn_cost == 0:
            raise ValueError("costs must be non-negative and not both zero")
        self.constraint = constraint
        self.threshold = float(threshold)
        self.fp_cost = float(fp_cost)
        self.fn_cost = float(fn_cost)
        self.mixing_rates_: Optional[Dict[str, Dict[int, float]]] = None
        self.result_: Optional[EqualizedOddsPostprocessingResult] = None

    def fit(self, y_true, groups, y_pred=None, y_scores=None) -> "EqualizedOddsPostprocessor":
        y_true, y_hat, used_threshold = _resolve_predictions(
            y_true, y_pred, y_scores, self.threshold
        )
        groups = np.asarray(groups).ravel().astype(str)
        if groups.shape[0] != y_true.shape[0]:
            raise ValueError("groups must have the same length as y_true")
        labels = sorted(np.unique(groups).tolist())
        if len(labels) < 2:
            raise ValueError("need at least two groups")
        n = y_true.shape[0]

        stats = {}
        for g in labels:
            mask = groups == g
            tpr, fpr, n_pos, n_neg = _rates(y_true[mask], y_hat[mask])
            stats[g] = (tpr, fpr, n_pos, n_neg, int(mask.sum()))

        # Variables: [p_g0, p_g1] for each group (in label order).
        k = len(labels)
        c = np.zeros(2 * k)
        for i, g in enumerate(labels):
            tpr, fpr, n_pos, n_neg, _ = stats[g]
            # Expected FP count: n_neg * FPR(Ỹ); expected FN: n_pos * (1 - TPR(Ỹ)).
            c[2 * i] += self.fp_cost * n_neg * (1 - fpr) - self.fn_cost * n_pos * (1 - tpr)
            c[2 * i + 1] += self.fp_cost * n_neg * fpr - self.fn_cost * n_pos * tpr
        c /= n

        def coeffs(i: int, rate: float) -> np.ndarray:
            row = np.zeros(2 * k)
            row[2 * i] = 1 - rate
            row[2 * i + 1] = rate
            return row

        a_eq = []
        ref = labels[0]
        for i, g in enumerate(labels[1:], start=1):
            a_eq.append(coeffs(i, stats[g][0]) - coeffs(0, stats[ref][0]))
            if self.constraint == "equalized_odds":
                a_eq.append(coeffs(i, stats[g][1]) - coeffs(0, stats[ref][1]))
        a_eq = np.asarray(a_eq)
        res = linprog(
            c,
            A_eq=a_eq,
            b_eq=np.zeros(a_eq.shape[0]),
            bounds=[(0.0, 1.0)] * (2 * k),
            method="highs",
        )
        if not res.success:  # pragma: no cover - the LP is always feasible (p = const)
            raise RuntimeError(f"linear program failed: {res.message}")
        p = np.clip(res.x, 0.0, 1.0)

        self.mixing_rates_ = {
            g: {0: float(p[2 * i]), 1: float(p[2 * i + 1])} for i, g in enumerate(labels)
        }

        group_rates: Dict[str, GroupRates] = {}
        exp_fp = exp_fn = 0.0
        for i, g in enumerate(labels):
            tpr, fpr, n_pos, n_neg, size = stats[g]
            p0, p1 = p[2 * i], p[2 * i + 1]
            tpr_after = float(p1 * tpr + p0 * (1 - tpr))
            fpr_after = float(p1 * fpr + p0 * (1 - fpr))
            exp_fp += n_neg * fpr_after
            exp_fn += n_pos * (1 - tpr_after)
            group_rates[g] = GroupRates(
                size=size,
                base_rate=n_pos / size,
                tpr_before=tpr,
                fpr_before=fpr,
                tpr_after=tpr_after,
                fpr_after=fpr_after,
                p_flip_to_positive=float(p0),
                p_keep_positive=float(p1),
            )

        def gap(attr: str) -> float:
            vals = [getattr(r, attr) for r in group_rates.values()]
            return float(max(vals) - min(vals))

        self.result_ = EqualizedOddsPostprocessingResult(
            constraint=self.constraint,
            threshold=used_threshold,
            n_samples=int(n),
            groups=group_rates,
            tpr_gap_before=gap("tpr_before"),
            fpr_gap_before=gap("fpr_before"),
            tpr_gap_after=gap("tpr_after"),
            fpr_gap_after=gap("fpr_after"),
            accuracy_before=float(np.mean(y_true == y_hat)),
            expected_accuracy_after=float(1.0 - (exp_fp + exp_fn) / n),
            expected_cost_after=float((self.fp_cost * exp_fp + self.fn_cost * exp_fn) / n),
            fp_cost=self.fp_cost,
            fn_cost=self.fn_cost,
        )
        return self

    def _base_predictions(self, y_pred, y_scores) -> np.ndarray:
        if y_pred is not None:
            arr = np.asarray(y_pred).ravel()
            if not np.all(np.isin(arr, [0, 1])):
                raise ValueError("y_pred must contain only 0/1")
            return arr.astype(int)
        if y_scores is None:
            raise ValueError("Provide y_pred or y_scores")
        scores = np.asarray(y_scores, dtype=float).ravel()
        if np.any(~np.isfinite(scores)):
            raise ValueError("y_scores must be finite")
        return (scores >= self.threshold).astype(int)

    def positive_probability(self, groups, y_pred=None, y_scores=None) -> np.ndarray:
        """Deterministic ``P(Ỹ = 1)`` for each row."""
        if self.mixing_rates_ is None:
            raise RuntimeError("fit must be called first")
        y_hat = self._base_predictions(y_pred, y_scores)
        groups = np.asarray(groups).ravel().astype(str)
        if groups.shape[0] != y_hat.shape[0]:
            raise ValueError("groups must have the same length as the predictions")
        unknown = sorted(set(groups.tolist()) - set(self.mixing_rates_))
        if unknown:
            raise ValueError(f"unseen groups: {unknown}")
        out = np.empty(y_hat.shape[0], dtype=float)
        for g, rates in self.mixing_rates_.items():
            mask = groups == g
            out[mask] = np.where(y_hat[mask] == 1, rates[1], rates[0])
        return out

    def predict(self, groups, y_pred=None, y_scores=None, seed: Optional[int] = None) -> np.ndarray:
        """Randomized post-processed 0/1 decisions."""
        prob = self.positive_probability(groups, y_pred=y_pred, y_scores=y_scores)
        rng = np.random.default_rng(seed)
        return (rng.uniform(size=prob.shape[0]) < prob).astype(int)


def fit_equalized_odds_postprocessing(
    y_true,
    groups,
    y_pred=None,
    y_scores=None,
    *,
    constraint: str = "equalized_odds",
    threshold: float = 0.5,
    fp_cost: float = 1.0,
    fn_cost: float = 1.0,
) -> EqualizedOddsPostprocessingResult:
    """Convenience wrapper returning only the fitted result."""
    post = EqualizedOddsPostprocessor(
        constraint=constraint, threshold=threshold, fp_cost=fp_cost, fn_cost=fn_cost
    ).fit(y_true, groups, y_pred=y_pred, y_scores=y_scores)
    return post.result_
