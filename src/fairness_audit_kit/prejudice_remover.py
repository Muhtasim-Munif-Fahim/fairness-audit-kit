"""
Prejudice Remover regularizer (Kamishima, Akaho, Asoh & Sakuma, ECML PKDD 2012).

An *in-processing* method: logistic regression trained with an extra
penalty on the (empirical) mutual information between the predicted label
and the sensitive attribute ``S``::

    loss(w) = -sum_i log M(y_i | x_i, s_i)  +  eta * PI(w)  +  (C / 2) ||w||^2

    PI = sum_i sum_{y in {0,1}} M(y | x_i, s_i) * log( Pr^[y | s_i] / Pr^[y] )

with ``Pr^[y | s] = mean_{i: s_i = s} M(y | x_i, s)`` and
``Pr^[y] = mean_i M(y | x_i, s_i)``. ``PI`` is ``n`` times the plug-in
estimate of ``I(Y_hat; S)``, so ``eta = 0`` is ordinary L2-regularised
logistic regression and larger ``eta`` trades accuracy for independence
of the score from the group (demographic parity).

As in the original paper the model has one weight vector per sensitive
group (``per_group=True``); with ``per_group=False`` a single weight vector
is shared and ``S`` only enters through the penalty.

The gradient is analytic. Because ``Pr^[y | s]`` and ``Pr^[y]`` are the
means of the model outputs, their own derivative contributions cancel and::

    dPI / dp_j = logit(Pr^[1 | s_j]) - logit(Pr^[1])

which makes L-BFGS fast even for many features.

``normalized_prejudice_index`` reports the paper's NPI,
``I(Y; S) / sqrt(H(Y) H(S))`` in ``[0, 1]``, for any hard predictions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.optimize import minimize

__all__ = [
    "PrejudiceRemover",
    "PrejudiceRemoverResult",
    "fit_prejudice_remover",
    "normalized_prejudice_index",
    "render_prejudice_remover_report",
]

_EPS = 1e-12


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def _logit(q: np.ndarray) -> np.ndarray:
    q = np.clip(q, _EPS, 1.0 - _EPS)
    return np.log(q) - np.log1p(-q)


def normalized_prejudice_index(y_pred, groups) -> float:
    """NPI = I(Y; S) / sqrt(H(Y) H(S)) of discrete predictions and groups (0 if undefined)."""
    y = np.asarray(y_pred)
    s = np.asarray(groups)
    if y.shape != s.shape or y.ndim != 1 or y.size == 0:
        raise ValueError("y_pred and groups must be 1-D arrays of the same non-zero length")
    _, yi = np.unique(y, return_inverse=True)
    _, si = np.unique(s, return_inverse=True)
    joint = np.zeros((yi.max() + 1, si.max() + 1))
    np.add.at(joint, (yi, si), 1.0)
    joint /= joint.sum()
    py, ps = joint.sum(axis=1), joint.sum(axis=0)
    nz = joint > 0
    mi = float((joint[nz] * np.log(joint[nz] / np.outer(py, ps)[nz])).sum())
    hy = float(-(py[py > 0] * np.log(py[py > 0])).sum())
    hs = float(-(ps[ps > 0] * np.log(ps[ps > 0])).sum())
    if hy <= 0 or hs <= 0:
        return 0.0
    return max(0.0, mi / np.sqrt(hy * hs))


class PrejudiceRemover:
    """Logistic regression with Kamishima's prejudice-remover penalty.

    Parameters
    ----------
    eta : float
        Strength of the mutual-information penalty (``0`` = plain logistic regression).
    C : float
        L2 penalty coefficient on the weights (intercepts are not penalised).
    per_group : bool
        One weight vector per sensitive group (paper default) or a shared one.
    fit_intercept : bool
    max_iter : int
        L-BFGS iteration cap.
    """

    def __init__(
        self,
        eta: float = 1.0,
        C: float = 1.0,
        per_group: bool = True,
        fit_intercept: bool = True,
        max_iter: int = 500,
    ):
        self.eta = eta
        self.C = C
        self.per_group = per_group
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter

    # -- helpers -----------------------------------------------------------
    def _design(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2:
            raise ValueError("X must be 2-D")
        if not np.isfinite(X).all():
            raise ValueError("X must be finite")
        if self.fit_intercept:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        return X

    def _group_index(self, groups, n: int) -> np.ndarray:
        g = np.asarray(groups)
        if g.shape != (n,):
            raise ValueError("groups must be 1-D with one entry per row of X")
        idx = np.empty(n, dtype=int)
        lookup = {v: k for k, v in enumerate(self.groups_)}
        for i, v in enumerate(g.tolist()):
            if v not in lookup:
                raise ValueError(f"unknown group {v!r}")
            idx[i] = lookup[v]
        return idx

    def _scores(self, W: np.ndarray, Z: np.ndarray, gi: np.ndarray) -> np.ndarray:
        if self.per_group:
            return np.einsum("ij,ij->i", Z, W[gi])
        return Z @ W[0]

    # -- objective ---------------------------------------------------------
    def _objective(self, theta, Z, y, gi, n_groups):
        d = Z.shape[1]
        W = theta.reshape(-1, d)
        z = self._scores(W, Z, gi)
        p = _sigmoid(z)
        # negative log-likelihood (stable): sum log(1 + e^z) - y z
        nll = float(np.sum(np.logaddexp(0.0, z) - y * z))
        dz = p - y
        counts = np.bincount(gi, minlength=n_groups).astype(float)
        q_s = np.bincount(gi, weights=p, minlength=n_groups) / np.maximum(counts, 1.0)
        q = p.mean()
        qs_i = np.clip(q_s[gi], _EPS, 1 - _EPS)
        qc = np.clip(q, _EPS, 1 - _EPS)
        pi = float(np.sum(p * np.log(qs_i / qc) + (1 - p) * np.log((1 - qs_i) / (1 - qc))))
        dz = dz + self.eta * (_logit(q_s)[gi] - _logit(np.array(q))) * p * (1 - p)
        mask = np.ones(d)
        if self.fit_intercept:
            mask[-1] = 0.0
        reg = 0.5 * self.C * float(np.sum((W * mask) ** 2))
        if self.per_group:
            G = np.zeros_like(W)
            np.add.at(G, gi, dz[:, None] * Z)
        else:
            G = (dz @ Z)[None, :]
        G = G + self.C * W * mask
        self._last_pi = pi
        return nll + self.eta * pi + reg, G.ravel()

    # -- public API --------------------------------------------------------
    def fit(self, X, y, groups):
        if not np.isfinite(self.eta) or self.eta < 0:
            raise ValueError("eta must be a non-negative number")
        if not np.isfinite(self.C) or self.C < 0:
            raise ValueError("C must be a non-negative number")
        Z = self._design(X)
        y_arr = np.asarray(y)
        if y_arr.shape != (Z.shape[0],):
            raise ValueError("y must be 1-D with one entry per row of X")
        self.classes_ = np.unique(y_arr)
        if self.classes_.size != 2:
            raise ValueError("PrejudiceRemover needs exactly two classes in y")
        yb = (y_arr == self.classes_[1]).astype(float)
        self.groups_ = list(np.unique(np.asarray(groups)).tolist())
        if len(self.groups_) < 2:
            raise ValueError("groups must contain at least two sensitive groups")
        gi = self._group_index(groups, Z.shape[0])
        n_models = len(self.groups_) if self.per_group else 1
        theta0 = np.zeros(n_models * Z.shape[1])
        res = minimize(
            self._objective,
            theta0,
            args=(Z, yb, gi, len(self.groups_)),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": int(self.max_iter)},
        )
        self.coef_ = res.x.reshape(n_models, Z.shape[1])
        self.converged_ = bool(res.success)
        self.n_iter_ = int(res.nit)
        self.objective_ = float(res.fun)
        self._objective(res.x, Z, yb, gi, len(self.groups_))
        self.prejudice_index_ = self._last_pi / Z.shape[0]
        return self

    def decision_function(self, X, groups) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise RuntimeError("PrejudiceRemover is not fitted yet")
        Z = self._design(X)
        if Z.shape[1] != self.coef_.shape[1]:
            raise ValueError("X has a different number of features than during fit")
        gi = self._group_index(groups, Z.shape[0])
        return self._scores(self.coef_, Z, gi)

    def predict_proba(self, X, groups) -> np.ndarray:
        p = _sigmoid(self.decision_function(X, groups))
        return np.column_stack([1 - p, p])

    def predict(self, X, groups, threshold: float = 0.5) -> np.ndarray:
        return np.where(self.predict_proba(X, groups)[:, 1] >= threshold, self.classes_[1], self.classes_[0])


@dataclass
class PrejudiceRemoverResult:
    """One row of an ``eta`` sweep."""

    eta: float
    accuracy: float
    positive_rates: Dict[str, float]
    demographic_parity_gap: float
    normalized_prejudice_index: float
    prejudice_index: float
    converged: bool
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eta": self.eta,
            "accuracy": self.accuracy,
            "positive_rates": dict(self.positive_rates),
            "demographic_parity_gap": self.demographic_parity_gap,
            "normalized_prejudice_index": self.normalized_prejudice_index,
            "prejudice_index": self.prejudice_index,
            "converged": self.converged,
            "notes": list(self.notes),
        }


def fit_prejudice_remover(
    X,
    y,
    groups,
    etas=(0.0, 1.0, 5.0, 25.0),
    *,
    X_eval=None,
    y_eval=None,
    groups_eval=None,
    C: float = 1.0,
    per_group: bool = True,
) -> List[PrejudiceRemoverResult]:
    """Fit a :class:`PrejudiceRemover` per ``eta`` and report the accuracy / parity trade-off.

    Metrics are computed on ``(X_eval, y_eval, groups_eval)`` when given,
    otherwise on the training data.
    """
    if X_eval is None:
        X_eval, y_eval, groups_eval = X, y, groups
    elif y_eval is None or groups_eval is None:
        raise ValueError("X_eval requires y_eval and groups_eval")
    y_eval = np.asarray(y_eval)
    g_eval = np.asarray(groups_eval)
    out: List[PrejudiceRemoverResult] = []
    for eta in etas:
        model = PrejudiceRemover(eta=float(eta), C=C, per_group=per_group).fit(X, y, groups)
        pred = model.predict(X_eval, g_eval)
        pos = model.classes_[1]
        rates = {str(gv): float(np.mean(pred[g_eval == gv] == pos)) for gv in model.groups_ if np.any(g_eval == gv)}
        notes = [] if model.converged_ else ["L-BFGS did not report convergence"]
        out.append(
            PrejudiceRemoverResult(
                eta=float(eta),
                accuracy=float(np.mean(pred == y_eval)),
                positive_rates=rates,
                demographic_parity_gap=float(max(rates.values()) - min(rates.values())),
                normalized_prejudice_index=normalized_prejudice_index(pred, g_eval),
                prejudice_index=float(model.prejudice_index_),
                converged=model.converged_,
                notes=notes,
            )
        )
    return out


def render_prejudice_remover_report(results: List[PrejudiceRemoverResult]) -> str:
    """Markdown table of an ``eta`` sweep."""
    lines = [
        "# Prejudice Remover (Kamishima et al., 2012)",
        "",
        "| eta | accuracy | DP gap | NPI | train PI |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.eta:g} | {r.accuracy:.4f} | {r.demographic_parity_gap:.4f} | "
            f"{r.normalized_prejudice_index:.4f} | {r.prejudice_index:.5f} |"
        )
    notes = [n for r in results for n in r.notes]
    if notes:
        lines += ["", "Notes:"] + [f"- {n}" for n in sorted(set(notes))]
    return "\n".join(lines) + "\n"
