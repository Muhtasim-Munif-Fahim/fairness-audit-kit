"""
Fairness constraints via decision-boundary covariance (Zafar, Valera,
Gomez-Rodriguez & Gummadi, AISTATS 2017).

An *in-processing* method for linear classifiers. Instead of the
non-convex constraint "positive rates are equal across groups", Zafar et
al. bound the empirical covariance between the sensitive attribute and the
signed distance to the decision boundary::

    Cov(z, d_theta(x)) ~= (1/N) sum_i (z_i - z_bar) * theta^T x_i

which is *linear* in ``theta``. For more than two groups one constraint is
used per group indicator ``z = 1[s = g]``. Two formulations are provided:

* ``mode="fairness"`` (maximize accuracy under fairness constraints)::

      min_theta  L(theta) + (C / 2) ||w||^2     s.t.  |Cov_g(theta)| <= c  for every group g

  ``c = 0`` asks for a boundary uncorrelated with the group; large ``c`` is
  ordinary L2 logistic regression.

* ``mode="accuracy"`` (maximize fairness under accuracy constraints)::

      min_theta  sum_g Cov_g(theta)^2     s.t.  L(theta) <= (1 + gamma) L(theta*)

  where ``theta*`` is the unconstrained optimum, so the loss may grow by at
  most a fraction ``gamma``. (The paper minimizes ``|Cov|``; the squared form
  has the same minimizers and is smooth for SLSQP.)

Both are solved with SLSQP using analytic gradients. The sensitive attribute
is only used for the constraint, never as a model input, so predictions do
not need it.

``p_percent_rule`` reports the p%-rule ``min(P(yhat=1|s=a)/P(yhat=1|s=b))``
used by the paper (the US "80% rule" for disparate impact).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
from scipy.optimize import minimize

__all__ = [
    "FairnessConstrainedClassifier",
    "FairnessConstraintResult",
    "boundary_covariance",
    "fit_fairness_constraints",
    "p_percent_rule",
    "render_fairness_constraints_report",
]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def _indicators(groups, levels) -> np.ndarray:
    g = np.asarray(groups)
    Zs = np.column_stack([(g == v).astype(float) for v in levels])
    # two groups: one constraint suffices (the second is its negation)
    return Zs[:, :1] if len(levels) == 2 else Zs


def boundary_covariance(scores, groups) -> Dict[str, float]:
    """Empirical ``Cov(1[s = g], score)`` for each group ``g``."""
    d = np.asarray(scores, dtype=float)
    g = np.asarray(groups)
    if d.shape != g.shape or d.ndim != 1 or d.size == 0:
        raise ValueError("scores and groups must be 1-D arrays of the same non-zero length")
    out = {}
    for v in np.unique(g):
        z = (g == v).astype(float)
        out[str(v)] = float(np.mean((z - z.mean()) * d))
    return out


def p_percent_rule(y_pred, groups, positive=1) -> float:
    """Smallest ratio of positive-prediction rates between any two groups (1 = parity)."""
    y = np.asarray(y_pred)
    g = np.asarray(groups)
    if y.shape != g.shape or y.ndim != 1 or y.size == 0:
        raise ValueError("y_pred and groups must be 1-D arrays of the same non-zero length")
    rates = [float(np.mean(y[g == v] == positive)) for v in np.unique(g)]
    hi = max(rates)
    if hi == 0:
        return 1.0
    return min(rates) / hi


class FairnessConstrainedClassifier:
    """Logistic regression with Zafar et al.'s covariance fairness constraints.

    Parameters
    ----------
    mode : {"fairness", "accuracy"}
    c : float
        Covariance bound for ``mode="fairness"``.
    gamma : float
        Allowed relative loss increase for ``mode="accuracy"``.
    C : float
        L2 penalty on the weights (intercept not penalized); the data term is
        the *mean* log-loss, so this matches scikit-learn's
        ``LogisticRegression(C=1 / (n * C))``.
    fit_intercept : bool
    max_iter : int
    """

    def __init__(
        self,
        mode: str = "fairness",
        c: float = 0.0,
        gamma: float = 0.1,
        C: float = 1e-3,
        fit_intercept: bool = True,
        max_iter: int = 500,
    ):
        self.mode = mode
        self.c = c
        self.gamma = gamma
        self.C = C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter

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

    def _loss(self, theta, Z, y, mask):
        z = Z @ theta
        n = Z.shape[0]
        val = float(np.sum(np.logaddexp(0.0, z) - y * z)) / n + 0.5 * self.C * float(np.sum((theta * mask) ** 2))
        grad = Z.T @ (_sigmoid(z) - y) / n + self.C * theta * mask
        return val, grad

    def fit(self, X, y, groups):
        if self.mode not in ("fairness", "accuracy"):
            raise ValueError("mode must be 'fairness' or 'accuracy'")
        if not np.isfinite(self.c) or self.c < 0:
            raise ValueError("c must be a non-negative number")
        if not np.isfinite(self.gamma) or self.gamma < 0:
            raise ValueError("gamma must be a non-negative number")
        if not np.isfinite(self.C) or self.C < 0:
            raise ValueError("C must be a non-negative number")
        Z = self._design(X)
        n, d = Z.shape
        y_arr = np.asarray(y)
        if y_arr.shape != (n,):
            raise ValueError("y must be 1-D with one entry per row of X")
        self.classes_ = np.unique(y_arr)
        if self.classes_.size != 2:
            raise ValueError("FairnessConstrainedClassifier needs exactly two classes in y")
        yb = (y_arr == self.classes_[1]).astype(float)
        g = np.asarray(groups)
        if g.shape != (n,):
            raise ValueError("groups must be 1-D with one entry per row of X")
        self.groups_ = list(np.unique(g).tolist())
        if len(self.groups_) < 2:
            raise ValueError("groups must contain at least two sensitive groups")
        S = _indicators(g, self.groups_)
        # covariance is linear: cov = A @ theta with A = (S - mean)^T Z / n
        A = (S - S.mean(axis=0)).T @ Z / n
        mask = np.ones(d)
        if self.fit_intercept:
            mask[-1] = 0.0

        theta0 = np.zeros(d)
        base = minimize(self._loss, theta0, args=(Z, yb, mask), jac=True, method="L-BFGS-B",
                        options={"maxiter": int(self.max_iter)})
        self.unconstrained_coef_ = base.x
        self.unconstrained_loss_ = float(base.fun)

        if self.mode == "fairness":
            c = float(self.c)
            cons = [
                {"type": "ineq", "fun": lambda t: c - A @ t, "jac": lambda t: -A},
                {"type": "ineq", "fun": lambda t: c + A @ t, "jac": lambda t: A},
            ]
            res = minimize(self._loss, base.x, args=(Z, yb, mask), jac=True, method="SLSQP",
                           constraints=cons, options={"maxiter": int(self.max_iter), "ftol": 1e-10})
        else:
            bound = (1.0 + float(self.gamma)) * self.unconstrained_loss_

            def obj(t):
                cv = A @ t
                return float(cv @ cv), 2.0 * A.T @ cv

            cons = [{
                "type": "ineq",
                "fun": lambda t: bound - self._loss(t, Z, yb, mask)[0],
                "jac": lambda t: -self._loss(t, Z, yb, mask)[1],
            }]
            res = minimize(obj, base.x, jac=True, method="SLSQP", constraints=cons,
                           options={"maxiter": int(self.max_iter), "ftol": 1e-14})
        self.coef_ = res.x
        self.converged_ = bool(res.success)
        self.n_iter_ = int(res.nit)
        self.loss_ = float(self._loss(res.x, Z, yb, mask)[0])
        self.covariance_ = boundary_covariance(Z @ res.x, g)
        return self

    def decision_function(self, X) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise RuntimeError("FairnessConstrainedClassifier is not fitted yet")
        Z = self._design(X)
        if Z.shape[1] != self.coef_.shape[0]:
            raise ValueError("X has a different number of features than during fit")
        return Z @ self.coef_

    def predict_proba(self, X) -> np.ndarray:
        p = _sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])


@dataclass
class FairnessConstraintResult:
    """One row of a ``c`` (or ``gamma``) sweep."""

    mode: str
    level: float
    accuracy: float
    positive_rates: Dict[str, float]
    p_percent: float
    max_abs_covariance: float
    converged: bool
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "level": self.level,
            "accuracy": self.accuracy,
            "positive_rates": dict(self.positive_rates),
            "p_percent": self.p_percent,
            "max_abs_covariance": self.max_abs_covariance,
            "converged": self.converged,
            "notes": list(self.notes),
        }


def fit_fairness_constraints(
    X,
    y,
    groups,
    levels=(None, 0.1, 0.01, 0.0),
    *,
    mode: str = "fairness",
    X_eval=None,
    y_eval=None,
    groups_eval=None,
    C: float = 1e-3,
) -> List[FairnessConstraintResult]:
    """Sweep the covariance bound ``c`` (``mode="fairness"``) or ``gamma`` (``mode="accuracy"``).

    ``None`` in ``levels`` means unconstrained logistic regression (reported
    with ``level = inf``). Metrics use the evaluation split when given.
    """
    if X_eval is None:
        X_eval, y_eval, groups_eval = X, y, groups
    elif y_eval is None or groups_eval is None:
        raise ValueError("X_eval requires y_eval and groups_eval")
    y_eval = np.asarray(y_eval)
    g_eval = np.asarray(groups_eval)
    out: List[FairnessConstraintResult] = []
    for lv in levels:
        if lv is None:
            model = FairnessConstrainedClassifier(mode="fairness", c=1e12, C=C).fit(X, y, groups)
            level = float("inf")
        elif mode == "fairness":
            model = FairnessConstrainedClassifier(mode="fairness", c=float(lv), C=C).fit(X, y, groups)
            level = float(lv)
        else:
            model = FairnessConstrainedClassifier(mode="accuracy", gamma=float(lv), C=C).fit(X, y, groups)
            level = float(lv)
        pred = model.predict(X_eval)
        pos = model.classes_[1]
        rates = {str(v): float(np.mean(pred[g_eval == v] == pos)) for v in np.unique(g_eval)}
        out.append(
            FairnessConstraintResult(
                mode="unconstrained" if lv is None else mode,
                level=level,
                accuracy=float(np.mean(pred == y_eval)),
                positive_rates=rates,
                p_percent=p_percent_rule(pred, g_eval, positive=pos),
                max_abs_covariance=float(max(abs(v) for v in model.covariance_.values())),
                converged=model.converged_,
                notes=[] if model.converged_ else ["SLSQP did not report convergence"],
            )
        )
    return out


def render_fairness_constraints_report(results: List[FairnessConstraintResult]) -> str:
    """Markdown table of a fairness-constraints sweep."""
    lines = [
        "# Fairness constraints (Zafar et al., 2017)",
        "",
        "| mode | level | accuracy | p%-rule | max abs Cov | positive rates |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in results:
        rates = ", ".join(f"{k}: {v:.3f}" for k, v in r.positive_rates.items())
        lines.append(
            f"| {r.mode} | {r.level:g} | {r.accuracy:.4f} | {r.p_percent:.3f} | "
            f"{r.max_abs_covariance:.5f} | {rates} |"
        )
    notes = sorted({n for r in results for n in r.notes})
    if notes:
        lines += ["", "Notes:"] + [f"- {n}" for n in notes]
    return "\n".join(lines) + "\n"
