"""
Exponentiated-gradient reduction for fair classification (Agarwal, Beygelzimer,
Dudík, Langford and Wallach, ICML 2018).

Post-processing methods in this kit adjust the scores of a model that is
already trained.  The *reductions* approach instead trains the model under a
fairness constraint, using any classifier that accepts ``sample_weight``.

The constraint is written as moment inequalities.  For **demographic parity**
every group ``a`` must satisfy

    γ_a(h) = E[h(X) | A = a] − E[h(X)],         |γ_a(h)| ≤ ε,

and for **equalized odds** the same holds conditionally on each label,

    γ_{y,a}(h) = E[h(X) | A = a, Y = y] − E[h(X) | Y = y].

The search is over *randomized* classifiers ``Q`` (distributions over base
classifiers).  Each inequality is split into a ``+`` and a ``−`` copy and the
saddle point of the Lagrangian

    L(Q, λ) = err(Q) + λᵀ(γ̂(Q) − ε),      λ ≥ 0, ‖λ‖₁ ≤ B

is found by a two-player game:

* the **λ-player** runs exponentiated gradient,
  ``λ = B·exp(θ) / (1 + Σ exp(θ))`` with ``θ += η (γ̂(h_t) − ε)``;
* the **Q-player** best-responds with a *cost-sensitive* classification
  problem.  Predicting 1 instead of 0 on example ``i`` changes the Lagrangian
  by ``c_i = (1 − 2y_i)/n + λ_{e_i,a_i}/(n p_{e_i,a_i}) − Σ_a' λ_{e_i,a'}/(n p_{e_i})``,
  so the base learner is fit on labels ``1[c_i < 0]`` with weights ``|c_i|``.

The algorithm stops when the duality gap of the averaged iterates drops below
``nu``.  As in the paper, the final mixture is then re-weighted by a small
linear program over all hypotheses found: minimise ``Σ q_h err_h + B·s``
subject to ``Σ q_h γ̂_h − s ≤ ε``.  The result is usually a sparse mixture of
two or three classifiers.

At prediction time only the features are needed (not the sensitive
attribute).  ``predict_proba`` returns ``Σ q_h h(x)`` and ``predict`` samples
from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

CONSTRAINTS = ("demographic_parity", "equalized_odds")


class _ConstantClassifier:
    """Base-learner stand-in used when a best response has a single label."""

    def __init__(self, value: int):
        self.value = int(value)

    def predict(self, X):
        return np.full(len(X), self.value, dtype=int)


def _default_estimator():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=1000)


def _clone(estimator):
    from sklearn.base import clone

    return clone(estimator)


class _Moment:
    """Group-conditional moments for DP (one event) or EO (event = label)."""

    def __init__(self, constraint: str, y: np.ndarray, groups: np.ndarray):
        if constraint not in CONSTRAINTS:
            raise ValueError(f"constraint must be one of {CONSTRAINTS}")
        self.constraint = constraint
        self.n = len(y)
        self.events = np.zeros(self.n, dtype=int) if constraint == "demographic_parity" else y.astype(int)
        self.group_values, self.group_idx = np.unique(groups, return_inverse=True)
        self.event_values = np.unique(self.events)
        if len(self.group_values) < 2:
            raise ValueError("need at least two groups")
        self.keys = []  # (event, group index)
        for e in self.event_values:
            for g in range(len(self.group_values)):
                mask = (self.events == e) & (self.group_idx == g)
                if mask.any():
                    self.keys.append((int(e), g))
        self.k = len(self.keys)
        self.p_eg = np.array([np.mean((self.events == e) & (self.group_idx == g)) for e, g in self.keys])
        self.p_e = {int(e): float(np.mean(self.events == e)) for e in self.event_values}

    def gamma(self, h: np.ndarray) -> np.ndarray:
        """Signed moment vector ``γ`` (length k) for (possibly fractional) predictions ``h``."""
        out = np.empty(self.k)
        for j, (e, g) in enumerate(self.keys):
            ev = self.events == e
            out[j] = h[ev & (self.group_idx == g)].mean() - h[ev].mean()
        return out

    def signed_weights(self, lam: np.ndarray) -> np.ndarray:
        """Per-example derivative of ``λᵀγ`` with respect to ``h(x_i)`` (``λ`` signed, length k)."""
        w = np.zeros(self.n)
        for j, (e, g) in enumerate(self.keys):
            ev = self.events == e
            w[ev & (self.group_idx == g)] += lam[j] / (self.n * self.p_eg[j])
            w[ev] -= lam[j] / (self.n * self.p_e[e])
        return w

    def labels(self) -> List[str]:
        if self.constraint == "demographic_parity":
            return [str(self.group_values[g]) for _, g in self.keys]
        return [f"y={e},{self.group_values[g]}" for e, g in self.keys]


@dataclass
class ExponentiatedGradientResult:
    """Training-set summary of the fitted randomized classifier."""

    constraint: str
    eps: float
    bound: float
    n_iter: int
    converged: bool
    duality_gap: float
    n_hypotheses: int
    weights: List[float]
    error_before: float
    error_after: float
    violation_before: float
    violation_after: float
    moments_before: Dict[str, float] = field(default_factory=dict)
    moments_after: Dict[str, float] = field(default_factory=dict)
    gap_history: List[float] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return self.violation_after <= self.eps + 1e-6

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["feasible"] = self.feasible
        return d


class ExponentiatedGradient:
    """Agarwal et al. (2018) exponentiated-gradient reduction.

    Parameters
    ----------
    estimator : classifier with ``fit(X, y, sample_weight=...)`` and ``predict``
        Base learner (default: scikit-learn ``LogisticRegression``).
    constraint : {"demographic_parity", "equalized_odds"}
    eps : float
        Allowed absolute moment violation ``ε``.
    max_iter : int
        Maximum number of game rounds.
    nu : float, optional
        Duality-gap tolerance.  Defaults to fairlearn's heuristic
        ``0.5 · std(|h₀(x) − y|) / √n`` for the unconstrained classifier ``h₀``.  The gap is checked for both
        the averaged EG iterates and the LP re-weighting (with its dual ``λ``);
        the smaller one is used.
    eta0 : float
        Base learning rate; the step is ``eta0 / B``.
    bound : float, optional
        ``B``, the ℓ₁ bound on ``λ``.  Defaults to ``1 / eps``.
    """

    def __init__(
        self,
        estimator=None,
        constraint: str = "demographic_parity",
        eps: float = 0.01,
        max_iter: int = 50,
        nu: Optional[float] = None,
        eta0: float = 2.0,
        bound: Optional[float] = None,
    ):
        if constraint not in CONSTRAINTS:
            raise ValueError(f"constraint must be one of {CONSTRAINTS}")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if eta0 <= 0:
            raise ValueError("eta0 must be positive")
        self.estimator = estimator
        self.constraint = constraint
        self.eps = float(eps)
        self.max_iter = int(max_iter)
        self.nu = None if nu is None else float(nu)
        self.bound = 1.0 / self.eps if bound is None else float(bound)
        if self.bound <= 0:
            raise ValueError("bound must be positive")
        self.eta0 = float(eta0)

    # -- internals ---------------------------------------------------------
    def _best_response(self, X, y, lam_signed):
        n = len(y)
        cost = (1.0 - 2.0 * y) / n + self._moment.signed_weights(lam_signed)
        labels = (cost < 0).astype(int)
        weights = np.abs(cost) * n
        if labels.min() == labels.max() or weights.sum() <= 0:
            value = labels[0] if weights.sum() > 0 else int(y.mean() >= 0.5)
            clf = _ConstantClassifier(value)
        else:
            clf = _clone(self._base).fit(X, labels, sample_weight=weights)
        pred = np.asarray(clf.predict(X), dtype=float)
        return clf, pred

    def _lagrangian(self, err, gamma_vec, lam):
        return err + float(lam @ (gamma_vec - self.eps))

    def _gap(self, X, y, q_err, q_gamma, lam, gamma_hat, signed) -> float:
        """Duality gap ``max(L(Q, λ*) − L(Q, λ), L(Q, λ) − L(h(λ), λ))``."""
        L = self._lagrangian(q_err, q_gamma, lam)
        viol = q_gamma - self.eps
        # best λ against Q: all mass B on the worst violated constraint (or none)
        L_high = q_err + (self.bound * viol.max() if viol.max() > 0 else 0.0)
        _, pred = self._best_response(X, y, signed(lam))
        L_low = self._lagrangian(float(np.mean(np.abs(pred - y))), gamma_hat(pred), lam)
        return float(max(L_high - L, L - L_low, 0.0))

    def fit(self, X, y, sensitive_features):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y).ravel()
        if not np.isin(y, [0, 1]).all():
            raise ValueError("y must be binary 0/1")
        y = y.astype(float)
        groups = np.asarray(sensitive_features).ravel()
        if not (len(X) == len(y) == len(groups)):
            raise ValueError("X, y and sensitive_features must have the same length")
        self._base = self.estimator if self.estimator is not None else _default_estimator()
        self._moment = m = _Moment(self.constraint, y.astype(int), groups)
        k2 = 2 * m.k

        def gamma_hat(pred):
            g = m.gamma(pred)
            return np.concatenate([g, -g])

        def signed(lam):
            return lam[: m.k] - lam[m.k:]

        B, eta = self.bound, self.eta0 / self.bound
        theta = np.zeros(k2)
        hyps, errs, gammas = [], [], []
        lam_sum = np.zeros(k2)
        gap_history: List[float] = []
        converged = False
        gap = float("inf")
        last_gap = float("inf")
        t = 0
        nu = self.nu
        for t in range(1, self.max_iter + 1):
            e = np.exp(theta - theta.max())
            lam = B * e / (np.exp(-theta.max()) + e.sum())
            clf, pred = self._best_response(X, y, signed(lam))
            hyps.append(clf)
            errs.append(float(np.mean(np.abs(pred - y))))
            gammas.append(gamma_hat(pred))
            lam_sum += lam
            if nu is None:
                nu = 0.5 * float(np.std(np.abs(hyps[0].predict(X) - y), ddof=1)) / np.sqrt(len(y))
            # (1) averaged EG iterates Q̄, λ̄
            q_err = float(np.mean(errs))
            q_gamma = np.mean(gammas, axis=0)
            lam_bar = lam_sum / t
            gap_eg = self._gap(X, y, q_err, q_gamma, lam_bar, gamma_hat, signed)
            # (2) LP re-weighting of the hypotheses found so far, with its dual λ
            q, lam_lp = self._solve_lp(np.array(errs), np.array(gammas))
            lp_gamma = q @ np.array(gammas)
            gap_lp = self._gap(X, y, float(q @ np.array(errs)), lp_gamma, lam_lp, gamma_hat, signed)
            gap = min(gap_eg, gap_lp)
            gap_history.append(float(gap))
            if gap <= nu:
                converged = True
                break
            if t > 1 and gap_eg > last_gap:
                eta *= 0.8  # shrink the step when the EG gap grows (as fairlearn does)
            last_gap = gap_eg
            theta += eta * (gammas[-1] - self.eps)

        self.weights_, self.lambda_ = self._solve_lp(np.array(errs), np.array(gammas))
        self.hypotheses_ = hyps
        self.n_iter_ = t
        self.nu_ = float(nu)
        pred0 = np.asarray(hyps[0].predict(X), dtype=float)
        proba = self.predict_proba(X)
        labels = m.labels()
        g0, g1 = m.gamma(pred0), m.gamma(proba)
        self.result_ = ExponentiatedGradientResult(
            constraint=self.constraint,
            eps=self.eps,
            bound=B,
            n_iter=t,
            converged=converged,
            duality_gap=float(gap),
            n_hypotheses=int(np.sum(self.weights_ > 1e-9)),
            weights=[float(w) for w in self.weights_],
            error_before=float(np.mean(np.abs(pred0 - y))),
            error_after=float(np.mean(np.abs(proba - y))),
            violation_before=float(np.max(np.abs(g0))),
            violation_after=float(np.max(np.abs(g1))),
            moments_before={lab: float(v) for lab, v in zip(labels, g0)},
            moments_after={lab: float(v) for lab, v in zip(labels, g1)},
            gap_history=gap_history,
        )
        return self

    def _solve_lp(self, errs: np.ndarray, gammas: np.ndarray) -> np.ndarray:
        """``min Σ q err + B s`` s.t. ``Γᵀq − s ≤ ε``, ``Σ q = 1``, ``q, s ≥ 0``.

        Returns the mixture weights and the LP dual ``λ`` of the moment constraints.
        """
        from scipy.optimize import linprog

        h, k2 = gammas.shape
        c = np.concatenate([errs, [self.bound]])
        A_ub = np.hstack([gammas.T, -np.ones((k2, 1))])
        b_ub = np.full(k2, self.eps)
        A_eq = np.concatenate([np.ones(h), [0.0]]).reshape(1, -1)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=[(0, None)] * (h + 1), method="highs")
        if not res.success:  # pragma: no cover - LP is always feasible (s absorbs violations)
            return np.full(h, 1.0 / h), np.zeros(k2)
        q = np.clip(res.x[:h], 0, None)
        q[q < 1e-12] = 0.0
        lam = np.clip(-np.asarray(res.ineqlin.marginals, dtype=float), 0.0, None)
        return q / q.sum(), lam

    # -- prediction --------------------------------------------------------
    def _check_fitted(self):
        if not hasattr(self, "weights_"):
            raise RuntimeError("ExponentiatedGradient is not fitted")

    def predict_proba(self, X) -> np.ndarray:
        """Probability of predicting 1 under the randomized classifier."""
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        p = np.zeros(len(X))
        for w, h in zip(self.weights_, self.hypotheses_):
            if w > 0:
                p += w * np.asarray(h.predict(X), dtype=float)
        return np.clip(p, 0.0, 1.0)

    def predict(self, X, random_state=None) -> np.ndarray:
        """Randomized 0/1 decisions sampled from :meth:`predict_proba`."""
        p = self.predict_proba(X)
        rng = np.random.default_rng(random_state)
        return (rng.random(len(p)) < p).astype(int)


def fit_exponentiated_gradient(X, y, sensitive_features, **kwargs) -> ExponentiatedGradientResult:
    """Fit :class:`ExponentiatedGradient` and return its training summary."""
    return ExponentiatedGradient(**kwargs).fit(X, y, sensitive_features).result_


def render_exponentiated_gradient_report(
    result: ExponentiatedGradientResult, title: str = "Exponentiated Gradient Reduction Report"
) -> str:
    """Markdown summary of an :class:`ExponentiatedGradientResult`."""
    lines = [
        f"# {title}",
        "",
        f"Constraint: **{result.constraint.replace('_', ' ')}**, ε = {result.eps:.4f}, B = {result.bound:.2f}",
        "",
        f"- Iterations: {result.n_iter} ({'converged' if result.converged else 'stopped at max_iter'}, duality gap {result.duality_gap:.2e})",
        f"- Hypotheses in final mixture: {result.n_hypotheses}",
        f"- Feasible on training data: {'yes' if result.feasible else 'no'}",
        "",
        "| | Unconstrained | Reduction |",
        "|---|---|---|",
        f"| Error | {result.error_before:.4f} | {result.error_after:.4f} |",
        f"| Max moment violation | {result.violation_before:.4f} | {result.violation_after:.4f} |",
        "",
        "## Moments γ",
        "",
        "| Moment | Unconstrained | Reduction |",
        "|---|---|---|",
    ]
    for key in result.moments_before:
        lines.append(f"| {key} | {result.moments_before[key]:+.4f} | {result.moments_after[key]:+.4f} |")
    lines.append("")
    return "\n".join(lines)
