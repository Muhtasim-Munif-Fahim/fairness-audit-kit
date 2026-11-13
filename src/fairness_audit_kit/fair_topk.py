"""FA*IR: fair top-k ranking with a ranked group fairness guarantee.

Zehlike et al. (CIKM 2017) define *ranked group fairness*: a top-k ranking
is fair for a protected group with target proportion ``p`` if, for **every**
prefix ``i = 1..k``, the number of protected items ``tau_i`` is not
significantly below what a Bernoulli(``p``) process would produce, i.e.

    BinomCDF(tau_i; i, p) > alpha_c.

This gives the *mtable* ``m(i)`` = minimum protected items required in the top
``i``. Because the test is applied ``k`` times, a fair process would fail at
least one prefix more often than ``alpha``. FA*IR therefore uses an adjusted
significance ``alpha_c <= alpha``. It is chosen so that the exact probability
that a random Bernoulli(``p``) ranking violates the mtable somewhere,
computed here by dynamic programming over prefix counts, is at most ``alpha``.

:func:`fair_topk` is the FA*IR algorithm. It keeps two score-sorted queues
(protected / non-protected). At position ``i`` it takes the best protected
candidate if the prefix is short of ``m(i)``, and otherwise the best of the
two queue heads. Within each group the score order is preserved. The result
is the utility-maximising ranking that satisfies the mtable.
:func:`ranked_group_fairness_test` checks any given ranking against the
same condition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from scipy import stats

__all__ = [
    "fair_mtable",
    "mtable_failure_probability",
    "adjust_alpha",
    "RankedGroupFairnessTest",
    "ranked_group_fairness_test",
    "FairTopKResult",
    "fair_topk",
]


def _check_k(k) -> int:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    return int(k)


def _check_p(p) -> float:
    p = float(p)
    if not 0.0 <= p < 1.0:
        raise ValueError("p must be in [0, 1)")
    return p


def _check_alpha(alpha) -> float:
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return alpha


def fair_mtable(k: int, p: float, alpha: float) -> np.ndarray:
    """Minimum protected count ``m(i)`` for prefixes ``i = 1..k`` (length-k int array).

    ``m(i)`` is the smallest ``t`` with ``BinomCDF(t; i, p) > alpha``.
    """
    k, p, alpha = _check_k(k), _check_p(p), _check_alpha(alpha)
    i = np.arange(1, k + 1)
    m = stats.binom.ppf(alpha, i, p).astype(int)
    m = np.maximum(m, 0)
    # ppf gives CDF >= alpha; the condition is strict (> alpha).
    bump = stats.binom.cdf(m, i, p) <= alpha
    m = m + bump.astype(int)
    return m


def mtable_failure_probability(mtable: Sequence[int], p: float) -> float:
    """Exact probability that a Bernoulli(``p``) ranking violates ``mtable`` at some prefix."""
    m = np.asarray(mtable, dtype=int)
    p = _check_p(p)
    k = m.size
    # dist[t] = P(t protected so far and every earlier prefix satisfied)
    dist = np.zeros(k + 1)
    dist[0] = 1.0
    for i in range(k):
        new = np.zeros(k + 1)
        new[: i + 1] += dist[: i + 1] * (1.0 - p)
        new[1 : i + 2] += dist[: i + 1] * p
        new[: m[i]] = 0.0
        dist = new
    return float(max(0.0, 1.0 - dist.sum()))


def adjust_alpha(k: int, p: float, alpha: float, tol: float = 1e-6, max_iter: int = 100) -> float:
    """Largest ``alpha_c <= alpha`` whose mtable fails a fair ranking with probability <= ``alpha``."""
    k, p, alpha = _check_k(k), _check_p(p), _check_alpha(alpha)
    if mtable_failure_probability(fair_mtable(k, p, alpha), p) <= alpha:
        return alpha
    lo, hi = 0.0, alpha
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if mid <= 0.0:
            break
        if mtable_failure_probability(fair_mtable(k, p, mid), p) <= alpha:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return lo if lo > 0.0 else tol


@dataclass
class RankedGroupFairnessTest:
    k: int
    p: float
    alpha: float
    alpha_adjusted: float
    mtable: np.ndarray
    protected_counts: np.ndarray
    fair: bool
    first_violation: Optional[int]  # 1-based prefix length, None if fair
    failure_probability: float


def _resolve_alpha(k, p, alpha, adjust):
    return adjust_alpha(k, p, alpha) if adjust else _check_alpha(alpha)


def ranked_group_fairness_test(
    protected_in_rank_order: Sequence[bool], p: float, alpha: float = 0.1, *, adjust: bool = True
) -> RankedGroupFairnessTest:
    """Check a ranking (protected flags, top first) against the FA*IR condition."""
    flags = np.asarray(protected_in_rank_order, dtype=bool)
    if flags.ndim != 1 or flags.size == 0:
        raise ValueError("protected_in_rank_order must be a non-empty 1-D sequence")
    k = flags.size
    p = _check_p(p)
    a_c = _resolve_alpha(k, p, alpha, adjust)
    m = fair_mtable(k, p, a_c)
    counts = np.cumsum(flags)
    bad = np.flatnonzero(counts < m)
    return RankedGroupFairnessTest(
        k=k, p=p, alpha=float(alpha), alpha_adjusted=a_c, mtable=m, protected_counts=counts,
        fair=bool(bad.size == 0), first_violation=int(bad[0]) + 1 if bad.size else None,
        failure_probability=mtable_failure_probability(m, p),
    )


@dataclass
class FairTopKResult:
    ranking: np.ndarray  # candidate indices, top first
    protected: np.ndarray  # protected flag per ranked position
    scores: np.ndarray  # score per ranked position
    mtable: np.ndarray
    alpha_adjusted: float
    feasible: bool
    n_promoted: int  # protected items placed above a higher-scoring non-protected candidate
    utility_ratio: float  # sum of scores vs. colour-blind top-k
    colorblind_ranking: np.ndarray
    notes: List[str] = field(default_factory=list)


def fair_topk(
    scores: Sequence[float],
    protected: Sequence[bool],
    k: int,
    p: float,
    alpha: float = 0.1,
    *,
    adjust: bool = True,
) -> FairTopKResult:
    """FA*IR top-k selection (Zehlike et al. 2017, Algorithm 2).

    ``scores`` are candidate qualifications (higher is better); ``protected``
    marks the protected group. Ties in score are broken by original index.
    If there are too few protected candidates to meet the mtable, the
    ranking uses all of them and ``feasible`` is ``False``.
    """
    s = np.asarray(scores, dtype=float)
    g = np.asarray(protected, dtype=bool)
    if s.ndim != 1 or g.shape != s.shape:
        raise ValueError("scores and protected must be 1-D sequences of equal length")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    k = _check_k(k)
    if k > s.size:
        raise ValueError("k cannot exceed the number of candidates")
    p = _check_p(p)
    a_c = _resolve_alpha(k, p, alpha, adjust)
    m = fair_mtable(k, p, a_c)
    order = np.lexsort((np.arange(s.size), -s))
    prot_q = [int(i) for i in order if g[i]]
    non_q = [int(i) for i in order if not g[i]]
    ranking: List[int] = []
    n_prot = 0
    feasible = True
    pi = ni = 0
    for pos in range(k):
        need = n_prot < m[pos]
        if need and pi < len(prot_q):
            take, pi = prot_q[pi], pi + 1
        else:
            if need:
                feasible = False
            if pi < len(prot_q) and (ni >= len(non_q) or (s[prot_q[pi]], -prot_q[pi]) > (s[non_q[ni]], -non_q[ni])):
                take, pi = prot_q[pi], pi + 1
            else:
                take, ni = non_q[ni], ni + 1
        n_prot += int(g[take])
        ranking.append(take)
    ranking_arr = np.array(ranking, dtype=int)
    blind = order[:k]
    # A protected item is "promoted" if some non-protected candidate with a
    # strictly higher score is ranked below it or left out of the top k.
    promoted = 0
    placed = set()
    for j in ranking_arr:
        placed.add(int(j))
        if g[j] and any(s[x] > s[j] for x in non_q if x not in placed):
            promoted += 1
    blind_sum = float(s[blind].sum())
    res = FairTopKResult(
        ranking=ranking_arr,
        protected=g[ranking_arr],
        scores=s[ranking_arr],
        mtable=m,
        alpha_adjusted=a_c,
        feasible=feasible,
        n_promoted=int(promoted),
        utility_ratio=float(s[ranking_arr].sum() / blind_sum) if blind_sum != 0 else 1.0,
        colorblind_ranking=blind,
    )
    if not feasible:
        res.notes.append("not enough protected candidates to satisfy the mtable")
    return res
