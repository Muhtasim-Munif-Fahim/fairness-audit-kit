"""
Fairness metrics for rankings (search results, recommendations, shortlists).

Classification metrics in this kit look at one decision per person. In a
ranking, what matters is *where* each person lands, because attention drops
off quickly with position. This module audits a single ranking, given as the
sequence of sensitive-group labels in rank order (top first):

* **Exposure** (Singh & Joachims, 2018). Position ``r`` (1-based) receives
  attention ``v_r``. The default is the DCG discount ``1 / log2(1 + r)``;
  ``"rbp"`` uses ``p^(r-1)`` and ``"reciprocal"`` uses ``1/r``. A group's
  exposure is the mean ``v_r`` of its members. The *demographic parity of
  exposure* gap is max minus min group exposure, and the ratio is min over max.
* **Disparate treatment ratio** (Singh & Joachims, 2018). When relevance
  scores are given, exposure should be proportional to merit:
  ``(Exp_g / U_g)``, where ``U_g`` is the group's mean relevance. Reported as
  min over max (1.0 = exposure tracks merit).
* **rND / rKL** (Yang & Stoyanovich, 2017). For a binary protected group,
  compare the protected share in each top-``i`` prefix
  (``i = step, 2*step, ...``) with its overall share. Use the absolute
  difference (rND) or KL divergence (rKL), discounted by ``1/log2(i)`` and
  normalised by the value of the most unfair ranking of the same
  composition (all protected items at the bottom or the top, whichever is
  worse). The result is in [0, 1].
* **NDKL** (Geyik et al., 2019, LinkedIn Talent Search). For every prefix,
  take the KL divergence of the multi-group distribution in the prefix from
  a target distribution (overall proportions by default), discounted by
  ``1/log2(i + 1)`` and normalised by the sum of the discounts. 0 means
  every prefix matches the target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "position_weights",
    "group_exposure",
    "rnd",
    "rkl",
    "ndkl",
    "RankingFairnessResult",
    "compute_ranking_fairness",
    "render_ranking_fairness_report",
]

_EPS = 1e-12


def position_weights(n: int, kind: str = "log", p: float = 0.85) -> np.ndarray:
    """Attention weight of positions 1..n."""
    if n <= 0:
        raise ValueError("n must be positive")
    r = np.arange(1, n + 1, dtype=float)
    if kind == "log":
        return 1.0 / np.log2(1.0 + r)
    if kind == "reciprocal":
        return 1.0 / r
    if kind == "rbp":
        if not 0.0 < p < 1.0:
            raise ValueError("p must be in (0, 1)")
        return p ** (r - 1.0)
    raise ValueError("kind must be 'log', 'reciprocal' or 'rbp'")


def _labels(groups: Sequence) -> np.ndarray:
    arr = np.asarray([str(g) for g in groups], dtype=object)
    if arr.size == 0:
        raise ValueError("ranking must not be empty")
    return arr


def group_exposure(groups: Sequence, kind: str = "log", p: float = 0.85) -> Dict[str, float]:
    """Mean position weight received by each group's members."""
    g = _labels(groups)
    w = position_weights(len(g), kind, p)
    return {str(k): float(w[g == k].mean()) for k in sorted(set(g))}


def _protected_mask(groups: Sequence, protected) -> np.ndarray:
    g = _labels(groups)
    mask = g == str(protected)
    if not mask.any() or mask.all():
        raise ValueError("protected group must be present and not the whole ranking")
    return mask


def _prefix_cutoffs(n: int, step: int) -> List[int]:
    if step <= 0:
        raise ValueError("step must be positive")
    cuts = list(range(step, n + 1, step))
    if not cuts:
        raise ValueError("ranking is shorter than step")
    return cuts


def _bin_kl(p: float, q: float) -> float:
    p = min(max(p, _EPS), 1 - _EPS)
    q = min(max(q, _EPS), 1 - _EPS)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def _raw_set_metric(mask: np.ndarray, step: int, metric: str) -> float:
    n = mask.size
    share = mask.mean()
    total = 0.0
    for i in _prefix_cutoffs(n, step):
        if i < 2:
            continue  # 1/log2(1) undefined; Yang & Stoyanovich start at 10
        frac = mask[:i].mean()
        val = abs(frac - share) if metric == "nd" else _bin_kl(frac, share)
        total += val / np.log2(i)
    return total


def _normalised(mask: np.ndarray, step: int, metric: str) -> float:
    n, k = mask.size, int(mask.sum())
    bottom = np.r_[np.zeros(n - k, bool), np.ones(k, bool)]
    top = bottom[::-1]
    z = max(_raw_set_metric(bottom, step, metric), _raw_set_metric(top, step, metric))
    if z <= 0:
        return 0.0
    return float(_raw_set_metric(mask, step, metric) / z)


def rnd(groups: Sequence, protected, step: int = 10) -> float:
    """Normalised discounted difference (Yang & Stoyanovich, 2017), in [0, 1]."""
    return _normalised(_protected_mask(groups, protected), step, "nd")


def rkl(groups: Sequence, protected, step: int = 10) -> float:
    """Normalised discounted KL divergence (Yang & Stoyanovich, 2017), in [0, 1]."""
    return _normalised(_protected_mask(groups, protected), step, "kl")


def ndkl(groups: Sequence, target: Optional[Mapping[str, float]] = None) -> float:
    """Normalised discounted cumulative KL divergence (Geyik et al., 2019)."""
    g = _labels(groups)
    cats = sorted(set(g))
    if target is None:
        q = np.array([np.mean(g == c) for c in cats])
    else:
        tgt = {str(k): float(v) for k, v in target.items()}
        cats = sorted(set(cats) | set(tgt))
        q = np.array([tgt.get(c, 0.0) for c in cats])
        if (q < 0).any() or q.sum() <= 0:
            raise ValueError("target must be non-negative with positive mass")
        q = q / q.sum()
    onehot = np.stack([(g == c).astype(float) for c in cats], axis=1)
    counts = np.cumsum(onehot, axis=0)
    i = np.arange(1, g.size + 1, dtype=float)
    d = counts / i[:, None]
    qq = np.clip(q, _EPS, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = np.where(d > 0, d * np.log(d / qq), 0.0).sum(axis=1)
    disc = 1.0 / np.log2(i + 1.0)
    return float((disc * kl).sum() / disc.sum())


@dataclass
class RankingFairnessResult:
    n_items: int
    group_size: Dict[str, int]
    exposure: Dict[str, float]
    exposure_share: Dict[str, float]
    exposure_difference: float
    exposure_ratio: float
    mean_relevance: Optional[Dict[str, float]] = None
    disparate_treatment_ratio: Optional[float] = None
    ndkl: float = 0.0
    rnd: Optional[float] = None
    rkl: Optional[float] = None
    protected: Optional[str] = None
    notes: List[str] = field(default_factory=list)


def compute_ranking_fairness(
    groups: Sequence,
    relevance: Optional[Sequence[float]] = None,
    *,
    protected=None,
    weighting: str = "log",
    p: float = 0.85,
    step: int = 10,
    target: Optional[Mapping[str, float]] = None,
) -> RankingFairnessResult:
    """Audit one ranking. ``groups[i]`` is the group of the item at rank i+1."""
    g = _labels(groups)
    exp = group_exposure(g, weighting, p)
    sizes = {k: int((g == k).sum()) for k in exp}
    total = sum(exp[k] * sizes[k] for k in exp)
    share = {k: exp[k] * sizes[k] / total for k in exp}
    vals = list(exp.values())
    res = RankingFairnessResult(
        n_items=int(g.size),
        group_size=sizes,
        exposure=exp,
        exposure_share=share,
        exposure_difference=float(max(vals) - min(vals)),
        exposure_ratio=float(min(vals) / max(vals)),
        ndkl=ndkl(g, target),
    )
    if len(exp) < 2:
        res.notes.append("only one group present; disparity metrics are trivially fair")
    if relevance is not None:
        rel = np.asarray(relevance, dtype=float)
        if rel.shape != (g.size,):
            raise ValueError("relevance must have one value per ranked item")
        if (rel < 0).any() or not np.isfinite(rel).all():
            raise ValueError("relevance must be finite and non-negative")
        mrel = {k: float(rel[g == k].mean()) for k in exp}
        res.mean_relevance = mrel
        per_merit = [exp[k] / mrel[k] for k in exp if mrel[k] > 0]
        if len(per_merit) >= 2:
            res.disparate_treatment_ratio = float(min(per_merit) / max(per_merit))
        else:
            res.notes.append("disparate treatment ratio needs >= 2 groups with positive relevance")
    if protected is not None:
        res.protected = str(protected)
        res.rnd = rnd(g, protected, step)
        res.rkl = rkl(g, protected, step)
    return res


def render_ranking_fairness_report(result: RankingFairnessResult) -> str:
    lines = ["# Ranking fairness audit", "", f"Items ranked: {result.n_items}", ""]
    lines += ["| group | n | mean exposure | exposure share |", "|---|---:|---:|---:|"]
    for k in result.exposure:
        lines.append(
            f"| {k} | {result.group_size[k]} | {result.exposure[k]:.4f} | {result.exposure_share[k]:.3f} |"
        )
    lines += [
        "",
        f"- Exposure difference (max - min): {result.exposure_difference:.4f}",
        f"- Exposure ratio (min / max): {result.exposure_ratio:.3f}",
        f"- NDKL: {result.ndkl:.4f}",
    ]
    if result.disparate_treatment_ratio is not None:
        lines.append(f"- Disparate treatment ratio: {result.disparate_treatment_ratio:.3f}")
    if result.rnd is not None:
        lines.append(f"- rND ({result.protected}): {result.rnd:.3f}")
        lines.append(f"- rKL ({result.protected}): {result.rkl:.3f}")
    for n in result.notes:
        lines.append(f"- Note: {n}")
    return "\n".join(lines) + "\n"
