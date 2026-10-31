"""
Massaging: label-changing pre-processing (Kamiran & Calders, KAIS 2012).

Reweighing keeps labels and changes sample weights. Massaging instead
relabels the fewest training examples needed to remove the base-rate gap
between a privileged and an unprivileged group:

1. Rank every sample with a scorer of ``P(Y = positive | x)`` (any
   probabilistic classifier trained *without* the sensitive attribute, or
   precomputed scores).
2. With discrimination ``disc = P(Y=+ | priv) - P(Y=+ | unpriv) > 0``, the
   number of swaps that equalises the two base rates is

       M = round(disc * n_priv * n_unpriv / (n_priv + n_unpriv)).

3. *Promote* the ``M`` unprivileged negatives with the highest scores
   (closest to the decision boundary) to positive, and *demote* the ``M``
   privileged positives with the lowest scores to negative.

The overall positive rate is unchanged because the number of promotions
equals the number of demotions. Picking the borderline examples keeps the
accuracy cost of the relabelling as small as possible.

If the unprivileged group is the one with the higher base rate (``disc < 0``)
the roles flip automatically. ``M`` is capped by the number of available
candidates in each pool.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class MassagingResult:
    """Massaged labels and diagnostics."""

    labels: np.ndarray
    n_samples: int
    n_changes: int
    promoted_indices: np.ndarray
    demoted_indices: np.ndarray
    privileged_group: Any
    unprivileged_group: Any
    positive_label: Any
    negative_label: Any
    base_rates_before: Dict[str, float]
    base_rates_after: Dict[str, float]
    base_rate_gap_before: float
    base_rate_gap_after: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "n_samples": self.n_samples,
            "n_changes": self.n_changes,
            "n_label_flips": 2 * self.n_changes,
            "promoted_indices": self.promoted_indices.tolist(),
            "demoted_indices": self.demoted_indices.tolist(),
            "privileged_group": str(self.privileged_group),
            "unprivileged_group": str(self.unprivileged_group),
            "positive_label": str(self.positive_label),
            "negative_label": str(self.negative_label),
            "base_rates_before": self.base_rates_before,
            "base_rates_after": self.base_rates_after,
            "base_rate_gap_before": self.base_rate_gap_before,
            "base_rate_gap_after": self.base_rate_gap_after,
            "notes": list(self.notes),
        }


def _rates(y: np.ndarray, g: np.ndarray, groups_order, positive_label) -> Dict[str, float]:
    return {str(gv): float(np.mean(y[g == gv] == positive_label)) for gv in groups_order}


def compute_massaging(
    y_true,
    groups,
    scores,
    privileged_group: Optional[Any] = None,
    positive_label: Optional[Any] = None,
) -> MassagingResult:
    """
    Relabel the minimal number of borderline samples to equalise base rates.

    Args:
        y_true: Binary training labels.
        groups: Sensitive attribute with exactly two values.
        scores: Ranking scores, higher = more likely positive (e.g. a
            classifier's ``predict_proba[:, 1]``).
        privileged_group: Group expected to have the higher base rate.
            Defaults to the group whose observed base rate is higher.
        positive_label: Favourable label. Defaults to ``1`` when present,
            otherwise the larger label.

    Returns:
        MassagingResult with the new ``labels`` array (input is not modified).
    """
    y = np.asarray(y_true)
    g = np.asarray(groups)
    s = np.asarray(scores, dtype=float)
    if y.ndim != 1 or g.ndim != 1 or s.ndim != 1:
        raise ValueError("y_true, groups and scores must be 1-D")
    if not (len(y) == len(g) == len(s)):
        raise ValueError("y_true, groups and scores must have the same length")
    if len(y) == 0:
        raise ValueError("need at least one sample")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")

    labels = sorted(np.unique(y).tolist(), key=lambda v: (str(type(v)), v))
    if len(labels) != 2:
        raise ValueError(f"massaging needs binary labels, got {len(labels)} distinct values")
    if positive_label is None:
        positive_label = 1 if 1 in labels else labels[-1]
    elif positive_label not in labels:
        raise ValueError(f"positive_label {positive_label!r} not found in y_true")
    negative_label = labels[0] if labels[1] == positive_label else labels[1]

    group_values = sorted(np.unique(g).tolist(), key=str)
    if len(group_values) != 2:
        raise ValueError(f"massaging needs exactly two groups, got {len(group_values)}")
    before = _rates(y, g, group_values, positive_label)
    if privileged_group is None:
        privileged_group = max(group_values, key=lambda gv: before[str(gv)])
    elif privileged_group not in group_values:
        raise ValueError(f"privileged_group {privileged_group!r} not found in groups")
    unprivileged_group = group_values[1] if group_values[0] == privileged_group else group_values[0]

    notes: List[str] = []
    hi, lo = privileged_group, unprivileged_group
    if before[str(hi)] < before[str(lo)]:
        notes.append(
            f"group {lo!r} has the higher base rate; promoting {hi!r} and demoting {lo!r} instead"
        )
        hi, lo = lo, hi

    n_hi = int(np.sum(g == hi))
    n_lo = int(np.sum(g == lo))
    disc = before[str(hi)] - before[str(lo)]
    m = int(round(disc * n_hi * n_lo / (n_hi + n_lo)))

    pos = y == positive_label
    promote_pool = np.flatnonzero((g == lo) & ~pos)
    demote_pool = np.flatnonzero((g == hi) & pos)
    cap = min(len(promote_pool), len(demote_pool))
    if m > cap:
        notes.append(f"needed {m} swaps but only {cap} candidates available; capped")
        m = cap

    # Stable tie-breaking by index keeps the result deterministic.
    promote_order = promote_pool[np.lexsort((promote_pool, -s[promote_pool]))]
    demote_order = demote_pool[np.lexsort((demote_pool, s[demote_pool]))]
    promoted = np.sort(promote_order[:m])
    demoted = np.sort(demote_order[:m])

    new_y = y.copy()
    new_y[promoted] = positive_label
    new_y[demoted] = negative_label
    after = _rates(new_y, g, group_values, positive_label)

    def _gap(r: Dict[str, float]) -> float:
        return float(max(r.values()) - min(r.values()))

    return MassagingResult(
        labels=new_y,
        n_samples=len(y),
        n_changes=m,
        promoted_indices=promoted,
        demoted_indices=demoted,
        privileged_group=privileged_group,
        unprivileged_group=unprivileged_group,
        positive_label=positive_label,
        negative_label=negative_label,
        base_rates_before=before,
        base_rates_after=after,
        base_rate_gap_before=_gap(before),
        base_rate_gap_after=_gap(after),
        notes=notes,
    )


def fit_with_massaging(
    estimator,
    X,
    y_true,
    groups,
    ranker=None,
    privileged_group: Optional[Any] = None,
    positive_label: Optional[Any] = None,
    **fit_params,
):
    """
    Massage labels with a ranker, then fit ``estimator`` on the new labels.

    ``ranker`` defaults to scikit-learn's ``LogisticRegression``; it is fitted
    on ``(X, y_true)`` and its positive-class probability is the ranking
    score. Returns ``(fitted_estimator, MassagingResult)``.
    """
    y = np.asarray(y_true)
    if ranker is None:
        from sklearn.linear_model import LogisticRegression

        ranker = LogisticRegression(max_iter=1000)
    ranker.fit(X, y)
    classes = list(ranker.classes_)
    labels = sorted(np.unique(y).tolist(), key=lambda v: (str(type(v)), v))
    pos_label = positive_label if positive_label is not None else (1 if 1 in labels else labels[-1])
    scores = ranker.predict_proba(X)[:, classes.index(pos_label)]
    result = compute_massaging(y, groups, scores, privileged_group, pos_label)
    estimator.fit(X, result.labels, **fit_params)
    return estimator, result


def render_massaging_report(result: MassagingResult) -> str:
    """Markdown summary of a massaging run."""
    lines = [
        "# Massaging (label relabelling) report",
        "",
        f"- Samples: {result.n_samples}",
        f"- Privileged group: `{result.privileged_group}`; unprivileged: `{result.unprivileged_group}`",
        f"- Promotions / demotions: {result.n_changes} / {result.n_changes}",
        f"- Base-rate gap: {result.base_rate_gap_before:.4f} -> {result.base_rate_gap_after:.4f}",
        "",
        "| Group | Base rate before | Base rate after |",
        "|---|---|---|",
    ]
    for key in result.base_rates_before:
        lines.append(
            f"| {key} | {result.base_rates_before[key]:.4f} | {result.base_rates_after[key]:.4f} |"
        )
    for note in result.notes:
        lines.append(f"\n> {note}")
    return "\n".join(lines) + "\n"
