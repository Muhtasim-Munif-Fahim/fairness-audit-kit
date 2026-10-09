"""Tests for Absolute Between-ROC Area (ABROCA)."""

from __future__ import annotations

import numpy as np
import pytest

from fairness_audit_kit.abroca import compute_abroca


def _two_group_scores(seed: int = 0, gap: float = 0.0):
    rng = np.random.default_rng(seed)
    n = 200
    y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=float)
    # Base scores correlate with label; group B gets an additive gap on positives.
    scores = y + 0.3 * rng.normal(size=n)
    groups = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    # Shuffle within? Keep simple: first half A, second B — but labels are
    # also first-half 0 second-half 1 which confounds. Rebuild properly.
    y = rng.integers(0, 2, size=n).astype(float)
    groups = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    scores = y + 0.5 * rng.normal(size=n)
    scores = scores.copy()
    scores[groups == "B"] = scores[groups == "B"] + gap * y[groups == "B"]
    return y, scores, groups


def test_identical_groups_near_zero():
    rng = np.random.default_rng(1)
    n = 300
    y = rng.integers(0, 2, size=n).astype(float)
    scores = y + 0.2 * rng.normal(size=n)
    # Same generative process, different group labels — ABROCA should be small.
    groups = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    # Force identical scores/labels by mirroring A onto B.
    y[n // 2 :] = y[: n // 2]
    scores[n // 2 :] = scores[: n // 2]
    result = compute_abroca(y, scores, groups)
    assert result.max_abroca == pytest.approx(0.0, abs=1e-9)
    assert result.mean_abroca == pytest.approx(0.0, abs=1e-9)


def test_separated_groups_positive():
    y, scores, groups = _two_group_scores(seed=2, gap=2.0)
    result = compute_abroca(y, scores, groups)
    assert result.n_groups == 2
    assert result.max_abroca > 0.01
    assert 0.0 <= result.max_abroca <= 1.0 + 1e-6


def test_pairwise_and_auc_present():
    y, scores, groups = _two_group_scores(seed=3, gap=0.5)
    result = compute_abroca(y, scores, groups)
    assert len(result.pairwise) == 1
    pair = result.pairwise[0]
    assert pair.abroca == result.max_abroca
    assert set(result.group_roc) == {"A", "B"}
    for roc in result.group_roc.values():
        assert roc.fpr[0] == pytest.approx(0.0)
        assert roc.fpr[-1] == pytest.approx(1.0)
        assert np.all(np.diff(roc.fpr) >= -1e-12)


def test_three_groups():
    rng = np.random.default_rng(4)
    n = 300
    y = rng.integers(0, 2, size=n).astype(float)
    scores = y + 0.3 * rng.normal(size=n)
    groups = np.array(["A"] * 100 + ["B"] * 100 + ["C"] * 100)
    result = compute_abroca(y, scores, groups)
    assert result.n_groups == 3
    assert len(result.pairwise) == 3
    assert result.max_abroca >= result.mean_abroca


def test_to_dict():
    y, scores, groups = _two_group_scores(seed=5, gap=0.1)
    d = compute_abroca(y, scores, groups).to_dict()
    assert "max_abroca" in d
    assert "pairwise" in d
    assert "group_roc" in d


def test_requires_two_groups():
    with pytest.raises(ValueError, match="two groups"):
        compute_abroca([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], ["A", "A", "A", "A"])


def test_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        compute_abroca([0, 1], [0.1], ["A", "B"])


def test_non_binary_labels():
    with pytest.raises(ValueError, match="binary"):
        compute_abroca([0, 2, 0, 1], [0.1, 0.9, 0.2, 0.8], ["A", "A", "B", "B"])
