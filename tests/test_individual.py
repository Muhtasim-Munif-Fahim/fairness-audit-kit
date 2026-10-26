"""Tests for individual fairness (kNN consistency, Lipschitz violations)."""

import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import NearestNeighbors

from fairness_audit_kit import (
    compute_individual_fairness,
    consistency_score,
    render_individual_fairness_report,
)
from fairness_audit_kit.cli import main


def test_consistency_is_one_for_a_function_of_the_neighbourhood():
    X = np.repeat(np.array([[0.0], [10.0], [20.0]]), 6, axis=0)
    y = np.repeat([0, 1, 0], 6)
    assert consistency_score(X, y, n_neighbors=5) == pytest.approx(1.0)


def test_consistency_matches_brute_force_and_aif360_convention():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = (rng.uniform(size=60) > 0.5).astype(float)
    k = 4
    # Zemel definition excludes the point itself.
    D = np.linalg.norm(X[:, None] - X[None], axis=2)
    np.fill_diagonal(D, np.inf)
    nbrs = np.argsort(D, axis=1)[:, :k]
    expected = 1 - np.mean(np.abs(y - y[nbrs].mean(axis=1)))
    assert consistency_score(X, y, k, standardize=False) == pytest.approx(expected)
    # AIF360: NearestNeighbors(k).kneighbors(X) includes self.
    _, idx = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)
    aif = 1 - np.mean(np.abs(y - y[idx].mean(axis=1)))
    assert consistency_score(X, y, k, include_self=True, standardize=False) == pytest.approx(aif)


def test_random_labels_are_less_consistent_than_smooth_ones():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 2))
    smooth = (X[:, 0] > 0).astype(int)
    noisy = rng.integers(0, 2, size=300)
    assert consistency_score(X, smooth) > consistency_score(X, noisy) + 0.3


def test_group_consistency_detects_a_group_treated_arbitrarily():
    rng = np.random.default_rng(2)
    n = 400
    X = rng.normal(size=(n, 2))
    group = np.where(rng.uniform(size=n) < 0.5, "a", "b")
    y = (X[:, 0] > 0).astype(int)
    flip = (group == "b") & (rng.uniform(size=n) < 0.4)
    y[flip] = 1 - y[flip]
    res = compute_individual_fairness(X, y, group, n_neighbors=5)
    assert res.group_consistency["a"] > res.group_consistency["b"] + 0.1
    assert res.consistency_gap == pytest.approx(
        res.group_consistency["a"] - res.group_consistency["b"]
    )
    assert res.group_size["a"] + res.group_size["b"] == n
    assert res.lipschitz_constant is None and res.violation_rate is None


def test_lipschitz_audit():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    f = np.array([0.0, 0.5, 3.0, 3.2])
    res = compute_individual_fairness(
        X, f, n_neighbors=1, standardize=False, lipschitz_constant=1.0
    )
    # Unique 1-NN pairs: (0,1), (1,2) or (1,0)..., (2,3).
    assert res.n_pairs >= 2
    assert res.max_ratio == pytest.approx(2.5)
    worst = res.worst_violations[0]
    assert (worst.i, worst.j) == (1, 2)
    assert worst.output_gap == pytest.approx(2.5)
    assert all(v.ratio > 1.0 for v in res.worst_violations)
    lenient = compute_individual_fairness(
        X, f, n_neighbors=1, standardize=False, lipschitz_constant=3.0
    )
    assert lenient.violation_rate == 0.0 and lenient.worst_violations == []
    with pytest.raises(ValueError):
        compute_individual_fairness(X, f, n_neighbors=1, lipschitz_constant=-1.0)


def test_duplicate_rows_and_validation():
    X = np.zeros((5, 2))
    y = np.array([1, 1, 1, 1, 0])
    res = compute_individual_fairness(X, y, n_neighbors=2, lipschitz_constant=1.0)
    assert 0.0 <= res.consistency <= 1.0
    assert res.max_ratio == np.inf
    with pytest.raises(ValueError, match="too large"):
        consistency_score(X, y, n_neighbors=5)
    with pytest.raises(ValueError):
        consistency_score(X, y[:3])
    with pytest.raises(ValueError):
        consistency_score(np.array([[np.nan]]), [1])
    with pytest.raises(ValueError):
        consistency_score(X, y, n_neighbors=0)


def test_report_renders():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, 2))
    y = rng.uniform(size=50)
    res = compute_individual_fairness(X, y, np.repeat(["x", "y"], 25), lipschitz_constant=0.1)
    text = render_individual_fairness_report(res)
    assert "# Individual Fairness Report" in text
    assert "| x |" in text and "Worst Lipschitz violations" in text
    d = res.to_dict()
    assert set(d["group_consistency"]) == {"x", "y"}


def test_cli_individual(tmp_path, monkeypatch, capsys):
    rng = np.random.default_rng(4)
    n = 120
    df = pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "sensitive_attr": rng.integers(0, 2, size=n),
        "target": rng.integers(0, 2, size=n),
        "y_score": rng.uniform(size=n),
    })
    df["y_pred"] = (df["f1"] > 0).astype(int)
    data = tmp_path / "preds.csv"
    df.to_csv(data, index=False)
    out = tmp_path / "ind.md"
    monkeypatch.setattr(sys, "argv", [
        "fairness-audit", "individual", "--data", str(data), "--output", str(out),
        "--lipschitz", "0.5",
    ])
    main()
    printed = capsys.readouterr().out
    expected = consistency_score(df[["f1", "f2"]].values, df["y_pred"].values)
    assert f"{expected:.4f}" in printed
    assert "violation rate" in printed
    assert "Individual Fairness Report" in out.read_text()

    monkeypatch.setattr(sys, "argv", [
        "fairness-audit", "individual", "--data", str(data), "--output", str(out),
        "--features", "f1", "--use-scores", "--exclude", "y_pred",
    ])
    main()
    assert "(1 features" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", [
        "fairness-audit", "individual", "--data", str(data), "--output", str(out),
        "--features", "nope",
    ])
    with pytest.raises(SystemExit):
        main()
