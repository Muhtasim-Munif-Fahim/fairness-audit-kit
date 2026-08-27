"""
Tests for synthetic biased dataset generator.
"""

import numpy as np
import pandas as pd
import pytest
from fairness_audit_kit.generator import (
    BiasedDataset,
    generate_biased_dataset,
    generate_multiple_biased_datasets,
    _inject_label_flip,
    _inject_feature_bias,
    _inject_correlation_bias,
)


class TestBiasedDataset:
    def test_to_dataframe(self):
        dataset = BiasedDataset(
            X=np.array([[1, 2], [3, 4]]),
            y=np.array([0, 1]),
            groups=np.array([0, 1]),
            feature_names=["f1", "f2"],
            group_names=["g0", "g1"],
            bias_info={}
        )
        df = dataset.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["f1", "f2", "target", "sensitive_attr"]
        assert len(df) == 2

    def test_to_csv(self, tmp_path):
        dataset = BiasedDataset(
            X=np.array([[1, 2], [3, 4]]),
            y=np.array([0, 1]),
            groups=np.array([0, 1]),
            feature_names=["f1", "f2"],
            group_names=["g0", "g1"],
            bias_info={}
        )
        path = tmp_path / "test.csv"
        dataset.to_csv(str(path))
        df = pd.read_csv(path)
        assert len(df) == 2


class TestInjectLabelFlip:
    def test_basic_flip(self):
        y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_biased, info = _inject_label_flip(y, groups, flip_rate=0.5, target_group=0, random_state=42)
        
        assert info["type"] == "label_flip"
        assert info["flip_rate"] == 0.5
        assert info["target_group"] == "0"
        assert info["n_flipped"] == 2  # 4 samples in group 0, 50% = 2
        # Check that flips occurred
        assert np.sum(y_biased != y) == 2

    def test_zero_flip_rate(self):
        y = np.array([0, 1, 0, 1])
        groups = np.array([0, 0, 1, 1])
        y_biased, info = _inject_label_flip(y, groups, flip_rate=0.0, target_group=0, random_state=42)
        
        np.testing.assert_array_equal(y_biased, y)
        assert info["n_flipped"] == 0

    def test_full_flip_rate(self):
        y = np.array([0, 0, 1, 1])
        groups = np.array([0, 0, 1, 1])
        y_biased, info = _inject_label_flip(y, groups, flip_rate=1.0, target_group=0, random_state=42)
        
        assert info["n_flipped"] == 2
        # All flipped
        assert np.all(y_biased[:2] == 1 - y[:2])

    def test_target_group_not_affected(self):
        y = np.array([0, 0, 1, 1])
        groups = np.array([0, 0, 1, 1])
        y_biased, info = _inject_label_flip(y, groups, flip_rate=1.0, target_group=0, random_state=42)
        
        # Group 1 should be unchanged
        np.testing.assert_array_equal(y_biased[2:], y[2:])


class TestInjectFeatureBias:
    def test_basic_shift(self):
        X = np.zeros((8, 5))
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        X_biased, info = _inject_feature_bias(X, groups, bias_strength=1.0, target_group=0, n_features=2, random_state=42)
        
        assert info["type"] == "feature_bias"
        assert info["bias_strength"] == 1.0
        assert info["target_group"] == "0"
        assert len(info["biased_features"]) == 2
        # Group 0 features should be shifted
        for feat_idx in info["biased_features"]:
            assert not np.allclose(X_biased[:4, feat_idx], 0)

    def test_zero_strength(self):
        X = np.zeros((4, 3))
        groups = np.array([0, 0, 1, 1])
        X_biased, info = _inject_feature_bias(X, groups, bias_strength=0.0, target_group=0, n_features=1, random_state=42)
        
        np.testing.assert_array_equal(X_biased, X)

    def test_group_1_not_affected(self):
        X = np.zeros((4, 3))
        groups = np.array([0, 0, 1, 1])
        X_biased, info = _inject_feature_bias(X, groups, bias_strength=1.0, target_group=0, n_features=2, random_state=42)
        
        # Group 1 should be unchanged
        np.testing.assert_array_equal(X_biased[2:], X[2:])


class TestInjectCorrelationBias:
    def test_basic_correlation(self):
        X = np.zeros((8, 5))
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        X_biased, info = _inject_correlation_bias(X, y, groups, correlation_strength=0.8, target_group=0, random_state=42)
        
        assert info["type"] == "correlation_bias"
        assert info["correlation_strength"] == 0.8
        assert info["target_group"] == "0"
        assert "biased_feature" in info
        # Group 0 feature should correlate with y
        feat_idx = info["biased_feature"]
        assert not np.allclose(X_biased[:4, feat_idx], 0)


class TestGenerateBiasedDataset:
    def test_label_flip_default(self):
        dataset = generate_biased_dataset(
            n_samples=100,
            n_features=10,
            bias_type="label_flip",
            bias_params={"flip_rate": 0.3, "target_group": 0},
            random_state=42
        )
        
        assert isinstance(dataset, BiasedDataset)
        assert dataset.X.shape == (100, 10)
        assert dataset.y.shape == (100,)
        assert dataset.groups.shape == (100,)
        assert len(dataset.feature_names) == 10
        assert dataset.bias_info["bias_type"] == "label_flip"

    def test_feature_bias(self):
        dataset = generate_biased_dataset(
            n_samples=100,
            n_features=10,
            bias_type="feature_bias",
            bias_params={"bias_strength": 1.5, "target_group": 1, "n_features": 2},
            random_state=42
        )
        
        assert dataset.bias_info["bias_type"] == "feature_bias"
        assert len(dataset.bias_info["details"][0]["biased_features"]) == 2

    def test_correlation_bias(self):
        dataset = generate_biased_dataset(
            n_samples=100,
            n_features=10,
            bias_type="correlation_bias",
            bias_params={"correlation_strength": 0.9, "target_group": 0},
            random_state=42
        )
        
        assert dataset.bias_info["bias_type"] == "correlation_bias"

    def test_combined_bias(self):
        dataset = generate_biased_dataset(
            n_samples=100,
            n_features=10,
            bias_type="combined",
            bias_params={
                "label_flip": {"flip_rate": 0.2, "target_group": 0},
                "feature_bias": {"bias_strength": 1.0, "target_group": 1, "n_features": 1},
            },
            random_state=42
        )
        
        assert dataset.bias_info["bias_type"] == "combined"
        assert len(dataset.bias_info["details"]) == 2

    def test_invalid_bias_type(self):
        with pytest.raises(ValueError, match="Unknown bias_type"):
            generate_biased_dataset(n_samples=100, bias_type="invalid_type", random_state=42)

    def test_group_proportions(self):
        dataset = generate_biased_dataset(
            n_samples=100,
            n_groups=3,
            group_proportions=[0.5, 0.3, 0.2],
            random_state=42
        )
        
        unique, counts = np.unique(dataset.groups, return_counts=True)
        assert len(unique) == 3
        # Check approximate proportions
        assert abs(counts[0] / 100 - 0.5) < 0.05
        assert abs(counts[1] / 100 - 0.3) < 0.05
        assert abs(counts[2] / 100 - 0.2) < 0.05

    def test_invalid_proportions(self):
        with pytest.raises(AssertionError, match="sum to 1"):
            generate_biased_dataset(n_samples=100, group_proportions=[0.5, 0.5, 0.5], random_state=42)

    def test_reproducibility(self):
        ds1 = generate_biased_dataset(n_samples=100, n_features=10, bias_type="label_flip", random_state=42)
        ds2 = generate_biased_dataset(n_samples=100, n_features=10, bias_type="label_flip", random_state=42)
        
        np.testing.assert_array_equal(ds1.X, ds2.X)
        np.testing.assert_array_equal(ds1.y, ds2.y)
        np.testing.assert_array_equal(ds1.groups, ds2.groups)

    def test_different_seeds_different_data(self):
        ds1 = generate_biased_dataset(n_samples=100, n_features=10, random_state=42)
        ds2 = generate_biased_dataset(n_samples=100, n_features=10, random_state=43)
        
        assert not np.array_equal(ds1.X, ds2.X)


class TestGenerateMultipleBiasedDatasets:
    def test_multiple_datasets(self):
        datasets = generate_multiple_biased_datasets(
            n_datasets=3,
            n_samples=100,
            n_features=10,
            bias_type="label_flip",
            bias_params={"flip_rate": 0.2},
            random_state=42
        )
        
        assert len(datasets) == 3
        for ds in datasets:
            assert isinstance(ds, BiasedDataset)
            assert ds.X.shape == (100, 10)

    def test_different_seeds(self):
        datasets = generate_multiple_biased_datasets(
            n_datasets=3,
            n_samples=100,
            n_features=10,
            random_state=42
        )
        
        # Each dataset should have different data due to different seeds
        for i in range(3):
            for j in range(i+1, 3):
                assert not np.array_equal(datasets[i].X, datasets[j].X)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
