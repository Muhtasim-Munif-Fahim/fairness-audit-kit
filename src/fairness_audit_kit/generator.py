"""
Synthetic biased dataset generator.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.datasets import make_classification


@dataclass
class BiasedDataset:
    """Container for generated biased dataset."""
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_names: List[str]
    group_names: List[str]
    bias_info: Dict

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        df = pd.DataFrame(self.X, columns=self.feature_names)
        df["target"] = self.y
        df["sensitive_attr"] = self.groups
        return df

    def to_csv(self, path: str) -> None:
        """Save to CSV file."""
        self.to_dataframe().to_csv(path, index=False)


def _inject_label_flip(
    y: np.ndarray,
    groups: np.ndarray,
    flip_rate: float,
    target_group: Union[int, str],
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, Dict]:
    """Inject label flip bias for a specific group."""
    rng = np.random.RandomState(random_state)
    y_biased = y.copy()
    bias_info = {"type": "label_flip", "flip_rate": flip_rate, "target_group": str(target_group)}
    
    mask = groups == target_group
    n_target = np.sum(mask)
    if n_target > 0:
        n_flip = int(n_target * flip_rate)
        flip_indices = rng.choice(np.where(mask)[0], size=n_flip, replace=False)
        y_biased[flip_indices] = 1 - y_biased[flip_indices]
        bias_info["n_flipped"] = n_flip
        bias_info["n_target_group"] = int(n_target)
    
    return y_biased, bias_info

def _inject_feature_bias(
    X: np.ndarray,
    groups: np.ndarray,
    bias_strength: float,
    target_group: Union[int, str],
    n_features: int = 1,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, Dict]:
    """Inject feature bias by shifting feature values for a specific group."""
    rng = np.random.RandomState(random_state)
    X_biased = X.copy()
    bias_info = {"type": "feature_bias", "bias_strength": bias_strength, "target_group": str(target_group)}
    
    mask = groups == target_group
    n_target = np.sum(mask)
    if n_target > 0 and n_features > 0:
        feature_indices = rng.choice(X.shape[1], size=min(n_features, X.shape[1]), replace=False)
        for feat_idx in feature_indices:
            shift = rng.normal(0, bias_strength)
            X_biased[mask, feat_idx] += shift
        bias_info["biased_features"] = feature_indices.tolist()
        bias_info["shifts"] = {str(i): float(rng.normal(0, bias_strength)) for i in feature_indices}
        bias_info["n_target_group"] = int(n_target)
    
    return X_biased, bias_info


def _inject_correlation_bias(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    correlation_strength: float,
    target_group: Union[int, str],
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, Dict]:
    """Inject bias by making a feature correlated with the target for a specific group."""
    rng = np.random.RandomState(random_state)
    X_biased = X.copy()
    bias_info = {"type": "correlation_bias", "correlation_strength": correlation_strength, "target_group": str(target_group)}
    
    mask = groups == target_group
    n_target = np.sum(mask)
    if n_target > 0:
        feat_idx = rng.choice(X.shape[1])
        noise_scale = 1.0 - correlation_strength
        X_biased[mask, feat_idx] = y[mask] + rng.normal(0, noise_scale, size=n_target)
        bias_info["biased_feature"] = int(feat_idx)
        bias_info["n_target_group"] = int(n_target)
    
    return X_biased, bias_info

def generate_biased_dataset(
    n_samples: int = 1000,
    n_features: int = 10,
    n_informative: int = 2,
    n_redundant: int = 1,
    n_groups: int = 2,
    group_proportions: Optional[List[float]] = None,
    bias_type: str = "label_flip",
    bias_params: Optional[Dict] = None,
    random_state: Optional[int] = None,
) -> BiasedDataset:
    """
    Generate a synthetic dataset with controllable bias.

    Args:
        n_samples: Total number of samples
        n_features: Number of features
        n_informative: Number of informative features
        n_redundant: Number of redundant features
        n_groups: Number of sensitive groups (2 or more)
        group_proportions: Proportion of samples per group (must sum to 1)
        bias_type: Type of bias to inject ("label_flip", "feature_bias", "correlation_bias", "combined")
        bias_params: Parameters for bias injection
        random_state: Random seed for reproducibility

    Returns:
        BiasedDataset object with features, labels, groups, and metadata
    """
    rng = np.random.RandomState(random_state)
    
    if group_proportions is None:
        group_proportions = [1.0 / n_groups] * n_groups
    group_proportions = np.array(group_proportions)
    assert abs(group_proportions.sum() - 1.0) < 1e-6, "group_proportions must sum to 1"
    assert len(group_proportions) == n_groups, "group_proportions length must match n_groups"
    
    if bias_params is None:
        bias_params = {}
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_clusters_per_class=1,
        weights=[0.5, 0.5],
        random_state=random_state,
    )
    
    group_sizes = (group_proportions * n_samples).astype(int)
    group_sizes[-1] = n_samples - group_sizes[:-1].sum()
    
    groups = np.repeat(np.arange(n_groups), group_sizes)
    rng.shuffle(groups)
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    group_names = [f"group_{i}" for i in range(n_groups)]
    
    bias_info = {"bias_type": bias_type, "details": []}
    
    if bias_type == "label_flip":
        flip_rate = bias_params.get("flip_rate", 0.3)
        target_group = bias_params.get("target_group", 0)
        y, info = _inject_label_flip(y, groups, flip_rate, target_group, random_state)
        bias_info["details"].append(info)
        
    elif bias_type == "feature_bias":
        bias_strength = bias_params.get("bias_strength", 1.0)
        target_group = bias_params.get("target_group", 0)
        n_biased_features = bias_params.get("n_features", 1)
        X, info = _inject_feature_bias(X, groups, bias_strength, target_group, n_biased_features, random_state)
        bias_info["details"].append(info)
        
    elif bias_type == "correlation_bias":
        correlation_strength = bias_params.get("correlation_strength", 0.7)
        target_group = bias_params.get("target_group", 0)
        X, info = _inject_correlation_bias(X, y, groups, correlation_strength, target_group, random_state)
        bias_info["details"].append(info)
        
    elif bias_type == "combined":
        if "label_flip" in bias_params:
            y, info = _inject_label_flip(y, groups, **bias_params["label_flip"], random_state=random_state)
            bias_info["details"].append(info)
        if "feature_bias" in bias_params:
            X, info = _inject_feature_bias(X, groups, **bias_params["feature_bias"], random_state=random_state)
            bias_info["details"].append(info)
        if "correlation_bias" in bias_params:
            X, info = _inject_correlation_bias(X, y, groups, **bias_params["correlation_bias"], random_state=random_state)
            bias_info["details"].append(info)
    
    else:
        raise ValueError(f"Unknown bias_type: {bias_type}")
    
    return BiasedDataset(
        X=X,
        y=y,
        groups=groups,
        feature_names=feature_names,
        group_names=group_names,
        bias_info=bias_info,
    )

def generate_multiple_biased_datasets(
    n_datasets: int = 5,
    **kwargs
) -> List[BiasedDataset]:
    """Generate multiple biased datasets with different random seeds."""
    datasets = []
    base_seed = kwargs.pop("random_state", 42)
    for i in range(n_datasets):
        seed = base_seed + i if base_seed is not None else None
        datasets.append(generate_biased_dataset(random_state=seed, **kwargs))
    return datasets


__all__ = [
    "BiasedDataset",
    "generate_biased_dataset",
    "generate_multiple_biased_datasets",
]
