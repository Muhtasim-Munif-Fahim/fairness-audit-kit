"""
Demo script for Fairness Audit Kit.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import tempfile
import os

from fairness_audit_kit import (
    generate_biased_dataset,
    compute_fairness_metrics,
    compute_theil_metrics,
    compute_calibration_metrics,
    compute_odds_parity_metrics,
    optimize_thresholds,
    render_metrics_report,
    render_theil_report,
    render_calibration_report,
    render_odds_parity_report,
    render_optimization_report,
)


def run_demo():
    print("=" * 60)
    print("Fairness Audit Kit Demo")
    print("=" * 60)
    
    # 1. Generate biased dataset
    print("\n1. Generating synthetic biased dataset...")
    dataset = generate_biased_dataset(
        n_samples=2000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_groups=2,
        group_proportions=[0.5, 0.5],
        bias_type="label_flip",
        bias_params={"flip_rate": 0.3, "target_group": 0},
        random_state=42,
    )
    
    print(f"   Generated {len(dataset.X)} samples with {dataset.X.shape[1]} features")
    print(f"   Groups: {np.unique(dataset.groups)}")
    print(f"   Bias info: {dataset.bias_info}")
    
    # Split data
    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        dataset.X, dataset.y, dataset.groups, test_size=0.3, random_state=42, stratify=dataset.y
    )
    
    # 2. Train a model
    print("\n2. Training Logistic Regression model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Get predictions
    y_scores = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    print(f"   Train accuracy: {model.score(X_train, y_train):.4f}")
    print(f"   Test accuracy: {model.score(X_test, y_test):.4f}")
    
    # 3. Evaluate fairness with default threshold (0.5)
    print("\n3. Evaluating fairness with default threshold (0.5)...")
    metrics = compute_fairness_metrics(y_test, y_pred, g_test)
    report = render_metrics_report(metrics, "Fairness Evaluation (Default Threshold)")
    print(report)

    print("\n3b. Theil / generalized entropy inequality...")
    theil = compute_theil_metrics(y_test, y_pred, g_test)
    theil_report = render_theil_report(theil)
    print(theil_report)

    print("\n3c. Reliability / expected calibration error...")
    calibration = compute_calibration_metrics(y_test, y_scores, g_test)
    cal_report = render_calibration_report(calibration)
    print(cal_report)

    print("\n3d. Equalized odds and predictive parity (per-group rates and pairwise gaps)...")
    odds = compute_odds_parity_metrics(y_test, y_pred, g_test)
    odds_report = render_odds_parity_report(odds)
    print(odds_report)
    
    # 4. Optimize thresholds for equalized odds
    print("\n4. Optimizing thresholds for equalized odds constraint...")
    result = optimize_thresholds(
        y_true=y_test,
        y_scores=y_scores,
        groups=g_test,
        constraint="equalized_odds",
        constraint_threshold=0.1,
        grid_resolution=51,
        objective="balanced_accuracy",
    )
    
    opt_report = render_optimization_report(result, "Threshold Optimization (Equalized Odds)")
    print(opt_report)
    
    # 5. Apply optimized thresholds and re-evaluate
    print("\n5. Applying optimized thresholds...")
    y_pred_opt = np.zeros_like(y_pred)
    for g, thresh in result.thresholds.items():
        mask = g_test == int(g)
        y_pred_opt[mask] = (y_scores[mask] >= thresh).astype(int)
    
    metrics_opt = compute_fairness_metrics(y_test, y_pred_opt, g_test)
    report_opt = render_metrics_report(metrics_opt, "Fairness Evaluation (Optimized Thresholds)")
    print(report_opt)
    
    # 6. Compare
    print("\n6. Comparison: Default vs Optimized Thresholds")
    print("-" * 60)
    print(f"{'Metric':<35} {'Default':>10} {'Optimized':>10} {'Improvement':>12}")
    print("-" * 60)
    
    comparisons = [
        ("Demographic Parity Diff", metrics.demographic_parity_difference, metrics_opt.demographic_parity_difference),
        ("Equal Opportunity Diff", metrics.equal_opportunity_difference, metrics_opt.equal_opportunity_difference),
        ("Equalized Odds Diff", metrics.equalized_odds_difference, metrics_opt.equalized_odds_difference),
        ("Disparate Impact Ratio", metrics.disparate_impact_ratio, metrics_opt.disparate_impact_ratio),
        ("Theil benefit between-group", theil.benefit.between_group,
         compute_theil_metrics(y_test, y_pred_opt, g_test).benefit.between_group),
    ]
    
    for name, default_val, opt_val in comparisons:
        if default_val != 0:
            improvement = (default_val - opt_val) / abs(default_val) * 100
            imp_str = f"{improvement:+.1f}%"
        else:
            imp_str = "N/A"
        print(f"{name:<35} {default_val:>10.4f} {opt_val:>10.4f} {imp_str:>12}")
    
    print("-" * 60)
    
    # 7. Save demo outputs
    print("\n7. Saving demo outputs...")
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "demo_data.csv")
        dataset.to_csv(data_path)
        print(f"   Dataset saved to: {data_path}")
        
        report_path = os.path.join(tmpdir, "fairness_report.md")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"   Default threshold report: {report_path}")
        
        opt_report_path = os.path.join(tmpdir, "optimization_report.md")
        with open(opt_report_path, "w") as f:
            f.write(opt_report)
        print(f"   Optimization report: {opt_report_path}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
