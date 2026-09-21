"""
CLI entry point.
"""

import argparse
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from fairness_audit_kit import (
    generate_biased_dataset,
    compute_fairness_metrics,
    compute_intersectional_metrics,
    compute_generalized_entropy,
    compute_calibration_metrics,
    compute_odds_parity_metrics,
    optimize_thresholds,
    render_metrics_report,
    render_intersectional_report,
    render_theil_report,
    render_calibration_report,
    render_odds_parity_report,
    render_optimization_report,
)


def cmd_generate(args):
    """Generate synthetic biased dataset."""
    dataset = generate_biased_dataset(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_informative=args.n_informative,
        n_redundant=args.n_redundant,
        n_groups=args.n_groups,
        group_proportions=args.group_proportions,
        bias_type=args.bias_type,
        bias_params=args.bias_params,
        random_state=args.random_state,
    )
    dataset.to_csv(args.output)
    print(f"Generated dataset with {len(dataset.X)} samples, {dataset.X.shape[1]} features, {args.n_groups} groups")
    print(f"Bias type: {dataset.bias_info["bias_type"]}")
    print(f"Saved to {args.output}")


def cmd_predict(args):
    """Generate predictions using a model."""
    # Load data
    df = pd.read_csv(args.data)
    X = df.drop(columns=[args.target_col, args.group_col]).values
    groups = df[args.group_col].values
    
    # Load model
    with open(args.model, "rb") as f:
        model = pickle.load(f)
    
    # Predict
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X)
        # Normalize to [0, 1]
        y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-8)
    else:
        y_scores = model.predict(X).astype(float)
    
    # Apply threshold
    if args.thresholds:
        # Per-group thresholds
        thresholds = {k: float(v) for k, v in [t.split("=") for t in args.thresholds]}
        y_pred = np.zeros_like(y_scores, dtype=int)
        for g, thresh in thresholds.items():
            mask = groups == g
            y_pred[mask] = (y_scores[mask] >= thresh).astype(int)
    else:
        # Global threshold
        y_pred = (y_scores >= args.threshold).astype(int)
    
    # Save predictions
    out_df = df.copy()
    out_df["y_pred"] = y_pred
    out_df["y_score"] = y_scores
    out_df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")


def cmd_evaluate(args):
    """Evaluate fairness metrics."""
    df = pd.read_csv(args.data)
    y_true = df[args.target_col].values
    y_pred = df[args.pred_col].values
    groups = df[args.group_col].values
    
    metrics = compute_fairness_metrics(y_true, y_pred, groups)
    report = render_metrics_report(metrics, args.title)

    odds = compute_odds_parity_metrics(y_true, y_pred, groups)
    odds_report = render_odds_parity_report(odds)
    report = report.rstrip() + "\n\n---\n\n" + odds_report
    print(
        f"Equalized odds difference={odds.equalized_odds_difference:.4f}, "
        f"TPR difference={odds.tpr_difference:.4f}, "
        f"FPR difference={odds.fpr_difference:.4f}, "
        f"PPV difference={odds.ppv_difference:.4f}"
    )

    entropy_alpha = getattr(args, "entropy_alpha", 1.0)
    entropy = compute_generalized_entropy(
        y_true, y_pred, groups, alpha=entropy_alpha
    )
    theil_report = render_theil_report(entropy)
    report = report.rstrip() + "\n\n---\n\n" + theil_report
    print(
        f"Theil/GEI (alpha={entropy.alpha:g}) benefit between-group="
        f"{entropy.benefit.between_group:.4f}, "
        f"error between-group={entropy.error.between_group:.4f}"
    )

    score_col = getattr(args, "score_col", "y_score")
    if score_col and score_col in df.columns:
        y_scores = df[score_col].values
        n_bins = getattr(args, "calibration_bins", 10)
        strategy = getattr(args, "calibration_strategy", "uniform")
        calibration = compute_calibration_metrics(
            y_true, y_scores, groups, n_bins=n_bins, strategy=strategy
        )
        cal_report = render_calibration_report(calibration)
        report = report.rstrip() + "\n\n---\n\n" + cal_report
        print(
            f"ECE={calibration.ece:.4f} (n_bins={calibration.n_bins}, "
            f"strategy={calibration.strategy}), "
            f"ECE difference={calibration.ece_difference:.4f}"
        )

    group_col_2 = getattr(args, "group_col_2", None)
    if group_col_2:
        groups_b = df[group_col_2].values
        inter = compute_intersectional_metrics(y_true, y_pred, groups, groups_b)
        inter_report = render_intersectional_report(
            inter,
            title="Intersectional Fairness Report",
            groups_a_name=args.group_col,
            groups_b_name=group_col_2,
        )
        report = report.rstrip() + "\n\n---\n\n" + inter_report
        print(
            f"Worst intersection: {inter.worst_intersection.group} "
            f"(n={inter.worst_intersection.n_samples}, "
            f"positive_rate={inter.worst_intersection.positive_rate:.4f})"
        )
    
    with open(args.output, "w") as f:
        f.write(report)
    print(f"Report saved to {args.output}")


def cmd_optimize(args):
    """Optimize thresholds for fairness."""
    df = pd.read_csv(args.data)
    y_true = df[args.target_col].values
    y_scores = df[args.score_col].values
    groups = df[args.group_col].values
    
    result = optimize_thresholds(
        y_true=y_true,
        y_scores=y_scores,
        groups=groups,
        constraint=args.constraint,
        constraint_threshold=args.constraint_threshold,
        grid_resolution=args.grid_resolution,
        objective=args.objective,
    )
    
    report = render_optimization_report(result, args.title)
    
    with open(args.output, "w") as f:
        f.write(report)
    print(f"Optimization report saved to {args.output}")
    print(f"Optimal thresholds: {result.thresholds}")
    print(f"Constraint satisfied: {result.constraint_satisfied}")


def main():
    parser = argparse.ArgumentParser(
        prog="fairness-audit",
        description="Fairness Audit Kit - Model fairness and bias evaluation toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic biased dataset")
    gen_parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples")
    gen_parser.add_argument("--n-features", type=int, default=10, help="Number of features")
    gen_parser.add_argument("--n-informative", type=int, default=2, help="Number of informative features")
    gen_parser.add_argument("--n-redundant", type=int, default=1, help="Number of redundant features")
    gen_parser.add_argument("--n-groups", type=int, default=2, help="Number of sensitive groups")
    gen_parser.add_argument("--group-proportions", type=float, nargs="+", default=None, help="Group proportions")
    gen_parser.add_argument("--bias-type", type=str, default="label_flip", 
                           choices=["label_flip", "feature_bias", "correlation_bias", "combined"],
                           help="Type of bias to inject")
    gen_parser.add_argument("--bias-params", type=str, default="{}", help="Bias parameters as JSON")
    gen_parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    gen_parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    gen_parser.set_defaults(func=cmd_generate)
    
    # Predict command
    pred_parser = subparsers.add_parser("predict", help="Generate predictions from model")
    pred_parser.add_argument("--model", type=str, required=True, help="Model pickle file")
    pred_parser.add_argument("--data", type=str, required=True, help="Input data CSV")
    pred_parser.add_argument("--target-col", type=str, default="target", help="Target column name")
    pred_parser.add_argument("--group-col", type=str, default="sensitive_attr", help="Group column name")
    pred_parser.add_argument("--threshold", type=float, default=0.5, help="Global threshold")
    pred_parser.add_argument("--thresholds", type=str, nargs="+", default=None, 
                            help="Per-group thresholds as GROUP=VALUE pairs")
    pred_parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    pred_parser.set_defaults(func=cmd_predict)
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate fairness metrics")
    eval_parser.add_argument("--data", type=str, required=True, help="Data CSV with predictions")
    eval_parser.add_argument("--target-col", type=str, default="target", help="True label column")
    eval_parser.add_argument("--pred-col", type=str, default="y_pred", help="Prediction column")
    eval_parser.add_argument("--group-col", type=str, default="sensitive_attr", help="Group column")
    eval_parser.add_argument(
        "--group-col-2",
        type=str,
        default=None,
        help="Second sensitive attribute column. When set, also evaluate "
             "intersectional fairness on the cross of --group-col and this column.",
    )
    eval_parser.add_argument(
        "--entropy-alpha",
        type=float,
        default=1.0,
        help="Generalized entropy alpha for Theil/GEI inequality "
             "(1=Theil index, 0=mean log deviation, 2=half squared CV).",
    )
    eval_parser.add_argument(
        "--score-col",
        type=str,
        default="y_score",
        help="Predicted probability column in [0, 1]. When present, the "
             "report includes ECE and a reliability diagram table overall "
             "and per group.",
    )
    eval_parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Number of bins for ECE / reliability diagrams (default: 10).",
    )
    eval_parser.add_argument(
        "--calibration-strategy",
        type=str,
        default="uniform",
        choices=["uniform", "quantile"],
        help="Binning strategy: uniform (equal-width) or quantile (equal-mass).",
    )
    eval_parser.add_argument("--title", type=str, default="Fairness Evaluation Report", help="Report title")
    eval_parser.add_argument("--output", type=str, required=True, help="Output report path")
    eval_parser.set_defaults(func=cmd_evaluate)
    
    # Optimize command
    opt_parser = subparsers.add_parser("optimize", help="Optimize thresholds for fairness")
    opt_parser.add_argument("--data", type=str, required=True, help="Data CSV with scores")
    opt_parser.add_argument("--target-col", type=str, default="target", help="True label column")
    opt_parser.add_argument("--score-col", type=str, default="y_score", help="Score column")
    opt_parser.add_argument("--group-col", type=str, default="sensitive_attr", help="Group column")
    opt_parser.add_argument("--constraint", type=str, default="equalized_odds",
                           choices=["demographic_parity", "equal_opportunity", "equalized_odds"],
                           help="Fairness constraint")
    opt_parser.add_argument("--constraint-threshold", type=float, default=0.1, help="Max constraint value")
    opt_parser.add_argument("--grid-resolution", type=int, default=51, help="Grid resolution")
    opt_parser.add_argument("--objective", type=str, default="accuracy",
                           choices=["accuracy", "balanced_accuracy", "f1"],
                           help="Optimization objective")
    opt_parser.add_argument("--title", type=str, default="Threshold Optimization Report", help="Report title")
    opt_parser.add_argument("--output", type=str, required=True, help="Output report path")
    opt_parser.set_defaults(func=cmd_optimize)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Parse bias_params JSON
    if hasattr(args, "bias_params") and args.bias_params:
        import json
        args.bias_params = json.loads(args.bias_params)
    
    args.func(args)


if __name__ == "__main__":
    main()
