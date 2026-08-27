"""
Markdown report renderer.
"""

from typing import Dict, List, Any
from fairness_audit_kit.metrics import FairnessMetrics
from fairness_audit_kit.optimizer import OptimizationResult


def render_metrics_report(metrics: FairnessMetrics, title: str = "Fairness Evaluation Report") -> str:
    """Render fairness metrics as Markdown report."""
    lines = [
        f"# {title}",
        "",
        "## Summary",
        "",
        f"- **Demographic Parity Difference*: {metrics.demographic_parity_difference:.4f}",
        f"- **Equal Opportunity Difference*: {metrics.equal_opportunity_difference:.4f}",
        f"- **Equalized Odds Difference*: {metrics.equalized_odds_difference:.4f}",
        f"- **Disparate Impact Ratio*: {metrics.disparate_impact_ratio:.4f}",
        "",
        "## Calibration by Group",
        "",
        "| Group | Calibration (P(y=1|y_hat=1) / P(y=1)) |",
        "|-------|---------------------------------------------|",
    ]
    for group, cal in metrics.calibration_by_group.items():
        lines.append(f"| {group} | {cal:.4f} |")
    
    lines.extend([
        "",
        "## Confusion Matrices by Group",
        "",
    ])
    
    for group, cm in metrics.confusion_matrices.items():
        lines.extend([
            f"### Group: {group}",
            "",
            "| | Predicted 0 | Predicted 1 |",
            "|---|---|---|",
            f"| Actual 0 | {cm["tn"]} | {cm["fp"]} |",
            f"| Actual 1 | {cm["fn"]} | {cm["tp"]} |",
            "",
            f"- **TPR (Recall)*: {cm["tp"] / (cm["tp"] + cm["fn"]) if (cm["tp"] + cm["fn"]) > 0 else 0:.4f}",
            f"- **FPR*: {cm["fp"] / (cm["fp"] + cm["tn"]) if (cm["fp"] + cm["tn"]) > 0 else 0:.4f}",
            f"- **Precision*: {cm["tp"] / (cm["tp"] + cm["fp"]) if (cm["tp"] + cm["fp"]) > 0 else 0:.4f}",
            "",
        ])
    
    lines.extend([
        "## Interpretation",
        "",
        "- **Demographic Parity Difference*: Difference in positive prediction rates between groups. 0 = perfect parity.",
        "- **Equal Opportunity Difference*: Difference in true positive rates between groups. 0 = equal opportunity.",
        "- **Equalized Odds Difference*: Max of TPR and FPR differences. 0 = equalized odds.",
        "- **Disparate Impact Ratio*: Ratio of positive rates (min/max). 1.0 = no disparate impact; < 0.8 often considered problematic.",
        "- **Calibration*: Ratio of actual positives to predicted positives per group. 1.0 = well calibrated.",
        "",
    ])
    
    return "\n".join(lines)


def render_optimization_report(result: OptimizationResult, title: str = "Threshold Optimization Report") -> str:
    """Render optimization result as Markdown report."""
    lines = [
        f"# {title}",
        "",
        "## Optimal Thresholds",
        "",
        "| Group | Threshold |",
        "|-------|-----------|",
    ]
    for group, thresh in result.thresholds.items():
        lines.append(f"| {group} | {thresh:.4f} |")
    
    lines.extend([
        "",
        f"**Objective Value*: {result.objective_value:.4f}",
        f"**Constraint Satisfied*: {'Yes' if result.constraint_satisfied else 'No'}",
        "",
        "## Fairness Metrics at Optimal Thresholds",
        "",
    ])
    lines.append(render_metrics_report(result.metrics, "").replace("# Fairness Evaluation Report", "").strip())
    
    lines.extend([
        "",
        "## Search History Summary",
        "",
        f"- **Total Configurations Evaluated*: {len(result.search_history)}",
        f"- **Constraint Satisfied Configurations*: {sum(1 for h in result.search_history if h['constraint_satisfied'])}",
        "",
    ])
    
    return "\n".join(lines)


def render_comparison_report(
    metrics_list: List[FairnessMetrics],
    labels: List[str],
    title: str = "Fairness Comparison Report",
) -> str:
    """Render comparison of multiple metrics as Markdown."""
    if len(metrics_list) != len(labels):
        raise ValueError("metrics_list and labels must have same length")
    
    lines = [
        f"# {title}",
        "",
        "## Metric Comparison",
        "",
        "| Metric | " + " | ".join(labels) + " |",
        "|--------|" + "|".join(["----------"] * len(labels)) + "|",
    ]
    
    metric_names = [
        ("Demographic Parity Diff", lambda m: m.demographic_parity_difference),
        ("Equal Opportunity Diff", lambda m: m.equal_opportunity_difference),
        ("Equalized Odds Diff", lambda m: m.equalized_odds_difference),
        ("Disparate Impact Ratio", lambda m: m.disparate_impact_ratio),
    ]
    
    for name, fn in metric_names:
        values = [f"{fn(m):.4f}" for m in metrics_list]
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    
    lines.extend([
        "",
        "## Per-Group Calibration Comparison",
        "",
    ])
    
    # Get all groups
    all_groups = set()
    for m in metrics_list:
        all_groups.update(m.calibration_by_group.keys())
    all_groups = sorted(all_groups)
    
    lines.append("| Group | " + " | ".join(labels) + " |")
    lines.append("|-------|" + "|".join(["----------"] * len(labels)) + "|")
    
    for group in all_groups:
        values = []
        for m in metrics_list:
            cal = m.calibration_by_group.get(group, 0)
            values.append(f"{cal:.4f}")
        lines.append(f"| {group} | " + " | ".join(values) + " |")
    
    return "\n".join(lines)


__all__ = [
    "render_metrics_report",
    "render_optimization_report",
    "render_comparison_report",
]
