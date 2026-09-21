"""
Markdown report renderer.
"""

from typing import Dict, List, Any
from fairness_audit_kit.metrics import FairnessMetrics
from fairness_audit_kit.optimizer import OptimizationResult
from fairness_audit_kit.intersectional import IntersectionalFairnessResult
from fairness_audit_kit.theil import GeneralizedEntropyResult
from fairness_audit_kit.calibration import CalibrationResult, ReliabilityBin
from fairness_audit_kit.odds_parity import OddsParityResult, GroupRateTable


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


def render_theil_report(
    result: GeneralizedEntropyResult,
    title: str = "Theil / Generalized Entropy Report",
) -> str:
    """Render Theil / generalized entropy inequality as Markdown."""
    alpha = result.alpha
    if abs(alpha - 1.0) < 1e-12:
        index_name = "Theil index (generalized entropy with alpha = 1)"
    elif abs(alpha) < 1e-12:
        index_name = "mean log deviation (generalized entropy with alpha = 0)"
    elif abs(alpha - 2.0) < 1e-12:
        index_name = "half the squared coefficient of variation (alpha = 2)"
    else:
        index_name = f"generalized entropy index with alpha = {alpha:g}"

    def _breakdown_lines(name: str, breakdown) -> List[str]:
        return [
            f"## {name}",
            "",
            f"- **Overall**: {breakdown.overall:.4f}",
            f"- **Between-Group**: {breakdown.between_group:.4f}",
            f"- **Within-Group**: {breakdown.within_group:.4f}",
            "",
        ]

    lines = [
        f"# {title}",
        "",
        f"Inequality measured with the **{index_name}**. "
        f"{result.n_groups} groups, {result.n_samples} samples.",
        "",
        f"- **Mean Benefit**: {result.mean_benefit:.4f}",
        f"- **Mean Error Rate**: {result.mean_error_rate:.4f}",
        "",
    ]
    lines.extend(_breakdown_lines("Benefit Inequality", result.benefit))
    lines.extend(_breakdown_lines("Error Inequality", result.error))
    lines.extend([
        "## Rates by Group",
        "",
        "| Group | N | Mean Benefit | Error Rate |",
        "|-------|---|--------------|------------|",
    ])
    for group in sorted(result.group_size.keys(), key=str):
        cell = str(group).replace("|", "\\|")
        lines.append(
            f"| {cell} | {result.group_size[group]} | "
            f"{result.group_mean_benefit[group]:.4f} | "
            f"{result.group_error_rate[group]:.4f} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- **Theil / generalized entropy**: 0 means perfect equality. Larger values "
        "mean more inequality in the chosen outcome.",
        "- **Benefit**: `b_i = 1 + 1[y_hat = favorable] - 1[y = favorable]` "
        "(Speicher et al. / AIF360). Correct predictions score 1, false positives 2, "
        "false negatives 0.",
        "- **Error**: 1 if the prediction is wrong, 0 otherwise. Between-group error "
        "inequality is the Theil/GEI of group error rates (size-weighted).",
        "- **Between-Group**: inequality after giving every person their group's mean "
        "outcome. This is the group-fairness term.",
        "- **Within-Group**: residual individual inequality inside groups. "
        "Overall = between-group + within-group.",
        "- **alpha**: 1 = Theil index (default), 0 = mean log deviation, "
        "2 = half the squared coefficient of variation.",
        "",
    ])

    return "\n".join(lines)


def _reliability_table(bins: List[ReliabilityBin]) -> List[str]:
    lines = [
        "| Bin | Range | N | Mean Confidence | Observed Rate | Gap |",
        "|-----|-------|---|-----------------|---------------|-----|",
    ]
    if not bins:
        lines.append("| — | — | 0 | — | — | — |")
        return lines
    for b in bins:
        lines.append(
            f"| {b.bin_index} | [{b.lower:.2f}, {b.upper:.2f}] | {b.n_samples} | "
            f"{b.mean_confidence:.4f} | {b.observed_positive_rate:.4f} | {b.gap:.4f} |"
        )
    return lines


def render_calibration_report(
    result: CalibrationResult,
    title: str = "Reliability / Calibration Report",
) -> str:
    """Render ECE and reliability diagram tables as Markdown."""
    strategy_label = (
        "equal-width (uniform)"
        if result.strategy == "uniform"
        else "equal-mass (quantile)"
    )

    lines = [
        f"# {title}",
        "",
        f"{result.n_bins} {strategy_label} bins, {result.n_groups} groups, "
        f"{result.n_samples} samples.",
        "",
        f"- **Overall ECE**: {result.ece:.4f}",
        f"- **Overall MCE**: {result.mce:.4f}",
        f"- **ECE Difference (max − min group)**: {result.ece_difference:.4f}",
        "",
        "## ECE by Group",
        "",
        "| Group | N | ECE | MCE |",
        "|-------|---|-----|-----|",
    ]
    for group in sorted(result.group_size.keys(), key=str):
        cell = str(group).replace("|", "\\|")
        lines.append(
            f"| {cell} | {result.group_size[group]} | "
            f"{result.group_ece[group]:.4f} | {result.group_mce[group]:.4f} |"
        )

    lines.extend([
        "",
        "## Overall Reliability Diagram",
        "",
    ])
    lines.extend(_reliability_table(result.reliability_bins))
    lines.extend(["", "## Reliability by Group", ""])

    for group in sorted(result.group_reliability_bins.keys(), key=str):
        cell = str(group).replace("|", "\\|")
        lines.extend([f"### Group: {cell}", ""])
        lines.extend(_reliability_table(result.group_reliability_bins[group]))
        lines.append("")

    lines.extend([
        "## Interpretation",
        "",
        "- **ECE (Expected Calibration Error)**: sample-weighted average of "
        "|observed positive rate − mean predicted probability| across bins. "
        "0 means perfectly calibrated.",
        "- **MCE (Maximum Calibration Error)**: largest single-bin gap. "
        "Highlights a badly calibrated region that ECE may dilute.",
        "- **Reliability diagram**: each row is a bin. Well-calibrated "
        "predictions have mean confidence ≈ observed rate (gap near 0). "
        "Above the diagonal is under-confidence; below is over-confidence.",
        "- **ECE by group**: a model can look calibrated overall while one "
        "sensitive group is systematically over- or under-confident.",
        "- **ECE difference**: max group ECE − min group ECE. 0 means every "
        "group is equally (mis)calibrated.",
        "- **uniform** bins are equal-width on [0, 1] (standard ECE). "
        "**quantile** bins have (approximately) equal sample counts.",
        "",
    ])

    return "\n".join(lines)


def _pairwise_gap_table(table: GroupRateTable, heading: str) -> List[str]:
    lines = [
        f"### {heading}",
        "",
        "| Group A | Group B | Rate A | Rate B | Gap | Abs Gap |",
        "|---------|---------|--------|--------|-----|---------|",
    ]
    if not table.pairwise_gaps:
        lines.append("| — | — | — | — | — | — |")
        return lines
    for gap in table.pairwise_gaps:
        a = str(gap.group_a).replace("|", "\\|")
        b = str(gap.group_b).replace("|", "\\|")
        lines.append(
            f"| {a} | {b} | {gap.rate_a:.4f} | {gap.rate_b:.4f} | "
            f"{gap.gap:.4f} | {gap.abs_gap:.4f} |"
        )
    return lines


def render_odds_parity_report(
    result: OddsParityResult,
    title: str = "Equalized Odds / Predictive Parity Report",
) -> str:
    """Render per-group TPR/FPR/PPV rates and pairwise gaps as Markdown."""
    lines = [
        f"# {title}",
        "",
        f"{result.n_groups} groups, {result.n_samples} samples.",
        "",
        f"- **TPR Difference (Equal Opportunity)**: {result.tpr_difference:.4f}",
        f"- **FPR Difference**: {result.fpr_difference:.4f}",
        f"- **Equalized Odds Difference**: {result.equalized_odds_difference:.4f}",
        f"- **PPV Difference (Predictive Parity)**: {result.ppv_difference:.4f}",
        "",
        "## Rates by Group",
        "",
        "| Group | N | TPR | FPR | PPV |",
        "|-------|---|-----|-----|-----|",
    ]
    tpr_rates = result.equalized_odds.tpr.group_rates
    fpr_rates = result.equalized_odds.fpr.group_rates
    ppv_rates = result.predictive_parity.ppv.group_rates
    for group in sorted(result.group_size.keys(), key=str):
        cell = str(group).replace("|", "\\|")
        lines.append(
            f"| {cell} | {result.group_size[group]} | "
            f"{tpr_rates[group]:.4f} | {fpr_rates[group]:.4f} | "
            f"{ppv_rates[group]:.4f} |"
        )

    lines.extend(["", "## Pairwise Gaps", ""])
    lines.extend(_pairwise_gap_table(result.equalized_odds.tpr, "True Positive Rate"))
    lines.extend([""])
    lines.extend(_pairwise_gap_table(result.equalized_odds.fpr, "False Positive Rate"))
    lines.extend([""])
    lines.extend(
        _pairwise_gap_table(
            result.predictive_parity.ppv, "Positive Predictive Value"
        )
    )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- **Equalized odds** (Hardt, Price, and Srebro, NIPS 2016): TPR and FPR "
        "equal across groups. Difference is max(TPR gap, FPR gap). 0 = equalized odds.",
        "- **TPR difference** is equal opportunity (Hardt et al.). It matches "
        "`compute_fairness_metrics().equal_opportunity_difference`.",
        "- **Equalized odds difference** matches "
        "`compute_fairness_metrics().equalized_odds_difference`. This report also "
        "lists every pairwise gap, not only the max-minus-min summary.",
        "- **Predictive parity** (Chouldechova, 2017): PPV / precision equal "
        "across groups. 0 = predictive parity. This was not previously exposed.",
        "- **Pairwise gap** is rate(group A) − rate(group B) with labels sorted "
        "lexicographically. |Gap| is the absolute disparity for that pair.",
        "- Undefined rates (no actual positives for TPR, no actual negatives for "
        "FPR, no predicted positives for PPV) are reported as 0, matching the "
        "other group metrics in this kit.",
        "",
    ])
    return "\n".join(lines)


def render_intersectional_report(
    result: IntersectionalFairnessResult,
    title: str = "Intersectional Fairness Report",
    groups_a_name: str = "group_a",
    groups_b_name: str = "group_b",
) -> str:
    """Render intersectional fairness metrics, including the worst intersection."""
    worst = result.worst_intersection
    criterion_label = {
        "positive_rate": "positive prediction rate",
        "tpr": "true positive rate",
    }.get(worst.criterion, worst.criterion)

    lines = [
        f"# {title}",
        "",
        f"Evaluated on the cross of `{groups_a_name}` × `{groups_b_name}` "
        f"({result.n_intersections} intersections).",
        "",
        "## Worst Intersection",
        "",
        f"- **Group**: `{worst.group}`",
        f"- **Samples**: {worst.n_samples}",
        f"- **Positive Rate**: {worst.positive_rate:.4f}",
        f"- **True Positive Rate**: {worst.tpr:.4f}",
        f"- **False Positive Rate**: {worst.fpr:.4f}",
        f"- **Criterion**: lowest {criterion_label}",
        f"- **Gap from Best Intersection**: {worst.gap_from_best:.4f}",
        "",
        "The worst intersection is the subgroup with the lowest value of the "
        "chosen criterion (default: positive prediction rate).",
        "",
        "## Intersectional Metrics",
        "",
        f"- **Demographic Parity Difference**: {result.metrics.demographic_parity_difference:.4f}",
        f"- **Equal Opportunity Difference**: {result.metrics.equal_opportunity_difference:.4f}",
        f"- **Equalized Odds Difference**: {result.metrics.equalized_odds_difference:.4f}",
        f"- **Disparate Impact Ratio**: {result.metrics.disparate_impact_ratio:.4f}",
        "",
        "## Rates by Intersection",
        "",
        "| Intersection | N | Positive Rate | TPR | FPR | Precision | Calibration |",
        "|--------------|---|---------------|-----|-----|-----------|-------------|",
    ]

    for group in result.intersection_labels:
        rates = result.group_rates[group]
        cell = str(group).replace("|", "\\|")
        lines.append(
            f"| {cell} | {rates['total']} | {rates['positive_rate']:.4f} | "
            f"{rates['tpr']:.4f} | {rates['fpr']:.4f} | {rates['precision']:.4f} | "
            f"{rates['calibration']:.4f} |"
        )

    lines.extend([
        "",
        "## Confusion Matrices by Intersection",
        "",
    ])

    for group in result.intersection_labels:
        cm = result.metrics.confusion_matrices[group]
        lines.extend([
            f"### Intersection: {group}",
            "",
            "| | Predicted 0 | Predicted 1 |",
            "|---|---|---|",
            f"| Actual 0 | {cm['tn']} | {cm['fp']} |",
            f"| Actual 1 | {cm['fn']} | {cm['tp']} |",
            "",
        ])

    lines.extend([
        "## Interpretation",
        "",
        "- Intersectional metrics apply the same group fairness definitions "
        "to the Cartesian product of two sensitive attributes.",
        "- A small gap on each attribute separately can hide a large gap at "
        "an intersection (fairness gerrymandering).",
        "- **Worst Intersection**: subgroup with the lowest positive rate "
        "(or TPR if `worst_by='tpr'`). Gap from best is max − min on that criterion.",
        "- **Demographic Parity Difference**: max − min positive prediction rate across intersections. 0 = perfect parity.",
        "- **Equal Opportunity Difference**: max − min true positive rate across intersections. 0 = equal opportunity.",
        "- **Equalized Odds Difference**: max of TPR and FPR differences across intersections. 0 = equalized odds.",
        "- **Disparate Impact Ratio**: min positive rate / max positive rate across intersections. 1.0 = no disparate impact.",
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
    "render_intersectional_report",
    "render_theil_report",
    "render_calibration_report",
    "render_odds_parity_report",
    "render_optimization_report",
    "render_comparison_report",
]
