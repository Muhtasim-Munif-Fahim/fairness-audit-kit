# Fairness Audit Kit

A model fairness and bias evaluation toolkit for machine learning models.

## Features

- **Group Fairness Metrics**: Demographic Parity Difference, Equal Opportunity Difference, Equalized Odds Difference, Disparate Impact Ratio, Calibration by Group
- **Intersectional Subgroup Fairness**: Evaluate the same group metrics on the cross of two sensitive attributes and surface the worst-off intersection
- **Confusion Matrices by Group**: Per-group confusion matrices with configurable thresholds
- **Synthetic Biased Dataset Generator**: Controlled label flip, feature bias, and correlation bias injection
- **Threshold Optimization**: Grid search over thresholds per group for fairness constraints (demographic parity, equal opportunity, equalized odds)
- **Markdown Evaluation Reports**: Human-readable fairness evaluation reports
- **CLI Interface**: End-to-end generate -> predict -> evaluate -> report workflow
- **Python API**: Full programmatic access to all functionality

## Installation

`ash
pip install -r requirements.txt
`

Or install in development mode:

`ash
pip install -e .
`

## Quick Start

`ash
# Run the demo
python examples/run_demo.py

# Or use the CLI
fairness-audit generate --n-samples 1000 --bias-type label_flip --output data.csv
fairness-audit predict --model-path model.pkl --data-path data.csv --output preds.csv
fairness-audit evaluate --preds-path preds.csv --data-path data.csv --group-col sensitive_attr --output report.md
fairness-audit evaluate --data preds.csv --group-col gender --group-col-2 race --output intersectional_report.md
fairness-audit optimize --data-path preds.csv --score-col y_score --constraint equalized_odds --output opt_report.md
`

## Module Structure

`
src/fairness_audit_kit/
├── __init__.py          # Main exports
├── metrics.py           # Group fairness metrics
├── intersectional.py    # Intersectional subgroup fairness
├── optimizer.py         # Threshold optimization
├── generator.py         # Synthetic biased dataset generator
├── report.py            # Markdown report rendering
└── cli.py               # Argparse CLI wiring
`

## Python API Usage

### Generate Biased Dataset

`python
from fairness_audit_kit import generate_biased_dataset

dataset = generate_biased_dataset(
    n_samples=1000,
    n_features=10,
    bias_type="label_flip",
    bias_params={"flip_rate": 0.3, "target_group": 0},
    random_state=42,
)

# Access data
X = dataset.X          # Features
y = dataset.y          # Labels
groups = dataset.groups # Group membership
df = dataset.to_dataframe()  # As pandas DataFrame
dataset.to_csv("data.csv")   # Save to CSV
`

### Compute Fairness Metrics

`python
from fairness_audit_kit import compute_fairness_metrics

metrics = compute_fairness_metrics(y_true, y_pred, groups)

print(f"Demographic Parity Diff: {metrics.demographic_parity_difference:.4f}")
print(f"Equal Opportunity Diff: {metrics.equal_opportunity_difference:.4f}")
print(f"Equalized Odds Diff: {metrics.equalized_odds_difference:.4f}")
print(f"Disparate Impact Ratio: {metrics.disparate_impact_ratio:.4f}")
print(f"Calibration by group: {metrics.calibration_by_group}")
`

### Intersectional Subgroup Fairness

Evaluate the same group metrics on the Cartesian product of two sensitive attributes. This catches disparities that disappear when each attribute is audited on its own.

```python
from fairness_audit_kit import (
    compute_intersectional_metrics,
    render_intersectional_report,
)

result = compute_intersectional_metrics(y_true, y_pred, gender, race)

print(f"Intersections: {result.n_intersections}")
print(f"Demographic Parity Diff: {result.metrics.demographic_parity_difference:.4f}")
print(f"Equal Opportunity Diff: {result.metrics.equal_opportunity_difference:.4f}")
print(f"Equalized Odds Diff: {result.metrics.equalized_odds_difference:.4f}")
print(f"Disparate Impact Ratio: {result.metrics.disparate_impact_ratio:.4f}")
print(f"Worst intersection: {result.worst_intersection.group}")
print(f"  positive rate: {result.worst_intersection.positive_rate:.4f}")
print(f"  gap from best: {result.worst_intersection.gap_from_best:.4f}")

report = render_intersectional_report(
    result, groups_a_name="gender", groups_b_name="race"
)
```

Intersection labels are `{group_a}/{group_b}` (override with `separator=`). The worst intersection is the subgroup with the lowest positive prediction rate by default, or the lowest TPR when `worst_by="tpr"`.

### Optimize Thresholds

`python
from fairness_audit_kit import optimize_thresholds

result = optimize_thresholds(
    y_true=y_test,
    y_scores=y_scores,
    groups=g_test,
    constraint="equalized_odds",
    constraint_threshold=0.1,
    grid_resolution=51,
    objective="balanced_accuracy",
)

print(f"Optimal thresholds: {result.thresholds}")
print(f"Constraint satisfied: {result.constraint_satisfied}")
`

### Generate Reports

`python
from fairness_audit_kit import render_metrics_report, render_optimization_report

# Metrics report
report = render_metrics_report(metrics, "My Fairness Report")

# Optimization report
opt_report = render_optimization_report(result, "Threshold Optimization")
`

## CLI Reference

### airness-audit generate

Generate synthetic biased dataset.

`ash
fairness-audit generate [OPTIONS] --output PATH

Options:
  --n-samples INT         Number of samples (default: 1000)
  --n-features INT        Number of features (default: 10)
  --n-informative INT     Number of informative features (default: 2)
  --n-redundant INT       Number of redundant features (default: 1)
  --n-groups INT          Number of sensitive groups (default: 2)
  --group-proportions     Group proportions as space-separated floats
  --bias-type TYPE        Bias type: label_flip, feature_bias, correlation_bias, combined
  --bias-params JSON      Bias parameters as JSON string
  --random-state INT      Random seed (default: 42)
  --output PATH           Output CSV path (required)
`

### airness-audit predict

Generate predictions using a trained model.

`ash
fairness-audit predict [OPTIONS] --model-path PATH --data-path PATH --output PATH

Options:
  --model-path PATH       Model pickle file (required)
  --data-path PATH        Input data CSV (required)
  --target-col STR        Target column name (default: target)
  --group-col STR         Group column name (default: sensitive_attr)
  --threshold FLOAT       Global threshold (default: 0.5)
  --thresholds            Per-group thresholds as GROUP=VALUE pairs
  --output PATH           Output CSV path (required)
`

### airness-audit evaluate

Evaluate fairness metrics on predictions.

`ash
fairness-audit evaluate [OPTIONS] --data-path PATH --output PATH

Options:
  --data-path PATH        Data CSV with predictions (required)
  --target-col STR        True label column (default: target)
  --pred-col STR          Prediction column (default: y_pred)
  --group-col STR         Group column (default: sensitive_attr)
  --group-col-2 STR       Second sensitive attribute. When set, also evaluate
                          intersectional fairness on the cross of --group-col
                          and this column, and include the worst intersection
                          in the report
  --title STR             Report title (default: "Fairness Evaluation Report")
  --output PATH           Output report path (required)
`

### airness-audit optimize

Optimize per-group thresholds under fairness constraints.

`ash
fairness-audit optimize [OPTIONS] --data-path PATH --output PATH

Options:
  --data-path PATH        Data CSV with scores (required)
  --target-col STR        True label column (default: target)
  --score-col STR         Score column (default: y_score)
  --group-col STR         Group column (default: sensitive_attr)
  --constraint TYPE       Constraint: demographic_parity, equal_opportunity, equalized_odds
  --constraint-threshold  Max constraint value (default: 0.1)
  --grid-resolution INT   Grid resolution (default: 51)
  --objective TYPE        Objective: accuracy, balanced_accuracy, f1
  --title STR             Report title (default: "Threshold Optimization Report")
  --output PATH           Output report path (required)
`

## Bias Types

| Type | Description | Parameters |
|------|-------------|------------|
| label_flip | Flip labels for a target group | lip_rate, 	arget_group |
| eature_bias | Shift feature values for a target group | ias_strength, 	arget_group, 
_features |
| correlation_bias | Make feature correlated with label for target group | correlation_strength, 	arget_group |
| combined | Combine multiple bias types | Dict with any of the above |

## Fairness Metrics

| Metric | Description | Ideal |
|--------|-------------|-------|
| Demographic Parity Difference | Max - min positive prediction rate across groups | 0 |
| Equal Opportunity Difference | Max - min true positive rate across groups | 0 |
| Equalized Odds Difference | Max of TPR diff and FPR diff across groups | 0 |
| Disparate Impact Ratio | Min positive rate / max positive rate | 1.0 |
| Calibration by Group | P(y=1|y_hat=1) / P(y=1) per group | 1.0 |
| Intersectional metrics | Same four group metrics on the cross of two attributes | same as above |
| Worst Intersection | Intersection with the lowest positive rate (or TPR) | n/a |

## Requirements

- Python 3.8+
- numpy >= 1.21.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0
- pytest >= 7.0.0 (for testing)

## Testing

`ash
python -m pytest tests -v
`

## License

MIT License - see LICENSE file for details.
