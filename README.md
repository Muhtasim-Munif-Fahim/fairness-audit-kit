# Fairness Audit Kit

A model fairness and bias evaluation toolkit for machine learning models.

## Features

- **Group Fairness Metrics**: Demographic Parity Difference, Equal Opportunity Difference, Equalized Odds Difference, Disparate Impact Ratio, Calibration by Group
- **Equalized Odds / Predictive Parity Gaps**: Per-group TPR, FPR, and PPV with every pairwise gap (Hardt et al.; Chouldechova). Optional scores are thresholded when hard labels are omitted
- **Theil / Generalized Entropy**: Inequality of classification benefit or error across groups, with between/within decomposition (Theil is alpha=1)
- **Reliability / Calibration by Group**: Expected Calibration Error (ECE) overall and per sensitive group, plus binned reliability diagram tables
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
├── odds_parity.py       # Equalized odds (TPR/FPR) and predictive parity (PPV) gaps
├── theil.py             # Theil index / generalized entropy inequality
├── calibration.py       # ECE and reliability diagram data by group
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

### Equalized Odds and Predictive Parity

`compute_fairness_metrics` already reports **scalar** max-minus-min TPR (equal opportunity) and equalized-odds summaries, plus demographic parity and disparate impact. This module adds the **per-group rates and every pairwise gap**, and **predictive parity (PPV)** which was not previously exposed.

Equalized odds (Hardt, Price, and Srebro, NIPS 2016) requires TPR and FPR to match across groups. Predictive parity (Chouldechova, 2017) requires PPV / precision to match. Pass binary `y_true` and `y_pred`, or omit `y_pred` and pass scores (thresholded at 0.5 by default).

```python
from fairness_audit_kit import (
    compute_equalized_odds,
    compute_predictive_parity,
    compute_odds_parity_metrics,
    render_odds_parity_report,
)

eo = compute_equalized_odds(y_true, y_pred, groups)
print(f"TPR by group: {eo.tpr.group_rates}")
print(f"FPR by group: {eo.fpr.group_rates}")
print(f"TPR pairwise gaps: {eo.tpr.pairwise_gaps}")
print(f"Equalized odds difference: {eo.equalized_odds_difference:.4f}")

pp = compute_predictive_parity(y_true, y_pred, groups)
print(f"PPV by group: {pp.ppv.group_rates}")
print(f"PPV difference: {pp.ppv_difference:.4f}")

# Scores only — hard labels are derived at the given threshold
eo_from_scores = compute_equalized_odds(
    y_true, None, groups, y_scores=y_scores, threshold=0.5
)

result = compute_odds_parity_metrics(y_true, y_pred, groups)
report = render_odds_parity_report(result)
```

`fairness-audit evaluate` always appends this section (it only needs labels and predictions).

Demographic parity / statistical parity difference and disparate impact ratio stay on `compute_fairness_metrics` (`demographic_parity_difference`, `disparate_impact_ratio`). They were not duplicated here because those scalars already exist.

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

### Theil Index / Generalized Entropy

Measure inequality of **benefit** (Speicher et al. / AIF360: `b_i = 1 + 1[y_hat=favorable] - 1[y=favorable]`) or **error** (0/1 misclassification) across individuals and groups. The index decomposes into a between-group term (each person is assigned their group's mean outcome) and a within-group residual. Default `alpha=1` is the Theil index; `alpha=2` is half the squared coefficient of variation.

```python
from fairness_audit_kit import (
    compute_theil_metrics,
    compute_generalized_entropy,
    render_theil_report,
    theil_index,
)

result = compute_theil_metrics(y_true, y_pred, groups)
print(f"Benefit Theil overall: {result.benefit.overall:.4f}")
print(f"Benefit Theil between-group: {result.benefit.between_group:.4f}")
print(f"Error Theil between-group: {result.error.between_group:.4f}")
print(f"Group error rates: {result.group_error_rate}")

gei = compute_generalized_entropy(y_true, y_pred, groups, alpha=2.0)
print(f"GEI(alpha=2) benefit between-group: {gei.benefit.between_group:.4f}")

report = render_theil_report(result)
```

`theil_index(values)` and `generalized_entropy_index(values, alpha=...)` also work on any non-negative array (for example a vector of group rates).

### Reliability / Expected Calibration Error

Measure how well predicted probabilities match observed frequencies, overall and per sensitive group. Scores are binned (default: 10 equal-width bins on `[0, 1]`). Each bin reports mean confidence, observed positive rate, and the gap. **ECE** is the sample-weighted average of those gaps; **MCE** is the largest single-bin gap. `ece_difference` is max group ECE minus min group ECE.

This is distinct from the hard-label `calibration_by_group` ratio in `compute_fairness_metrics` (`P(y=1) / P(y_hat=1)`). ECE needs predicted probabilities.

```python
from fairness_audit_kit import (
    compute_calibration_metrics,
    expected_calibration_error,
    compute_reliability_bins,
    render_calibration_report,
)

result = compute_calibration_metrics(y_true, y_scores, groups)
print(f"Overall ECE: {result.ece:.4f}")
print(f"Overall MCE: {result.mce:.4f}")
print(f"ECE by group: {result.group_ece}")
print(f"ECE difference: {result.ece_difference:.4f}")

for row in result.reliability_bins:
    print(
        f"[{row.lower:.2f}, {row.upper:.2f}] n={row.n_samples} "
        f"conf={row.mean_confidence:.3f} obs={row.observed_positive_rate:.3f} "
        f"gap={row.gap:.3f}"
    )

ece = expected_calibration_error(y_true, y_scores, n_bins=15, strategy="quantile")
bins = compute_reliability_bins(y_true, y_scores, n_bins=15, strategy="quantile")

report = render_calibration_report(result)
```

`fairness-audit evaluate` appends this section when the CSV has a score column (default `y_score`).

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
                          The evaluate report always includes equalized odds
                          (TPR/FPR) and predictive parity (PPV) per-group
                          rates and pairwise gaps
  --entropy-alpha FLOAT   Generalized entropy alpha (default: 1 = Theil).
                          0 = mean log deviation, 2 = half squared CV.
                          The evaluate report always includes benefit and
                          error inequality with between/within decomposition
  --score-col STR         Predicted probability column (default: y_score).
                          When present, the report includes ECE and a
                          reliability diagram table overall and per group
  --calibration-bins INT  Number of ECE / reliability bins (default: 10)
  --calibration-strategy  uniform (equal-width, default) or quantile
                          (equal-mass)
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
| TPR / FPR pairwise gaps | Per-group true/false positive rates and every pair gap | 0 |
| Predictive parity (PPV gap) | Max − min positive predictive value (precision) across groups | 0 |
| Disparate Impact Ratio | Min positive rate / max positive rate | 1.0 |
| Calibration by Group | P(y=1|y_hat=1) / P(y=1) per group | 1.0 |
| Theil index | Generalized entropy of benefit or error with alpha=1 | 0 |
| Generalized entropy (alpha) | Inequality of benefit or error; alpha=2 is half squared CV | 0 |
| Between-group Theil/GEI | Theil/GEI after assigning each person their group mean | 0 |
| ECE | Sample-weighted |observed rate − mean confidence| across probability bins | 0 |
| ECE by group | ECE computed on each sensitive group | 0 |
| ECE difference | Max group ECE − min group ECE | 0 |
| Reliability diagram | Per-bin mean confidence, observed rate, count, and gap | gap 0 |
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
