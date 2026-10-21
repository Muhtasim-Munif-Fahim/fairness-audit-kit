# Fairness Audit Kit

A model fairness and bias evaluation toolkit for machine learning models.

## Features

- **Group Fairness Metrics**: Demographic Parity Difference, Equal Opportunity Difference, Equalized Odds Difference, Disparate Impact Ratio, Calibration by Group
- **ABROCA (Absolute Between-ROC Area)**: Integrate `|TPR_a - TPR_b|` over FPR for score-based group disparity (Gardner et al., LAK 2019)
- **Equalized Odds / Predictive Parity Gaps**: Per-group TPR, FPR, and PPV with every pairwise gap (Hardt et al.; Chouldechova). Optional scores are thresholded when hard labels are omitted
- **Demographic Parity / Disparate Impact**: Per-group positive prediction rates, statistical parity difference (max − min), and disparate impact ratio (min / max). Needs only binary predictions and a sensitive attribute
- **Theil / Generalized Entropy**: Inequality of classification benefit or error across groups, with between/within decomposition (Theil is alpha=1)
- **Counterfactual Fairness Proxy**: Compare factual vs sensitive-attribute-flipped scores (Kusner et al. proxy without a causal graph)
- **Reliability / Calibration by Group**: Expected Calibration Error (ECE) overall and per sensitive group, plus binned reliability diagram tables
- **Intersectional Subgroup Fairness**: Evaluate the same group metrics on the cross of two sensitive attributes and surface the worst-off intersection
- **Confusion Matrices by Group**: Per-group confusion matrices with configurable thresholds
- **Synthetic Biased Dataset Generator**: Controlled label flip, feature bias, and correlation bias injection
- **Treatment Equality / Accuracy Equality**: Per-group FN/FP ratios (Berk et al.) and accuracy gaps, with a `treatment` CLI
- **Reweighing (bias mitigation)**: Kamiran–Calders pre-processing weights `w(g, y) = P(g)·P(y) / P(g, y)`. They make the label independent of the sensitive attribute, so every group has the same weighted base rate. Pass them as `sample_weight` when training
- **Disparate Impact Remover (feature repair)**: Feldman et al. (KDD 2015) quantile repair of numeric features toward the per-quantile median of the group distributions, with partial repair levels, rank preservation, before/after KS separability diagnostics and a `repair` CLI
- **Reject Option Classification (post-processing)**: Kamiran, Karim & Zhang (ICDM 2012). Inside a critical band `(τ−θ, τ+θ]` around the threshold, the unprivileged group gets the favourable label and the privileged group the unfavourable one. `τ` and `θ` are grid-searched for the best balanced accuracy subject to `|statistical parity / average odds / equal opportunity difference| ≤ bound`. Matches AIF360's `RejectOptionClassification`; available as the `reject-option` CLI
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
fairness-audit treatment --data preds.csv --output treatment_report.md
fairness-audit postprocess --data preds.csv --output fair_preds.csv --report postprocess.md
`

## Module Structure

`
src/fairness_audit_kit/
├── __init__.py          # Main exports
├── metrics.py           # Group fairness metrics
├── odds_parity.py       # Equalized odds (TPR/FPR) and predictive parity (PPV) gaps
├── demographic_parity.py # Statistical parity difference and disparate impact ratio
├── theil.py             # Theil index / generalized entropy inequality
├── calibration.py       # ECE and reliability diagram data by group
├── counterfactual.py    # Sensitive-flip counterfactual fairness proxy
├── intersectional.py    # Intersectional subgroup fairness
├── optimizer.py         # Threshold optimization
├── reweighing.py        # Kamiran-Calders reweighing (pre-processing mitigation)
├── repair.py            # Feldman et al. disparate impact remover (feature repair)
├── postprocessing.py    # Hardt et al. equalized-odds post-processing (LP)
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


### ABROCA

`compute_abroca` summarises ranking disparity between groups by integrating
the absolute gap between their ROC curves over false-positive rate
(Gardner et al., LAK 2019). Unlike a single equalized-odds snapshot, it
uses the full score ranking.

```python
from fairness_audit_kit import compute_abroca

result = compute_abroca(y_true, y_scores, groups)
print(result.max_abroca, result.mean_abroca)
for pair in result.pairwise:
    print(pair.group_a, pair.group_b, pair.abroca)
```

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

Per-group positive rates, statistical parity difference, and the disparate impact ratio are computed by `compute_demographic_parity` (predictions and a sensitive attribute only; labels are not required). The scalar summaries on `compute_fairness_metrics` (`demographic_parity_difference`, `disparate_impact_ratio`) use the same definitions.

### Demographic Parity and Disparate Impact

Demographic parity (Dwork et al., ITCS 2012), also called statistical parity, requires the positive prediction rate to be equal across groups. The metric uses predictions and the sensitive attribute only. Pass binary `y_pred` and a sensitive attribute, or omit `y_pred` and pass scores (thresholded at 0.5 by default).

Statistical parity difference (SPD) is max positive rate minus min positive rate. It matches `compute_fairness_metrics().demographic_parity_difference` (also exposed as `demographic_parity_difference` on the result). Disparate impact ratio is min rate / max rate (Feldman et al., KDD 2015). 1.0 means the rates match; below 0.8 is the usual four-fifths-rule flag. When every group has a zero positive rate, the ratio is 1.0.

```python
from fairness_audit_kit import (
    compute_demographic_parity,
    render_demographic_parity_report,
)

parity = compute_demographic_parity(y_pred, groups)
print(f"Positive rates: {parity.positive_rates}")
print(f"SPD: {parity.statistical_parity_difference:.4f}")
print(f"Disparate impact: {parity.disparate_impact_ratio:.4f}")
print(f"Pairwise gaps: {parity.positive_rate.pairwise_gaps}")

# Scores only — hard labels are derived at the given threshold
parity_from_scores = compute_demographic_parity(
    None, groups, y_scores=y_scores, threshold=0.5
)

report = render_demographic_parity_report(parity)
```

`fairness-audit evaluate` always appends this section (it only needs predictions and the group column).

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

### Disparate Impact Remover (Feature Repair)

`DisparateImpactRemover` (Feldman, Friedler, Moeller, Scheidegger &
Venkatasubramanian, "Certifying and Removing Disparate Impact", KDD 2015)
edits numeric features so their distribution no longer reveals the group.
Each value moves toward the per-quantile median of the group distributions,
`x' = (1 - lambda) x + lambda F_A^-1(F_g(x))` with
`F_A^-1(u) = median_g F_g^-1(u)`. Order within each group is preserved, so
features keep their predictive ranking. `repair_level=1` maps every group
onto the same distribution; smaller values interpolate.

```python
from fairness_audit_kit import DisparateImpactRemover, render_disparate_impact_repair_report

rem = DisparateImpactRemover(repair_level=1.0).fit(X_train, g_train)
X_train_fair = rem.transform(X_train, g_train)
X_test_fair = rem.transform(X_test, g_test)          # CDFs from the training groups
diag = rem.diagnose(X_train, g_train)
diag.ks_before, diag.ks_after                        # max pairwise KS per feature
print(render_disparate_impact_repair_report(diag))
```

### Reweighing (Pre-processing Mitigation)

`compute_reweighing_weights` gives every `(group, label)` cell the weight
`P(g)·P(y) / P(g, y)` (Kamiran & Calders, 2012). The weights sum to `n`, and
under them the label is independent of the sensitive attribute: the weighted
positive rate is equal across groups. Features and labels are left unchanged.

```python
from sklearn.linear_model import LogisticRegression
from fairness_audit_kit import compute_reweighing_weights, fit_with_reweighing, render_reweighing_report

res = compute_reweighing_weights(y_train, g_train)
print(res.base_rate_gap_before, res.base_rate_gap_after)   # e.g. 0.21 -> 0.0
model = LogisticRegression().fit(X_train, y_train, sample_weight=res.weights)

# or in one step
model, res = fit_with_reweighing(LogisticRegression(), X_train, y_train, g_train)
print(render_reweighing_report(res))
```

Cells with no samples cannot be fixed by weighting. They are listed in
`res.empty_cells`.

### Equalized-Odds Post-processing (Hardt et al.)

`EqualizedOddsPostprocessor` derives a randomized predictor from any frozen
classifier's decisions and the group: each group keeps a positive with
probability `P(Ỹ=1 | Ŷ=1, A=g)` and flips a negative with probability
`P(Ỹ=1 | Ŷ=0, A=g)`. The mixing rates are found by a linear program that
minimizes expected (cost-weighted) error subject to equal TPR and FPR
(`constraint="equalized_odds"`) or equal TPR only (`"equal_opportunity"`).

```python
from fairness_audit_kit import EqualizedOddsPostprocessor, render_equalized_odds_postprocessing_report

post = EqualizedOddsPostprocessor(constraint="equalized_odds").fit(y_val, g_val, y_scores=s_val)
post.mixing_rates_                       # {"a": {0: p0, 1: p1}, "b": {...}}
fair = post.predict(g_test, y_scores=s_test, seed=0)        # randomized 0/1
prob = post.positive_probability(g_test, y_scores=s_test)   # deterministic P(Ỹ=1)
print(render_equalized_odds_postprocessing_report(post.result_))
```

`result_` reports TPR/FPR gaps and accuracy before and after (exact
expectations). Fit on a validation split, not the training data.

### Reject Option Classification (Kamiran et al.)

A deterministic post-processing step for a scored classifier. Decisions near the threshold are where the model is least sure, so inside the band `(τ−θ, τ+θ]` the unprivileged group is given the favourable label and the privileged group the unfavourable one. Outside the band, `ŷ = 1[s > τ]`.

The grid follows AIF360. `τ` takes `n_thresholds` values in `[0.01, 0.99]`, and for each `τ`, `θ` takes `n_margins` values in `[0, min(τ, 1−τ)]`. The chosen pair maximises balanced accuracy subject to `|metric| ≤ bound`. Supported metrics are `statistical_parity`, `average_odds` and `equal_opportunity`, each computed as unprivileged minus privileged. The chosen `(τ, θ)` and the resulting decisions match aif360 0.6.1 exactly: this was checked offline on 36 problems, and reference values are pinned in the tests.

```python
from fairness_audit_kit import RejectOptionClassifier, render_reject_option_report

roc = RejectOptionClassifier(privileged="male", metric="average_odds", bound=0.05).fit(y_val, g_val, s_val)
roc.threshold_, roc.margin_
y_fair = roc.predict(g_test, s_test)
res = roc.result_          # before/after SPD, AOD, EOD, balanced accuracy, accuracy; Pareto frontier
print(render_reject_option_report(res))
```

```bash
fairness-audit reject-option --data scores.csv --privileged 1 --metric statistical_parity \
    --bound 0.05 --output roc_decisions.csv --report roc.md
```

Note that the rule uses the sensitive attribute at decision time, which is disparate treatment and may be restricted in some domains.

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

Evaluate fairness metrics on predictions. The report always includes
equalized odds (TPR/FPR) and predictive parity (PPV) per-group rates
and pairwise gaps, demographic parity (per-group positive rates,
statistical parity difference, and disparate impact ratio), plus
Theil / generalized entropy.

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

### fairness-audit reweigh

Write Kamiran–Calders reweighing weights as a new column of the training CSV.

```bash
fairness-audit reweigh --data train.csv --output train_weighted.csv --report reweighing.md

Options:
  --data PATH             Training data CSV (required)
  --target-col STR        Label column (default: target)
  --group-col STR         Group column (default: sensitive_attr)
  --weight-col STR        Output weight column (default: sample_weight)
  --output PATH           Output CSV path (required)
  --report PATH           Optional Markdown report path
```

### fairness-audit repair

Repair numeric features with the Feldman et al. disparate impact remover and write the repaired CSV.

```bash
fairness-audit repair --data train.csv --output train_repaired.csv --repair-level 0.8 --report repair.md

Options:
  --data PATH             Data CSV (required)
  --group-col STR         Group column (default: sensitive_attr)
  --features COL [...]    Columns to repair (default: all numeric except group / --exclude)
  --exclude COL [...]     Never repaired by default (default: target y_pred y_score)
  --repair-level FLOAT    lambda in [0, 1] (default: 1.0 = full repair)
  --output PATH           Output CSV path (required)
  --report PATH           Optional Markdown report path
```

### fairness-audit postprocess

Fit Hardt et al. post-processing and write randomized fair decisions plus their probabilities.

```bash
fairness-audit postprocess --data preds.csv --output fair_preds.csv --report postprocess.md

Options:
  --data PATH             CSV with labels, groups and y_pred or y_score (required)
  --constraint STR        equalized_odds (default) or equal_opportunity
  --fp-cost / --fn-cost   Relative error costs (default: 1.0 / 1.0)
  --threshold FLOAT       Score threshold when y_pred is absent (default: 0.5)
  --seed INT              Seed for the randomized decisions
  --prob-col / --output-col  Output columns (default: fair_positive_prob / y_pred_fair)
  --output PATH           Output CSV path (required)
  --report PATH           Optional Markdown report path
```

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
| Statistical parity difference | Same unsigned gap, with per-group positive rates and pairwise gaps. Labels not required | 0 |
| Positive rate by group | P(y_pred = 1 \| group) for the sensitive attribute | equal across groups |
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
