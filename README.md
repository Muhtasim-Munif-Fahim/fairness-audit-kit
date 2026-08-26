# Fairness Audit Kit

A model fairness and bias evaluation toolkit for machine learning models.

## Features

- **Group Fairness Metrics**: Demographic Parity Difference, Equal Opportunity Difference, Equalized Odds Difference, Disparate Impact Ratio, Calibration by Group
- **Confusion Matrices by Group**: Per-group confusion matrices with configurable thresholds
- **Synthetic Biased Dataset Generator**: Controlled label flip and feature bias injection
- **Threshold Optimization**: Grid search over thresholds per group for fairness constraints
- **Markdown Evaluation Reports**: Human-readable fairness evaluation reports
- **CLI Interface**: End-to-end generate ? predict ? evaluate ? eport workflow

## Installation

`ash
pip install -r requirements.txt
`

## Quick Start

`ash
# Run the demo
python examples/run_demo.py

# Or use the CLI
fairness-audit generate --n-samples 1000 --bias-type label_flip --output data.csv
fairness-audit predict --model-path model.pkl --data-path data.csv --output preds.csv
fairness-audit evaluate --preds-path preds.csv --data-path data.csv --group-col sensitive_attr --output report.md
`

## Module Structure

`
src/fairness_audit_kit/
+-- __init__.py
+-- metrics.py          # Group fairness metrics
+-- optimizer.py        # Threshold optimization
+-- generator.py        # Synthetic biased dataset generator
+-- report.py           # Markdown report renderer
+-- cli.py              # Argparse CLI wiring
`

## Requirements

- Python 3.8+
- numpy
- pandas
- scipy
- scikit-learn
- pytest (for testing)

## Testing

`ash
python -m pytest tests -v
`

## License

MIT License - see LICENSE file for details.
