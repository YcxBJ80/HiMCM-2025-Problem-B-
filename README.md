# Entropy-Based Energy Scoring Pipeline

This repository contains a fully scripted workflow that ingests the raw SEDS Excel workbooks in this directory, cleans every indicator into a tidy format, and builds an entropy-weight composite scoring model. The workflow also exports intermediate datasets, model artifacts, and publication-ready figures.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_entropy_model.py --data-dir . --output-dir outputs --year 2023
```

### Key arguments

- `--data-dir`: Directory containing the original Excel files (default `.`).
- `--output-dir`: Folder for cleaned data, model artifacts, and figures (default `outputs`).
- `--year`: Target year for the feature matrix and entropy model (default `2023`).
- `--min-feature-coverage`: Minimum fraction of states that must have data for a feature to be retained (default `0.85`).
- `--min-state-coverage`: Minimum fraction of retained features that a state must have to receive a score (default `0.80`).

## Outputs

Running the pipeline populates the `outputs/` directory with:

- `intermediate/indicator_timeseries.csv`: Long-format table with every state–year–indicator record.
- `model/feature_matrix_<year>.csv`: Final modeling matrix (states × indicators).
- `model/indicator_weights_<year>.csv`: Entropy-derived weights for each indicator.
- `model/state_scores_<year>.csv`: Composite scores (0–100) per state.
- `model/entropy_model_<year>.json`: Serialized model configuration (min/max values, weights, orientation).
- `model/feature_stats_<year>.csv`: Human-readable snapshot of each indicator’s scaling metadata.
- `figures/*.png`: Ready-to-use charts (weight ranking, entropy ranking, score distribution, top states).

## Extending the Model

1. Inspect `outputs/intermediate/indicator_metadata.csv` to review the automatically detected indicators.
2. Adjust the orientation (`benefit` vs `cost`) rules inside `src/entropy_model.py` if a metric’s direction should be reversed.
3. Re-run `src/run_entropy_model.py` to regenerate the model artifacts and figures.

The exported `entropy_model_<year>.json` file can be loaded in any downstream tool to score new observations: normalize each feature with the stored min/max values, apply the cost/benefit direction, multiply by the saved weights, and rescale to the 0–100 score range.
