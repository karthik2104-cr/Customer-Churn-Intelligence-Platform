Generated training output lives in this directory and is not committed.

`python -m ml.training.train` writes:

- `model.joblib` — inference pipeline and explanation background
- `metadata.json` — comparison, thresholds, and held-out metrics
- `segments.joblib` — k-means scaler and centers
- `eda_report.md`, `eda_report.json`, `data_quality.json`, and `figures/`

The API reads these files from the paths in `.env`, or from this directory by default.
