# Trained Models & Artifacts Directory

This directory stores trained model binaries, preprocessors, label encoders, and evaluation metadata.

## Artifact Manifest

- `best_model.joblib`: Serialized top-performing classifier (Random Forest or XGBoost).
- `random_forest.joblib`: Serialized tuned Random Forest model pipeline.
- `xgboost.joblib`: Serialized tuned XGBoost model pipeline.
- `preprocessor.joblib`: Fitted `ColumnTransformer` including imputation, one-hot encoding, and scaling.
- `label_encoder.joblib`: Fitted target encoder (`LabelEncoder`) mapping class names (`Slight`, `Serious`, `Fatal`) to numeric identifiers (`0, 1, 2`).
- `model_metadata.json`: Machine-readable metadata capturing:
  - Evaluation timestamp
  - Feature names and types
  - Preprocessing configurations
  - Random seed
  - Measured test metrics (Accuracy, Macro/Weighted F1, Recall, Precision, ROC-AUC)
  - Model hyperparameters
- `evaluation_reports/`: Contains JSON metric breakdowns and per-class classification reports.

## How to Regenerate Artifacts

To retrain and re-evaluate all models with reproducible seeds:

```bash
python -m src.train
```

To run standalone evaluation on test data:

```bash
python -m src.evaluate
```

