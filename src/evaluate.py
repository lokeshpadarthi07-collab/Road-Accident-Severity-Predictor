"""
Model Evaluation Module for Road Accident Severity Predictor.

Computes comprehensive classification metrics, handles multiclass ROC-AUC,
evaluates per-class performance on imbalanced severity classes,
and generates machine-readable evaluation reports and comparison tables.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from sklearn.dummy import DummyClassifier

logger = logging.getLogger(__name__)


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics for accident severity prediction.
    """
    if class_names is None:
        class_names = [str(cls) for cls in np.unique(y_true)]

    acc = float(accuracy_score(y_true, y_pred))

    # Macro & Weighted metrics
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Per-class metrics
    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    per_class_metrics = {}
    for i, name in enumerate(class_names):
        per_class_metrics[name] = {
            "precision": float(round(p_class[i], 4)),
            "recall": float(round(r_class[i], 4)),
            "f1_score": float(round(f1_class[i], 4)),
            "support": int(support_class[i])
        }

    cm = confusion_matrix(y_true, y_pred)

    # Multiclass ROC-AUC (One-vs-Rest) if probabilities are provided
    roc_auc_ovr = None
    if y_proba is not None and len(class_names) > 1:
        try:
            roc_auc_ovr = float(round(roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="macro"
            ), 4))
        except Exception as e:
            logger.warning(f"ROC-AUC computation skipped: {e}")

    results = {
        "accuracy": float(round(acc, 4)),
        "macro_f1": float(round(f1_macro, 4)),
        "weighted_f1": float(round(f1_weighted, 4)),
        "macro_precision": float(round(p_macro, 4)),
        "macro_recall": float(round(r_macro, 4)),
        "roc_auc_ovr": roc_auc_ovr,
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names
    }

    return results


def evaluate_baseline_majority(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Evaluate a Dummy Majority-Class Baseline.
    Crucial for showing why raw accuracy is misleading on imbalanced datasets.
    """
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_pred = dummy.predict(X_test)
    y_proba = dummy.predict_proba(X_test)

    baseline_metrics = evaluate_classifier(y_test, y_pred, y_proba, class_names=class_names)
    baseline_metrics["model_name"] = "Dummy Majority Baseline"
    return baseline_metrics


def generate_comparison_table(models_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a clean, comparative pandas DataFrame across evaluated models.
    """
    rows = []
    for name, res in models_results.items():
        row = {
            "Model": name,
            "Accuracy": f"{res.get('accuracy', 0.0) * 100:.2f}%",
            "Macro F1": f"{res.get('macro_f1', 0.0):.4f}",
            "Weighted F1": f"{res.get('weighted_f1', 0.0):.4f}",
            "Macro Recall": f"{res.get('macro_recall', 0.0):.4f}",
            "ROC-AUC (OvR)": f"{res.get('roc_auc_ovr', 0.0):.4f}" if res.get('roc_auc_ovr') is not None else "N/A"
        }
        # Add minority class recall (Fatal and Serious)
        per_class = res.get("per_class", {})
        if "Fatal" in per_class:
            row["Fatal Recall"] = f"{per_class['Fatal']['recall'] * 100:.1f}%"
        if "Serious" in per_class:
            row["Serious Recall"] = f"{per_class['Serious']['recall'] * 100:.1f}%"
        rows.append(row)

    df_comp = pd.DataFrame(rows)
    return df_comp


def save_evaluation_report(
    results: Dict[str, Any],
    report_path: str = "models/evaluation_reports/evaluation_results.json"
) -> None:
    """Save evaluation metrics to disk as JSON."""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved evaluation report to {report_path}")


def load_evaluation_report(
    report_path: str = "models/evaluation_reports/evaluation_results.json"
) -> Optional[Dict[str, Any]]:
    """Load evaluation metrics JSON from disk if available."""
    if not os.path.exists(report_path):
        return None
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    """Execute standalone evaluation of trained models on processed test data."""
    import joblib
    from src.data_loader import load_config
    from src.preprocessing import load_preprocessor_artifacts

    config = load_config()
    test_path = config["paths"]["processed_test"]
    if not os.path.exists(test_path):
        print(f"Test data not found at {test_path}. Run 'python -m src.train' first.")
        return

    test_df = pd.read_parquet(test_path)
    target_col = config["data"]["target_column"]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    preprocessor, target_encoder = load_preprocessor_artifacts(
        config["paths"]["preprocessor_path"],
        config["paths"]["label_encoder_path"]
    )

    X_test_trans = preprocessor.transform(X_test)
    y_test_enc = target_encoder.transform(y_test)
    class_names = list(target_encoder.classes_)

    results_summary = {}

    # 1. Best / Champion Model
    best_model_path = config["paths"]["best_model_path"]
    if os.path.exists(best_model_path):
        best_model = joblib.load(best_model_path)
        y_pred = best_model.predict(X_test_trans)
        y_proba = best_model.predict_proba(X_test_trans) if hasattr(best_model, "predict_proba") else None
        res = evaluate_classifier(y_test_enc, y_pred, y_proba, class_names=class_names)
        results_summary["Champion Model"] = res

    # 2. Random Forest
    rf_path = os.path.join(config["paths"]["models_dir"], "random_forest.joblib")
    if os.path.exists(rf_path):
        rf_model = joblib.load(rf_path)
        y_pred = rf_model.predict(X_test_trans)
        y_proba = rf_model.predict_proba(X_test_trans) if hasattr(rf_model, "predict_proba") else None
        res = evaluate_classifier(y_test_enc, y_pred, y_proba, class_names=class_names)
        results_summary["Random Forest"] = res

    # 3. XGBoost
    xgb_path = os.path.join(config["paths"]["models_dir"], "xgboost.joblib")
    if os.path.exists(xgb_path):
        xgb_model = joblib.load(xgb_path)
        y_pred = xgb_model.predict(X_test_trans)
        y_proba = xgb_model.predict_proba(X_test_trans) if hasattr(xgb_model, "predict_proba") else None
        res = evaluate_classifier(y_test_enc, y_pred, y_proba, class_names=class_names)
        results_summary["XGBoost"] = res

    comp_df = generate_comparison_table(results_summary)
    print("\n" + "="*70)
    print("ROAD ACCIDENT SEVERITY PREDICTOR — MODEL EVALUATION BENCHMARK")
    print("="*70)
    print(comp_df.to_string(index=False))
    print("="*70 + "\n")


if __name__ == "__main__":
    main()


