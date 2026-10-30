"""
Training Pipeline for Road Accident Severity Predictor.

Trains Random Forest and XGBoost classifiers on pre-accident features with stratified splitting,
handles severe class imbalance, performs evaluation, selects the best model based on Macro-F1,
serializes models, and saves comprehensive metadata.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from src.data_loader import load_dataset, load_config, validate_dataset
from src.features import AccidentFeatureEngineer
from src.preprocessing import (
    build_preprocessor,
    TargetEncoder,
    save_preprocessor_artifacts,
    get_feature_names_out
)
from src.evaluate import (
    evaluate_classifier,
    evaluate_baseline_majority,
    generate_comparison_table,
    save_evaluation_report
)
from src.visualization import (
    plot_class_distribution,
    plot_accidents_by_hour,
    plot_accidents_by_day,
    plot_environmental_severity,
    plot_speed_vs_severity,
    plot_confusion_matrix_heatmap,
    plot_feature_importance
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def train_pipeline(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Execute end-to-end training, model comparison, artifact persistence, and metadata recording.
    """
    config = load_config(config_path)
    random_state = config.get("project", {}).get("random_state", 42)
    models_dir = config.get("paths", {}).get("models_dir", "models")
    os.makedirs(models_dir, exist_ok=True)
    figures_dir = config.get("paths", {}).get("figures_dir", "reports/figures")
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Load data
    logger.info("Step 1: Loading accident dataset...")
    df_raw = load_dataset(config=config)

    # 2. Generate Exploratory Visualizations
    logger.info("Step 2: Generating exploratory data analysis visualizations...")
    try:
        plot_class_distribution(df_raw, output_dir=figures_dir)
        plot_accidents_by_day(df_raw, output_dir=figures_dir)
        plot_environmental_severity(df_raw, "weather_conditions", "Severity Breakdown by Weather Conditions", output_dir=figures_dir, save_filename="04_weather_severity.png")
        plot_environmental_severity(df_raw, "road_surface_conditions", "Severity Breakdown by Road Surface", output_dir=figures_dir, save_filename="05_road_surface_severity.png")
        plot_environmental_severity(df_raw, "light_conditions", "Severity Breakdown by Light Conditions", output_dir=figures_dir, save_filename="06_light_severity.png")
        plot_speed_vs_severity(df_raw, output_dir=figures_dir)
    except Exception as e:
        logger.warning(f"EDA plot generation non-fatal issue: {e}")

    # 3. Feature Engineering
    logger.info("Step 3: Engineering pre-accident features...")
    feature_engineer = AccidentFeatureEngineer(time_column=config["data"]["time_feature"])
    df_featured = feature_engineer.transform(df_raw)

    # Hourly plot if hour_of_day is present
    if "hour_of_day" in df_featured.columns:
        plot_accidents_by_hour(df_featured, output_dir=figures_dir)

    target_col = config["data"]["target_column"]
    X = df_featured.drop(columns=[target_col])
    y = df_featured[target_col]

    # 4. Stratified Train/Test Split
    test_size = config["data"].get("test_size", 0.20)
    logger.info(f"Step 4: Performing stratified train/test split (test_size={test_size}, seed={random_state})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y if config["data"].get("stratify", True) else None,
        random_state=random_state
    )

    # Save processed splits
    os.makedirs("data/processed", exist_ok=True)
    train_export = pd.concat([X_train, y_train], axis=1)
    test_export = pd.concat([X_test, y_test], axis=1)
    train_export.to_parquet("data/processed/train_processed.parquet", index=False)
    test_export.to_parquet("data/processed/test_processed.parquet", index=False)
    logger.info(f"Saved processed train ({len(X_train)}) and test ({len(X_test)}) datasets to data/processed/")

    # 5. Fit Preprocessor and Target Encoder
    logger.info("Step 5: Fitting ColumnTransformer and TargetEncoder on training split...")
    cat_cols = [c for c in config["data"]["categorical_features"] if c in X_train.columns]
    num_cols = [c for c in config["data"]["numerical_features"] if c in X_train.columns]

    # Additional engineered numerical columns
    engineered_num = ["is_rush_hour", "is_night", "is_weekend", "is_adverse_weather",
                       "is_adverse_surface", "is_poor_lighting", "is_vulnerable_vehicle", "is_high_speed"]
    for c in engineered_num:
        if c in X_train.columns and c not in num_cols:
            num_cols.append(c)

    preprocessor = build_preprocessor(cat_cols, num_cols)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    feature_names = get_feature_names_out(preprocessor)

    target_encoder = TargetEncoder(class_order=config["data"].get("target_classes", ["Slight", "Serious", "Fatal"]))
    y_train_enc = target_encoder.fit_transform(y_train)
    y_test_enc = target_encoder.transform(y_test)
    class_names = list(target_encoder.classes_)

    # Save preprocessor artifacts
    save_preprocessor_artifacts(
        preprocessor,
        target_encoder,
        config["paths"]["preprocessor_path"],
        config["paths"]["label_encoder_path"]
    )

    evaluation_summary: Dict[str, Any] = {}

    # 6. Baseline: Dummy Majority Classifier
    logger.info("Step 6: Evaluating baseline dummy majority classifier...")
    baseline_res = evaluate_baseline_majority(X_train_trans, y_train_enc, X_test_trans, y_test_enc, class_names=class_names)
    evaluation_summary["Dummy Majority Baseline"] = baseline_res

    # 7. Model 1: Random Forest Classifier
    logger.info("Step 7: Training Random Forest Classifier...")
    rf_cfg = config.get("models", {}).get("random_forest", {})
    rf = RandomForestClassifier(
        n_estimators=rf_cfg.get("n_estimators", 200),
        max_depth=rf_cfg.get("max_depth", 12),
        min_samples_split=rf_cfg.get("min_samples_split", 5),
        min_samples_leaf=rf_cfg.get("min_samples_leaf", 2),
        class_weight=rf_cfg.get("class_weight", "balanced"),
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train_trans, y_train_enc)
    y_pred_rf = rf.predict(X_test_trans)
    y_proba_rf = rf.predict_proba(X_test_trans)

    rf_eval = evaluate_classifier(y_test_enc, y_pred_rf, y_proba_rf, class_names=class_names)
    rf_eval["model_name"] = "Random Forest"
    evaluation_summary["Random Forest"] = rf_eval

    # Save Random Forest artifacts
    joblib.dump(rf, os.path.join(models_dir, "random_forest.joblib"))
    plot_confusion_matrix_heatmap(np.array(rf_eval["confusion_matrix"]), class_names, "Random Forest", output_dir=figures_dir)
    if hasattr(rf, "feature_importances_") and feature_names:
        plot_feature_importance(rf.feature_importances_, feature_names, "Random Forest", output_dir=figures_dir)

    # 8. Model 2: XGBoost Classifier
    xgb = None
    if HAS_XGBOOST:
        logger.info("Step 8: Training XGBoost Classifier...")
        xgb_cfg = config.get("models", {}).get("xgboost", {})
        # Compute balanced sample weights for XGBoost multi-class training
        sample_weights = compute_sample_weight("balanced", y_train_enc)

        xgb = XGBClassifier(
            n_estimators=xgb_cfg.get("n_estimators", 150),
            max_depth=xgb_cfg.get("max_depth", 5),
            learning_rate=xgb_cfg.get("learning_rate", 0.08),
            subsample=xgb_cfg.get("subsample", 0.85),
            colsample_bytree=xgb_cfg.get("colsample_bytree", 0.85),
            eval_metric=xgb_cfg.get("eval_metric", "mlogloss"),
            random_state=random_state,
            n_jobs=-1
        )
        xgb.fit(X_train_trans, y_train_enc, sample_weight=sample_weights)
        y_pred_xgb = xgb.predict(X_test_trans)
        y_proba_xgb = xgb.predict_proba(X_test_trans)

        xgb_eval = evaluate_classifier(y_test_enc, y_pred_xgb, y_proba_xgb, class_names=class_names)
        xgb_eval["model_name"] = "XGBoost"
        evaluation_summary["XGBoost"] = xgb_eval

        joblib.dump(xgb, os.path.join(models_dir, "xgboost.joblib"))
        plot_confusion_matrix_heatmap(np.array(xgb_eval["confusion_matrix"]), class_names, "XGBoost", output_dir=figures_dir)
        if hasattr(xgb, "feature_importances_") and feature_names:
            plot_feature_importance(xgb.feature_importances_, feature_names, "XGBoost", output_dir=figures_dir)
    else:
        logger.warning("XGBoost is not installed. Skipping XGBoost training and proceeding with Random Forest.")

    # 9. Model Selection: Select top model based on Macro-F1 (imbalance-aware)
    logger.info("Step 9: Selecting top performing model based on Macro F1-score...")
    candidates = ["Random Forest"]
    if HAS_XGBOOST and xgb is not None:
        candidates.append("XGBoost")

    best_model_name = max(candidates, key=lambda m: evaluation_summary[m]["macro_f1"])
    best_model = rf if best_model_name == "Random Forest" else xgb
    logger.info(f"Top model selected: {best_model_name} (Macro F1: {evaluation_summary[best_model_name]['macro_f1']:.4f})")

    # Save best model to config path
    joblib.dump(best_model, config["paths"]["best_model_path"])

    # 10. Record Metadata and Save Evaluation Reports
    comparison_df = generate_comparison_table(evaluation_summary)
    logger.info("\n" + comparison_df.to_string(index=False))

    metadata = {
        "project_name": config["project"]["name"],
        "project_version": config["project"]["version"],
        "evaluation_timestamp": datetime.now().isoformat(),
        "random_state": random_state,
        "dataset": {
            "total_records": len(df_raw),
            "train_records": len(X_train),
            "test_records": len(X_test),
            "target_column": target_col,
            "classes": class_names,
            "class_distribution": df_raw[target_col].value_counts().to_dict()
        },
        "features": {
            "raw_categorical": cat_cols,
            "raw_numerical": num_cols,
            "engineered_features": engineered_num,
            "encoded_feature_count": len(feature_names),
            "feature_names": feature_names
        },
        "best_model": {
            "name": best_model_name,
            "criterion": "Highest Macro F1-score",
            "accuracy": evaluation_summary[best_model_name]["accuracy"],
            "macro_f1": evaluation_summary[best_model_name]["macro_f1"],
            "weighted_f1": evaluation_summary[best_model_name]["weighted_f1"],
            "roc_auc_ovr": evaluation_summary[best_model_name]["roc_auc_ovr"],
            "saved_path": config["paths"]["best_model_path"]
        },
        "all_models_evaluation": evaluation_summary
    }

    metadata_path = config["paths"]["metadata_path"]
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved complete metadata to: {metadata_path}")

    save_evaluation_report(evaluation_summary, os.path.join(config["paths"]["evaluation_dir"], "evaluation_results.json"))

    return {
        "best_model_name": best_model_name,
        "evaluation_summary": evaluation_summary,
        "comparison_table": comparison_df,
        "metadata": metadata
    }


if __name__ == "__main__":
    train_pipeline()
