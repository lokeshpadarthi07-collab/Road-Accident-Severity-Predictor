"""
Prediction Pipeline for Road Accident Severity Predictor.

Provides end-to-end inference for single instances and batch DataFrames.
Includes input validation, risk factor attribution, calibrated probabilities,
and experimental decision-support disclaimers.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import joblib

from src.features import AccidentFeatureEngineer
from src.preprocessing import load_preprocessor_artifacts
from src.data_loader import load_config

logger = logging.getLogger(__name__)

DISCLAIMER_TEXT = (
    "DISCLAIMER: Predictions produced by this application are experimental, "
    "statistical decision-support estimates. They must NOT be treated as guaranteed "
    "outcomes or used as the sole basis for emergency dispatch, insurance liability, "
    "or road safety infrastructure enforcement."
)


class AccidentSeverityPredictor:
    """
    Inference engine that wraps preprocessors, models, and explanations.
    """

    def __init__(
        self,
        model_path: str = "models/best_model.joblib",
        preprocessor_path: str = "models/preprocessor.joblib",
        encoder_path: str = "models/label_encoder.joblib",
        metadata_path: str = "models/model_metadata.json"
    ):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.encoder_path = encoder_path
        self.metadata_path = metadata_path

        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        self.metadata = None
        self.feature_engineer = AccidentFeatureEngineer()

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load required model and preprocessing binaries."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model binary not found at '{self.model_path}'. "
                "Please train the model first by running: python -m src.train"
            )

        self.model = joblib.load(self.model_path)
        self.preprocessor, self.label_encoder = load_preprocessor_artifacts(
            self.preprocessor_path, self.encoder_path
        )

        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")

    def validate_input(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate single input dictionary against expected schema and domains."""
        errors = []

        # Validate speed limit
        speed = data.get("speed_limit")
        if speed is not None:
            try:
                s_val = float(speed)
                if s_val < 10 or s_val > 100:
                    errors.append(f"Speed limit ({s_val}) is outside plausible range (10 - 100 mph).")
            except (ValueError, TypeError):
                errors.append(f"Invalid speed limit value: {speed}")

        # Validate number of vehicles
        vehicles = data.get("number_of_vehicles")
        if vehicles is not None:
            try:
                v_val = int(vehicles)
                if v_val < 1 or v_val > 20:
                    errors.append(f"Number of vehicles ({v_val}) must be between 1 and 20.")
            except (ValueError, TypeError):
                errors.append(f"Invalid vehicle count value: {vehicles}")

        return (len(errors) == 0, errors)

    def extract_risk_factors(self, input_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Identify key contributing risk factors in the input features."""
        factors = []

        speed = float(input_data.get("speed_limit", 30))
        if speed >= 60:
            factors.append({
                "factor": f"High Speed Limit ({int(speed)} mph)",
                "level": "High",
                "impact": "Significantly amplifies collision impact energy and fatal injury probability."
            })
        elif speed >= 50:
            factors.append({
                "factor": f"Elevated Speed Limit ({int(speed)} mph)",
                "level": "Medium",
                "impact": "Increases stopping distance and severity potential."
            })

        vehicle = str(input_data.get("vehicle_type", "Car"))
        if vehicle == "Motorcycle":
            factors.append({
                "factor": "Vulnerable Road User (Motorcycle)",
                "level": "High",
                "impact": "Absence of passenger cabin protection sharply elevates severe injury risk."
            })

        light = str(input_data.get("light_conditions", "Daylight"))
        if "Darkness - no lighting" in light or "Darkness - lights unlit" in light:
            factors.append({
                "factor": f"Poor Visibility ({light})",
                "level": "High",
                "impact": "Severely degrades reaction window and obstacle recognition."
            })

        road_surface = str(input_data.get("road_surface_conditions", "Dry"))
        if road_surface in ["Frost or ice", "Flood over 3cm", "Snow"]:
            factors.append({
                "factor": f"Hazardous Road Surface ({road_surface})",
                "level": "High",
                "impact": "Substantially reduces tire adhesion and increases skid potential."
            })

        weather = str(input_data.get("weather_conditions", "Fine no high winds"))
        if weather in ["Raining + high winds", "Fog or mist", "Snowing"]:
            factors.append({
                "factor": f"Adverse Weather ({weather})",
                "level": "Medium",
                "impact": "Decreases driver visibility and vehicle control."
            })

        area = str(input_data.get("urban_or_rural_area", "Urban"))
        if area == "Rural":
            factors.append({
                "factor": "Rural Road Setting",
                "level": "Medium",
                "impact": "Rural roadways typically have higher travel speeds and longer emergency transit times."
            })

        return factors

    def predict(self, raw_data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Execute prediction pipeline on input data.
        """
        if isinstance(raw_data, dict):
            is_valid, errors = self.validate_input(raw_data)
            if not is_valid:
                raise ValueError(f"Input validation failed: {'; '.join(errors)}")
            df = pd.DataFrame([raw_data])
            single_instance = True
        elif isinstance(raw_data, pd.DataFrame):
            df = raw_data.copy()
            single_instance = len(df) == 1
        else:
            raise TypeError("Expected input to be a dict or a pandas DataFrame.")

        # Apply feature engineering
        df_engineered = self.feature_engineer.transform(df)

        # Drop non-feature columns if accidentally present
        if "accident_severity" in df_engineered.columns:
            df_engineered = df_engineered.drop(columns=["accident_severity"])

        # Transform using fitted preprocessor
        X_trans = self.preprocessor.transform(df_engineered)

        # Generate model predictions
        preds_enc = self.model.predict(X_trans)
        preds_class = self.label_encoder.inverse_transform(preds_enc)

        # Compute probabilities if supported
        probas = None
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(X_trans)

        class_names = list(self.label_encoder.classes_)

        if single_instance:
            predicted_class = str(preds_class[0])
            prob_dict = {}
            if probas is not None:
                for idx, cname in enumerate(class_names):
                    prob_dict[cname] = float(round(probas[0][idx], 4))

            # Determine risk tier
            risk_tier = "Moderate"
            if predicted_class == "Fatal" or prob_dict.get("Fatal", 0) > 0.15:
                risk_tier = "Critical"
            elif predicted_class == "Serious" or prob_dict.get("Serious", 0) > 0.35:
                risk_tier = "Elevated"
            else:
                risk_tier = "Standard / Slight"

            risk_factors = self.extract_risk_factors(df.iloc[0].to_dict())

            return {
                "predicted_severity": predicted_class,
                "risk_tier": risk_tier,
                "probabilities": prob_dict,
                "classes": class_names,
                "risk_factors": risk_factors,
                "disclaimer": DISCLAIMER_TEXT
            }
        else:
            # Batch results
            batch_results = []
            for i in range(len(df)):
                res = {
                    "predicted_severity": str(preds_class[i]),
                    "probabilities": {class_names[j]: float(round(probas[i][j], 4)) for j in range(len(class_names))} if probas is not None else {}
                }
                batch_results.append(res)
            return {
                "batch_predictions": batch_results,
                "total_records": len(df),
                "disclaimer": DISCLAIMER_TEXT
            }


if __name__ == "__main__":
    # Test sample prediction
    sample_input = {
        "accident_time": "22:45",
        "day_of_week": "Friday",
        "speed_limit": 60,
        "weather_conditions": "Raining + high winds",
        "road_surface_conditions": "Wet or damp",
        "light_conditions": "Darkness - no lighting",
        "road_type": "Single carriageway",
        "urban_or_rural_area": "Rural",
        "junction_detail": "Not at junction",
        "number_of_vehicles": 2,
        "driver_age_band": "18-25",
        "vehicle_type": "Motorcycle"
    }

    try:
        predictor = AccidentSeverityPredictor()
        result = predictor.predict(sample_input)
        print("Sample Prediction Output:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Prediction test note: {e}")
