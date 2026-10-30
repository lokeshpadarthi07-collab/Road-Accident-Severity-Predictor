"""
Data Loading and Validation Module for Road Accident Severity Predictor.

Handles data acquisition, schema validation, leak detection, class distribution analysis,
and reproducible generation of calibrated demonstration data when external datasets are not provided.
"""

import os
import logging
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Known post-accident features that constitute data leakage if used for pre-collision severity estimation
POTENTIAL_LEAKAGE_COLUMNS = [
    "number_of_casualties",
    "casualty_severity",
    "hospital_status",
    "hospital_time",
    "police_response_time",
    "vehicle_damage_cost",
    "airbag_deployed",
    "towed_away",
    "insurance_claim_amount"
]


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration safely."""
    if not os.path.exists(config_path):
        # Fallback to looking in parent directory if called from src or tests
        alt_path = os.path.join(os.path.dirname(__file__), "..", config_path)
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def generate_demonstration_dataset(
    output_path: str = "data/raw/road_accidents_demo.csv",
    n_samples: int = 3000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate a calibrated benchmark demonstration dataset based on realistic
    distributions from open road safety datasets (e.g. UK DfT STATS19).

    Ensures pre-accident features only (preventing leakage) with realistic class imbalance:
    - Slight: ~79%
    - Serious: ~18%
    - Fatal: ~3%
    """
    logger.info(f"Generating calibrated benchmark demonstration dataset ({n_samples} samples, seed={random_state})...")
    rng = np.random.default_rng(random_state)

    # Pre-accident categories
    weather_choices = [
        "Fine no high winds", "Raining no high winds", "Raining + high winds",
        "Snowing", "Fog or mist", "Other"
    ]
    weather_probs = [0.72, 0.16, 0.04, 0.03, 0.03, 0.02]

    road_surface_choices = ["Dry", "Wet or damp", "Snow", "Frost or ice", "Flood over 3cm"]
    road_surface_probs = [0.65, 0.26, 0.03, 0.04, 0.02]

    light_choices = [
        "Daylight", "Darkness - lights lit", "Darkness - lights unlit", "Darkness - no lighting"
    ]
    light_probs = [0.68, 0.18, 0.05, 0.09]

    road_type_choices = ["Single carriageway", "Dual carriageway", "Roundabout", "One way street", "Slip road"]
    road_type_probs = [0.70, 0.16, 0.08, 0.04, 0.02]

    urban_rural_choices = ["Urban", "Rural"]
    urban_rural_probs = [0.62, 0.38]

    junction_choices = [
        "Not at junction", "T or staggered junction", "Crossroads", "Roundabout", "More than 4 arms"
    ]
    junction_probs = [0.42, 0.30, 0.15, 0.09, 0.04]

    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_probs = [0.14, 0.14, 0.15, 0.15, 0.17, 0.13, 0.12]

    driver_age_choices = ["18-25", "26-35", "36-55", "56-65", "65+"]
    driver_age_probs = [0.22, 0.31, 0.28, 0.11, 0.08]

    vehicle_choices = ["Car", "Motorcycle", "Bus", "Goods vehicle", "Other"]
    vehicle_probs = [0.74, 0.10, 0.05, 0.08, 0.03]

    # Sample base attributes
    weather = rng.choice(weather_choices, size=n_samples, p=weather_probs)
    road_surface = rng.choice(road_surface_choices, size=n_samples, p=road_surface_probs)
    light = rng.choice(light_choices, size=n_samples, p=light_probs)
    road_type = rng.choice(road_type_choices, size=n_samples, p=road_type_probs)
    urban_rural = rng.choice(urban_rural_choices, size=n_samples, p=urban_rural_probs)
    junction = rng.choice(junction_choices, size=n_samples, p=junction_probs)
    day = rng.choice(days_of_week, size=n_samples, p=day_probs)
    driver_age = rng.choice(driver_age_choices, size=n_samples, p=driver_age_probs)
    vehicle = rng.choice(vehicle_choices, size=n_samples, p=vehicle_probs)

    # Speeds correlated with road type and urban/rural
    speed_limits = []
    for ur, rt in zip(urban_rural, road_type):
        if ur == "Urban":
            speed_limits.append(rng.choice([20, 30, 40], p=[0.20, 0.70, 0.10]))
        else:
            if rt in ["Dual carriageway", "Slip road"]:
                speed_limits.append(rng.choice([50, 60, 70], p=[0.15, 0.35, 0.50]))
            else:
                speed_limits.append(rng.choice([40, 50, 60], p=[0.20, 0.35, 0.45]))
    speed_limits = np.array(speed_limits)

    # Number of vehicles (1 to 5)
    num_vehicles = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.35, 0.52, 0.10, 0.03])

    # Times (hours 0-23 and minutes 0-59)
    # Peak accident hours: morning rush (07-09), evening rush (16-19), late night weekend
    hour_weights = [
        0.015, 0.010, 0.008, 0.007, 0.010, 0.020,  # 00-05
        0.040, 0.075, 0.090, 0.055, 0.045, 0.045,  # 06-11
        0.050, 0.050, 0.055, 0.065, 0.085, 0.090,  # 12-17
        0.070, 0.045, 0.030, 0.025, 0.020, 0.015   # 18-23
    ]
    hour_weights = np.array(hour_weights) / sum(hour_weights)
    hours = rng.choice(np.arange(24), size=n_samples, p=hour_weights)
    minutes = rng.integers(0, 60, size=n_samples)
    times = [f"{h:02d}:{m:02d}" for h, m in zip(hours, minutes)]

    # Calculate severity probability based on domain risk factors
    # Base log-odds: Slight (dominant), Serious (moderate), Fatal (rare)
    severities = []
    for i in range(n_samples):
        # Base risk score
        risk = 0.0

        # Speed risk
        if speed_limits[i] >= 60:
            risk += 1.3
        elif speed_limits[i] >= 50:
            risk += 0.8
        elif speed_limits[i] <= 20:
            risk -= 0.6

        # Road condition risk
        if road_surface[i] in ["Frost or ice", "Flood over 3cm"]:
            risk += 0.9
        elif road_surface[i] in ["Snow", "Wet or damp"]:
            risk += 0.4

        # Light condition risk
        if light[i] in ["Darkness - no lighting", "Darkness - lights unlit"]:
            risk += 1.0
        elif light[i] == "Darkness - lights lit":
            risk += 0.3

        # Vulnerable vehicle risk (motorcycles have higher casualty severity)
        if vehicle[i] == "Motorcycle":
            risk += 1.4
        elif vehicle[i] == "Goods vehicle":
            risk += 0.5

        # Location risk (rural higher speed)
        if urban_rural[i] == "Rural":
            risk += 0.7

        # Young or elderly drivers
        if driver_age[i] in ["18-25", "65+"]:
            risk += 0.4

        # Weather
        if weather[i] in ["Raining + high winds", "Fog or mist"]:
            risk += 0.6

        # Convert risk score to 3-class probabilities
        # Base probabilities: Slight ~ 80%, Serious ~ 17%, Fatal ~ 3%
        p_slight = np.clip(0.85 - (risk * 0.10), 0.35, 0.95)
        p_serious = np.clip(0.12 + (risk * 0.07), 0.04, 0.45)
        p_fatal = np.clip(0.03 + (risk * 0.03), 0.01, 0.20)

        # Normalize probabilities
        total_p = p_slight + p_serious + p_fatal
        p_slight /= total_p
        p_serious /= total_p
        p_fatal /= total_p

        severity_choice = rng.choice(["Slight", "Serious", "Fatal"], p=[p_slight, p_serious, p_fatal])
        severities.append(severity_choice)

    # Realistic small percentage of missing values (to test data imputation)
    # Road surface missing ~1.5%
    missing_surface_idx = rng.choice(n_samples, size=int(n_samples * 0.015), replace=False)
    road_surface_list = list(road_surface)
    for idx in missing_surface_idx:
        road_surface_list[idx] = np.nan

    # Driver age missing ~1.0%
    missing_age_idx = rng.choice(n_samples, size=int(n_samples * 0.010), replace=False)
    driver_age_list = list(driver_age)
    for idx in missing_age_idx:
        driver_age_list[idx] = np.nan

    # Build DataFrame
    df = pd.DataFrame({
        "accident_id": [f"ACC-2026-{i+1:05d}" for i in range(n_samples)],
        "accident_time": times,
        "day_of_week": day,
        "speed_limit": speed_limits,
        "weather_conditions": weather,
        "road_surface_conditions": road_surface_list,
        "light_conditions": light,
        "road_type": road_type,
        "urban_or_rural_area": urban_rural,
        "junction_detail": junction,
        "number_of_vehicles": num_vehicles,
        "driver_age_band": driver_age_list,
        "vehicle_type": vehicle,
        "accident_severity": severities
    })

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved benchmark demonstration dataset to: {output_path}")

    return df


def validate_dataset(
    df: pd.DataFrame,
    target_column: str = "accident_severity",
    required_features: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Validate dataset structure, data types, missing values, duplicates,
    class distribution, and detect potential target leakage.
    """
    report: Dict[str, Any] = {
        "is_valid": True,
        "warnings": [],
        "errors": [],
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_distribution": {},
        "leakage_detected": []
    }

    # 1. Target column existence
    if target_column not in df.columns:
        report["is_valid"] = False
        report["errors"].append(f"Target column '{target_column}' is missing from the dataset.")
    else:
        counts = df[target_column].value_counts().to_dict()
        proportions = df[target_column].value_counts(normalize=True).round(4).to_dict()
        report["class_distribution"] = {
            "counts": counts,
            "proportions": proportions,
            "unique_classes": list(counts.keys())
        }

        # Check for severe class imbalance warning
        min_class_pct = min(proportions.values()) * 100
        if min_class_pct < 5.0:
            report["warnings"].append(
                f"Minority class represents only {min_class_pct:.2f}% of the target. "
                "Class-weighted loss, stratified splitting, and macro-F1 evaluation are required."
            )

    # 2. Check required features if specified
    if required_features:
        missing_features = [col for col in required_features if col not in df.columns]
        if missing_features:
            report["is_valid"] = False
            report["errors"].append(f"Missing required features: {missing_features}")

    # 3. Check for potential target leakage
    for col in df.columns:
        col_lower = col.lower().strip()
        for leak_term in POTENTIAL_LEAKAGE_COLUMNS:
            if leak_term in col_lower:
                report["leakage_detected"].append(col)
                report["warnings"].append(
                    f"Potential target leakage column detected: '{col}'. "
                    "Pre-accident severity prediction should avoid post-accident observations."
                )

    # 4. Check for duplicate rows
    if report["duplicate_rows"] > 0:
        report["warnings"].append(f"Dataset contains {report['duplicate_rows']} duplicate rows.")

    return report


def load_dataset(filepath: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Load dataset from specified path or config path.
    If default demo path does not exist, automatically generates the calibrated demo dataset.
    """
    if config is None:
        try:
            config = load_config()
        except Exception:
            config = {}

    target_path = filepath or config.get("paths", {}).get("raw_data", "data/raw/road_accidents_demo.csv")

    if not os.path.exists(target_path):
        logger.warning(f"Dataset at '{target_path}' not found. Generating default benchmark demonstration dataset...")
        df = generate_demonstration_dataset(output_path=target_path, n_samples=3000, random_state=42)
    else:
        logger.info(f"Loading existing dataset from: {target_path}")
        df = pd.read_csv(target_path)

    # Validate
    target_col = config.get("data", {}).get("target_column", "accident_severity")
    validation = validate_dataset(df, target_column=target_col)

    if not validation["is_valid"]:
        raise ValueError(f"Dataset validation failed: {validation['errors']}")

    logger.info(f"Loaded {len(df)} records across {len(df.columns)} columns successfully.")
    return df


if __name__ == "__main__":
    cfg = load_config()
    data = load_dataset()
    val = validate_dataset(data)
    print("Dataset Validation Summary:")
    print(f"Total Rows: {val['total_rows']}")
    print(f"Class Distribution: {val['class_distribution']['counts']}")
    print(f"Missing Values: {val['missing_values']}")
