"""
Unit Tests for Data Validation and Loading.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_loader import validate_dataset, generate_demonstration_dataset, POTENTIAL_LEAKAGE_COLUMNS


def test_generate_demonstration_dataset():
    """Verify demo dataset generates expected sample count and columns."""
    df = generate_demonstration_dataset(output_path=None, n_samples=200, random_state=42)
    assert len(df) == 200
    assert "accident_severity" in df.columns
    assert "speed_limit" in df.columns
    assert "weather_conditions" in df.columns

    # Verify classes present
    unique_classes = set(df["accident_severity"].unique())
    assert "Slight" in unique_classes
    assert "Serious" in unique_classes


def test_validate_dataset_valid():
    """Verify clean dataset passes validation."""
    df = pd.DataFrame({
        "accident_severity": ["Slight", "Serious", "Fatal", "Slight"],
        "speed_limit": [30, 60, 70, 30],
        "weather_conditions": ["Fine no high winds", "Raining", "Snowing", "Fine no high winds"]
    })
    report = validate_dataset(df, target_column="accident_severity")
    assert report["is_valid"] is True
    assert len(report["errors"]) == 0
    assert report["total_rows"] == 4


def test_validate_dataset_missing_target():
    """Verify validation fails when target column is missing."""
    df = pd.DataFrame({
        "speed_limit": [30, 60],
        "weather_conditions": ["Fine no high winds", "Raining"]
    })
    report = validate_dataset(df, target_column="accident_severity")
    assert report["is_valid"] is False
    assert any("missing" in err.lower() for err in report["errors"])


def test_validate_dataset_leakage_detection():
    """Verify detection of post-accident target leakage columns."""
    df = pd.DataFrame({
        "accident_severity": ["Slight", "Serious"],
        "speed_limit": [30, 60],
        "number_of_casualties": [1, 3],  # Post-accident leakage!
        "hospital_status": ["Treated", "Admitted"]  # Post-accident leakage!
    })
    report = validate_dataset(df, target_column="accident_severity")
    assert len(report["leakage_detected"]) == 2
    assert "number_of_casualties" in report["leakage_detected"]
    assert "hospital_status" in report["leakage_detected"]
    assert any("leakage" in w.lower() for w in report["warnings"])

