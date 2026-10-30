"""
Unit Tests for Prediction Inference Pipeline.
"""

import os
import pytest
import pandas as pd
from src.predict import AccidentSeverityPredictor, DISCLAIMER_TEXT


def test_input_validation():
    """Verify schema bounds checking for inputs."""
    # When model is not trained yet, testing validator standalone
    try:
        predictor = AccidentSeverityPredictor()
    except FileNotFoundError:
        # If model artifacts are not yet saved during initial test discovery, pass gracefully
        pytest.skip("Model artifacts not yet generated. Run training first.")

    valid_input = {"speed_limit": 50, "number_of_vehicles": 2}
    is_valid, errs = predictor.validate_input(valid_input)
    assert is_valid is True
    assert len(errs) == 0

    invalid_speed = {"speed_limit": 250, "number_of_vehicles": 2}
    is_valid, errs = predictor.validate_input(invalid_speed)
    assert is_valid is False
    assert any("speed limit" in e.lower() for e in errs)

    invalid_vehicles = {"speed_limit": 40, "number_of_vehicles": -1}
    is_valid, errs = predictor.validate_input(invalid_vehicles)
    assert is_valid is False
    assert any("vehicles" in e.lower() for e in errs)


def test_prediction_output_structure():
    """Verify prediction output returns required fields and probability constraints."""
    try:
        predictor = AccidentSeverityPredictor()
    except FileNotFoundError:
        pytest.skip("Model artifacts not yet generated. Run training first.")

    sample = {
        "accident_time": "18:30",
        "day_of_week": "Friday",
        "speed_limit": 60,
        "weather_conditions": "Raining + high winds",
        "road_surface_conditions": "Wet or damp",
        "light_conditions": "Darkness - lights lit",
        "road_type": "Single carriageway",
        "urban_or_rural_area": "Rural",
        "junction_detail": "T or staggered junction",
        "number_of_vehicles": 2,
        "driver_age_band": "26-35",
        "vehicle_type": "Car"
    }

    result = predictor.predict(sample)

    assert "predicted_severity" in result
    assert result["predicted_severity"] in ["Slight", "Serious", "Fatal"]
    assert "probabilities" in result
    assert "risk_factors" in result
    assert "disclaimer" in result
    assert result["disclaimer"] == DISCLAIMER_TEXT

    # Probabilities should sum to approximately 1.0
    probs = result["probabilities"]
    assert abs(sum(probs.values()) - 1.0) < 0.05
