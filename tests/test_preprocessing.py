"""
Unit Tests for Preprocessing and Feature Engineering.
"""

import pytest
import pandas as pd
import numpy as np
from src.features import AccidentFeatureEngineer, extract_hour_from_time
from src.preprocessing import build_preprocessor, TargetEncoder, get_feature_names_out


def test_extract_hour_from_time():
    """Verify time parser extracts correct 24h hour integer."""
    assert extract_hour_from_time("08:30") == 8
    assert extract_hour_from_time("17:45") == 17
    assert extract_hour_from_time("00:15") == 0
    assert extract_hour_from_time("23:59") == 23
    assert extract_hour_from_time(np.nan) == 12
    assert extract_hour_from_time("invalid") == 12


def test_accident_feature_engineer():
    """Verify feature engineering derives flags and drops identifiers."""
    df = pd.DataFrame({
        "accident_id": ["ACC-01", "ACC-02"],
        "accident_time": ["08:15", "23:45"],
        "day_of_week": ["Monday", "Saturday"],
        "weather_conditions": ["Fine no high winds", "Raining + high winds"],
        "road_surface_conditions": ["Dry", "Frost or ice"],
        "light_conditions": ["Daylight", "Darkness - no lighting"],
        "vehicle_type": ["Car", "Motorcycle"],
        "speed_limit": [30, 70]
    })

    engineer = AccidentFeatureEngineer()
    df_transformed = engineer.transform(df)

    # accident_id should be dropped
    assert "accident_id" not in df_transformed.columns
    # time should be converted to hour_of_day
    assert "hour_of_day" in df_transformed.columns
    assert list(df_transformed["hour_of_day"]) == [8, 23]
    # flags
    assert list(df_transformed["is_rush_hour"]) == [1, 0]
    assert list(df_transformed["is_night"]) == [0, 1]
    assert list(df_transformed["is_weekend"]) == [0, 1]
    assert list(df_transformed["is_adverse_weather"]) == [0, 1]
    assert list(df_transformed["is_adverse_surface"]) == [0, 1]
    assert list(df_transformed["is_poor_lighting"]) == [0, 1]
    assert list(df_transformed["is_vulnerable_vehicle"]) == [0, 1]
    assert list(df_transformed["is_high_speed"]) == [0, 1]


def test_target_encoder():
    """Verify custom TargetEncoder maintains consistent order and inverse mapping."""
    encoder = TargetEncoder(class_order=["Slight", "Serious", "Fatal"])
    series = pd.Series(["Slight", "Fatal", "Serious", "Slight"])
    y_enc = encoder.fit_transform(series)

    assert list(y_enc) == [0, 2, 1, 0]

    y_inv = encoder.inverse_transform(y_enc)
    assert list(y_inv) == ["Slight", "Fatal", "Serious", "Slight"]


def test_column_transformer_unknown_category():
    """Verify preprocessor handles unknown categorical levels gracefully without crashing."""
    df_train = pd.DataFrame({
        "weather": ["Fine", "Rain", "Snow"],
        "speed": [30, 50, 70]
    })
    df_test = pd.DataFrame({
        "weather": ["Fog", "Fine"],  # 'Fog' was not in train!
        "speed": [40, 30]
    })

    preprocessor = build_preprocessor(categorical_features=["weather"], numerical_features=["speed"])
    X_train_trans = preprocessor.fit_transform(df_train)
    X_test_trans = preprocessor.transform(df_test)

    assert X_train_trans.shape[0] == 3
    assert X_test_trans.shape[0] == 2
    assert not np.isnan(X_test_trans).any()

