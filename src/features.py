"""
Feature Engineering Module for Road Accident Severity Predictor.

Extracts domain-driven pre-accident features while strictly preventing target leakage.
Handles temporal extraction, environmental risk flags, and interaction terms.
"""

import re
import logging
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


def extract_hour_from_time(time_val: Any) -> int:
    """Extract integer hour (0-23) from time string or return a default (12)."""
    if pd.isna(time_val):
        return 12
    if isinstance(time_val, (int, float)):
        try:
            val = int(time_val)
            return max(0, min(23, val))
        except (ValueError, TypeError):
            return 12

    s = str(time_val).strip()
    match = re.search(r"(\d{1,2}):?(\d{2})?", s)
    if match:
        try:
            hour = int(match.group(1))
            return max(0, min(23, hour))
        except (ValueError, TypeError):
            return 12
    return 12


class AccidentFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible feature engineering transformer for accident records.
    Derives domain features strictly from pre-accident conditions.
    """

    def __init__(
        self,
        time_column: str = "accident_time",
        drop_id_columns: bool = True,
        id_columns: Optional[List[str]] = None
    ):
        self.time_column = time_column
        self.drop_id_columns = drop_id_columns
        self.id_columns = id_columns or ["accident_id", "id", "incident_id"]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit method (stateless transformer)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw DataFrame into engineered feature set.
        """
        df = X.copy()

        # 1. Drop identifier columns if present
        if self.drop_id_columns:
            for col in self.id_columns:
                if col in df.columns:
                    df = df.drop(columns=[col])

        # 2. Extract hour of day from time column if present
        if self.time_column in df.columns and "hour_of_day" not in df.columns:
            df["hour_of_day"] = df[self.time_column].apply(extract_hour_from_time).astype(int)
            df = df.drop(columns=[self.time_column])
        elif "hour_of_day" not in df.columns:
            # Default midday if time column missing
            df["hour_of_day"] = 12

        # 3. Derive rush hour flag (07:00-09:59 or 16:00-18:59)
        hour = df["hour_of_day"]
        df["is_rush_hour"] = (((hour >= 7) & (hour <= 9)) | ((hour >= 16) & (hour <= 18))).astype(int)

        # 4. Derive night-time flag (21:00-05:59)
        df["is_night"] = ((hour >= 21) | (hour <= 5)).astype(int)

        # 5. Derive weekend flag from day_of_week
        if "day_of_week" in df.columns:
            weekend_days = ["Saturday", "Sunday"]
            df["is_weekend"] = df["day_of_week"].astype(str).str.strip().isin(weekend_days).astype(int)
        else:
            df["is_weekend"] = 0

        # 6. Environmental risk flags
        if "weather_conditions" in df.columns:
            adverse_weather = ["Raining + high winds", "Snowing", "Fog or mist"]
            df["is_adverse_weather"] = (
                df["weather_conditions"].astype(str).isin(adverse_weather).astype(int)
            )
        else:
            df["is_adverse_weather"] = 0

        if "road_surface_conditions" in df.columns:
            adverse_surface = ["Snow", "Frost or ice", "Flood over 3cm"]
            df["is_adverse_surface"] = (
                df["road_surface_conditions"].astype(str).isin(adverse_surface).astype(int)
            )
        else:
            df["is_adverse_surface"] = 0

        if "light_conditions" in df.columns:
            poor_light = ["Darkness - no lighting", "Darkness - lights unlit"]
            df["is_poor_lighting"] = (
                df["light_conditions"].astype(str).isin(poor_light).astype(int)
            )
        else:
            df["is_poor_lighting"] = 0

        # 7. Vulnerable vehicle flag (motorcycles have higher casualty severity)
        if "vehicle_type" in df.columns:
            df["is_vulnerable_vehicle"] = (
                (df["vehicle_type"].astype(str) == "Motorcycle").astype(int)
            )
        else:
            df["is_vulnerable_vehicle"] = 0

        # 8. High-speed road flag
        if "speed_limit" in df.columns:
            df["is_high_speed"] = (df["speed_limit"] >= 60).astype(int)
        else:
            df["is_high_speed"] = 0

        return df


def engineer_features(df: pd.DataFrame, time_column: str = "accident_time") -> pd.DataFrame:
    """Helper functional interface to engineer features."""
    transformer = AccidentFeatureEngineer(time_column=time_column)
    return transformer.transform(df)
