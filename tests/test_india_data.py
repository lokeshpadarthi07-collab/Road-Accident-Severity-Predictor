"""
Tests for India Highway Network Routes & MoRTH Accident Blackspots Module.
"""

import os
import pytest
import pandas as pd
from src.india_data import (
    get_india_highway_routes,
    get_highway_corridors_list,
    generate_india_accidents_dataset,
    load_india_accidents_dataset,
)


def test_get_india_highway_routes():
    """Verify that highway route polylines are properly structured for PyDeck PathLayer."""
    routes = get_india_highway_routes()
    assert len(routes) >= 8, f"Expected at least 8 major Indian highway routes, found {len(routes)}"

    for r in routes:
        assert "highway_id" in r
        assert "highway_name" in r
        assert "path" in r
        assert "length_km" in r
        assert "color" in r
        assert len(r["color"]) == 4  # RGBA
        assert len(r["path"]) >= 2, f"Path for {r['highway_id']} must have at least 2 waypoints"

        # Verify each waypoint is [lon, lat] within India bounds (approx 68-98 E, 7-38 N)
        for pt in r["path"]:
            assert len(pt) == 2
            lon, lat = pt[0], pt[1]
            assert 65.0 <= lon <= 98.0, f"Longitude {lon} out of India bounds for {r['highway_id']}"
            assert 6.0 <= lat <= 38.0, f"Latitude {lat} out of India bounds for {r['highway_id']}"


def test_get_highway_corridors_list():
    """Verify that MoRTH blackspot corridors have required danger attributes."""
    corridors = get_highway_corridors_list()
    assert len(corridors) >= 10

    for c in corridors:
        assert "morth_blackspot_id" in c
        assert "highway_name" in c
        assert "area_name" in c
        assert "latitude" in c
        assert "longitude" in c
        assert "posted_speed" in c
        assert "advisory_speed" in c
        assert c["advisory_speed"] <= c["posted_speed"], "Advisory speed must be <= posted speed"
        assert "primary_cause" in c


def test_generate_india_accidents_dataset(tmp_path):
    """Test generating a small synthetic India blackspots dataset."""
    test_csv = str(tmp_path / "test_india_accidents.csv")
    df = generate_india_accidents_dataset(output_path=test_csv, n_samples=50, random_state=42)

    assert len(df) == 50
    assert os.path.exists(test_csv)
    expected_cols = [
        "accident_id", "highway_id", "highway_name", "state", "area_name",
        "latitude", "longitude", "posted_speed_kmh", "advisory_safe_speed_kmh",
        "danger_level", "recorded_speed_kmh", "vehicle_type", "accident_severity",
        "primary_cause", "morth_blackspot_id"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"

    assert set(df["accident_severity"].unique()).issubset({"Slight", "Serious", "Fatal"})
