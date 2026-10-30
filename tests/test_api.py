"""
Unit Tests for FastAPI Microservice Endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from api import app, startup_event


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI test client and trigger startup event."""
    startup_event()
    with TestClient(app) as test_client:
        yield test_client


def test_api_root(client):
    """Verify root endpoint returns system status."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "version" in data


def test_api_health(client):
    """Verify health endpoint reports model status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_api_predict(client):
    """Verify single-incident prediction endpoint."""
    sample_payload = {
        "accident_time": "19:45",
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
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_severity"] in ["Slight", "Serious", "Fatal"]
    assert "probabilities" in data
    assert "risk_factors" in data
    assert "disclaimer" in data


def test_api_predict_invalid_speed(client):
    """Verify validation error when speed limit is out of bounds."""
    invalid_payload = {
        "accident_time": "12:00",
        "day_of_week": "Monday",
        "speed_limit": 500,  # Invalid!
        "weather_conditions": "Fine no high winds",
        "road_surface_conditions": "Dry",
        "light_conditions": "Daylight",
        "road_type": "Roundabout",
        "urban_or_rural_area": "Urban",
        "junction_detail": "Roundabout",
        "number_of_vehicles": 1,
        "driver_age_band": "26-35",
        "vehicle_type": "Car"
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # Unprocessable Entity (Pydantic validation failure)
