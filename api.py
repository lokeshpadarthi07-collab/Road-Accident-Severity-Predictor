"""
Production REST API for Road Accident Severity Predictor.
Built with FastAPI and Pydantic v2.
"""

import os
import json
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.predict import AccidentSeverityPredictor, DISCLAIMER_TEXT
from src.data_loader import load_config

# Initialize FastAPI application
app = FastAPI(
    title="Road Accident Severity Predictor API",
    description=(
        "Production-grade REST microservice providing real-time and batch "
        "road accident severity estimation based on pre-collision factors."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load predictor at startup
predictor: Optional[AccidentSeverityPredictor] = None


@app.on_event("startup")
def startup_event():
    """Load model artifacts on application startup."""
    global predictor
    try:
        predictor = AccidentSeverityPredictor()
    except Exception as e:
        print(f"Warning: Predictor failed to load at startup: {e}")


class IncidentInput(BaseModel):
    """Schema for single incident pre-accident attributes."""
    accident_time: str = Field(..., description="Timestamp of accident in HH:MM format", example="18:30")
    day_of_week: str = Field(..., description="Day of the week", example="Friday")
    speed_limit: int = Field(..., ge=10, le=100, description="Posted speed limit in mph", example=60)
    weather_conditions: str = Field(..., description="Weather condition", example="Raining + high winds")
    road_surface_conditions: str = Field(..., description="Road surface condition", example="Wet or damp")
    light_conditions: str = Field(..., description="Lighting condition", example="Darkness - no lighting")
    road_type: str = Field(..., description="Roadway geometry", example="Single carriageway")
    urban_or_rural_area: str = Field(..., description="Urban or rural classification", example="Rural")
    junction_detail: str = Field(..., description="Junction geometry", example="Not at junction")
    number_of_vehicles: int = Field(..., ge=1, le=20, description="Number of vehicles involved", example=2)
    driver_age_band: str = Field(..., description="Age band of primary driver", example="18-25")
    vehicle_type: str = Field(..., description="Primary vehicle type", example="Motorcycle")


class RiskFactor(BaseModel):
    factor: str
    level: str
    impact: str


class PredictionResponse(BaseModel):
    predicted_severity: str
    risk_tier: str
    probabilities: Dict[str, float]
    classes: List[str]
    risk_factors: List[RiskFactor]
    disclaimer: str


class BatchIncidentInput(BaseModel):
    incidents: List[IncidentInput]


@app.get("/", tags=["Root"])
def root():
    """Root landing endpoint with system status."""
    return {
        "service": "Road Accident Severity Predictor API",
        "version": "1.0.0",
        "status": "online",
        "docs_url": "/docs"
    }


@app.get("/health", tags=["Monitoring"])
def health_check():
    """Health check endpoint for liveness and model status."""
    is_ready = predictor is not None and predictor.model is not None
    return {
        "status": "healthy" if is_ready else "degraded",
        "model_loaded": is_ready,
        "champion_model": predictor.metadata.get("best_model", {}).get("name", "Unknown") if predictor and predictor.metadata else "None"
    }


@app.get("/model_info", tags=["Monitoring"])
def model_info():
    """Retrieve active champion model metadata, features, and benchmark metrics."""
    if predictor is None or predictor.metadata is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model metadata is not loaded.")
    return predictor.metadata


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_severity(payload: IncidentInput):
    """
    Predict road accident severity from pre-collision features for a single incident.
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor engine is not available. Please ensure models are trained."
        )

    try:
        input_dict = payload.model_dump()
        result = predictor.predict(input_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/batch_predict", tags=["Inference"])
def batch_predict_severity(payload: BatchIncidentInput):
    """
    Batch predict severity for multiple accident incident records.
    """
    if predictor is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Predictor engine is not available.")

    try:
        import pandas as pd
        records = [item.model_dump() for item in payload.incidents]
        df = pd.DataFrame(records)
        result = predictor.predict(df)
        return result
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
