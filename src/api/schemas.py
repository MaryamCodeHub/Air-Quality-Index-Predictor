"""
Pydantic Schemas — API Request/Response Models
================================================
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    horizon: int = Field(24, description="Forecast horizon in hours (24, 48, or 72)")


class PredictResponse(BaseModel):
    city: str = "Islamabad"
    horizon_hours: int
    predicted_aqi: float
    model_used: str
    health_advisory: Dict[str, Any]
    timestamp: str


class MetricsResponse(BaseModel):
    models: List[Dict[str, Any]]


class DriftResponse(BaseModel):
    status: str
    total_features: int
    drifted_features: int
    drift_ratio: float
    drifted_feature_names: List[str]
    timestamp: str


class CurrentResponse(BaseModel):
    location: str
    current_aqi: Optional[float]
    category: str
    temperature: Optional[float]
    humidity: Optional[float]
    wind_speed: Optional[float]
    aqi_source: str
    weather_source: str
    last_updated: str
    is_stale: bool


class HealthAdviceResponse(BaseModel):
    city: str = "Islamabad"
    current_aqi: Optional[float]
    level: str
    color: str
    advice: str


class RetrainResponse(BaseModel):
    status: str
    message: str
    models_trained: int
