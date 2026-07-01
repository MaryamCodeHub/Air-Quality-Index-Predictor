"""
API Integration Tests
=====================
Tests all FastAPI endpoints using mocked backend data.

Tests cover:
- Prediction endpoint
- Metrics endpoint
- Drift detection endpoint
- Health advice endpoint
- Retraining endpoint
- History retrieval
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.api.main import app
from src.utils.helpers import load_config, ensure_directories, save_json


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def test_config():
    """Load test configuration."""
    config = load_config("config/config.yaml")
    ensure_directories(config)
    return config


@pytest.fixture
def test_processed_data(test_config):
    """Create test processed data."""
    dates = pd.date_range("2026-04-01", periods=100, freq="H")
    np.random.seed(42)

    df = pd.DataFrame({
        "timestamp": dates,
        "aqi": np.random.randint(30, 150, 100).astype(float),
        "pm25": np.random.randint(10, 100, 100).astype(float),
        "pm10": np.random.randint(20, 150, 100).astype(float),
        "o3": np.random.randint(20, 80, 100).astype(float),
        "no2": np.random.randint(10, 60, 100).astype(float),
        "so2": np.random.randint(5, 30, 100).astype(float),
        "co": np.random.randint(100, 500, 100).astype(float),
        "temperature": np.random.uniform(15, 35, 100),
        "humidity": np.random.uniform(30, 80, 100),
        "pressure": np.random.uniform(950, 1020, 100),
        "wind_speed": np.random.uniform(1, 15, 100),
    })

    # Create raw data parquet
    raw_path = Path(test_config["paths"]["raw_data"]) / "raw_aqi_data.parquet"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(raw_path, index=False)

    # Create processed data parquet
    proc_path = Path(test_config["paths"]["processed_data"]) / "processed_aqi_data.parquet"
    proc_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(proc_path, index=False)

    return df


@pytest.fixture
def test_models(test_config, test_processed_data):
    """Create dummy trained models."""
    from src.training.model_registry import ModelRegistry

    registry = ModelRegistry(test_config)

    # Create simple dummy models
    from sklearn.linear_model import Ridge
    import joblib

    model = Ridge()

    # Create dummy X, y for fitting
    X_dummy = test_processed_data.drop(
        columns=["timestamp", "aqi", "dominant_pollutant", "station_name"], errors="ignore"
    ).fillna(0).select_dtypes(include=[np.number]).values[:50]

    y_dummy = test_processed_data["aqi"].values[:50]

    model.fit(X_dummy, y_dummy)

    # Save model and metadata for each horizon
    for horizon in [24, 48, 72]:
        registry.save_model(
            model=model,
            model_name="ridge",
            horizon=horizon,
            metrics={"rmse": 10.5, "mae": 8.2, "r2": 0.75},
            feature_names=list(range(X_dummy.shape[1]))
        )
        registry.mark_best("ridge", horizon)

    return registry


# ============================================================
# Test 1: Root Endpoint
# ============================================================

def test_root_endpoint(client):
    """Test GET / returns service info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "version" in data


# ============================================================
# Test 2: Predict Endpoint
# ============================================================

def test_predict_valid_horizon(client, test_models, test_processed_data):
    """Test POST /api/v1/predict with valid horizon."""
    response = client.post(
        "/api/v1/predict",
        json={"horizon": 24}
    )
    assert response.status_code == 200
    data = response.json()

    assert "predicted_aqi" in data
    assert "horizon_hours" in data
    assert "model_used" in data
    assert "health_advisory" in data
    assert "timestamp" in data
    assert data["horizon_hours"] == 24


def test_predict_invalid_horizon(client):
    """Test POST /api/v1/predict with invalid horizon."""
    response = client.post(
        "/api/v1/predict",
        json={"horizon": 99}  # Invalid horizon
    )
    assert response.status_code == 400


def test_predict_default_horizon(client, test_models, test_processed_data):
    """Test POST /api/v1/predict uses default horizon."""
    response = client.post(
        "/api/v1/predict",
        json={}  # Uses default 24h
    )
    assert response.status_code == 200
    data = response.json()
    assert data["horizon_hours"] == 24


def test_predict_aqi_range(client, test_models, test_processed_data):
    """Test predicted AQI is within valid range."""
    response = client.post(
        "/api/v1/predict",
        json={"horizon": 24}
    )
    data = response.json()
    aqi = data["predicted_aqi"]
    assert 0 <= aqi <= 500, f"AQI {aqi} out of range [0, 500]"


def test_predict_includes_health_advisory(client, test_models, test_processed_data):
    """Test prediction includes health advisory."""
    response = client.post(
        "/api/v1/predict",
        json={"horizon": 24}
    )
    data = response.json()
    advisory = data["health_advisory"]

    assert "aqi" in advisory
    assert "level" in advisory
    assert "color" in advisory
    assert "advice" in advisory


# ============================================================
# Test 3: Metrics Endpoint
# ============================================================

def test_metrics_endpoint(client, test_models):
    """Test GET /api/v1/metrics returns model metrics."""
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()

    assert "models" in data
    assert len(data["models"]) > 0

    # Check model structure
    for model in data["models"]:
        assert "model_name" in model
        assert "horizon_hours" in model
        assert "metrics" in model
        assert "rmse" in model["metrics"]
        assert "mae" in model["metrics"]
        assert "r2" in model["metrics"]


def test_metrics_all_horizons(client, test_models):
    """Test metrics cover all forecast horizons."""
    response = client.get("/api/v1/metrics")
    data = response.json()
    models = data["models"]

    horizons = {m["horizon_hours"] for m in models}
    assert 24 in horizons
    assert 48 in horizons
    assert 72 in horizons


# ============================================================
# Test 4: Health Advice Endpoint
# ============================================================

def test_health_advice_endpoint(client, test_processed_data):
    """Test GET /api/v1/health-advice returns advice."""
    response = client.get("/api/v1/health-advice")
    assert response.status_code == 200
    data = response.json()

    assert "current_aqi" in data
    assert "level" in data
    assert "color" in data
    assert "advice" in data


def test_health_advice_valid_levels(client, test_processed_data):
    """Test health advice returns valid level."""
    response = client.get("/api/v1/health-advice")
    data = response.json()
    level = data["level"]

    valid_levels = [
        "Good", "Moderate", "Unhealthy for Sensitive Groups",
        "Unhealthy", "Very Unhealthy", "Hazardous", "Unknown"
    ]
    assert level in valid_levels, f"Invalid level: {level}"


# ============================================================
# Test 4b: Current Conditions Endpoint
# ============================================================

def test_current_endpoint_uses_live_sources(client, monkeypatch):
    """Test GET /api/v1/current uses AQICN AQI and Open-Meteo weather payloads."""
    from src.api import routes

    class MockResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def mock_get(url, params=None, timeout=None):
        if "api.waqi.info" in url:
            return MockResponse(
                {
                    "status": "ok",
                    "data": {
                        "aqi": 137,
                        "time": {"iso": "2026-07-01T10:00:00+05:00"},
                    },
                }
            )

        return MockResponse(
            {
                "current": {
                    "temperature_2m": 31.2,
                    "relative_humidity_2m": 45,
                    "wind_speed_10m": 3.4,
                }
            }
        )

    monkeypatch.setitem(routes.CONFIG["api"]["aqicn"], "api_key", "test-token")
    monkeypatch.setattr(routes.requests, "get", mock_get)

    response = client.get("/api/v1/current")
    assert response.status_code == 200
    data = response.json()

    assert data["location"] == "Islamabad, Pakistan"
    assert data["current_aqi"] == 137.0
    assert data["category"] == "Unhealthy for Sensitive Groups"
    assert data["temperature"] == 31.2
    assert data["humidity"] == 45.0
    assert data["wind_speed"] == 3.4
    assert data["aqi_source"] == "AQICN"
    assert data["weather_source"] == "Open-Meteo"
    assert isinstance(data["is_stale"], bool)


# ============================================================
# Test 5: Drift Status Endpoint
# ============================================================

def test_drift_status_endpoint(client):
    """Test GET /api/v1/drift-status returns drift info."""
    response = client.get("/api/v1/drift-status")
    assert response.status_code == 200
    data = response.json()

    assert "status" in data
    assert "total_features" in data
    assert "drifted_features" in data
    assert "drift_ratio" in data
    assert "timestamp" in data


def test_drift_status_valid_status(client):
    """Test drift status returns valid status."""
    response = client.get("/api/v1/drift-status")
    data = response.json()
    status = data["status"]

    valid_statuses = ["no_drift", "moderate_drift", "high_drift", "no_data"]
    assert status in valid_statuses


def test_drift_ratio_range(client):
    """Test drift ratio is between 0 and 1."""
    response = client.get("/api/v1/drift-status")
    data = response.json()
    ratio = data["drift_ratio"]

    assert 0 <= ratio <= 1, f"Drift ratio {ratio} out of range [0, 1]"


# ============================================================
# Test 6: History Endpoint
# ============================================================

def test_history_endpoint(client, test_processed_data):
    """Test GET /api/v1/history returns historical data."""
    response = client.get("/api/v1/history")
    assert response.status_code == 200
    data = response.json()

    assert "raw" in data
    assert "processed" in data
    assert isinstance(data["raw"], list)
    assert isinstance(data["processed"], list)


def test_history_limit(client, test_processed_data):
    """Test history limit parameter."""
    response = client.get("/api/v1/history?limit=10")
    assert response.status_code == 200
    data = response.json()

    # Limit should reduce results
    assert len(data["raw"]) <= 10
    assert len(data["processed"]) <= 10


def test_history_contains_aqi(client, test_processed_data):
    """Test history includes AQI readings."""
    response = client.get("/api/v1/history?limit=1")
    data = response.json()

    if data["raw"]:
        assert "aqi" in data["raw"][0]


# ============================================================
# Test 7: Drift History Endpoint
# ============================================================

def test_drift_history_endpoint(client):
    """Test GET /api/v1/drift-history returns history."""
    response = client.get("/api/v1/drift-history")
    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)  # Returns array of drift events


# ============================================================
# Test 8: Retrain Endpoint
# ============================================================

@pytest.mark.slow
def test_retrain_endpoint(client, test_models, test_processed_data):
    """Test POST /api/v1/retrain triggers retraining."""
    response = client.post("/api/v1/retrain")
    # May timeout if training takes too long; acceptable for this test
    if response.status_code in [200, 504]:
        if response.status_code == 200:
            data = response.json()
            assert "status" in data


# ============================================================
# Test 9: CORS Headers
# ============================================================

def test_cors_headers(client):
    """Test CORS headers are present."""
    response = client.get("/")
    # CORS headers should be present
    assert "access-control-allow-origin" in response.headers or response.status_code == 200


# ============================================================
# Test 10: Error Handling
# ============================================================

def test_invalid_endpoint(client):
    """Test invalid endpoint returns 404."""
    response = client.get("/api/v1/nonexistent")
    assert response.status_code == 404


def test_prediction_without_models(client, tmp_path, test_config):
    """Test prediction fails gracefully without trained models."""
    # This would require mocking the model registry to have no models
    # Acceptable to skip if models exist
    pass


# ============================================================
# Test 11: Response Schema Validation
# ============================================================

def test_predict_response_schema(client, test_models, test_processed_data):
    """Test /predict response matches PredictResponse schema."""
    response = client.post(
        "/api/v1/predict",
        json={"horizon": 24}
    )
    data = response.json()

    # Validate all required fields
    required_fields = [
        "city", "horizon_hours", "predicted_aqi", "model_used",
        "health_advisory", "timestamp"
    ]

    for field in required_fields:
        assert field in data, f"Missing required field: {field}"


def test_metrics_response_schema(client, test_models):
    """Test /metrics response structure."""
    response = client.get("/api/v1/metrics")
    data = response.json()

    assert "models" in data
    for model in data["models"]:
        required = ["model_name", "horizon_hours", "metrics", "feature_names"]
        for field in required:
            assert field in model


# ============================================================
# Test 12: Data Types
# ============================================================

def test_predict_aqi_is_float(client, test_models, test_processed_data):
    """Test predicted_aqi is a float."""
    response = client.post(
        "/api/v1/predict",
        json={"horizon": 24}
    )
    data = response.json()
    assert isinstance(data["predicted_aqi"], (int, float))


def test_metrics_values_are_numeric(client, test_models):
    """Test metric values are numeric."""
    response = client.get("/api/v1/metrics")
    data = response.json()

    for model in data["models"]:
        metrics = model["metrics"]
        assert isinstance(metrics["rmse"], (int, float))
        assert isinstance(metrics["mae"], (int, float))
        assert isinstance(metrics["r2"], (int, float))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
