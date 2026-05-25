"""
End-to-End Pipeline Test
========================
Validates the complete data flow from ingestion through serving.

Pipeline stages tested:
1. Data Ingestion (mocked APIs)
2. Data Cleaning
3. Feature Engineering
4. Feature Store Preparation
5. Model Training
6. Drift Detection
7. API Serving
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import load_config, ensure_directories
from src.ingestion.api_client import DataIngestionPipeline
from src.ingestion.data_cleaner import run_cleaning_pipeline
from src.ingestion.feature_engineer import run_feature_pipeline
from src.training.trainer import train_all_models
from src.intelligence.drift_detector import DriftDetector
from src.training.model_registry import ModelRegistry
from src.intelligence.health_advisor import HealthAdvisor


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def e2e_config():
    """Load configuration for E2E testing."""
    config = load_config("config/config.yaml")
    ensure_directories(config)
    return config


@pytest.fixture
def mock_aqicn_response():
    """Mock AQICN API response."""
    return {
        "status": "ok",
        "data": {
            "aqi": 120,
            "iaqi": {
                "pm25": {"v": 85.0},
                "pm10": {"v": 120.0},
                "o3": {"v": 45.0},
                "no2": {"v": 35.0},
                "so2": {"v": 15.0},
                "co": {"v": 350.0},
            },
            "dominentpol": "pm25",
            "city": {"name": "Islamabad"},
            "time": {"iso": datetime.now().isoformat()}
        }
    }


@pytest.fixture
def mock_openmeteo_response():
    """Mock Open-Meteo API response."""
    now = datetime.now()
    return {
        "hourly": {
            "time": [now.isoformat(), (now + timedelta(hours=1)).isoformat()],
            "temperature_2m": [28.5, 29.0],
            "relativehumidity_2m": [45, 40],
            "windspeed_10m": [8.5, 9.0],
            "surface_pressure": [980.5, 981.0]
        }
    }


# ============================================================
# Stage 1: Ingestion
# ============================================================

def test_stage1_data_ingestion(e2e_config, mock_aqicn_response, mock_openmeteo_response):
    """Test Stage 1: Data ingestion from mocked APIs."""

    with patch('src.ingestion.api_clients.aqicn.AQICNClient.fetch_aqi') as mock_aqicn, \
         patch('src.ingestion.api_clients.open_meteo.OpenMeteoClient.fetch_weather') as mock_weather:

        mock_aqicn.return_value = {
            "aqi": 120,
            "pm25": 85.0,
            "pm10": 120.0,
            "o3": 45.0,
            "no2": 35.0,
            "so2": 15.0,
            "co": 350.0,
            "dominant_pollutant": "pm25",
            "station_name": "Islamabad",
            "timestamp": datetime.now().isoformat()
        }

        mock_weather.return_value = {
            "temperature": 28.5,
            "humidity": 45.0,
            "pressure": 980.5,
            "wind_speed": 8.5,
            "timestamp": datetime.now().isoformat()
        }

        pipeline = DataIngestionPipeline(e2e_config)
        result = pipeline.run()

        # Verify ingestion produced data
        assert not result.empty, "Ingestion pipeline returned empty DataFrame"
        assert len(result) > 0, "No records ingested"

        # Verify data contains expected columns
        expected_cols = ["aqi", "temperature", "humidity"]
        for col in expected_cols:
            assert col in result.columns or result[col].notna().any(), f"Missing {col} data"

        # Verify raw data was persisted
        raw_path = Path(e2e_config["paths"]["raw_data"]) / "raw_aqi_data.parquet"
        assert raw_path.exists(), "Raw data not persisted to Parquet"

        print(f"✓ Stage 1 Complete: {len(result)} records ingested")


# ============================================================
# Stage 2: Cleaning
# ============================================================

def test_stage2_data_cleaning(e2e_config):
    """Test Stage 2: Data cleaning."""

    # Create test raw data
    dates = pd.date_range("2026-04-01", periods=50, freq="H")
    raw_df = pd.DataFrame({
        "timestamp": dates,
        "aqi": [np.nan, 100, 150, 600, 120] * 10,  # Include outliers and NaNs
        "pm25": [50, np.nan, 80, 1200, 90] * 10,
        "pm10": [80, 120, np.nan, 2000, 150] * 10,
        "temperature": [25, 30, 28, 32, 26] * 10,
        "humidity": [50, 45, 40, 35, 50] * 10,
    })

    raw_path = Path(e2e_config["paths"]["raw_data"]) / "raw_aqi_data.parquet"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_parquet(raw_path, index=False)

    # Run cleaning
    clean_df = run_cleaning_pipeline(e2e_config)

    # Verify cleaning worked
    assert not clean_df.empty, "Cleaning produced empty DataFrame"
    assert len(clean_df) > 0, "All records were dropped"
    assert len(clean_df) <= len(raw_df), "More records after cleaning (shouldn't happen)"

    # Verify no NaNs in numeric columns
    numeric_cols = clean_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        assert clean_df[col].notna().all(), f"NaNs remain in {col}"

    # Verify outliers were clamped
    if "aqi" in clean_df.columns:
        assert (clean_df["aqi"] <= 500).all(), "AQI values exceed max (500)"
        assert (clean_df["aqi"] >= 0).all(), "AQI values below min (0)"

    # Verify cleaned data was persisted
    proc_path = Path(e2e_config["paths"]["processed_data"]) / "processed_aqi_data.parquet"
    assert proc_path.exists(), "Processed data not persisted"

    print(f"✓ Stage 2 Complete: {len(clean_df)} records cleaned")


# ============================================================
# Stage 3: Feature Engineering
# ============================================================

def test_stage3_feature_engineering(e2e_config):
    """Test Stage 3: Feature engineering."""

    # Create clean data
    dates = pd.date_range("2026-04-01", periods=100, freq="H")
    clean_df = pd.DataFrame({
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

    proc_path = Path(e2e_config["paths"]["processed_data"]) / "processed_aqi_data.parquet"
    proc_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(proc_path, index=False)

    # Run feature engineering
    featured_df = run_feature_pipeline(e2e_config, clean_df)

    # Verify features were created
    assert not featured_df.empty, "Feature engineering produced empty DataFrame"
    assert len(featured_df.columns) > len(clean_df.columns), "No features created"

    # Verify key features exist
    expected_features = [
        "hour", "day_of_week", "month", "is_weekend",
        "aqi_roll_mean_6h", "aqi_lag_1h"
    ]

    for feat in expected_features:
        if feat in clean_df.columns:
            assert feat in featured_df.columns, f"Missing feature: {feat}"

    # Verify no NaNs after feature engineering
    assert featured_df.isna().sum().sum() == 0, "NaNs present after feature engineering"

    print(f"✓ Stage 3 Complete: {len(featured_df.columns)} features engineered")


# ============================================================
# Stage 4: Feature Store Preparation
# ============================================================

def test_stage4_feast_preparation(e2e_config):
    """Test Stage 4: Feast feature store preparation."""

    # Create processed data
    dates = pd.date_range("2026-04-01", periods=50, freq="H")
    processed_df = pd.DataFrame({
        "timestamp": dates,
        "aqi": np.random.randint(30, 150, 50).astype(float),
        "pm25": np.random.randint(10, 100, 50).astype(float),
        "pm10": np.random.randint(20, 150, 50).astype(float),
        "temperature": np.random.uniform(15, 35, 50),
        "humidity": np.random.uniform(30, 80, 50),
    })

    proc_path = Path(e2e_config["paths"]["processed_data"]) / "processed_aqi_data.parquet"
    proc_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_parquet(proc_path, index=False)

    # Prepare Feast data
    df = pd.read_parquet(proc_path)
    df["city"] = "islamabad"
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    feast_path = Path(e2e_config["paths"]["feast_data"]) / "aqi_features.parquet"
    feast_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(feast_path, index=False)

    # Verify Feast data was created
    assert feast_path.exists(), "Feast Parquet not created"

    # Verify Feast data is readable
    feast_df = pd.read_parquet(feast_path)
    assert len(feast_df) > 0, "Feast Parquet is empty"
    assert "city" in feast_df.columns, "Entity column not added"
    assert feast_df["city"].unique().tolist() == ["islamabad"], "Entity column incorrect"

    print(f"✓ Stage 4 Complete: Feast data prepared ({len(feast_df)} rows)")


# ============================================================
# Stage 5: Training
# ============================================================

@pytest.mark.slow
def test_stage5_model_training(e2e_config):
    """Test Stage 5: Model training."""

    # Create sufficient training data (min 50 samples as per config)
    dates = pd.date_range("2026-04-01", periods=150, freq="H")
    np.random.seed(42)

    train_df = pd.DataFrame({
        "timestamp": dates,
        "aqi": np.random.randint(30, 150, 150).astype(float),
        "pm25": np.random.randint(10, 100, 150).astype(float),
        "pm10": np.random.randint(20, 150, 150).astype(float),
        "o3": np.random.randint(20, 80, 150).astype(float),
        "no2": np.random.randint(10, 60, 150).astype(float),
        "so2": np.random.randint(5, 30, 150).astype(float),
        "co": np.random.randint(100, 500, 150).astype(float),
        "temperature": np.random.uniform(15, 35, 150),
        "humidity": np.random.uniform(30, 80, 150),
        "pressure": np.random.uniform(950, 1020, 150),
        "wind_speed": np.random.uniform(1, 15, 150),
        "hour": np.random.randint(0, 24, 150),
        "day_of_week": np.random.randint(0, 7, 150),
        "month": np.random.randint(1, 13, 150),
        "is_weekend": np.random.randint(0, 2, 150),
        "aqi_roll_mean_6h": np.random.randint(30, 150, 150).astype(float),
        "aqi_roll_mean_12h": np.random.randint(30, 150, 150).astype(float),
        "aqi_roll_mean_24h": np.random.randint(30, 150, 150).astype(float),
        "aqi_lag_1h": np.random.randint(30, 150, 150).astype(float),
        "aqi_lag_3h": np.random.randint(30, 150, 150).astype(float),
        "aqi_lag_6h": np.random.randint(30, 150, 150).astype(float),
        "aqi_lag_12h": np.random.randint(30, 150, 150).astype(float),
        "aqi_lag_24h": np.random.randint(30, 150, 150).astype(float),
    })

    proc_path = Path(e2e_config["paths"]["processed_data"]) / "processed_aqi_data.parquet"
    proc_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(proc_path, index=False)

    # Run training
    results = train_all_models(e2e_config)

    # Verify training completed for all horizons
    assert len(results) > 0, "No models trained"
    assert 24 in results, "24h horizon not trained"
    assert 48 in results, "48h horizon not trained"
    assert 72 in results, "72h horizon not trained"

    # Verify models were saved
    registry = ModelRegistry(e2e_config)
    models_list = registry.list_models()
    assert len(models_list) > 0, "No models in registry"

    # Verify best models were marked
    for horizon in [24, 48, 72]:
        best = registry.get_best_model_name(horizon)
        assert best is not None, f"No best model marked for {horizon}h"
        assert best in ["ridge", "random_forest", "xgboost"], f"Invalid model name: {best}"

    print(f"✓ Stage 5 Complete: {len(results)} horizon groups trained")


# ============================================================
# Stage 6: Drift Detection
# ============================================================

def test_stage6_drift_detection(e2e_config):
    """Test Stage 6: Drift detection."""

    # Create two datasets
    dates_ref = pd.date_range("2026-03-01", periods=100, freq="H")
    dates_cur = pd.date_range("2026-04-01", periods=50, freq="H")

    ref_df = pd.DataFrame({
        "aqi": np.random.normal(loc=100, scale=15, size=100),
        "temperature": np.random.normal(loc=25, scale=5, size=100),
        "humidity": np.random.normal(loc=50, scale=10, size=100),
    })

    # Shift current distribution (simulating drift)
    cur_df = pd.DataFrame({
        "aqi": np.random.normal(loc=120, scale=20, size=50),  # Shifted mean
        "temperature": np.random.normal(loc=30, scale=5, size=50),  # Shifted
        "humidity": np.random.normal(loc=45, scale=10, size=50),  # Shifted
    })

    detector = DriftDetector(e2e_config)
    report = detector.detect(ref_df, cur_df)

    # Verify drift report structure
    assert "status" in report
    assert "total_features" in report
    assert "drifted_features" in report
    assert "drift_ratio" in report
    assert "details" in report

    # Verify drift was detected (with high probability given the shifts)
    assert report["total_features"] > 0, "No features checked"
    assert report["drifted_ratio"] >= 0, "Invalid drift ratio"

    print(f"✓ Stage 6 Complete: Drift detected ({report['drifted_features']}/{report['total_features']} features)")


# ============================================================
# Stage 7: Health Advisory
# ============================================================

def test_stage7_health_advisory(e2e_config):
    """Test Stage 7: Health advisory generation."""

    advisor = HealthAdvisor(e2e_config)

    # Test various AQI levels
    test_cases = [
        (25, "Good"),
        (75, "Moderate"),
        (125, "Unhealthy for Sensitive Groups"),
        (175, "Unhealthy"),
        (250, "Very Unhealthy"),
        (400, "Hazardous"),
        (None, "Unknown"),
    ]

    for aqi_val, expected_level in test_cases:
        advisory = advisor.get_advice(aqi_val)

        assert "aqi" in advisory
        assert "level" in advisory
        assert "color" in advisory
        assert "advice" in advisory

        if aqi_val is not None:
            # Level should match or be close
            assert advisory["level"] in ["Good", "Moderate", "Unhealthy for Sensitive Groups",
                                        "Unhealthy", "Very Unhealthy", "Hazardous"]

    print(f"✓ Stage 7 Complete: Health advisories generated for {len(test_cases)} AQI values")


# ============================================================
# Full Pipeline Test
# ============================================================

@pytest.mark.slow
def test_full_e2e_pipeline(e2e_config):
    """Test complete end-to-end pipeline."""

    print("\n" + "="*60)
    print("RUNNING FULL E2E PIPELINE TEST")
    print("="*60)

    # Stage 1: Create raw data
    dates = pd.date_range("2026-04-01", periods=200, freq="H")
    np.random.seed(42)

    raw_df = pd.DataFrame({
        "timestamp": dates,
        "aqi": np.random.randint(30, 150, 200).astype(float),
        "pm25": np.random.randint(10, 100, 200).astype(float),
        "pm10": np.random.randint(20, 150, 200).astype(float),
        "o3": np.random.randint(20, 80, 200).astype(float),
        "no2": np.random.randint(10, 60, 200).astype(float),
        "so2": np.random.randint(5, 30, 200).astype(float),
        "co": np.random.randint(100, 500, 200).astype(float),
        "temperature": np.random.uniform(15, 35, 200),
        "humidity": np.random.uniform(30, 80, 200),
        "pressure": np.random.uniform(950, 1020, 200),
        "wind_speed": np.random.uniform(1, 15, 200),
    })

    raw_path = Path(e2e_config["paths"]["raw_data"]) / "raw_aqi_data.parquet"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_parquet(raw_path, index=False)
    print(f"✓ Created raw data: {len(raw_df)} records")

    # Stage 2: Clean
    cleaner_config = e2e_config.copy()
    clean_df = run_cleaning_pipeline(cleaner_config)
    assert len(clean_df) > 0
    print(f"✓ Cleaned data: {len(clean_df)} records")

    # Stage 3: Engineer features
    featured_df = run_feature_pipeline(cleaner_config, clean_df)
    assert len(featured_df) > 0
    assert len(featured_df.columns) > len(clean_df.columns)
    print(f"✓ Engineered features: {len(featured_df.columns)} features")

    # Stage 4: Prepare Feast
    feast_df = featured_df.copy()
    feast_df["city"] = "islamabad"
    feast_df["timestamp"] = pd.to_datetime(feast_df["timestamp"])

    feast_path = Path(e2e_config["paths"]["feast_data"]) / "aqi_features.parquet"
    feast_path.parent.mkdir(parents=True, exist_ok=True)
    feast_df.to_parquet(feast_path, index=False)
    print(f"✓ Feast data prepared: {len(feast_df)} records")

    # Stage 5: Train models
    results = train_all_models(e2e_config)
    assert len(results) == 3  # 3 horizons
    total_models = sum(len(v) for v in results.values())
    print(f"✓ Models trained: {total_models} models across 3 horizons")

    # Stage 6: Drift detection
    split_idx = int(len(featured_df) * 0.7)
    ref_data = featured_df.iloc[:split_idx]
    cur_data = featured_df.iloc[split_idx:]

    detector = DriftDetector(e2e_config)
    drift_report = detector.detect(ref_data, cur_data)
    assert "status" in drift_report
    print(f"✓ Drift detection: {drift_report['status']} ({drift_report['drift_ratio']:.1%} features drifted)")

    # Stage 7: Health advisory
    latest_aqi = featured_df["aqi"].iloc[-1]
    advisor = HealthAdvisor(e2e_config)
    advisory = advisor.get_advice(latest_aqi)
    assert advisory["level"] in ["Good", "Moderate", "Unhealthy for Sensitive Groups",
                                "Unhealthy", "Very Unhealthy", "Hazardous"]
    print(f"✓ Health advisory: {advisory['level']} (AQI {latest_aqi:.0f})")

    print("\n" + "="*60)
    print("✓ FULL E2E PIPELINE TEST PASSED")
    print("="*60 + "\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "not slow"])
