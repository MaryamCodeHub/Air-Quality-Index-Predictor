"""
FastAPI Routes — Islamabad AQI System
=======================================
All API endpoints for prediction, metrics, drift, health advice, and retraining.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    DriftResponse,
    HealthAdviceResponse,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
    RetrainResponse,
)
from src.intelligence.drift_detector import DriftDetector
from src.intelligence.health_advisor import HealthAdvisor
from src.training.model_registry import ModelRegistry
from src.utils.helpers import load_config
from src.utils.logger import setup_logger

logger = setup_logger("api.routes")
router = APIRouter()

# Load config once at module level
CONFIG = load_config()


# ============================================================
# POST /predict — AQI Forecast
# ============================================================

@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Generate an AQI forecast for Islamabad at the specified horizon."""
    valid_horizons = CONFIG["training"]["forecast_horizons"]
    if req.horizon not in valid_horizons:
        raise HTTPException(400, f"Invalid horizon. Choose from: {valid_horizons}")

    registry = ModelRegistry(CONFIG)
    model, meta = registry.load_best_model(req.horizon)
    if model is None:
        raise HTTPException(503, f"No trained model for {req.horizon}h. Run 'python run.py train' first.")

    # ----- Feast Feature Store Integration -----
    try:
        from feast import FeatureStore
        from src.utils.helpers import get_project_root
        
        fs_dir = os.path.join(get_project_root(), "feature_store")
        store = FeatureStore(repo_path=fs_dir)
        
        feature_names_feast = [
            "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
            "temperature", "humidity", "pressure", "wind_speed",
            "hour", "day_of_week", "month", "season", "is_weekend",
            "aqi_roll_mean_6h", "aqi_roll_mean_12h", "aqi_roll_mean_24h",
            "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_12h", "aqi_lag_24h"
        ]
        features_to_fetch = [f"aqi_islamabad_features:{feat}" for feat in feature_names_feast]
        
        response = store.get_online_features(
            features=features_to_fetch,
            entity_rows=[{"city": "islamabad"}]
        ).to_dict()
        
        # Build the model feature input vector in the exact required order
        if meta is None:
            raise HTTPException(503, f"No model metadata for {req.horizon}h.")
        feature_names = meta["feature_names"]
        input_vector = []
        for feat in feature_names:
            val = response.get(feat, [None])[0]
            if val is None:
                # Try double underscore format Feast uses:
                val = response.get(f"aqi_islamabad_features__{feat}", [None])[0]
            if val is None:
                val = response.get(f"aqi_islamabad_features:{feat}", [None])[0]
            
            if val is None:
                val = 0.0
            input_vector.append(val)
        
        latest = np.array([input_vector])
        logger.info("Successfully fetched features from Feast online store")
    except Exception as exc:
        logger.warning(f"Failed to fetch features from Feast online store: {exc}. Falling back to direct Parquet reading.")
        # Load latest processed data for feature vector
        proc_path = os.path.join(CONFIG["paths"]["processed_data"], "processed_aqi_data.parquet")
        if not os.path.exists(proc_path):
            raise HTTPException(503, "No processed data available. Run 'python run.py ingest' first.")

        df = pd.read_parquet(proc_path)
        if meta is None:
            raise HTTPException(503, f"No model metadata for {req.horizon}h.")
        feature_names = meta["feature_names"]
        available = [c for c in feature_names if c in df.columns]

        # Use the most recent row as input
        latest = df[available].tail(1).values
        if latest.shape[1] < len(feature_names):
            # Pad missing features with 0
            padded = np.zeros((1, len(feature_names)))
            padded[:, :latest.shape[1]] = latest
            latest = padded

    predicted_aqi = float(model.predict(latest)[0])
    predicted_aqi = max(0, min(500, predicted_aqi))  # Clamp to valid range

    # Health advisory for the prediction
    advisor = HealthAdvisor(CONFIG)
    advisory = advisor.get_advice(predicted_aqi)

    if meta is None:
        raise HTTPException(503, f"No model metadata for {req.horizon}h.")
    
    return PredictResponse(
        horizon_hours=req.horizon,
        predicted_aqi=round(predicted_aqi, 1),
        model_used=meta["model_name"],
        health_advisory=advisory,
        timestamp=datetime.now().isoformat(),
    )


# ============================================================
# GET /metrics — Model Performance
# ============================================================

@router.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    """Return performance metrics for all trained models."""
    registry = ModelRegistry(CONFIG)
    models = registry.list_models()
    if not models:
        raise HTTPException(404, "No trained models found. Run 'python run.py train' first.")
    return MetricsResponse(models=models)


# ============================================================
# GET /drift-status — Drift Detection
# ============================================================

@router.get("/drift-status", response_model=DriftResponse)
def get_drift_status():
    """Return the latest drift detection status."""
    detector = DriftDetector(CONFIG)
    history = detector.get_drift_history()

    if not history:
        return DriftResponse(
            status="no_data",
            total_features=0,
            drifted_features=0,
            drift_ratio=0.0,
            drifted_feature_names=[],
            timestamp=datetime.now().isoformat(),
        )

    latest = history[-1]
    return DriftResponse(
        status=latest.get("status", "unknown"),
        total_features=latest.get("total_features", 0),
        drifted_features=latest.get("drifted_features", 0),
        drift_ratio=latest.get("drift_ratio", 0.0),
        drifted_feature_names=latest.get("drifted_feature_names", []),
        timestamp=latest.get("timestamp", datetime.now().isoformat()),
    )


# ============================================================
# GET /health-advice — Current Health Advisory
# ============================================================

@router.get("/health-advice", response_model=HealthAdviceResponse)
def get_health_advice():
    """Return health advisory based on the latest AQI reading for Islamabad."""
    raw_path = os.path.join(CONFIG["paths"]["raw_data"], "raw_aqi_data.parquet")
    if not os.path.exists(raw_path):
        raise HTTPException(503, "No AQI data available. Run 'python run.py ingest' first.")

    df = pd.read_parquet(raw_path)
    if "aqi" not in df.columns or df["aqi"].dropna().empty:
        raise HTTPException(503, "No AQI readings in data.")

    current_aqi = float(df["aqi"].dropna().iloc[-1])
    advisor = HealthAdvisor(CONFIG)
    advisory = advisor.get_advice(current_aqi)

    return HealthAdviceResponse(
        current_aqi=advisory["aqi"],
        level=advisory["level"],
        color=advisory["color"],
        advice=advisory["advice"],
    )


# ============================================================
# POST /retrain — Trigger Model Retraining
# ============================================================

@router.post("/retrain", response_model=RetrainResponse)
def retrain():
    """Trigger a full model retraining pipeline."""
    try:
        from src.training.trainer import train_all_models

        results = train_all_models(CONFIG)
        if not results:
            return RetrainResponse(
                status="failed",
                message="Training failed — check logs for details",
                models_trained=0,
            )

        n_models = sum(len(v) for v in results.values())
        return RetrainResponse(
            status="success",
            message=f"Retrained {n_models} models across {len(results)} horizons",
            models_trained=n_models,
        )
    except Exception as exc:
        logger.error(f"Retrain failed: {exc}")
        raise HTTPException(500, f"Retraining error: {exc}")


# ============================================================
# GET /history — Historical Data
# ============================================================

@router.get("/history")
def get_history(limit: int = 500):
    """Return historical raw and processed records for the dashboard."""
    raw_path = os.path.join(CONFIG["paths"]["raw_data"], "raw_aqi_data.parquet")
    proc_path = os.path.join(CONFIG["paths"]["processed_data"], "processed_aqi_data.parquet")

    raw_data = []
    processed_data = []

    if os.path.exists(raw_path):
        try:
            df_raw = pd.read_parquet(raw_path)
            # Convert timestamp to string if it is datetime
            if "timestamp" in df_raw.columns:
                df_raw["timestamp"] = df_raw["timestamp"].astype(str)
            df_raw = df_raw.replace({np.nan: None})
            raw_data = df_raw.tail(limit).to_dict(orient="records")
        except Exception as exc:
            logger.warning(f"Error reading raw data: {exc}")

    if os.path.exists(proc_path):
        try:
            df_proc = pd.read_parquet(proc_path)
            if "timestamp" in df_proc.columns:
                df_proc["timestamp"] = df_proc["timestamp"].astype(str)
            df_proc = df_proc.replace({np.nan: None})
            processed_data = df_proc.tail(limit).to_dict(orient="records")
        except Exception as exc:
            logger.warning(f"Error reading processed data: {exc}")

    return {
        "raw": raw_data,
        "processed": processed_data
    }


# ============================================================
# GET /drift-history — Drift History Data
# ============================================================

@router.get("/drift-history")
def get_drift_history():
    """Return the entire drift detection history."""
    detector = DriftDetector(CONFIG)
    return detector.get_drift_history()
