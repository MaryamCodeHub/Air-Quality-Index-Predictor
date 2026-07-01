"""
FastAPI Routes — Islamabad AQI System
=======================================
All API endpoints for prediction, metrics, drift, health advice, and retraining.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    CurrentResponse,
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
LOCATION = "Islamabad, Pakistan"
AQI_STALE_AFTER = timedelta(hours=6)


def _safe_float(value: Any) -> Optional[float]:
    """Convert API or stored values to float without raising."""
    try:
        if value is None or value == "-":
            return None
        numeric = float(value)
        if pd.isna(numeric):
            return None
        return numeric
    except (TypeError, ValueError):
        return None


def _parse_timestamp(value: Any) -> Optional[datetime]:
    """Parse timestamps as UTC-aware datetimes when possible."""
    if value is None or pd.isna(value):
        return None

    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None

    return parsed.to_pydatetime()


def _timestamp_iso(ts: Optional[datetime]) -> str:
    """Serialize timestamps consistently for API responses."""
    return (ts or datetime.now(timezone.utc)).isoformat()


def _aqi_is_stale(ts: Optional[datetime], live_fetch_failed: bool) -> bool:
    """Treat failed live AQICN fetches or old station data as stale."""
    if live_fetch_failed or ts is None:
        return True
    return datetime.now(timezone.utc) - ts > AQI_STALE_AFTER


def _fetch_live_aqicn() -> Dict[str, Any]:
    """Fetch the latest Islamabad AQI directly from AQICN."""
    aqicn = CONFIG["api"]["aqicn"]
    api_key = aqicn.get("api_key")
    if not api_key or api_key == "ENV":
        raise RuntimeError("AQICN_API_KEY is not configured")

    url = f"{aqicn['base_url']}/feed/{CONFIG['city']['id']}/"
    timeout = CONFIG.get("resilience", {}).get("request_timeout_seconds", 30)

    response = requests.get(url, params={"token": api_key}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    if payload.get("status") != "ok":
        raise RuntimeError(f"AQICN returned status={payload.get('status')}")

    data = payload.get("data") or {}
    current_aqi = _safe_float(data.get("aqi"))
    if current_aqi is None:
        raise RuntimeError("AQICN response did not include a usable AQI value")

    time_info = data.get("time") or {}
    observed_at = _parse_timestamp(time_info.get("iso") or time_info.get("s"))

    return {
        "current_aqi": round(current_aqi, 1),
        "last_updated": observed_at,
    }


def _fetch_open_meteo_current() -> Dict[str, Optional[float]]:
    """Fetch current Islamabad weather from Open-Meteo."""
    om = CONFIG["api"]["open_meteo"]
    timeout = CONFIG.get("resilience", {}).get("request_timeout_seconds", 30)
    params = {
        "latitude": CONFIG["city"]["lat"],
        "longitude": CONFIG["city"]["lon"],
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "timezone": "Asia/Karachi",
    }

    response = requests.get(om["forecast_url"], params=params, timeout=timeout)
    response.raise_for_status()
    current = (response.json() or {}).get("current") or {}

    return {
        "temperature": _safe_float(current.get("temperature_2m")),
        "humidity": _safe_float(current.get("relative_humidity_2m")),
        "wind_speed": _safe_float(current.get("wind_speed_10m")),
    }


def _latest_stored_reading() -> Optional[Dict[str, Any]]:
    """Read the latest stored raw AQI row as a stale fallback."""
    raw_path = os.path.join(CONFIG["paths"]["raw_data"], "raw_aqi_data.parquet")
    if not os.path.exists(raw_path):
        return None

    df = pd.read_parquet(raw_path)
    if "aqi" not in df.columns:
        return None

    df = df.copy()
    df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce")
    df = df.dropna(subset=["aqi"])
    if df.empty:
        return None

    if "timestamp" in df.columns:
        df["_parsed_timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
            utc=True,
        )
        df = df.sort_values("_parsed_timestamp", na_position="first")

    latest = df.iloc[-1]
    return {
        "current_aqi": round(float(latest.get("aqi")), 1),
        "last_updated": _parse_timestamp(latest.get("timestamp")),
        "temperature": _safe_float(latest.get("temperature") or latest.get("temp")),
        "humidity": _safe_float(latest.get("humidity")),
        "wind_speed": _safe_float(latest.get("wind_speed")),
    }


def _build_current_payload(include_weather: bool = True) -> Dict[str, Any]:
    """Build current AQI/weather payload, keeping live and forecast data separate."""
    live_fetch_failed = False

    try:
        aqi_data = _fetch_live_aqicn()
    except Exception as exc:
        live_fetch_failed = True
        logger.warning(f"Live AQICN current fetch failed: {exc}")
        aqi_data = _latest_stored_reading()
        if aqi_data is None:
            raise HTTPException(
                503,
                "Current AQI unavailable. Configure AQICN_API_KEY or run ingestion first.",
            )

    stored_fallback = _latest_stored_reading()
    if include_weather:
        try:
            weather_data = _fetch_open_meteo_current()
        except Exception as exc:
            logger.warning(f"Live Open-Meteo current fetch failed: {exc}")
            weather_data = {
                "temperature": (stored_fallback or {}).get("temperature"),
                "humidity": (stored_fallback or {}).get("humidity"),
                "wind_speed": (stored_fallback or {}).get("wind_speed"),
            }
    else:
        weather_data = {
            "temperature": None,
            "humidity": None,
            "wind_speed": None,
        }

    last_updated = aqi_data.get("last_updated")
    advisor = HealthAdvisor(CONFIG)
    advisory = advisor.get_advice(aqi_data.get("current_aqi"))

    return {
        "location": LOCATION,
        "current_aqi": aqi_data.get("current_aqi"),
        "category": advisory["level"],
        "temperature": weather_data.get("temperature"),
        "humidity": weather_data.get("humidity"),
        "wind_speed": weather_data.get("wind_speed"),
        "aqi_source": "AQICN",
        "weather_source": "Open-Meteo",
        "last_updated": _timestamp_iso(last_updated),
        "is_stale": _aqi_is_stale(last_updated, live_fetch_failed),
    }


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

@router.get("/current", response_model=CurrentResponse)
def get_current():
    """Return the latest live AQICN AQI and Open-Meteo weather for Islamabad."""
    return CurrentResponse(**_build_current_payload())


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
    """Return health advisory based on the same AQICN current AQI as /current."""
    current = _build_current_payload(include_weather=False)
    current_aqi = current.get("current_aqi")
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
