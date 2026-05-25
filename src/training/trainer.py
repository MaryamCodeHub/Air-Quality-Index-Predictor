"""
Model Training Pipeline — Islamabad AQI Forecasting
=====================================================
Trains Ridge Regression, Random Forest, and XGBoost models
for 24h, 48h, and 72h AQI forecasting horizons.
"""

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

from src.training.evaluator import evaluate_model
from src.training.model_registry import ModelRegistry
from src.utils.logger import setup_logger

logger = setup_logger("training.trainer")

# Features used for training (exclude target, timestamp, metadata)
EXCLUDE_COLS = [
    "timestamp", "ingestion_timestamp", "dominant_pollutant",
    "station_name",
    "aqi_change", "aqi_pct_change",  # derived from target
]


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return list of valid feature columns (numeric, non-target)."""
    exclude = set(EXCLUDE_COLS)
    features = []
    for col in df.columns:
        if col in exclude:
            continue
        # Exclude target columns (aqi_target_*)
        if col.startswith("aqi_target_"):
            continue
        if df[col].dtype in [np.float64, np.int64, np.float32, np.int32, float, int]:
            features.append(col)
    return features


def prepare_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Create the target column by shifting AQI by `horizon` rows.

    Each row's target is the AQI value `horizon` time steps in the future.
    """
    col_name = f"aqi_target_{horizon}h"
    df[col_name] = df["aqi"].shift(-horizon)
    # Drop rows where target is NaN (end of series)
    df = df.dropna(subset=[col_name])
    return df, col_name


def build_models(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build model instances from config parameters."""
    mc = config["training"]["models"]
    return {
        "ridge": Ridge(alpha=mc["ridge"]["alpha"]),
        "random_forest": RandomForestRegressor(
            n_estimators=mc["random_forest"]["n_estimators"],
            max_depth=mc["random_forest"]["max_depth"],
            min_samples_split=mc["random_forest"]["min_samples_split"],
            random_state=config["training"]["random_state"],
            n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=mc["xgboost"]["n_estimators"],
            max_depth=mc["xgboost"]["max_depth"],
            learning_rate=mc["xgboost"]["learning_rate"],
            subsample=mc["xgboost"]["subsample"],
            random_state=config["training"]["random_state"],
            verbosity=0,
        ),
    }


def train_all_models(config: Dict[str, Any]) -> Dict[str, Dict]:
    """
    Train all models across all forecast horizons.

    Returns:
        results dict: {horizon: {model_name: {metrics, model, features}}}
    """
    # Load processed data
    proc_path = os.path.join(config["paths"]["processed_data"], "processed_aqi_data.parquet")
    if not os.path.exists(proc_path):
        logger.error(f"Processed data not found: {proc_path}. Run ingestion first.")
        return {}

    df = pd.read_parquet(proc_path)
    min_samples = config["training"].get("min_samples_required", 50)

    if len(df) < min_samples:
        logger.error(
            f"Insufficient data: {len(df)} rows (need ≥{min_samples}). "
            f"Run 'python run.py ingest' more times to accumulate data."
        )
        return {}

    # ----- Hopsworks Feature Store Integration -----
    try:
        from src.feature_store import HopsworksConnector
        
        connector = HopsworksConnector(config)
        status = connector.get_feature_store_status()
        
        if status["connected"]:
            logger.info("Fetching historical features from Hopsworks feature store...")
            
            # List of features to retrieve (corresponds to trained features)
            feature_names = [
                "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
                "temperature", "humidity", "pressure", "wind_speed",
                "hour", "day_of_week", "month", "season", "is_weekend",
                "aqi_roll_mean_6h", "aqi_roll_mean_12h", "aqi_roll_mean_24h",
                "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_12h", "aqi_lag_24h"
            ]
            
            # Try to get features from Hopsworks
            hw_df = connector.get_features(
                feature_names=feature_names,
                feature_group_name="aqi_features_24h",
                version=1
            )
            
            if hw_df is not None and not hw_df.empty:
                df = hw_df
                logger.info("✓ Successfully fetched features from Hopsworks")
            else:
                logger.warning("No features returned from Hopsworks, using Parquet")
        else:
            logger.warning(f"Hopsworks not connected: {status}. Using Parquet fallback.")
            
    except Exception as exc:
        logger.warning(f"Failed to fetch features from Hopsworks: {exc}. Using Parquet fallback.")

    logger.info(f"Training data loaded: {len(df)} rows")

    horizons = config["training"]["forecast_horizons"]
    test_size = config["training"]["test_size"]
    random_state = config["training"]["random_state"]

    registry = ModelRegistry(config)
    all_results = {}

    for horizon in horizons:
        logger.info(f"\n{'='*50}")
        logger.info(f"TRAINING — {horizon}h forecast horizon")
        logger.info(f"{'='*50}")

        # Prepare target
        horizon_df, target_col = prepare_target(df.copy(), horizon)
        feature_cols = get_feature_columns(horizon_df)

        # Remove target-related columns from features
        feature_cols = [c for c in feature_cols if c != "aqi" and not c.startswith("aqi_target_")]

        X = horizon_df[feature_cols].values
        y = horizon_df[target_col].values

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )

        logger.info(f"Train: {len(X_train)} | Test: {len(X_test)} | Features: {len(feature_cols)}")

        models = build_models(config)
        horizon_results = {}
        best_rmse = float("inf")
        best_model_name = None

        for name, model in models.items():
            logger.info(f"Training {name} …")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = evaluate_model(y_test, y_pred)
            logger.info(
                f"  {name}: RMSE={metrics['rmse']:.4f} | "
                f"MAE={metrics['mae']:.4f} | R²={metrics['r2']:.4f}"
            )

            horizon_results[name] = {
                "model": model,
                "metrics": metrics,
                "features": feature_cols,
            }

            # Track best model by RMSE
            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_model_name = name

            # Save model to registry
            registry.save_model(
                model=model,
                model_name=name,
                horizon=horizon,
                metrics=metrics,
                feature_names=feature_cols,
            )

        logger.info(f"✓ Best model for {horizon}h: {best_model_name} (RMSE={best_rmse:.4f})")

        # Save best model marker
        registry.mark_best(best_model_name, horizon)

        all_results[horizon] = horizon_results

    logger.info("\nMODEL TRAINING — COMPLETE")
    return all_results
