"""
Model Training Pipeline — Islamabad AQI Forecasting
=====================================================
Trains Ridge Regression, Random Forest, and XGBoost models
for 24h, 48h, and 72h AQI forecasting horizons.

Data flow:
  1. PRIMARY: Feast feature store (get_features_for_training)
  2. FALLBACK: Parquet if Feast unavailable
  3. Training: Same models (Ridge, RF, XGBoost) on features
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
from src.feature_store.feast_integration import FeastIntegration
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
    
    Data source priority:
      1. Try Feast feature store (PRIMARY)
      2. Fall back to Parquet if Feast unavailable (SECONDARY)

    Returns:
        results dict: {horizon: {model_name: {metrics, model, features}}}
    """
    df = None
    data_source = None
    
    # ===== STEP 1: Try Feast (PRIMARY) =====
    try:
        logger.info("▶ Attempting to load training features from Feast (primary)...")
        feast = FeastIntegration(repo_path="feature_store")
        df, metadata = feast.get_features_for_training(lookback_days=30)
        
        if df is not None and len(df) > 0:
            data_source = "Feast"
            logger.info(
                f"✅ Feast successfully loaded {len(df)} training records "
                f"({len(df.columns)} features)"
            )
        else:
            logger.warning("Feast returned empty result, attempting Parquet fallback...")
            data_source = None
            
    except Exception as e:
        logger.warning(f"⚠ Feast retrieval failed: {type(e).__name__}: {e}")
        logger.info("ℹ Attempting Parquet fallback...")
        data_source = None
    
    # ===== STEP 2: Fallback to Parquet =====
    if df is None or len(df) == 0:
        try:
            logger.info("▶ Loading training features from Parquet (fallback)...")
            proc_path = os.path.join(config["paths"]["processed_data"], "processed_aqi_data.parquet")
            
            if not os.path.exists(proc_path):
                logger.error(f"Processed data not found: {proc_path}. Run ingestion first.")
                return {}
            
            df = pd.read_parquet(proc_path)
            data_source = "Parquet"
            logger.info(f"✅ Parquet fallback successful: {len(df)} records ({len(df.columns)} features)")
            
        except Exception as e:
            logger.error(f"❌ Parquet fallback also failed: {type(e).__name__}: {e}")
            return {}
    
    # ===== STEP 3: Validate data =====
    min_samples = config["training"].get("min_samples_required", 50)
    if len(df) < min_samples:
        logger.error(
            f"Insufficient data: {len(df)} rows (need ≥{min_samples}). "
            f"Run 'python run.py ingest' more times to accumulate data."
        )
        return {}

    logger.info(f"✅ Training data ready: {len(df)} rows from {data_source}")

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

    logger.info(f"\n✅ MODEL TRAINING COMPLETE (data from {data_source})")
    return all_results
