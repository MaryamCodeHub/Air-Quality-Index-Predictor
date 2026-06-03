"""
Feature Engineering — Single-City (Islamabad)
==============================================
Transforms cleaned data into ML-ready features.

Features:
  - Time: hour, day_of_week, month, season, is_weekend, cyclical encoding
  - Rolling: mean/std over 6h, 12h, 24h windows
  - Lag: AQI at t-1h, t-3h, t-6h, t-12h, t-24h
  - Rate of change: AQI diff + pct_change
  - Pollutant ratios: pm25/pm10, o3/no2
"""

import os
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("feature_engineering.feature_engineer")


class FeatureEngineer:
    """Generates ML features from cleaned Islamabad AQI + weather data."""

    def __init__(self, config: Dict[str, Any]):
        feat = config.get("features", {})
        self.rolling_windows = feat.get("rolling_windows", [6, 12, 24])
        self.lag_hours = feat.get("lag_hours", [1, 3, 6, 12, 24])

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            logger.warning("Empty DataFrame — skipping feature engineering")
            return df

        logger.info(f"Feature engineering START — {len(df)} rows")
        df = self._time_features(df)
        df = self._rolling_features(df)
        df = self._lag_features(df)
        df = self._rate_of_change(df)
        df = self._pollutant_ratios(df)
        logger.info(f"Feature engineering COMPLETE — {len(df.columns)} columns")
        return df

    def _time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" not in df.columns:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["day_of_month"] = df["timestamp"].dt.day
        df["month"] = df["timestamp"].dt.month
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["season"] = df["month"].map(
            {12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2,
             6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4}
        )
        # Cyclical encoding
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        return df

    def _rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        targets = [c for c in ["aqi", "pm25", "pm10"] if c in df.columns]
        for col in targets:
            for w in self.rolling_windows:
                df[f"{col}_roll_mean_{w}h"] = df[col].rolling(w, min_periods=1).mean()
                df[f"{col}_roll_std_{w}h"] = df[col].rolling(w, min_periods=1).std()
        return df

    def _lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "aqi" not in df.columns:
            return df
        for lag in self.lag_hours:
            df[f"aqi_lag_{lag}h"] = df["aqi"].shift(lag)
        return df

    def _rate_of_change(self, df: pd.DataFrame) -> pd.DataFrame:
        if "aqi" not in df.columns:
            return df
        df["aqi_change"] = df["aqi"].diff()
        df["aqi_pct_change"] = df["aqi"].pct_change()
        return df

    def _pollutant_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        if "pm25" in df.columns and "pm10" in df.columns:
            df["pm25_pm10_ratio"] = df["pm25"] / df["pm10"].replace(0, np.nan)
        if "o3" in df.columns and "no2" in df.columns:
            df["o3_no2_ratio"] = df["o3"] / df["no2"].replace(0, np.nan)
        return df


def run_feature_pipeline(config: Dict[str, Any], clean_df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features → save to processed Parquet."""
    fe = FeatureEngineer(config)
    featured = fe.engineer(clean_df)

    if featured.empty:
        return featured

    # Ensure the city entity column is added
    featured["city"] = "islamabad"

    # Ensure timestamp column is cast to datetime for Feast
    if "timestamp" in featured.columns:
        featured["timestamp"] = pd.to_datetime(featured["timestamp"])

    # Fill NaN carefully: numeric columns with 0, object/string columns with empty string
    numeric_cols = featured.select_dtypes(include=[np.number]).columns
    object_cols = featured.select_dtypes(include=['object']).columns
    
    for col in numeric_cols:
        featured[col] = featured[col].fillna(0)
    for col in object_cols:
        featured[col] = featured[col].fillna('')
    
    path = os.path.join(config["paths"]["processed_data"], "processed_aqi_data.parquet")
    featured.to_parquet(path, index=False)
    logger.info(f"Processed data saved → {path} ({len(featured)} rows)")
    return featured
