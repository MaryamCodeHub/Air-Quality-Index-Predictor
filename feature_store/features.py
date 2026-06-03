"""
Feast Feature Definitions — Islamabad AQI
==========================================
Defines the entity, data source, and feature views for the Feast feature store.

Entity: islamabad (single-city — no entity column needed, we use a constant)
Source: Parquet files from the processed data pipeline
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int32, Int64, String
from feast.value_type import ValueType


# ============================================================
# Entity: Single city — Islamabad
# ============================================================
# Even for a single-city system, Feast requires an entity.
# We use a constant "islamabad" entity key for all rows.

city_entity = Entity(
    name="city",
    description="City entity — fixed to Islamabad for this deployment",
    value_type=ValueType.STRING,
)


# ============================================================
# Data Source: Real processed Parquet file
# ============================================================
# Points to actual processed data from ingestion pipeline

aqi_source = FileSource(
    name="aqi_islamabad_source",
    path="../data/processed/processed_aqi_data.parquet",
    timestamp_field="timestamp",
)


# ============================================================
# Feature View: AQI + Weather + Engineered Features (All 53 columns)
# ============================================================
# Maps to all columns in processed_aqi_data.parquet

aqi_features = FeatureView(
    name="aqi_islamabad_features",
    entities=[city_entity],
    ttl=timedelta(hours=72),
    schema=[
        # Raw AQI readings
        Field(name="aqi", dtype=Float64),
        Field(name="dominant_pollutant", dtype=String),
        Field(name="pm25", dtype=Float64),
        Field(name="pm10", dtype=Float64),
        Field(name="o3", dtype=Float64),
        Field(name="no2", dtype=Float64),
        Field(name="so2", dtype=Float64),
        Field(name="co", dtype=Float64),
        # Station/time info
        Field(name="station_name", dtype=String),
        Field(name="time", dtype=String),
        # Weather features
        Field(name="temperature", dtype=Float64),
        Field(name="humidity", dtype=Int64),
        Field(name="pressure", dtype=Float64),
        Field(name="wind_speed", dtype=Float64),
        # Ingestion tracking
        Field(name="ingestion_timestamp", dtype=String),
        # Time features
        Field(name="hour", dtype=Int32),
        Field(name="day_of_week", dtype=Int32),
        Field(name="day_of_month", dtype=Int32),
        Field(name="month", dtype=Int32),
        Field(name="is_weekend", dtype=Int64),
        Field(name="season", dtype=Int64),
        # Trigonometric time features
        Field(name="hour_sin", dtype=Float64),
        Field(name="hour_cos", dtype=Float64),
        Field(name="month_sin", dtype=Float64),
        Field(name="month_cos", dtype=Float64),
        # Rolling mean features (AQI)
        Field(name="aqi_roll_mean_6h", dtype=Float64),
        Field(name="aqi_roll_std_6h", dtype=Float64),
        Field(name="aqi_roll_mean_12h", dtype=Float64),
        Field(name="aqi_roll_std_12h", dtype=Float64),
        Field(name="aqi_roll_mean_24h", dtype=Float64),
        Field(name="aqi_roll_std_24h", dtype=Float64),
        # Rolling mean features (PM2.5)
        Field(name="pm25_roll_mean_6h", dtype=Float64),
        Field(name="pm25_roll_std_6h", dtype=Float64),
        Field(name="pm25_roll_mean_12h", dtype=Float64),
        Field(name="pm25_roll_std_12h", dtype=Float64),
        Field(name="pm25_roll_mean_24h", dtype=Float64),
        Field(name="pm25_roll_std_24h", dtype=Float64),
        # Rolling mean features (PM10)
        Field(name="pm10_roll_mean_6h", dtype=Float64),
        Field(name="pm10_roll_std_6h", dtype=Float64),
        Field(name="pm10_roll_mean_12h", dtype=Float64),
        Field(name="pm10_roll_std_12h", dtype=Float64),
        Field(name="pm10_roll_mean_24h", dtype=Float64),
        Field(name="pm10_roll_std_24h", dtype=Float64),
        # Lag features
        Field(name="aqi_lag_1h", dtype=Float64),
        Field(name="aqi_lag_3h", dtype=Float64),
        Field(name="aqi_lag_6h", dtype=Float64),
        Field(name="aqi_lag_12h", dtype=Float64),
        Field(name="aqi_lag_24h", dtype=Float64),
        # Change features
        Field(name="aqi_change", dtype=Float64),
        Field(name="aqi_pct_change", dtype=Float64),
        # Ratio features
        Field(name="pm25_pm10_ratio", dtype=Float64),
        Field(name="o3_no2_ratio", dtype=Float64),
    ],
    source=aqi_source,
    online=True,
)
