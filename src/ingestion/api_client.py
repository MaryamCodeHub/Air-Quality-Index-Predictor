"""
API Client Module — Single-City (Islamabad)
============================================
Production-grade API clients orchestrator for AQICN and Open-Meteo.
Wraps resilience patterns and exposes standard interfaces.
"""

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import requests

from src.utils.helpers import load_config, ensure_directories
from src.utils.logger import setup_logger

# Import decoupled, resilient components
from src.ingestion.base_client import BaseAPIClient, CircuitBreaker
from src.ingestion.schema_mapper import SchemaMapper
from src.ingestion.api_clients.aqicn import AQICNClient
from src.ingestion.api_clients.openweather import OpenMeteoClient

logger = setup_logger("ingestion.api_client")


# ============================================================
# Unified Ingestion Pipeline — Islamabad Only
# ============================================================

class DataIngestionPipeline:
    """
    Orchestrates data collection for Islamabad from both APIs.

    Graceful degradation: if one API fails, the pipeline still
    produces a partial record from the other.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        ensure_directories(config)

        self.aqicn_client: Optional[AQICNClient] = None
        self.weather_client: Optional[OpenMeteoClient] = None

        try:
            self.aqicn_client = AQICNClient(config)
        except Exception as exc:
            logger.error(f"AQICN client init failed: {exc}")

        try:
            self.weather_client = OpenMeteoClient(config)
        except Exception as exc:
            logger.error(f"OpenMeteo client init failed: {exc}")

        if self.aqicn_client is None and self.weather_client is None:
            raise ValueError("Both API clients failed. Check config/config.yaml")

    def run(self) -> pd.DataFrame:
        """
        Fetch AQI + weather for Islamabad → merge → append to raw Parquet.

        Returns:
            DataFrame of all raw records (including historical appends)
        """
        logger.info("=" * 60)
        logger.info("DATA INGESTION — Islamabad")
        logger.info("=" * 60)

        record: Dict[str, Any] = {}

        # Fetch AQI
        if self.aqicn_client:
            aqi_data = self.aqicn_client.fetch_aqi()
            if aqi_data:
                record.update(aqi_data)

        # Fetch weather
        if self.weather_client:
            weather_data = self.weather_client.fetch_weather()
            if weather_data:
                record.update(weather_data)

        if not record:
            logger.error("No data from either API. Pipeline aborted.")
            return pd.DataFrame()

        # Ensure timestamp exists
        if "timestamp" not in record:
            record["timestamp"] = datetime.now().isoformat()

        record["ingestion_timestamp"] = datetime.now().isoformat()

        # Convert single record to DataFrame
        new_df = pd.DataFrame([record])

        # Append to existing raw Parquet
        raw_path = os.path.join(self.config["paths"]["raw_data"], "raw_aqi_data.parquet")
        if os.path.exists(raw_path):
            existing = pd.read_parquet(raw_path)
            merged = pd.concat([existing, new_df], ignore_index=True)
            logger.info(f"Appended — total {len(merged)} raw records")
        else:
            merged = new_df
            logger.info("Created new raw data file — 1 record")

        merged.to_parquet(raw_path, index=False)
        logger.info(f"Raw data saved → {raw_path}")
        logger.info("DATA INGESTION — COMPLETE")
        return merged
