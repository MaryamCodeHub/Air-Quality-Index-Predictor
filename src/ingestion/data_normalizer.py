"""
Data Normalization Layer — API-Agnostic and Safe
==================================================
Accepts multiple API payload inputs (AQICN, Open-Meteo, etc.) and
translates them dynamically to standardised internal formats.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd

from src.utils.logger import setup_logger
from src.ingestion.schema_mapper import SchemaMapper

logger = setup_logger("ingestion.data_normalizer")


class DataNormalizer:
    """
    Unified Data Normalization Layer.
    Translates various API structures into standardized internal schemas,
    resolving schema conflicts and enforcing fail-safe fallback designs.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config if config is not None else {}
        self.mapper = SchemaMapper(config)

    def normalize_aqicn(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize raw AQICN JSON response using dynamic SchemaMapper."""
        if not raw_data:
            return {}
            
        normalized = {}
        
        # Standard columns from config
        cols = list(self.config.get("schema_mapping", {}).keys())
        for col in cols:
            val = self.mapper.resolve_field(raw_data, col)
            normalized[col] = val

        # Handle fallback / specific validation
        # Resolve timestamp dynamically
        ts = self.resolve_timestamp(normalized, raw_data)
        normalized["timestamp"] = ts
        
        # Standardize AQI value
        aqi_val = normalized.get("aqi")
        if aqi_val is None or aqi_val == "-":
            normalized["aqi"] = None
        else:
            try:
                normalized["aqi"] = int(float(aqi_val))
            except (ValueError, TypeError):
                normalized["aqi"] = None

        if not normalized.get("dominant_pollutant"):
            normalized["dominant_pollutant"] = "unknown"
            
        if not normalized.get("station_name"):
            normalized["station_name"] = self.config.get("city", {}).get("name", "Islamabad")

        return normalized

    def normalize_open_meteo(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize raw Open-Meteo JSON response using dynamic SchemaMapper."""
        if not raw_data:
            return {}
            
        normalized = {}
        
        # Parse hourly arrays
        time_path = self.mapper.resolve_field(raw_data, "time")
        if not time_path and "hourly" in raw_data:
            time_path = raw_data.get("hourly", {}).get("time", [])

        if not time_path:
            logger.warning("[OpenMeteo] No time path found in hourly response!")
            return {}
            
        # Find index for closest hourly reading
        now = datetime.now()
        best_idx = -1
        for i, t in enumerate(time_path):
            try:
                ts = datetime.fromisoformat(t)
                if ts <= now:
                    best_idx = i
            except (ValueError, TypeError):
                continue
                
        # Resolve columns
        cols = list(self.config.get("schema_mapping", {}).keys())
        for col in cols:
            if col in ["timestamp", "time"]:
                continue
            val = self.mapper.resolve_field(raw_data, col)
            if isinstance(val, list) and len(val) > 0:
                if abs(best_idx) < len(val):
                    normalized[col] = val[best_idx]
                else:
                    normalized[col] = None
            else:
                normalized[col] = None

        return normalized

    def resolve_timestamp(self, normalized: Dict[str, Any], raw_data: Dict[str, Any]) -> str:
        """
        Dynamic Timestamp Resolver:
        Finds and validates timestamp from custom mappings with fallback priority order.
        Never crashes; falls back to current system time if all options are exhausted.
        """
        ts_val = normalized.get("timestamp")
        
        # Option 1: Configured keys or time_key / datetime_key direct config paths
        config_keys = []
        if self.config:
            config_keys.extend(self.config.get("schema_mapping", {}).get("timestamp", []))
            if "time_key" in self.config:
                config_keys.append(self.config["time_key"])
            if "datetime_key" in self.config:
                config_keys.append(self.config["datetime_key"])

        for key in config_keys:
            val = self.mapper._get_nested(raw_data, key)
            if isinstance(val, str):
                ts_val = val
                break
                
        # Option 2: Fallback priority check in raw keys
        if not ts_val:
            fallback_keys = ["timestamp", "time", "dt", "datetime", "data.time.iso", "ts"]
            for key in fallback_keys:
                val = self.mapper._get_nested(raw_data, key)
                if isinstance(val, str):
                    ts_val = val
                    break

        if not ts_val:
            ts_val = datetime.now().isoformat()
            
        # Validate format or parse
        try:
            parsed = pd.to_datetime(ts_val)
            if parsed.tz is not None:
                parsed = parsed.tz_localize(None)
            
            # Check if timestamp is older than 24 hours
            now = datetime.now()
            if (now - parsed).total_seconds() > 86400:
                logger.warning(f"Detected stale API timestamp '{ts_val}' (older than 24h). Overriding with current system time.")
                parsed = now
                
            return parsed.isoformat()
        except Exception as e:
            logger.warning(f"Error parsing timestamp '{ts_val}', falling back to current time. Reason: {e}")
            return datetime.now().isoformat()
