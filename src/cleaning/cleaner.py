"""
Enterprise Data Cleaning Pipeline
==================================
Cleans raw ingested data for feature engineering.
No schemas, column mappings, or outlier treatment methodologies are hardcoded.
Everything is resolved dynamically at runtime using config.yaml configurations,
with dynamic auto-inference fallbacks to ensure the pipeline never crashes.
"""

import os
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from src.utils.logger import setup_logger
from src.ingestion.schema_mapper import SchemaMapper
from src.cleaning.rules_engine import RulesEngine

logger = setup_logger("cleaning.cleaner")


class DataCleaner:
    """Enterprise-grade, fully config-driven schema-flexible Cleaner."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config if config is not None else {}
        self.mapper = SchemaMapper(config)
        self.rules_engine = RulesEngine(config)
        
        # Trigger runtime configuration validation layer
        self.validate_config()

    def validate_config(self) -> Dict[str, Any]:
        """
        Robust runtime validation of config.yaml structure.
        Reports missing, invalid, or suboptimal fields.
        Does NOT stop execution unless the configuration is completely unusable.
        """
        report = {"status": "OK", "warnings": [], "errors": []}
        
        if not self.config:
            report["status"] = "CRITICAL"
            report["errors"].append("Configuration dictionary is completely empty or None.")
            logger.error("[Config Validation] Configuration is completely empty!")
            return report

        # Check required structural paths
        required_paths = ["paths", "api", "city"]
        for path in required_paths:
            if path not in self.config:
                report["warnings"].append(f"Missing recommended structural section: '{path}' in config.")
                
        # Validate schema mapping
        schema_mapping = self.config.get("schema_mapping", {})
        if not schema_mapping:
            report["warnings"].append("No 'schema_mapping' defined in config.yaml. Dynamic columns won't be mapped.")
        else:
            if "timestamp" not in schema_mapping:
                report["warnings"].append("'schema_mapping' does not define a 'timestamp' mapping list.")

        # Validate cleaning rules
        cleaning_rules = self.config.get("cleaning_rules", {})
        if not cleaning_rules:
            report["warnings"].append("No 'cleaning_rules' defined in config.yaml. Default strategies will be applied.")
        else:
            for field, rules in cleaning_rules.items():
                if "outlier_method" not in rules:
                    report["warnings"].append(f"Field '{field}' cleaning rules miss 'outlier_method'. Fallback to 'clip'.")
                else:
                    method = rules["outlier_method"].lower()
                    if method not in ["iqr", "zscore", "clip"]:
                        report["warnings"].append(f"Invalid outlier strategy '{method}' for '{field}'. Fallback to 'clip'.")
                        
        if report["errors"]:
            report["status"] = "ERROR"
        elif report["warnings"]:
            report["status"] = "WARNING"
            
        # Log validation report
        logger.info(f"Runtime Configuration Validation: STATUS = {report['status']}")
        for w in report["warnings"]:
            logger.warning(f"[Config Validation Warning] {w}")
        for e in report["errors"]:
            logger.error(f"[Config Validation Error] {e}")
            
        return report

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the dynamic, resilient cleaning pipeline."""
        if df.empty:
            logger.warning("Empty DataFrame — nothing to clean")
            return df

        logger.info(f"Cleaning START — {len(df)} rows, {len(df.columns)} cols")
        n_before = len(df)

        # Step 1: Normalize column schemas dynamically
        df = self.mapper.map_dataframe(df)

        # Step 2: Parse and resolve dynamic timestamp
        df = self._parse_timestamps(df)

        # Step 3: Coerce numeric datatypes
        df = self._enforce_dtypes(df)

        # Step 4: Remove duplicate records
        df = self._remove_duplicates(df)

        # Step 5: Validate and clamp ranges using dynamic min/max bounds
        df = self._validate_ranges(df)

        # Step 6: Handle missing observations
        df = self._handle_missing(df)

        # Step 7: Apply pluggable strategy outlier handlers
        df = self._cap_outliers(df)

        logger.info(f"Cleaning COMPLETE — {len(df)} rows (dropped {n_before - len(df)})")
        return df

    def _detect_timestamp_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Resiliently detects a timestamp-like column inside the DataFrame.
        Looks at config.yaml specifications first, then falls back to auto-detecting.
        """
        config_keys = []
        if self.config:
            config_keys.extend(self.config.get("schema_mapping", {}).get("timestamp", []))
            if "time_key" in self.config:
                config_keys.append(self.config["time_key"])
            if "datetime_key" in self.config:
                config_keys.append(self.config["datetime_key"])

        # Check mapped options
        for key in config_keys:
            if key in df.columns:
                return key

        # Fallback auto-detection of standard names
        fallback_names = ["timestamp", "time", "date", "datetime", "dt", "ts", "ingestion_timestamp"]
        for name in fallback_names:
            for col in df.columns:
                if col.lower() == name.lower():
                    return col

        # Scan columns and check for datetime dtypes
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
                
        return None

    def _parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse, localize and sort by resolved timestamp column with full system falls."""
        detected_col = self._detect_timestamp_column(df)
        
        if detected_col:
            logger.info(f"Timestamp column resolved dynamically: '{detected_col}'")
            df = df.rename(columns={detected_col: "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)
            
            if df["timestamp"].isna().all():
                logger.warning("All parsed timestamps were invalid/NaT. Injecting current system timestamp as fallback.")
                df["timestamp"] = pd.Timestamp.now()
            else:
                bad = df["timestamp"].isna().sum()
                if bad > 0:
                    logger.warning(f"Dropped {bad} rows with bad timestamps")
                    df = df.dropna(subset=["timestamp"])
                df = df.sort_values("timestamp").reset_index(drop=True)
        else:
            logger.warning("No timestamp column detected from config or auto-discovery! Injecting system time as 'timestamp'.")
            df["timestamp"] = pd.to_datetime(pd.Timestamp.now())
            
        return df

    def _enforce_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enforces numeric types on configured and inferred numeric columns."""
        numeric_cols = self.rules_engine.get_numeric_columns(df)
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        if "timestamp" in df.columns:
            df = df.drop_duplicates(subset=["timestamp"], keep="last")
        removed = before - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate timestamps")
        return df

    def _validate_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resiliently validates features within standard bounds without crashing on missing columns."""
        numeric_cols = self.rules_engine.get_numeric_columns(df)
        for col in numeric_cols:
            if col in df.columns:
                lo, hi = self.rules_engine.get_bounds(col)
                mask = (df[col] < lo) | (df[col] > hi)
                n = mask.sum()
                if n > 0:
                    logger.warning(f"Range violation: '{col}': {n} values outside [{lo}, {hi}] → NaN")
                    df.loc[mask, col] = np.nan
            else:
                logger.debug(f"Configured numeric column '{col}' missing from DataFrame during range validation. Skipping.")
        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = self.rules_engine.get_numeric_columns(df)
        cols = [c for c in numeric_cols if c in df.columns]
        if not cols:
            return df
        before = df[cols].isna().sum().sum()
        df[cols] = df[cols].ffill().interpolate(method="linear").bfill()
        after = df[cols].isna().sum().sum()
        logger.info(f"Missing values handled: {before} → {after} (filled {before - after})")
        return df

    def _cap_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cap outliers using resolved outlier handlers per feature with fallback mapping."""
        numeric_cols = self.rules_engine.get_numeric_columns(df)
        cols = [c for c in numeric_cols if c in df.columns]
        total = 0
        for col in cols:
            handler = self.rules_engine.get_handler(col)
            rules = self.rules_engine.get_rules(col)
            
            # Apply dynamic handler strategy
            before = df[col].copy()
            df[col] = handler.handle(df[col], **rules)
            
            # Count changes
            changes = (before != df[col]).sum()
            total += changes
            
        if total > 0:
            logger.info(f"Dynamic Outliers: Capped {total} outlier values across features")
        return df


def run_cleaning_pipeline(config: Dict[str, Any]) -> pd.DataFrame:
    """Load raw Parquet → clean → return DataFrame."""
    raw_path = os.path.join(config["paths"]["raw_data"], "raw_aqi_data.parquet")
    if not os.path.exists(raw_path):
        logger.error(f"Raw data not found at {raw_path}. Run ingestion first.")
        return pd.DataFrame()

    raw_df = pd.read_parquet(raw_path)
    logger.info(f"Loaded {len(raw_df)} raw rows from {raw_path}")
    cleaner = DataCleaner(config)
    return cleaner.clean(raw_df)
