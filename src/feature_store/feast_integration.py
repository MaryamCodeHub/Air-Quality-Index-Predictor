"""
Feast Integration Module for AQI Pipeline

Bridges the gap between:
- Data ingestion pipeline (src/ingestion)
- Feature engineering pipeline (src/processing)
- Feast feature store (src/feature_store)
- Model training (src/training)

Provides:
- Feature registration automation
- Materialization scheduling
- Feature retrieval for training/inference
- Quality monitoring

Author: AQI Team
Date: May 2026
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
from datetime import datetime

from src.feature_store.feast_store import AQIFeastStore
from src.feature_store.feature_registry import get_registry


logger = logging.getLogger(__name__)


class FeastIntegration:
    """
    Production-grade Feast integration for the AQI pipeline.
    
    Responsibilities:
    1. Initialize Feast store
    2. Register raw and engineered features
    3. Materialize features for training
    4. Retrieve features for inference
    5. Monitor feature quality
    """
    
    def __init__(self, repo_path: str = "feature_store"):
        """Initialize Feast integration."""
        self.store = AQIFeastStore(repo_path=repo_path)
        self.registry = get_registry()
        self.last_materialization = None
        
        logger.info("✅ Feast Integration initialized")
    
    def ingest_and_register(
        self,
        raw_df: pd.DataFrame,
        engineered_df: pd.DataFrame
    ) -> dict:
        """
        Complete ingestion: load data → register features → materialize.
        
        Args:
            raw_df: DataFrame with raw sensor data (aqi, pm25, pm10, temp, humidity, wind_speed)
            engineered_df: DataFrame with all engineered features (60+ columns)
            
        Returns:
            Dictionary with ingestion results
        """
        logger.info("🔄 Feast Ingestion Pipeline Starting...")
        
        results = {
            "step_1_raw_registration": self._register_raw(raw_df),
            "step_2_engineered_registration": self._register_engineered(engineered_df),
            "step_3_materialization": self._materialize(engineered_df),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("✅ Feast Ingestion Complete")
        
        return results
    
    def _register_raw(self, df: pd.DataFrame) -> dict:
        """Register raw features from sensor data."""
        logger.info("📝 Step 1: Registering raw features...")
        
        result = self.store.register_raw_features(df)
        
        logger.info(f"✅ Raw features registered: {result}")
        
        return result
    
    def _register_engineered(self, df: pd.DataFrame) -> dict:
        """Register engineered features (60+)."""
        logger.info("📝 Step 2: Registering engineered features...")
        
        result = self.store.register_engineered_features(df)
        
        logger.info(f"✅ Engineered features registered: {result}")
        
        return result
    
    def _materialize(self, df: pd.DataFrame) -> dict:
        """Materialize all features to offline store (Parquet)."""
        logger.info("💾 Step 3: Materializing features...")
        
        result = self.store.materialize_features(df)
        self.last_materialization = result
        
        logger.info(f"✅ Materialization complete: {result['num_records']} records")
        
        return result
    
    def get_features_for_training(
        self,
        lookback_days: int = 30
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Retrieve features for model training.
        
        Returns all materialized features from the last N days.
        
        Args:
            lookback_days: Number of days to retrieve
            
        Returns:
            Tuple[features_df, metadata]
        """
        logger.info(f"📂 Retrieving training features (last {lookback_days} days)...")
        
        features_df = self.store.get_offline_features(num_days=lookback_days)
        
        metadata = {
            "num_records": len(features_df),
            "num_features": len(features_df.columns) if len(features_df) > 0 else 0,
            "timestamp": datetime.now().isoformat(),
            "lookback_days": lookback_days
        }
        
        logger.info(f"✅ Retrieved {metadata['num_records']} training records")
        
        return features_df, metadata
    
    def get_features_for_inference(
        self
    ) -> Tuple[dict, dict]:
        """
        Retrieve latest features for inference.
        
        Returns latest feature values for making predictions.
        
        Returns:
            Tuple[features_dict, metadata]
        """
        logger.info("🔄 Retrieving inference features (latest)...")
        
        features = self.store.get_online_features()
        
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "num_locations": len(features),
            "feature_timestamp": features.get("islamabad", {}).get("timestamp", "unknown") if features else "unknown"
        }
        
        logger.info(f"✅ Retrieved inference features for {metadata['num_locations']} locations")
        
        return features, metadata
    
    def feature_quality_report(self) -> dict:
        """
        Generate feature quality report.
        
        Checks:
        - Missing values
        - Value ranges
        - Data type correctness
        - Feature freshness
        """
        logger.info("📊 Generating feature quality report...")
        
        features_df, _ = self.get_features_for_training(lookback_days=1)
        
        if features_df.empty:
            logger.warning("No features available for quality check")
            return {"status": "no_data"}
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_features": len(features_df.columns),
            "total_records": len(features_df),
            "quality_checks": {
                "missing_values": self._check_missing_values(features_df),
                "data_types": self._check_data_types(features_df),
                "value_ranges": self._check_value_ranges(features_df),
                "freshness": self._check_freshness(features_df)
            }
        }
        
        logger.info(f"✅ Quality report complete: {report['quality_checks']}")
        
        return report
    
    def _check_missing_values(self, df: pd.DataFrame) -> dict:
        """Check for missing values."""
        missing = df.isnull().sum().to_dict()
        return {
            "columns_with_missing": {k: v for k, v in missing.items() if v > 0},
            "max_missing_pct": (df.isnull().sum().max() / len(df) * 100) if len(df) > 0 else 0
        }
    
    def _check_data_types(self, df: pd.DataFrame) -> dict:
        """Check data types are correct."""
        return {
            "dtypes": df.dtypes.astype(str).to_dict(),
            "numeric_columns": df.select_dtypes(include=['float64', 'int64']).shape[1],
            "object_columns": df.select_dtypes(include=['object']).shape[1]
        }
    
    def _check_value_ranges(self, df: pd.DataFrame) -> dict:
        """Check values are within expected ranges."""
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        
        return {
            "aqi_range": [float(numeric_df['aqi'].min()), float(numeric_df['aqi'].max())] if 'aqi' in numeric_df else None,
            "temp_range": [float(numeric_df['temp'].min()), float(numeric_df['temp'].max())] if 'temp' in numeric_df else None,
            "humidity_range": [float(numeric_df['humidity'].min()), float(numeric_df['humidity'].max())] if 'humidity' in numeric_df else None,
        }
    
    def _check_freshness(self, df: pd.DataFrame) -> dict:
        """Check how recent the data is."""
        if 'timestamp' not in df.columns:
            return {"status": "no_timestamp"}
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        latest = df['timestamp'].max()
        age_hours = (datetime.now() - latest).total_seconds() / 3600 if pd.notna(latest) else None
        
        return {
            "latest_timestamp": str(latest) if pd.notna(latest) else None,
            "age_hours": age_hours,
            "is_fresh": age_hours < 24 if age_hours else False
        }
    
    def export_feature_manifest(self, output_path: str = "docs/reports/feature_manifest.json") -> dict:
        """
        Export feature manifest (registry + materialization status).
        
        Useful for:
        - Data governance
        - Compliance audits
        - ML ops tracking
        - Feature discovery
        """
        logger.info("📋 Exporting feature manifest...")
        
        registry_stats = self.registry.get_statistics()
        store_stats = self.store.get_feature_statistics()
        
        manifest = {
            "exported_at": datetime.now().isoformat(),
            "registry": registry_stats,
            "store": store_stats,
            "last_materialization": self.last_materialization
        }
        
        # Save to file
        import json
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"✅ Feature manifest exported to {output_path}")
        
        return manifest
    
    def health_check(self) -> dict:
        """Complete health check of Feast integration."""
        logger.info("🏥 Running Feast integration health check...")
        
        health = {
            "timestamp": datetime.now().isoformat(),
            "store_health": self.store.health_check(),
            "registry_stats": self.registry.get_statistics(),
            "latest_materialization": self.last_materialization is not None
        }
        
        overall_status = (
            health["store_health"]["status"] == "healthy" and
            health["registry_stats"]["num_registered_views"] > 0 and
            health["latest_materialization"]
        )
        
        health["overall_status"] = "healthy" if overall_status else "degraded"
        
        logger.info(f"✅ Health check complete: {health['overall_status']}")
        
        return health
