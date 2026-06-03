"""
Feast Feature Store Integration for AQI Forecasting
- Feature Registry: Defines all available features
- Feature Materialization: Stores computed features
- Feature Retrieval: Gets features for inference/training
- Online/Offline Split: Supports both online and offline serving

Author: AQI Team
Date: June 2026
"""

import os
import json
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from feast import (
    FeatureStore, 
    FeatureView, 
    Entity, 
    ValueType,
)

logger = logging.getLogger(__name__)


class AQIFeastStore:
    """
    Production-grade Feast feature store for AQI forecasting.
    
    Responsibilities:
    - Initialize Feast repository
    - Define feature views (raw + engineered)
    - Materialize features to offline/online stores
    - Retrieve features for training/inference
    - Track feature lineage and metadata
    """
    
    def __init__(self, repo_path: str = "feature_store"):
        """
        Initialize Feast Feature Store.
        
        Args:
            repo_path: Path to Feast repository (default: feature_store/)
        """
        self.repo_path = Path(repo_path)
        self.repo_path.mkdir(exist_ok=True)
        
        self.store = None
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "features_registered": [],
            "materializations": [],
            "version": "1.0.0"
        }
        
        self._initialize_store()
        logger.info(f"✅ Feast Feature Store initialized at {self.repo_path}")
    
    def _initialize_store(self):
        """Initialize Feast store with local SQLite/file backend."""
        try:
            self.store = FeatureStore(repo_path=str(self.repo_path))
            logger.info("✅ Standard Feast store initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed standard Feast initialization: {e}")
            
    def register_raw_features(self, df: pd.DataFrame) -> Dict:
        """
        Register raw AQI/weather features in Feast metadata registry.
        (Features themselves are statically configured in features.py).
        """
        logger.info("📝 Logging raw features schema info...")
        
        result = {
            "status": "registered",
            "entity": "city",
            "view": "aqi_raw_features",
            "features": 6,
            "timestamp": datetime.now().isoformat()
        }
        
        self.metadata["features_registered"].append(result)
        logger.info("✅ Loged raw features schema info")
        return result
    
    def register_engineered_features(self, df: pd.DataFrame) -> Dict:
        """
        Register engineered features in Feast metadata registry.
        (Features themselves are statically configured in features.py).
        """
        logger.info("📝 Logging engineered features schema info...")
        
        result = {
            "status": "registered",
            "view": "aqi_engineered_features",
            "features": 30,
            "timestamp": datetime.now().isoformat()
        }
        
        self.metadata["features_registered"].append(result)
        logger.info("✅ Loged engineered features schema info")
        return result
    
    def materialize_features(
        self, 
        df: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        Materialize features to both Feast online store (SQLite) and offline Parquet backup.
        
        Args:
            df: Feature DataFrame
            start_date: Materialization start (default: data start)
            end_date: Materialization end (default: now)
            
        Returns:
            Dictionary with materialization status
        """
        logger.info("💾 Materializing features to Feast offline and online stores...")
        
        # 1. Run 'feast apply' via subprocess to make sure definitions are applied
        try:
            logger.info("Running 'feast apply' to register features.py definitions...")
            res = subprocess.run(["feast", "apply"], cwd=str(self.repo_path), check=True, capture_output=True, text=True)
            logger.info(f"Feast apply output:\n{res.stdout}")
        except Exception as e:
            logger.error(f"Failed to run 'feast apply': {e}")
            
        # 2. Re-initialize Feast store to pick up any changes
        self._initialize_store()
        
        # 3. Call standard materialize_incremental to SQLite online store
        try:
            m_end = end_date if end_date else datetime.now()
            logger.info(f"Running store.materialize_incremental up to {m_end}...")
            self.store.materialize_incremental(end_date=m_end)
            logger.info("✅ Feast online store materialization completed successfully")
        except Exception as e:
            logger.error(f"❌ Failed standard Feast materialization: {e}")
        
        # 4. Save features to custom Parquet backup directory (for backward compatibility)
        materialization_dir = self.repo_path / "materializations" / datetime.now().strftime("%Y%m%d_%H%M%S")
        materialization_dir.mkdir(parents=True, exist_ok=True)
        
        feature_path = materialization_dir / "features.parquet"
        df.to_parquet(feature_path)
        
        # Save metadata
        metadata = {
            "materialization_id": datetime.now().isoformat(),
            "num_records": len(df),
            "num_features": len(df.columns),
            "start_date": start_date.isoformat() if start_date else df['timestamp'].min().isoformat() if 'timestamp' in df.columns and len(df) > 0 else datetime.now().isoformat(),
            "end_date": end_date.isoformat() if end_date else df['timestamp'].max().isoformat() if 'timestamp' in df.columns and len(df) > 0 else datetime.now().isoformat(),
            "feature_path": str(feature_path),
            "columns": df.columns.tolist()
        }
        
        metadata_path = materialization_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.metadata["materializations"].append(metadata)
        logger.info(f"✅ Parquet materialization backup saved to {feature_path}")
        
        return metadata
    
    def get_offline_features(
        self,
        locations: List[str] = ["islamabad"],
        num_days: int = 30
    ) -> pd.DataFrame:
        """
        Retrieve features for offline training using Feast's get_historical_features.
        
        Args:
            locations: List of location IDs (entities)
            num_days: Number of days to retrieve
            
        Returns:
            DataFrame with offline features
        """
        logger.info(f"📂 Retrieving offline features via Feast for last {num_days} days...")
        
        try:
            # Re-initialize to ensure up-to-date config
            self._initialize_store()
            
            # Load processed data to get event timestamps and keys
            proc_path = "data/processed/processed_aqi_data.parquet"
            if not os.path.exists(proc_path):
                # Fallback to current directory check
                proc_path = os.path.join("data", "processed", "processed_aqi_data.parquet")
                
            if not os.path.exists(proc_path):
                logger.warning(f"Processed data file not found at {proc_path}")
                return pd.DataFrame()
                
            df_proc = pd.read_parquet(proc_path)
            if df_proc.empty:
                logger.warning("Processed data is empty, returning empty DataFrame")
                return pd.DataFrame()
                
            # Filter by lookback window
            df_proc["timestamp"] = pd.to_datetime(df_proc["timestamp"])
            cutoff = datetime.now() - timedelta(days=num_days)
            df_filtered = df_proc[df_proc["timestamp"] >= cutoff].copy()
            
            if df_filtered.empty:
                logger.warning(f"No records found within the last {num_days} days")
                return pd.DataFrame()
                
            # Ensure 'city' column is present
            if "city" not in df_filtered.columns:
                df_filtered["city"] = "islamabad"
                
            # Build entity df for Feast historical feature retrieval
            entity_df = pd.DataFrame({
                "city": df_filtered["city"].astype(str),
                "timestamp": df_filtered["timestamp"]
            })
            
            # Fetch FeatureView schema to build features request list
            fv = self.store.get_feature_view("aqi_islamabad_features")
            features = [f"aqi_islamabad_features:{field.name}" for field in fv.schema if field.name not in fv.entities]
            
            logger.info(f"Calling store.get_historical_features for {len(entity_df)} entity rows...")
            retrieval = self.store.get_historical_features(
                entity_df=entity_df,
                features=features
            )
            features_df = retrieval.to_df()
            
            # Sort by timestamp
            if "timestamp" in features_df.columns:
                features_df = features_df.sort_values("timestamp").reset_index(drop=True)
                
            logger.info(f"✅ Successfully retrieved {len(features_df)} rows from Feast offline store")
            return features_df
            
        except Exception as e:
            logger.warning(f"⚠️ Feast get_historical_features failed: {e}. Falling back to legacy Parquet retrieval.")
            
            # Fallback 1: read from custom materializations
            materializations_dir = self.repo_path / "materializations"
            if materializations_dir.exists():
                dirs = sorted(materializations_dir.iterdir())
                if dirs:
                    latest = dirs[-1]
                    feature_path = latest / "features.parquet"
                    if feature_path.exists():
                        df = pd.read_parquet(feature_path)
                        logger.info(f"✅ Retrieved {len(df)} features from legacy materialization backup")
                        return df
                        
            # Fallback 2: read directly from processed parquet
            proc_path = "data/processed/processed_aqi_data.parquet"
            if os.path.exists(proc_path):
                df = pd.read_parquet(proc_path)
                logger.info(f"✅ Retrieved {len(df)} features from direct processed parquet fallback")
                return df
                
            return pd.DataFrame()
    
    def get_online_features(self, locations: List[str] = ["islamabad"]) -> Dict:
        """
        Retrieve latest features for online inference from SQLite online store.
        
        Args:
            locations: Location IDs for inference
            
        Returns:
            Dictionary with latest feature values: {location: {feature_name: value}}
        """
        logger.info("🔄 Querying Feast online store for latest features...")
        
        try:
            # Re-initialize store
            self._initialize_store()
            
            fv = self.store.get_feature_view("aqi_islamabad_features")
            features = [f"aqi_islamabad_features:{field.name}" for field in fv.schema if field.name not in fv.entities]
            
            entity_rows = [{"city": loc} for loc in locations]
            
            logger.info(f"Calling store.get_online_features for entities {entity_rows}...")
            response = self.store.get_online_features(
                features=features,
                entity_rows=entity_rows
            )
            
            response_dict = response.to_dict()
            
            # Format output structure to match: {location: {feature_name: value}}
            result = {}
            for i, loc in enumerate(locations):
                loc_features = {}
                for key, values in response_dict.items():
                    loc_features[key] = values[i]
                result[loc] = loc_features
                
            logger.info(f"✅ Online features retrieved successfully for {locations}")
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ Feast get_online_features failed: {e}. Falling back to offline dataset.")
            
            # Fallback to retrieving latest values from get_offline_features
            try:
                df = self.get_offline_features(locations, num_days=30)
                if not df.empty:
                    features = {}
                    for location in locations:
                        latest = df.iloc[-1].to_dict()
                        features[location] = latest
                    return features
            except Exception as ex:
                logger.error(f"Online fallback failed: {ex}")
                
            return {}
    
    def get_feature_statistics(self) -> Dict:
        """Get statistics about registered and materialized features."""
        stats = {
            "created_at": self.metadata["created_at"],
            "version": self.metadata["version"],
            "num_registered_views": len(self.metadata["features_registered"]),
            "num_materializations": len(self.metadata["materializations"]),
            "registered_features": self.metadata["features_registered"],
            "materializations": self.metadata["materializations"],
        }
        
        logger.info(f"📊 Feature Store Stats: {len(stats['registered_features'])} views, {len(stats['materializations'])} materializations")
        return stats
    
    def export_metadata(self, output_path: str = "feature_store_metadata.json"):
        """Export feature store metadata for audit & compliance."""
        with open(output_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"✅ Feature store metadata exported to {output_path}")
    
    def health_check(self) -> Dict:
        """Verify feature store health and connectivity."""
        try:
            materializations_dir = self.repo_path / "materializations"
            num_materializations = len(list(materializations_dir.glob("*"))) if materializations_dir.exists() else 0
            
            # Simple Feast health query
            self._initialize_store()
            fvs = self.store.list_feature_views()
            
            health = {
                "status": "healthy",
                "store_path": str(self.repo_path),
                "materializations": num_materializations,
                "registered_features": len(fvs),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("✅ Feature Store Health: OK")
            return health
        except Exception as e:
            logger.error(f"❌ Feature Store Health Check Failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
