"""
Hopsworks Feature Store Integration
====================================
Cloud-based feature store for production ML pipelines.

Usage:
    connector = HopsworksConnector(config)
    connector.push_features(df, feature_group_name="aqi_features_24h")
    features_df = connector.get_features(feature_names, entity_ids)
"""

import os
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("feature_store.hopsworks")


class HopsworksConnector:
    """
    Manages connection to Hopsworks feature store.
    
    Handles:
    - Project connection
    - Feature group creation/update
    - Feature ingestion
    - Feature retrieval for training
    
    API Key: HOPSWORKS_API_KEY from .env
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Hopsworks connection.
        
        Args:
            config: Configuration dict with hopsworks section:
              hopsworks:
                project_name: "aqi_forecasting"
                api_key: "ENV"  (loaded from HOPSWORKS_API_KEY)
                host: "https://us-east-1.app.hopsworks.ai"
        """
        self.config = config
        self.project = None
        self.fs = None
        self.api_key = os.getenv("HOPSWORKS_API_KEY")
        
        # Get Hopsworks config
        hw_config = config.get("hopsworks", {})
        self.project_name = hw_config.get("project_name", "aqi_forecasting")
        self.host = hw_config.get("host", "https://us-east-1.app.hopsworks.ai")
        
        if not self.api_key:
            logger.warning(
                "HOPSWORKS_API_KEY not found in .env. "
                "Feature store integration disabled. Features will use Parquet fallback."
            )
            return
        
        self._connect()

    def _connect(self):
        """Connect to Hopsworks project and feature store."""
        try:
            import hopsworks
            
            logger.info(f"Connecting to Hopsworks project: {self.project_name}")
            
            project = hopsworks.login(
                host=self.host,
                api_key_value=self.api_key,
                project=self.project_name
            )
            
            self.project = project
            self.fs = project.get_feature_store()
            
            logger.info("✓ Connected to Hopsworks feature store")
            
        except ImportError:
            logger.warning(
                "hopsworks not installed. Install with: pip install hopsworks"
            )
        except Exception as e:
            logger.warning(f"Failed to connect to Hopsworks: {e}. Using Parquet fallback.")

    def push_features(
        self,
        df: pd.DataFrame,
        feature_group_name: str,
        version: int = 1,
        primary_key: Optional[List[str]] = None,
        event_time: str = "timestamp",
    ) -> bool:
        """
        Push features to Hopsworks feature store.
        
        Args:
            df: DataFrame with features
            feature_group_name: Name of feature group (e.g., "aqi_features_24h")
            version: Feature group version
            primary_key: List of primary key columns (default: ["city", "timestamp"])
            event_time: Event time column name
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.fs:
            logger.warning("Hopsworks not connected. Skipping feature push.")
            return False
        
        try:
            logger.info(
                f"Pushing {len(df)} rows to feature group: {feature_group_name}"
            )
            
            # Set defaults
            if primary_key is None:
                primary_key = ["city", "timestamp"] if "city" in df.columns else ["timestamp"]
            
            # Get or create feature group
            fg = self.fs.get_or_create_feature_group(
                name=feature_group_name,
                version=version,
                primary_key=primary_key,
                event_time=event_time,
                online_enabled=True,  # Enable online store for real-time predictions
                stream=False,
                description=f"AQI forecasting features for {feature_group_name}",
            )
            
            # Insert features
            fg.insert(df, write_options={"start_offline_materialization": True})
            
            logger.info(
                f"✓ {len(df)} features inserted into {feature_group_name} v{version}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to push features: {e}")
            return False

    def get_features(
        self,
        feature_names: List[str],
        entity_df: Optional[pd.DataFrame] = None,
        feature_group_name: str = "aqi_features_24h",
        version: int = 1,
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve features from Hopsworks for training.
        
        Args:
            feature_names: List of feature column names
            entity_df: DataFrame with entity keys (city, timestamp)
            feature_group_name: Name of feature group to retrieve from
            version: Feature group version
        
        Returns:
            DataFrame with requested features or None if failed
        """
        if not self.fs:
            logger.warning("Hopsworks not connected. Using Parquet fallback.")
            return None
        
        try:
            logger.info(
                f"Retrieving {len(feature_names)} features from {feature_group_name}"
            )
            
            # Get feature group
            fg = self.fs.get_feature_group(
                name=feature_group_name,
                version=version
            )
            
            # Query features
            if entity_df is not None:
                # Retrieve features for specific entities
                query = fg.select(feature_names)
                features_df = query.read()
                logger.info(f"✓ Retrieved {len(features_df)} rows from feature store")
                return features_df
            else:
                # Retrieve all features
                query = fg.select(feature_names)
                features_df = query.read()
                logger.info(f"✓ Retrieved {len(features_df)} rows from feature store")
                return features_df
            
        except Exception as e:
            logger.warning(f"Failed to retrieve features: {e}. Using Parquet fallback.")
            return None

    def get_online_features(
        self,
        feature_names: List[str],
        entity_values: Dict[str, Any],
        feature_group_name: str = "aqi_features_24h",
        version: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve features from online store for real-time predictions.
        
        Args:
            feature_names: List of feature names to retrieve
            entity_values: Dict with entity keys (e.g., {"city": "islamabad", "timestamp": "2026-05-24T15:00"})
            feature_group_name: Name of feature group
            version: Feature group version
        
        Returns:
            Dict with feature values or None if failed
        """
        if not self.fs:
            logger.warning("Hopsworks not connected. Cannot retrieve online features.")
            return None
        
        try:
            fg = self.fs.get_feature_group(
                name=feature_group_name,
                version=version
            )
            
            # Retrieve from online store
            features = fg.get_feature_view().get_feature_view(
                feature_names,
                entity_values
            )
            
            logger.info(f"✓ Retrieved online features for {entity_values}")
            return features
            
        except Exception as e:
            logger.debug(f"Online feature retrieval not available: {e}")
            return None

    def list_feature_groups(self) -> List[str]:
        """List all feature groups in project."""
        if not self.fs:
            return []
        
        try:
            fgs = self.fs.list_feature_groups()
            names = [fg.name for fg in fgs]
            logger.info(f"Found {len(names)} feature groups: {names}")
            return names
        except Exception as e:
            logger.warning(f"Failed to list feature groups: {e}")
            return []

    def get_feature_store_status(self) -> Dict[str, Any]:
        """Get current status of feature store connection."""
        return {
            "connected": self.fs is not None,
            "project_name": self.project_name if self.project else None,
            "has_api_key": bool(self.api_key),
            "host": self.host,
        }
