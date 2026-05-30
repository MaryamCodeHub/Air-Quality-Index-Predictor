"""
Feast Feature Registry for AQI Forecasting
Declarative feature definitions with lineage, schemas, and metadata.

This module serves as the Single Source of Truth (SSOT) for all features,
enabling feature discoverability, governance, and ML ops.

Author: AQI Team
Date: May 2026
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
import json
from datetime import datetime


class FeatureCategory(Enum):
    """Feature categories for organization and governance."""
    RAW_DATA = "raw_data"
    LAG_FEATURES = "lag_features"
    ROLLING_STATS = "rolling_statistics"
    CYCLICAL = "cyclical"
    TEMPORAL = "temporal"
    METEOROLOGICAL = "meteorological"
    INTERACTION = "interaction"
    SEASONAL = "seasonal"


@dataclass
class FeatureDefinition:
    """
    Feature definition with metadata, lineage, and governance info.
    
    Attributes:
        name: Unique feature name
        description: Human-readable description
        dtype: Data type (float, int, bool, string)
        category: Feature category (from FeatureCategory enum)
        source: Data source (sensor, computed, external)
        units: Unit of measurement (if applicable)
        range: Expected value range [min, max]
        missing_pct_allowed: Max allowed missing data percentage
        created_at: Creation timestamp
        updated_at: Last update timestamp
        owner: Team/person responsible
        dependencies: List of upstream features
        tags: Searchable tags
    """
    name: str
    description: str
    dtype: str
    category: FeatureCategory
    source: str
    units: Optional[str] = None
    range: Optional[tuple] = None
    missing_pct_allowed: float = 10.0
    created_at: str = None
    updated_at: str = None
    owner: str = "aqi-team"
    dependencies: List[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = datetime.now().isoformat()
        if self.dependencies is None:
            self.dependencies = []
        if self.tags is None:
            self.tags = []
    
    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "dtype": self.dtype,
            "category": self.category.value,
            "source": self.source,
            "units": self.units,
            "range": self.range,
            "missing_pct_allowed": self.missing_pct_allowed,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "owner": self.owner,
            "dependencies": self.dependencies,
            "tags": self.tags,
        }


class AQIFeatureRegistry:
    """
    Centralized registry of all AQI forecasting features.
    
    Enables:
    - Feature discovery via semantic search
    - Dependency tracking (DAG)
    - Data quality enforcement
    - Governance and compliance
    - ML lineage tracking
    """
    
    def __init__(self):
        self.features: Dict[str, FeatureDefinition] = {}
        self._register_all_features()
    
    def _register_all_features(self):
        """Register all 60+ features used in AQI forecasting."""
        
        # ===== RAW SENSOR DATA =====
        self.register(FeatureDefinition(
            name="aqi",
            description="Air Quality Index (0-500+, higher = worse)",
            dtype="float",
            category=FeatureCategory.RAW_DATA,
            source="aqicn_api",
            units="AQI points",
            range=(0, 500),
            tags=["target", "primary", "air_quality"]
        ))
        
        self.register(FeatureDefinition(
            name="pm25",
            description="PM2.5 concentration (fine particles)",
            dtype="float",
            category=FeatureCategory.RAW_DATA,
            source="aqicn_api",
            units="μg/m³",
            range=(0, 500),
            tags=["pollutant", "health", "primary"]
        ))
        
        self.register(FeatureDefinition(
            name="pm10",
            description="PM10 concentration (coarse particles)",
            dtype="float",
            category=FeatureCategory.RAW_DATA,
            source="aqicn_api",
            units="μg/m³",
            range=(0, 500),
            tags=["pollutant", "health"]
        ))
        
        self.register(FeatureDefinition(
            name="temp",
            description="Temperature in Celsius",
            dtype="float",
            category=FeatureCategory.METEOROLOGICAL,
            source="open_meteo_api",
            units="°C",
            range=(-10, 55),
            tags=["weather", "meteorological"]
        ))
        
        self.register(FeatureDefinition(
            name="humidity",
            description="Relative humidity percentage",
            dtype="float",
            category=FeatureCategory.METEOROLOGICAL,
            source="open_meteo_api",
            units="%",
            range=(0, 100),
            tags=["weather", "meteorological"]
        ))
        
        self.register(FeatureDefinition(
            name="wind_speed",
            description="Wind speed at 10m height",
            dtype="float",
            category=FeatureCategory.METEOROLOGICAL,
            source="open_meteo_api",
            units="m/s",
            range=(0, 30),
            tags=["weather", "dispersion"]
        ))
        
        # ===== LAG FEATURES (AQI) =====
        for lag_hour in [1, 2, 6, 24, 48, 168]:
            self.register(FeatureDefinition(
                name=f"aqi_lag_{lag_hour}h",
                description=f"AQI value {lag_hour} hours ago",
                dtype="float",
                category=FeatureCategory.LAG_FEATURES,
                source="computed",
                dependencies=["aqi"],
                tags=["temporal", "autoregressive", f"lag_{lag_hour}h"]
            ))
        
        # ===== LAG FEATURES (PM2.5) =====
        for lag_hour in [1, 2, 6, 24]:
            self.register(FeatureDefinition(
                name=f"pm25_lag_{lag_hour}h",
                description=f"PM2.5 value {lag_hour} hours ago",
                dtype="float",
                category=FeatureCategory.LAG_FEATURES,
                source="computed",
                dependencies=["pm25"],
                tags=["temporal", "pollutant"]
            ))
        
        # ===== LAG FEATURES (PM10) =====
        for lag_hour in [1, 2, 6, 24]:
            self.register(FeatureDefinition(
                name=f"pm10_lag_{lag_hour}h",
                description=f"PM10 value {lag_hour} hours ago",
                dtype="float",
                category=FeatureCategory.LAG_FEATURES,
                source="computed",
                dependencies=["pm10"],
                tags=["temporal", "pollutant"]
            ))
        
        # ===== ROLLING STATISTICS =====
        for window in [3, 6, 24]:
            for stat in ["mean", "std", "min", "max"]:
                self.register(FeatureDefinition(
                    name=f"aqi_{window}h_{stat}",
                    description=f"AQI {stat} over last {window} hours",
                    dtype="float",
                    category=FeatureCategory.ROLLING_STATS,
                    source="computed",
                    dependencies=["aqi"],
                    tags=["rolling", "statistical", f"window_{window}h"]
                ))
        
        # ===== CYCLICAL FEATURES =====
        cyclical_features = [
            ("hour_sin", "Sine of hour (24h cycle)", [0, 1]),
            ("hour_cos", "Cosine of hour (24h cycle)", [0, 1]),
            ("day_sin", "Sine of day (7-day cycle)", [0, 1]),
            ("day_cos", "Cosine of day (7-day cycle)", [0, 1]),
            ("month_sin", "Sine of month (365-day cycle)", [0, 1]),
            ("month_cos", "Cosine of month (365-day cycle)", [0, 1]),
        ]
        
        for name, desc, range_ in cyclical_features:
            self.register(FeatureDefinition(
                name=name,
                description=desc,
                dtype="float",
                category=FeatureCategory.CYCLICAL,
                source="computed",
                range=tuple(range_),
                tags=["temporal", "cyclical", "periodic"]
            ))
        
        # ===== TEMPORAL BINARY FEATURES =====
        self.register(FeatureDefinition(
            name="is_weekend",
            description="Whether timestamp is on weekend (1=yes, 0=no)",
            dtype="int",
            category=FeatureCategory.TEMPORAL,
            source="computed",
            range=(0, 1),
            tags=["temporal", "categorical"]
        ))
        
        self.register(FeatureDefinition(
            name="is_rush_hour",
            description="Whether timestamp is during rush hours (7-9am, 5-7pm)",
            dtype="int",
            category=FeatureCategory.TEMPORAL,
            source="computed",
            range=(0, 1),
            tags=["temporal", "human_activity"]
        ))
        
        # ===== METEOROLOGICAL INTERACTION FEATURES =====
        meteorological_features = [
            ("temp_normalized", "Temperature normalized to [-1, 1]", [-1, 1]),
            ("humidity_normalized", "Humidity normalized to [-1, 1]", [-1, 1]),
            ("atmospheric_stability", "Buoyancy/stability indicator", [0, 1]),
            ("pm_ratio", "PM2.5 to PM10 ratio", [0, 1]),
        ]
        
        for name, desc, range_ in meteorological_features:
            self.register(FeatureDefinition(
                name=name,
                description=desc,
                dtype="float",
                category=FeatureCategory.METEOROLOGICAL,
                source="computed",
                range=tuple(range_),
                dependencies=["temp", "humidity"] if "temp" in name or "humidity" in name else ["pm25", "pm10"],
                tags=["derived", "meteorological"]
            ))
        
        # ===== INTERACTION FEATURES =====
        interaction_features = [
            ("dispersion_factor", "Wind speed × stability (higher = better dispersion)", [0, 10]),
            ("aqi_momentum", "AQI change rate (rate of increase/decrease)", [-50, 50]),
            ("temp_humidity_interaction", "Temperature-humidity interaction effect", [-1, 1]),
            ("pm25_accumulation", "PM2.5 accumulation indicator", [0, 1]),
            ("aqi_change_1h", "AQI change in last hour", [-100, 100]),
            ("aqi_change_6h", "AQI change in last 6 hours", [-200, 200]),
        ]
        
        for name, desc, range_ in interaction_features:
            self.register(FeatureDefinition(
                name=name,
                description=desc,
                dtype="float",
                category=FeatureCategory.INTERACTION,
                source="computed",
                range=tuple(range_),
                tags=["derived", "interaction", "nonlinear"]
            ))
        
        # ===== SEASONAL FEATURES =====
        seasonal_features = [
            ("aqi_trend", "Long-term AQI trend (30-day rolling)", [-100, 100]),
            ("aqi_seasonal", "Seasonal AQI component from decomposition", [-50, 50]),
        ]
        
        for name, desc, range_ in seasonal_features:
            self.register(FeatureDefinition(
                name=name,
                description=desc,
                dtype="float",
                category=FeatureCategory.SEASONAL,
                source="computed",
                range=tuple(range_),
                dependencies=["aqi"],
                tags=["time_series", "seasonal", "decomposition"]
            ))
    
    def register(self, feature: FeatureDefinition):
        """Register a feature in the registry."""
        self.features[feature.name] = feature
    
    def get_feature(self, name: str) -> Optional[FeatureDefinition]:
        """Get feature definition by name."""
        return self.features.get(name)
    
    def list_by_category(self, category: FeatureCategory) -> List[FeatureDefinition]:
        """Get all features in a category."""
        return [f for f in self.features.values() if f.category == category]
    
    def search_by_tag(self, tag: str) -> List[FeatureDefinition]:
        """Search features by tag."""
        return [f for f in self.features.values() if tag in f.tags]
    
    def get_feature_lineage(self, feature_name: str) -> Dict:
        """Get upstream dependencies for a feature (lineage)."""
        feature = self.get_feature(feature_name)
        if not feature:
            return {}
        
        lineage = {
            "feature": feature_name,
            "direct_dependencies": feature.dependencies,
            "category": feature.category.value,
            "source": feature.source,
        }
        
        return lineage
    
    def export_registry(self, output_path: str = "feature_registry.json"):
        """Export registry as JSON for documentation & governance."""
        registry_data = {
            "exported_at": datetime.now().isoformat(),
            "num_features": len(self.features),
            "features": [f.to_dict() for f in self.features.values()],
            "categories": {cat.value: len(self.list_by_category(cat)) for cat in FeatureCategory},
        }
        
        with open(output_path, 'w') as f:
            json.dump(registry_data, f, indent=2)
        
        print(f"✅ Feature registry exported to {output_path}")
        return registry_data
    
    def get_statistics(self) -> Dict:
        """Get registry statistics."""
        return {
            "total_features": len(self.features),
            "by_category": {cat.value: len(self.list_by_category(cat)) for cat in FeatureCategory},
            "by_source": self._group_by_source(),
        }
    
    def _group_by_source(self) -> Dict[str, int]:
        """Count features by source."""
        sources = {}
        for feature in self.features.values():
            sources[feature.source] = sources.get(feature.source, 0) + 1
        return sources


# Global registry instance
_registry = None

def get_registry() -> AQIFeatureRegistry:
    """Get or create global feature registry (singleton)."""
    global _registry
    if _registry is None:
        _registry = AQIFeatureRegistry()
    return _registry
