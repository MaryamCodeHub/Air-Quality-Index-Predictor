"""
Rules Engine Module
====================
Parses cleaning_rules inside config.yaml and coordinates features
with their respective ranges and dynamic outlier strategy handlers.
Supports automatic column inference if configuration is incomplete.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.utils.logger import setup_logger
from src.cleaning.outliers import (
    IQROutlierHandler,
    ZScoreOutlierHandler,
    ClipOutlierHandler,
    OutlierStrategy
)

logger = setup_logger("cleaning.rules_engine")


class RulesEngine:
    """
    Parses and applies config-driven cleaning parameters,
    handling pluggable outlier methods per feature.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config if config is not None else {}
        self.rules = self.config.get("cleaning_rules", {})
        
        # Strategy lookup
        self._handlers = {
            "iqr": IQROutlierHandler(),
            "zscore": ZScoreOutlierHandler(),
            "clip": ClipOutlierHandler()
        }

    def get_rules(self, field: str) -> Dict[str, Any]:
        """Get cleaning rules dict for a given feature."""
        return self.rules.get(field, {})

    def get_handler(self, field: str) -> OutlierStrategy:
        """Resolve dynamic outlier handler strategy based on config rules."""
        rules = self.get_rules(field)
        method = rules.get("outlier_method", "clip").lower()
        if method not in self._handlers:
            logger.warning(f"Outlier method '{method}' not recognized for '{field}'. Falling back to 'clip' strategy.")
            return self._handlers["clip"]
        return self._handlers[method]

    def infer_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Automatically detect numeric columns from the DataFrame dtypes."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude common non-feature metadata columns if present
        metadata = ["timestamp", "ingestion_timestamp", "city"]
        return [c for c in numeric_cols if c not in metadata]

    def get_numeric_columns(self, df: Optional[pd.DataFrame] = None) -> List[str]:
        """Retrieve all numeric columns configured for processing, merged with inferred columns."""
        config_cols = list(self.rules.keys())
        if df is not None:
            inferred = self.infer_numeric_columns(df)
            merged = list(set(config_cols + inferred))
            return merged
        return config_cols

    def get_bounds(self, field: str) -> tuple:
        """Get validation bounds (min, max) for range checks."""
        rules = self.get_rules(field)
        lo = rules.get("min", -np.inf)
        hi = rules.get("max", np.inf)
        return lo, hi
