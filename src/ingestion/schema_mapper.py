"""
Schema Mapping Layer — Dynamic, Config-Driven, and Resilient
============================================================
Resolves variations in raw API keys and DataFrame schemas.
Uses dynamic checks and validation warnings to ensure pipeline robustness.
"""

from typing import Any, Dict, List, Optional
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("ingestion.schema_mapper")


class SchemaMapper:
    """
    MLOps-grade dynamic schema mapping layer.
    Allows mapping multiple variations of API field names to standardized internal names.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config if config is not None else {}
        self.schema_mapping = self.config.get("schema_mapping", {})

    def resolve_field(self, data: Dict[str, Any], standard_field: str) -> Any:
        """
        Dynamically finds the correct key in raw API data for a standardized field name,
        supporting nested dot-separated paths (e.g. 'data.iaqi.pm25.v').
        """
        keys_to_try = self.schema_mapping.get(standard_field, [standard_field])
        for key in keys_to_try:
            val = self._get_nested(data, key)
            if val is not None:
                return val
        return None

    def map_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Dynamically resolves and standardizes columns of a pandas DataFrame
        based on the registered schema mapping rules.
        """
        if df.empty:
            return df
            
        mapped_df = df.copy()
        
        rename_map = {}
        for std_col, variations in self.schema_mapping.items():
            for var in variations:
                if var in mapped_df.columns:
                    rename_map[var] = std_col
                    break
                    
        if rename_map:
            logger.info(f"Dynamically mapped columns: {rename_map}")
            rename_map = {k: v for k, v in rename_map.items() if k != v}
            mapped_df = mapped_df.rename(columns=rename_map)
            
        # Validation layer: Warn (don't crash) on schema mismatch or missing fields
        for std_col in self.schema_mapping.keys():
            if std_col not in mapped_df.columns:
                logger.warning(
                    f"[Schema Validation Warning] Standard column '{std_col}' was not detected in dataset. "
                    "Inference/imputation will be applied where applicable."
                )
                
        return mapped_df

    def _get_nested(self, d: Any, path: str) -> Any:
        """Safely fetch a nested dictionary value using a dot-separated string path."""
        if not path or not isinstance(d, dict):
            return None
        keys = path.split(".")
        curr = d
        for key in keys:
            if isinstance(curr, dict):
                curr = curr.get(key)
            else:
                return None
        return curr
