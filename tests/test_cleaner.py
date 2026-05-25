import pytest
import pandas as pd
import numpy as np
from src.cleaning.cleaner import DataCleaner

def test_data_cleaner_pipeline():
    # Setup custom minimal config for cleaner
    config = {
        "schema_mapping": {
            "timestamp": ["timestamp", "time"],
            "aqi": ["aqi"],
            "pm25": ["pm25"]
        },
        "cleaning_rules": {
            "aqi": {
                "outlier_method": "clip",
                "min": 0,
                "max": 500
            },
            "pm25": {
                "outlier_method": "clip",
                "min": 0,
                "max": 1000
            }
        }
    }
    
    # Setup dirty dummy DataFrame:
    # - Row 0 & 1 have duplicate timestamps (Row 0 should be dropped)
    # - Row 2 has aqi=600 (out of bounds, set to NaN, then filled via interpolation)
    # - Row 1 has pm25=None (imputed via interpolation)
    data = {
        "timestamp": ["2026-05-23T06:00:00Z", "2026-05-23T06:00:00Z", "2026-05-23T08:00:00Z"],
        "aqi": ["100", "150", "600"],  # string values to test numeric enforcement
        "pm25": [50.0, None, 1200.0]
    }
    df = pd.DataFrame(data)
    
    cleaner = DataCleaner(config)
    cleaned_df = cleaner.clean(df)
    
    # Verify duplicates were dropped
    assert len(cleaned_df) == 2
    
    # Verify timestamp ordering & parsing
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df["timestamp"])
    
    # Verify numeric coercion
    assert pd.api.types.is_numeric_dtype(cleaned_df["aqi"])
    assert pd.api.types.is_numeric_dtype(cleaned_df["pm25"])
    
    # Verify range validation bounds & outlier clamping (no NaN values remaining)
    assert not cleaned_df.isna().any().any()
    assert (cleaned_df["aqi"] <= 500).all()
    assert (cleaned_df["pm25"] <= 1000).all()
