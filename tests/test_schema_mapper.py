import pytest
import pandas as pd
from src.ingestion.schema_mapper import SchemaMapper

def test_schema_mapper_resolve_field():
    config = {
        "schema_mapping": {
            "pm25": ["pm25", "data.iaqi.pm25.v", "pm2_5"],
            "temperature": ["temperature", "temp"]
        }
    }
    mapper = SchemaMapper(config)
    
    # 1. Flat dictionary test
    data_flat = {"temp": 32.5}
    assert mapper.resolve_field(data_flat, "temperature") == 32.5
    
    # 2. Nested dictionary dot-notation test
    data_nested = {
        "data": {
            "iaqi": {
                "pm25": {
                    "v": 88.0
                }
            }
        }
    }
    assert mapper.resolve_field(data_nested, "pm25") == 88.0
    
    # 3. Missing resolution fallback test
    assert mapper.resolve_field(data_nested, "temperature") is None


def test_schema_mapper_map_dataframe():
    config = {
        "schema_mapping": {
            "pm25": ["pm25", "pm2_5"],
            "temperature": ["temperature", "temp", "hourly.temperature_2m"]
        }
    }
    mapper = SchemaMapper(config)
    
    # DataFrame with variations of naming
    df = pd.DataFrame({
        "pm2_5": [12.0, 15.0],
        "hourly.temperature_2m": [18.5, 21.0]
    })
    
    mapped_df = mapper.map_dataframe(df)
    
    # Verify columns were standardized
    assert "pm25" in mapped_df.columns
    assert "temperature" in mapped_df.columns
    assert "pm2_5" not in mapped_df.columns
    assert "hourly.temperature_2m" not in mapped_df.columns
    assert mapped_df.loc[0, "pm25"] == 12.0
