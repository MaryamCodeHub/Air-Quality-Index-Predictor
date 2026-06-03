"""
Essential pipeline tests: config, cleaning, feature engineering, model registry.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import load_config, ensure_directories
from src.ingestion.schema_mapper import SchemaMapper
from src.feature_engineering.feature_engineer import FeatureEngineer
from src.training.model_registry import ModelRegistry


@pytest.fixture
def config():
    cfg = load_config("configs/config.yaml")
    ensure_directories(cfg)
    return cfg


def test_config_loads(config):
    assert config["city"]["name"] == "Islamabad"
    assert config["paths"]["models"] == "artifacts/models"


def test_schema_mapper():
    mapper = SchemaMapper({"schema_mapping": {"aqi": ["aqi", "data.aqi"]}})
    assert mapper.resolve_field({"data": {"aqi": 42}}, "aqi") == 42


def test_feature_engineering(config):
    ts = pd.date_range("2024-01-01", periods=48, freq="h")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "aqi": range(48),
            "pm25": range(48),
            "pm10": range(48),
        }
    )
    out = FeatureEngineer(config).engineer(df)
    assert "aqi_lag_1h" in out.columns
    assert len(out) == 48


def test_model_registry_paths(config):
    reg = ModelRegistry(config)
    assert reg.model_dir == config["paths"]["models"]
