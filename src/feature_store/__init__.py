"""Feature Store Integration — Feast (Primary) + Parquet (Emergency Fallback)."""

from src.feature_store.feast_integration import FeastIntegration
from src.feature_store.feast_store import AQIFeastStore
from src.feature_store.feature_registry import get_registry

__all__ = ["FeastIntegration", "AQIFeastStore", "get_registry"]
