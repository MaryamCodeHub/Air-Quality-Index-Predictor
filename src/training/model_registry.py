"""
Model Registry
===============
Handles saving, loading, and tracking trained models with metadata.

Each model is saved as:
  models/{model_name}_{horizon}h.joblib   — serialized model
  models/{model_name}_{horizon}h.json     — metadata (metrics, features, timestamp)
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib

from src.utils.helpers import save_json, load_json
from src.utils.logger import setup_logger

logger = setup_logger("training.model_registry")


class ModelRegistry:
    """Manages model persistence and metadata tracking."""

    def __init__(self, config: Dict[str, Any]):
        self.model_dir = config["paths"]["models"]
        os.makedirs(self.model_dir, exist_ok=True)

    def save_model(
        self,
        model: Any,
        model_name: str,
        horizon: int,
        metrics: Dict[str, float],
        feature_names: List[str],
    ) -> str:
        """
        Save a trained model + metadata to disk.

        Returns:
            Path to the saved model file
        """
        base = f"{model_name}_{horizon}h"
        model_path = os.path.join(self.model_dir, f"{base}.joblib")
        meta_path = os.path.join(self.model_dir, f"{base}.json")

        # Save model
        joblib.dump(model, model_path)

        # Save metadata
        metadata = {
            "model_name": model_name,
            "horizon_hours": horizon,
            "metrics": metrics,
            "feature_names": feature_names,
            "trained_at": datetime.now().isoformat(),
            "model_path": model_path,
        }
        save_json(metadata, meta_path)

        logger.info(f"Saved {model_name} ({horizon}h) → {model_path}")
        return model_path

    def load_model(self, model_name: str, horizon: int) -> Optional[Any]:
        """Load a trained model from disk."""
        path = os.path.join(self.model_dir, f"{model_name}_{horizon}h.joblib")
        if not os.path.exists(path):
            logger.error(f"Model not found: {path}")
            return None
        return joblib.load(path)

    def load_metadata(self, model_name: str, horizon: int) -> Optional[Dict]:
        """Load model metadata from disk."""
        path = os.path.join(self.model_dir, f"{model_name}_{horizon}h.json")
        if not os.path.exists(path):
            return None
        return load_json(path)

    def mark_best(self, model_name: str, horizon: int) -> None:
        """Record which model is best for a given horizon."""
        best_path = os.path.join(self.model_dir, f"best_{horizon}h.json")
        save_json({"best_model": model_name, "horizon": horizon, "selected_at": datetime.now().isoformat()}, best_path)
        logger.info(f"Best model for {horizon}h: {model_name}")

    def get_best_model_name(self, horizon: int) -> Optional[str]:
        """Get the name of the best model for a horizon."""
        path = os.path.join(self.model_dir, f"best_{horizon}h.json")
        if not os.path.exists(path):
            return None
        data = load_json(path)
        return data.get("best_model")

    def load_best_model(self, horizon: int):
        """Load the best model for a given horizon."""
        name = self.get_best_model_name(horizon)
        if name is None:
            logger.error(f"No best model recorded for {horizon}h")
            return None, None
        model = self.load_model(name, horizon)
        meta = self.load_metadata(name, horizon)
        return model, meta

    def list_models(self) -> List[Dict]:
        """List all saved models with their metadata."""
        models = []
        for f in os.listdir(self.model_dir):
            if f.endswith(".json") and not f.startswith("best_"):
                meta = load_json(os.path.join(self.model_dir, f))
                models.append(meta)
        return models
