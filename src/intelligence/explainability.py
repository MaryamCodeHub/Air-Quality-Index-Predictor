"""
SHAP Explainability Module
===========================
Generates SHAP explanations for trained AQI forecasting models.

Uses TreeExplainer for XGBoost/RandomForest, LinearExplainer for Ridge.
Saves summary plots and feature importance to plots/ directory.
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt

from src.training.model_registry import ModelRegistry
from src.utils.logger import setup_logger

logger = setup_logger("intelligence.explainability")


class ExplainabilityEngine:
    """Generates SHAP-based model explanations for Islamabad AQI models."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.plots_dir = config["paths"]["plots"]
        os.makedirs(self.plots_dir, exist_ok=True)
        self.registry = ModelRegistry(config)

    def explain(self, horizon: int = 24, max_samples: int = 100) -> Optional[Dict]:
        """
        Generate SHAP explanation for the best model at a given horizon.

        Args:
            horizon: Forecast horizon (24, 48, or 72)
            max_samples: Max samples for SHAP computation (speed vs accuracy)

        Returns:
            Dict with feature importances and plot paths
        """
        model, meta = self.registry.load_best_model(horizon)
        if model is None:
            logger.error(f"No model found for {horizon}h horizon")
            return None

        model_name = meta["model_name"]
        feature_names = meta["feature_names"]
        logger.info(f"Explaining {model_name} ({horizon}h) with SHAP …")

        # Load processed data for background
        proc_path = os.path.join(self.config["paths"]["processed_data"], "processed_aqi_data.parquet")
        df = pd.read_parquet(proc_path)
        X = df[[c for c in feature_names if c in df.columns]].tail(max_samples).values

        # Select appropriate SHAP explainer
        try:
            if model_name in ("xgboost", "random_forest"):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
            else:
                # Ridge / Linear — use KernelExplainer with background sample
                background = shap.sample(X, min(50, len(X)))
                explainer = shap.KernelExplainer(model.predict, background)
                shap_values = explainer.shap_values(X, nsamples=100)
        except Exception as exc:
            logger.error(f"SHAP computation failed: {exc}")
            return None

        # Feature importance (mean absolute SHAP value)
        importance = np.abs(shap_values).mean(axis=0)
        used_features = [c for c in feature_names if c in df.columns]
        feat_importance = dict(zip(used_features[:len(importance)], importance.tolist()))
        feat_importance = dict(sorted(feat_importance.items(), key=lambda x: x[1], reverse=True))

        # Generate summary plot
        summary_path = os.path.join(self.plots_dir, f"shap_summary_{horizon}h.png")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, feature_names=used_features, show=False)
        plt.title(f"SHAP Summary — {model_name} ({horizon}h Forecast)")
        plt.tight_layout()
        plt.savefig(summary_path, dpi=150)
        plt.close()
        logger.info(f"SHAP summary plot saved → {summary_path}")

        # Generate bar plot
        bar_path = os.path.join(self.plots_dir, f"shap_bar_{horizon}h.png")
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, feature_names=used_features, plot_type="bar", show=False)
        plt.title(f"Feature Importance — {model_name} ({horizon}h)")
        plt.tight_layout()
        plt.savefig(bar_path, dpi=150)
        plt.close()
        logger.info(f"SHAP bar plot saved → {bar_path}")

        return {
            "model_name": model_name,
            "horizon": horizon,
            "feature_importance": feat_importance,
            "summary_plot": summary_path,
            "bar_plot": bar_path,
        }

    def explain_all_horizons(self) -> List[Dict]:
        """Generate explanations for all forecast horizons."""
        results = []
        for h in self.config["training"]["forecast_horizons"]:
            result = self.explain(horizon=h)
            if result:
                results.append(result)
        return results
