"""
Model Evaluator
================
Computes regression metrics: RMSE, MAE, R².
"""

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute standard regression metrics.

    Args:
        y_true: Ground truth values
        y_pred: Model predictions

    Returns:
        Dict with 'rmse', 'mae', 'r2' keys
    """
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
