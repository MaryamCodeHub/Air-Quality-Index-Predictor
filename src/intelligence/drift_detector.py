"""
Drift Detection Module
=======================
Detects feature distribution drift between training and live data
using the Kolmogorov-Smirnov (KS) statistical test.

Drift events are logged and persisted for monitoring.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.helpers import save_json, load_json
from src.utils.logger import setup_logger

logger = setup_logger("intelligence.drift_detector")


class DriftDetector:
    """
    Monitors feature distributions for statistical drift.

    Compares the current (live) data distribution against a reference
    (training) distribution using the KS-test.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ks_threshold = config["drift"]["ks_threshold"]
        self.min_samples = config["drift"]["min_samples"]
        self.drift_log_path = os.path.join(config["paths"]["logs"], "drift_events.json")

    def detect(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run KS-test drift detection across all numeric features.

        Args:
            reference_df: Training data (reference distribution)
            current_df: Current/live data
            feature_cols: Specific columns to check (default: all numeric)

        Returns:
            Drift report dict with per-feature results and overall status
        """
        if len(current_df) < self.min_samples:
            logger.warning(
                f"Insufficient current data ({len(current_df)} < {self.min_samples}) "
                f"for reliable drift detection"
            )
            return {"status": "insufficient_data", "details": {}}

        if feature_cols is None:
            feature_cols = [c for c in current_df.select_dtypes(include=[np.number]).columns]

        results = {}
        drifted_features = []

        for col in feature_cols:
            if col not in reference_df.columns or col not in current_df.columns:
                continue

            ref = reference_df[col].dropna().values
            cur = current_df[col].dropna().values

            if len(ref) == 0 or len(cur) == 0:
                continue

            # Two-sample Kolmogorov-Smirnov test
            ks_stat, p_value = stats.ks_2samp(ref, cur)

            is_drifted = ks_stat > self.ks_threshold
            results[col] = {
                "ks_statistic": round(float(ks_stat), 6),
                "p_value": round(float(p_value), 6),
                "threshold": self.ks_threshold,
                "is_drifted": is_drifted,
                "ref_mean": round(float(ref.mean()), 4),
                "cur_mean": round(float(cur.mean()), 4),
            }

            if is_drifted:
                drifted_features.append(col)

        # Overall drift status
        n_drifted = len(drifted_features)
        n_total = len(results)
        drift_ratio = n_drifted / max(n_total, 1)

        if drift_ratio > 0.3:
            status = "high_drift"
        elif n_drifted > 0:
            status = "moderate_drift"
        else:
            status = "no_drift"

        report = {
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "total_features": n_total,
            "drifted_features": n_drifted,
            "drift_ratio": round(drift_ratio, 4),
            "drifted_feature_names": drifted_features,
            "details": results,
        }

        logger.info(
            f"Drift status: {status} | "
            f"{n_drifted}/{n_total} features drifted (ratio={drift_ratio:.2%})"
        )

        # Log drift event
        self._log_drift_event(report)

        return report

    def _log_drift_event(self, report: Dict) -> None:
        """Append drift report to the persistent drift log."""
        events = []
        if os.path.exists(self.drift_log_path):
            try:
                events = load_json(self.drift_log_path)
                if not isinstance(events, list):
                    events = [events]
            except Exception:
                events = []

        # Keep only a summary in the log (not full details)
        summary = {k: v for k, v in report.items() if k != "details"}
        events.append(summary)

        # Keep last 100 events
        events = events[-100:]
        save_json(events, self.drift_log_path)

    def get_drift_history(self) -> List[Dict]:
        """Return the history of drift detection events."""
        if os.path.exists(self.drift_log_path):
            data = load_json(self.drift_log_path)
            return data if isinstance(data, list) else [data]
        return []
