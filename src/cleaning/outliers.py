"""
Outlier Strategy Pattern — Pluggable and Configurable
======================================================
Defines interchangeable strategies for treating distribution outliers:
  - IQR Method
  - Z-Score Method
  - Direct Clamping Method
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("cleaning.outliers")


class OutlierStrategy(ABC):
    """Abstract base strategy class for treating data outliers."""

    @abstractmethod
    def handle(self, series: pd.Series, **kwargs) -> pd.Series:
        """
        Processes a pandas Series, handling its outliers according to the strategy.

        Args:
            series: Raw numerical series
            kwargs: Strategy specific arguments (min, max, thresholds)

        Returns:
            Cleaned/clamped pandas Series
        """
        pass


class IQROutlierHandler(OutlierStrategy):
    """IQR Strategy: Caps outliers using standard Interquartile Range thresholds (1.5 * IQR)."""

    def handle(self, series: pd.Series, **kwargs) -> pd.Series:
        s = series.copy()
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            return s
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        return s.clip(lower=lo, upper=hi)


class ZScoreOutlierHandler(OutlierStrategy):
    """Z-Score Strategy: Clamps outliers utilizing mean standard deviation bounds (e.g. 3 * STD)."""

    def handle(self, series: pd.Series, **kwargs) -> pd.Series:
        s = series.copy()
        threshold = kwargs.get("zscore_threshold", 3.0)
        mean = s.mean()
        std = s.std()
        if pd.isna(std) or std == 0:
            return s
        lo = mean - threshold * std
        hi = mean + threshold * std
        return s.clip(lower=lo, upper=hi)


class ClipOutlierHandler(OutlierStrategy):
    """Direct Clip Strategy: Clamps distribution values straight to valid min/max specifications."""

    def handle(self, series: pd.Series, **kwargs) -> pd.Series:
        s = series.copy()
        lo = kwargs.get("min", -np.inf)
        hi = kwargs.get("max", np.inf)
        return s.clip(lower=lo, upper=hi)
