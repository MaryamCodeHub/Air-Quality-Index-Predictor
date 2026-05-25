import pytest
import pandas as pd
import numpy as np
from src.cleaning.outliers import (
    ClipOutlierHandler,
    IQROutlierHandler,
    ZScoreOutlierHandler
)

def test_clip_outlier_handler():
    s = pd.Series([-50.0, 10.0, 20.0, 30.0, 200.0])
    handler = ClipOutlierHandler()
    res = handler.handle(s, min=0.0, max=100.0)
    
    assert res.iloc[0] == 0.0
    assert res.iloc[1] == 10.0
    assert res.iloc[4] == 100.0


def test_iqr_outlier_handler():
    # Clean standard distribution with a single massive outlier (1000)
    s = pd.Series([10.0, 11.0, 12.0, 10.0, 11.0, 12.0, 1000.0])
    handler = IQROutlierHandler()
    res = handler.handle(s)
    
    # Verify that the outlier has been clamped significantly
    assert res.iloc[-1] < 50.0
    assert res.iloc[0] == 10.0


def test_zscore_outlier_handler():
    # Single massive outlier (1000)
    s = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1000.0])
    handler = ZScoreOutlierHandler()
    res = handler.handle(s, zscore_threshold=2.0)
    
    # Verify that standard deviation based clamping was triggered
    assert res.iloc[-1] < 500.0
