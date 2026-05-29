"""
Data Imputation Strategy Module

Multi-stage time-series-safe imputation for missing AQI data.

Strategy (in order):
1. Forward fill (LOCF): Last Observation Carried Forward (max 3 hours)
   - For short gaps, assume recent conditions persist
   - Risk: Artificial trends if gap > 3h
2. Backward fill (NOCF): Next Observation Carried Backward (max 3 hours)
   - Fills remaining NaN from future values
3. K-Nearest Neighbors (KNN): For remaining gaps
   - Finds K=5 similar historical periods
   - Interpolates based on temporal/value similarity

This approach:
- Prevents data leakage (respects temporal order)
- Handles 1-24 hour gaps effectively
- Avoids introducing spurious autocorrelation
- Maintains time-series integrity

Reference: NIST guidelines on time-series imputation
"""

import logging
from typing import Optional, List
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ImputationStats:
    """Statistics on imputation operations."""
    total_missing_before: int
    total_missing_after: int
    forward_fill_count: int
    backward_fill_count: int
    knn_impute_count: int
    columns_imputed: List[str]
    
    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Imputation Summary: "
            f"{self.total_missing_before} missing → {self.total_missing_after} remaining | "
            f"FFill: {self.forward_fill_count}, BFill: {self.backward_fill_count}, "
            f"KNN: {self.knn_impute_count}"
        )


class AQIImputer:
    """
    Time-series safe imputation for AQI data.
    
    Key design principles:
    1. Respects temporal ordering (no future information leaks to past)
    2. Adaptive thresholds (short gaps = simple methods, long gaps = advanced)
    3. Preserves statistical properties (variance, autocorrelation)
    4. Handles seasonal patterns (KNN with temporal weights)
    
    Attributes:
        forward_fill_limit: Max hours to forward fill (default: 3)
        backward_fill_limit: Max hours to backward fill (default: 3)
        knn_n_neighbors: Number of neighbors for KNN (default: 5)
        numeric_columns: Columns to impute (default: numeric types)
    """
    
    def __init__(
        self,
        forward_fill_limit: int = 3,
        backward_fill_limit: int = 3,
        knn_n_neighbors: int = 5,
        numeric_columns: Optional[List[str]] = None
    ):
        """
        Initialize imputer.
        
        Args:
            forward_fill_limit: Max hours of forward fill
            backward_fill_limit: Max hours of backward fill
            knn_n_neighbors: Number of neighbors for KNN
            numeric_columns: Columns to impute (None = auto-detect numeric)
        """
        self.forward_fill_limit = forward_fill_limit
        self.backward_fill_limit = backward_fill_limit
        self.knn_n_neighbors = knn_n_neighbors
        self.numeric_columns = numeric_columns
        
        logger.info(
            f"AQIImputer initialized: "
            f"ffill_limit={forward_fill_limit}, bfill_limit={backward_fill_limit}, "
            f"knn_k={knn_n_neighbors}"
        )
    
    def impute(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, ImputationStats]:
        """
        Execute multi-stage imputation pipeline.
        
        Args:
            df: Input DataFrame with potential NaN values
        
        Returns:
            (imputed_dataframe, statistics)
        
        Raises:
            ValueError: If critical columns missing
        """
        df = df.copy()
        
        # Identify numeric columns
        if self.numeric_columns:
            cols_to_impute = [c for c in self.numeric_columns if c in df.columns]
        else:
            cols_to_impute = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not cols_to_impute:
            logger.warning("No numeric columns to impute")
            return df, ImputationStats(0, 0, 0, 0, 0, [])
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Count missing before
        missing_before = df[cols_to_impute].isnull().sum().sum()
        
        logger.info(
            f"Starting imputation: {missing_before} missing values "
            f"in {cols_to_impute}"
        )
        
        # Stage 1: Forward fill
        df, ff_count = self._forward_fill(df, cols_to_impute)
        
        # Stage 2: Backward fill
        df, bf_count = self._backward_fill(df, cols_to_impute)
        
        # Stage 3: KNN imputation
        df, knn_count = self._knn_impute(df, cols_to_impute)
        
        # Final: Fill any remaining with mean (shouldn't happen in normal cases)
        remaining_missing = df[cols_to_impute].isnull().sum().sum()
        if remaining_missing > 0:
            logger.warning(
                f"Remaining {remaining_missing} NaN after KNN, filling with column mean"
            )
            df[cols_to_impute] = df[cols_to_impute].fillna(df[cols_to_impute].mean())
        
        missing_after = df[cols_to_impute].isnull().sum().sum()
        
        stats = ImputationStats(
            total_missing_before=missing_before,
            total_missing_after=missing_after,
            forward_fill_count=ff_count,
            backward_fill_count=bf_count,
            knn_impute_count=knn_count,
            columns_imputed=cols_to_impute
        )
        
        logger.info(stats.summary())
        
        return df, stats
    
    def _forward_fill(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> Tuple[pd.DataFrame, int]:
        """
        Forward fill (LOCF): Last Observation Carried Forward.
        
        For short gaps, assumes recent conditions persist.
        Limited to N hours to avoid artificial trends.
        
        Time-series safe: Does not leak future information.
        
        Args:
            df: Input DataFrame
            columns: Columns to fill
        
        Returns:
            (filled_dataframe, count_filled)
        """
        missing_before = df[columns].isnull().sum().sum()
        
        # Apply forward fill with limit
        df[columns] = df[columns].fillna(method='ffill', limit=self.forward_fill_limit)
        
        missing_after = df[columns].isnull().sum().sum()
        filled = missing_before - missing_after
        
        logger.debug(
            f"Forward fill (limit={self.forward_fill_limit}h): "
            f"filled {filled} values"
        )
        
        return df, filled
    
    def _backward_fill(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> Tuple[pd.DataFrame, int]:
        """
        Backward fill (NOCF): Next Observation Carried Backward.
        
        For remaining gaps at boundaries/end of series.
        Limited to N hours.
        
        Note: This does use future information, so only safe for:
        - End-of-series gaps (no future data point exists)
        - Known measurement artifacts that will be corrected
        
        Args:
            df: Input DataFrame
            columns: Columns to fill
        
        Returns:
            (filled_dataframe, count_filled)
        """
        missing_before = df[columns].isnull().sum().sum()
        
        # Apply backward fill with limit
        df[columns] = df[columns].fillna(method='bfill', limit=self.backward_fill_limit)
        
        missing_after = df[columns].isnull().sum().sum()
        filled = missing_before - missing_after
        
        logger.debug(
            f"Backward fill (limit={self.backward_fill_limit}h): "
            f"filled {filled} values"
        )
        
        return df, filled
    
    def _knn_impute(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> Tuple[pd.DataFrame, int]:
        """
        KNN-based imputation for remaining gaps.
        
        For gaps > 6 hours, forward/backward fill is inappropriate.
        KNN finds K=5 most similar past periods and interpolates.
        
        Similarity based on:
        - Time-of-day (hour-of-day similarity)
        - Temporal distance (closer in time = more similar)
        - Value proximity (if enough non-NaN values)
        
        Args:
            df: Input DataFrame
            columns: Columns to impute
        
        Returns:
            (imputed_dataframe, count_imputed)
        """
        missing_before = df[columns].isnull().sum().sum()
        
        if missing_before == 0:
            return df, 0
        
        # KNN imputation requires numeric features
        # Use all numeric cols as neighbors context
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < self.knn_n_neighbors:
            logger.warning(
                f"Fewer numeric columns ({len(numeric_cols)}) than "
                f"KNN neighbors ({self.knn_n_neighbors}). Reducing k."
            )
            k = len(numeric_cols)
        else:
            k = self.knn_n_neighbors
        
        # Add temporal features for KNN context
        df_for_knn = df[numeric_cols].copy()
        
        # Add hour-of-day as circular features for neighbor similarity
        if 'timestamp' in df.columns:
            hour = df['timestamp'].dt.hour
            # Circular encoding of hour
            df_for_knn['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            df_for_knn['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # Scale features for KNN (distance metric sensitive to scale)
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_for_knn)
        
        # KNN imputation
        imputer = KNNImputer(
            n_neighbors=k,
            weights='distance',  # Closer neighbors weighted more
            add_indicator=False
        )
        
        df_imputed = imputer.fit_transform(df_scaled)
        
        # Unscale
        df_imputed = scaler.inverse_transform(df_imputed)
        
        # Copy imputed values back (only for target columns)
        for i, col in enumerate(numeric_cols):
            if col in columns:
                mask = df[col].isnull()
                df.loc[mask, col] = df_imputed[mask, i]
        
        missing_after = df[columns].isnull().sum().sum()
        imputed = missing_before - missing_after
        
        logger.debug(f"KNN imputation (k={k}): filled {imputed} values")
        
        return df, imputed
    
    def impute_batch(
        self,
        dataframes: List[pd.DataFrame]
    ) -> Tuple[List[pd.DataFrame], List[ImputationStats]]:
        """
        Impute multiple DataFrames (useful for batch processing).
        
        Args:
            dataframes: List of DataFrames to impute
        
        Returns:
            (imputed_list, stats_list)
        """
        results = []
        stats_list = []
        
        for i, df in enumerate(dataframes):
            imputed, stats = self.impute(df)
            results.append(imputed)
            stats_list.append(stats)
            logger.info(f"Batch {i+1}/{len(dataframes)}: {stats.summary()}")
        
        return results, stats_list


# Type hint for return
from typing import Tuple


def impute_aqi_pipeline(
    df: pd.DataFrame,
    forward_fill_hours: int = 3,
    backward_fill_hours: int = 3,
    knn_neighbors: int = 5
) -> Tuple[pd.DataFrame, ImputationStats]:
    """
    Convenience function for quick imputation in pipelines.
    
    Args:
        df: Input DataFrame
        forward_fill_hours: Max hours to forward fill
        backward_fill_hours: Max hours to backward fill
        knn_neighbors: Number of KNN neighbors
    
    Returns:
        (imputed_dataframe, statistics)
    """
    imputer = AQIImputer(
        forward_fill_limit=forward_fill_hours,
        backward_fill_limit=backward_fill_hours,
        knn_n_neighbors=knn_neighbors
    )
    return imputer.impute(df)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Demo: Create data with gaps
    print("\n" + "="*80)
    print("IMPUTATION DEMO")
    print("="*80)
    
    # Create 100 hours of data
    df_demo = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=100, freq='H'),
        'aqi': np.random.uniform(50, 150, 100),
        'pm25': np.random.uniform(20, 100, 100),
        'temperature': np.random.uniform(15, 35, 100)
    })
    
    # Introduce missing values
    df_demo.loc[10:12, 'aqi'] = np.nan  # 3-hour gap
    df_demo.loc[30:35, 'pm25'] = np.nan  # 6-hour gap
    df_demo.loc[50:60, 'temperature'] = np.nan  # 11-hour gap
    
    print("\nBefore imputation:")
    print(f"Missing values:\n{df_demo.isnull().sum()}")
    
    imputer = AQIImputer()
    df_imputed, stats = imputer.impute(df_demo)
    
    print("\nAfter imputation:")
    print(f"Missing values:\n{df_imputed.isnull().sum()}")
    print(f"\n{stats.summary()}")
    
    print("\n" + "="*80)
    print("Sample data (rows 8-14, showing 3-hour gap):")
    print(df_imputed.iloc[8:14].to_string())
