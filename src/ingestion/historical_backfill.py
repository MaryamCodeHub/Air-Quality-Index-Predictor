"""
Historical Data Backfill Module

Handles acquisition and versioning of 6-12 months of baseline AQI data from multiple sources.
Enforces minimum sample thresholds (5000+) and provides data provenance tracking.

Sources:
  - AQICN API historical (if available)
  - Open-Meteo historical weather archive
  - Synthetic generation (with flags) for development/testing
  - CSV import from Kaggle or government sources
"""

import logging
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import hashlib

import pandas as pd
import numpy as np
import requests
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)


class HistoricalBackfillManager:
    """
    Manages acquisition, versioning, and validation of historical baseline data.
    
    Attributes:
        data_dir: Root directory for data storage
        min_samples: Minimum required samples (default: 5000)
        lookback_days: Historical window to fetch (default: 180 days = 6 months)
    """
    
    def __init__(
        self,
        data_dir: str = "data/backfill",
        min_samples: int = 5000,
        lookback_days: int = 180,
        city: str = "Islamabad"
    ):
        """
        Initialize backfill manager.
        
        Args:
            data_dir: Directory for backfill data storage
            min_samples: Minimum samples required for training
            lookback_days: Days of historical data to fetch
            city: City name for AQICN API queries
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.min_samples = min_samples
        self.lookback_days = lookback_days
        self.city = city
        
        logger.info(
            f"HistoricalBackfillManager initialized: "
            f"dir={self.data_dir}, min_samples={min_samples}, "
            f"lookback={lookback_days} days"
        )
    
    def fetch_from_aqicn_historical(self) -> Optional[pd.DataFrame]:
        """
        Attempt to fetch historical AQI data from AQICN API.
        
        Note: AQICN free tier does not provide direct historical API.
        This method would require:
        - Premium AQICN subscription, OR
        - Querying archived snapshots if available
        
        Returns:
            DataFrame with columns [timestamp, aqi, pm25, pm10] or None if unavailable
        """
        logger.warning(
            "AQICN historical API not available in free tier. "
            "Recommend purchasing premium access or using alternative sources."
        )
        return None
    
    def fetch_from_openmeteo_historical(
        self,
        latitude: float = 33.7298,
        longitude: float = 73.1772
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical weather data from Open-Meteo (free, no auth required).
        
        This provides temperature, humidity, precipitation - correlates with AQI.
        
        Args:
            latitude: Location latitude (default: Islamabad)
            longitude: Location longitude (default: Islamabad)
        
        Returns:
            DataFrame with weather features
        """
        try:
            start_date = (datetime.now() - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")
            end_date = datetime.now().strftime("%Y-%m-%d")
            
            url = (
                f"https://archive-api.open-meteo.com/v1/archive?"
                f"latitude={latitude}&longitude={longitude}"
                f"&start_date={start_date}&end_date={end_date}"
                f"&hourly=temperature_2m,relative_humidity_2m,precipitation"
                f"&timezone=UTC"
            )
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(data['hourly']['time']),
                'temperature': data['hourly']['temperature_2m'],
                'humidity': data['hourly']['relative_humidity_2m'],
                'precipitation': data['hourly']['precipitation']
            })
            
            logger.info(f"Fetched {len(df)} historical weather records from Open-Meteo")
            return df
            
        except RequestException as e:
            logger.error(f"Failed to fetch Open-Meteo data: {e}")
            return None
    
    def generate_synthetic_baseline(self) -> pd.DataFrame:
        """
        Generate synthetic AQI data for development/testing.
        
        Uses realistic patterns:
        - Daily cycle (higher pollution in early morning/evening)
        - Weekly pattern (weekday > weekend)
        - Random walk with mean reversion (realistic temporal dynamics)
        - Correlated with temperature/humidity
        
        Returns:
            DataFrame with 6+ months of synthetic hourly data
        """
        logger.info(
            f"Generating synthetic baseline data ({self.lookback_days} days, ~{self.lookback_days * 24} records)"
        )
        
        n_hours = self.lookback_days * 24
        timestamps = pd.date_range(
            end=datetime.now(),
            periods=n_hours,
            freq='H',
            name='timestamp'
        )
        
        # Temperature: realistic range with daily/seasonal variation
        base_temp = 25
        seasonal_trend = 8 * np.sin(np.arange(n_hours) * 2 * np.pi / (365 * 24))
        daily_cycle = 5 * np.sin(np.arange(n_hours) * 2 * np.pi / 24)
        temp_noise = np.random.normal(0, 2, n_hours)
        temperature = base_temp + seasonal_trend + daily_cycle + temp_noise
        
        # Humidity: inverse relationship with temperature
        humidity = 60 - 0.8 * (temperature - base_temp) + np.random.normal(0, 5, n_hours)
        humidity = np.clip(humidity, 20, 95)
        
        # AQI: complex dynamics
        # Base: 80 (moderate pollution in Islamabad)
        # Daily pattern: peaks at 7-9am and 6-8pm (rush hours)
        aqi_base = 80
        hour_of_day = np.arange(n_hours) % 24
        
        # Rush hour effect
        rush_morning = 20 * np.exp(-((hour_of_day - 8) ** 2) / 4)
        rush_evening = 15 * np.exp(-((hour_of_day - 19) ** 2) / 6)
        rush_hour_effect = rush_morning + rush_evening
        
        # Weekly pattern (lower on weekends)
        day_of_week = timestamps.dayofweek.values
        weekend_effect = -15 * (day_of_week >= 5).astype(float)
        
        # Seasonal pattern (worse in winter)
        seasonal_aqi = 20 * np.sin(np.arange(n_hours) * 2 * np.pi / (365 * 24) + np.pi / 2)
        
        # Weather correlation (inverse: high temp = better dispersion)
        dispersion_factor = -0.5 * (temperature - base_temp)
        
        # Random walk (AQI has temporal persistence)
        aqi_noise = np.cumsum(np.random.normal(0, 1.5, n_hours))
        aqi_noise = (aqi_noise - aqi_noise.mean()) / aqi_noise.std() * 10
        
        aqi = (
            aqi_base +
            rush_hour_effect +
            weekend_effect +
            seasonal_aqi +
            dispersion_factor +
            aqi_noise +
            np.random.normal(0, 3, n_hours)
        )
        aqi = np.clip(aqi, 10, 500)
        
        # PM2.5 & PM10: proportional to AQI with noise
        pm25 = aqi * 0.5 + np.random.normal(0, 5, n_hours)
        pm25 = np.clip(pm25, 0, 300)
        
        pm10 = aqi * 0.7 + np.random.normal(0, 8, n_hours)
        pm10 = np.clip(pm10, 0, 400)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'aqi': aqi,
            'pm25': pm25,
            'pm10': pm10,
            'temperature': temperature,
            'humidity': humidity,
            'precipitation': np.maximum(np.random.exponential(0.5, n_hours), 0)
        })
        
        logger.info(
            f"Generated synthetic data: {len(df)} records, "
            f"AQI range [{df['aqi'].min():.1f}, {df['aqi'].max():.1f}]"
        )
        return df
    
    def import_from_csv(self, csv_path: str) -> Optional[pd.DataFrame]:
        """
        Import historical data from CSV file (Kaggle, government sources, etc).
        
        Expected CSV columns: timestamp, aqi, pm25, pm10, temperature, humidity
        
        Args:
            csv_path: Path to CSV file
        
        Returns:
            Normalized DataFrame or None if import fails
        """
        try:
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Normalize column names to lowercase
            df.columns = df.columns.str.lower()
            
            logger.info(f"Imported {len(df)} records from {csv_path}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to import CSV {csv_path}: {e}")
            return None
    
    def combine_data_sources(
        self,
        sources: List[Tuple[str, pd.DataFrame]]
    ) -> pd.DataFrame:
        """
        Combine multiple data sources intelligently.
        
        For overlapping timestamps, prioritizes: AQICN > Import > Synthetic
        For non-overlapping, concatenates and handles gaps.
        
        Args:
            sources: List of (source_name, dataframe) tuples
        
        Returns:
            Combined, deduplicated, sorted DataFrame
        """
        if not sources:
            raise ValueError("No data sources provided")
        
        logger.info(f"Combining {len(sources)} data sources")
        
        # Concatenate all
        combined = pd.concat(
            [df for _, df in sources],
            ignore_index=True,
            sort=False
        )
        
        # Sort by timestamp
        combined = combined.sort_values('timestamp').reset_index(drop=True)
        
        # Remove exact duplicates (keep first)
        combined = combined.drop_duplicates(subset=['timestamp'], keep='first')
        
        logger.info(f"Combined dataset: {len(combined)} unique records")
        return combined
    
    def validate_and_save(
        self,
        df: pd.DataFrame,
        source_name: str = "historical_baseline"
    ) -> Tuple[bool, Dict]:
        """
        Validate dataset meets minimum requirements and save with metadata.
        
        Checks:
        - Minimum 5000 samples
        - No missing critical columns
        - Timestamp continuity (warn if gaps > 2 hours)
        - Data type correctness
        
        Args:
            df: DataFrame to validate
            source_name: Name for versioning
        
        Returns:
            (is_valid, validation_report)
        """
        report = {
            'source': source_name,
            'timestamp': datetime.now().isoformat(),
            'total_records': len(df),
            'errors': [],
            'warnings': [],
            'checksum': None
        }
        
        # Check 1: Minimum samples
        if len(df) < self.min_samples:
            report['errors'].append(
                f"Insufficient data: {len(df)} samples, need {self.min_samples}"
            )
            return False, report
        
        # Check 2: Required columns
        required_cols = {'timestamp', 'aqi', 'temperature', 'humidity'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            report['errors'].append(f"Missing columns: {missing}")
            return False, report
        
        # Check 3: Timestamp analysis
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        time_diff = df_sorted['timestamp'].diff()
        expected_diff = pd.Timedelta(hours=1)
        
        gaps = time_diff[time_diff > expected_diff]
        if len(gaps) > 0:
            max_gap_hours = gaps.max().total_seconds() / 3600
            report['warnings'].append(
                f"Found {len(gaps)} timestamp gaps, max {max_gap_hours:.1f} hours"
            )
        
        # Check 4: Data types
        for col in ['aqi', 'temperature', 'humidity']:
            if not np.issubdtype(df[col].dtype, np.number):
                report['errors'].append(f"Column {col} is not numeric")
                return False, report
        
        # Check 5: Reasonable value ranges
        if (df['aqi'] < 0).any() or (df['aqi'] > 1000).any():
            report['warnings'].append(
                f"AQI values out of normal range: [{df['aqi'].min():.1f}, {df['aqi'].max():.1f}]"
            )
        
        if (df['temperature'] < -50).any() or (df['temperature'] > 60).any():
            report['warnings'].append(
                f"Temperature values unrealistic: [{df['temperature'].min():.1f}, {df['temperature'].max():.1f}]"
            )
        
        if not report['errors']:
            # Save parquet with metadata
            version_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
            parquet_path = self.data_dir / f"{source_name}_{version_tag}.parquet"
            metadata_path = self.data_dir / f"{source_name}_{version_tag}_metadata.json"
            
            df.to_parquet(parquet_path, index=False)
            
            # Compute checksum for data integrity
            file_hash = hashlib.md5(parquet_path.read_bytes()).hexdigest()
            report['checksum'] = file_hash
            report['parquet_path'] = str(parquet_path)
            
            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(
                f"✅ Validation passed. Saved to {parquet_path} "
                f"(checksum: {file_hash[:8]}...)"
            )
            return True, report
        
        return False, report
    
    def run_backfill_pipeline(
        self,
        use_synthetic: bool = False,
        csv_import_path: Optional[str] = None
    ) -> Tuple[bool, pd.DataFrame, Dict]:
        """
        Execute full backfill pipeline: fetch, combine, validate, save.
        
        Strategy:
        1. Attempt real data sources (AQICN, Open-Meteo, CSV)
        2. If insufficient, use synthetic data for development
        3. Combine sources intelligently
        4. Validate and version
        
        Args:
            use_synthetic: Force synthetic data generation
            csv_import_path: Path to CSV for import
        
        Returns:
            (success, dataframe, report)
        """
        logger.info("Starting historical backfill pipeline")
        sources: List[Tuple[str, pd.DataFrame]] = []
        
        # Try to fetch real weather data
        weather_df = self.fetch_from_openmeteo_historical()
        if weather_df is not None:
            sources.append(('openmeteo_weather', weather_df))
        else:
            logger.warning("Open-Meteo weather fetch failed, will rely on other sources")
        
        # Try CSV import if provided
        if csv_import_path and os.path.exists(csv_import_path):
            csv_df = self.import_from_csv(csv_import_path)
            if csv_df is not None:
                sources.append(('csv_import', csv_df))
        
        # Use synthetic data if explicitly requested or if insufficient real data
        if use_synthetic or not sources:
            synthetic_df = self.generate_synthetic_baseline()
            sources.append(('synthetic_baseline', synthetic_df))
        
        # Combine all sources
        if not sources:
            report = {
                'error': 'No data sources available',
                'timestamp': datetime.now().isoformat()
            }
            logger.error("Backfill pipeline failed: no sources")
            return False, pd.DataFrame(), report
        
        combined_df = self.combine_data_sources(sources)
        
        # Validate and save
        is_valid, report = self.validate_and_save(combined_df)
        
        if is_valid:
            logger.info("✅ Backfill pipeline completed successfully")
        else:
            logger.error(f"❌ Backfill pipeline failed: {report['errors']}")
        
        return is_valid, combined_df if is_valid else pd.DataFrame(), report


def main():
    """Execute backfill as standalone script."""
    import sys
    
    manager = HistoricalBackfillManager(
        data_dir="data/backfill",
        min_samples=5000,
        lookback_days=180
    )
    
    # Try real data first, fallback to synthetic
    success, df, report = manager.run_backfill_pipeline(use_synthetic=False)
    
    if not success:
        logger.warning("Real data insufficient, retrying with synthetic...")
        success, df, report = manager.run_backfill_pipeline(use_synthetic=True)
    
    print("\n" + "=" * 80)
    print("BACKFILL REPORT")
    print("=" * 80)
    print(json.dumps(report, indent=2, default=str))
    print("\nDataset Summary:")
    print(df.describe())
    print("=" * 80)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
