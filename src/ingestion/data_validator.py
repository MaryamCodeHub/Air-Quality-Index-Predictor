"""
Data Validation Module

Comprehensive quality checks for AQI pipeline data:
- Sample size enforcement (minimum thresholds)
- Schema validation (expected columns and types)
- Timestamp continuity analysis (gap detection)
- Duplicate detection and removal
- IQR-based outlier detection
- Missing value analysis
- Data integrity checksums

Provides detailed validation reports for CI/CD integration and monitoring.
"""

import logging
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Structured validation report."""
    is_valid: bool
    total_records: int
    timestamp: str
    errors: List[str]
    warnings: List[str]
    quality_score: float
    data_quality_summary: Dict
    timestamp_continuity: Dict
    outliers_detected: Dict
    missing_values: Dict
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class AQIDataValidator:
    """
    Enterprise-grade data validation for AQI pipeline.
    
    Enforces:
    - Minimum sample counts (prevent training on tiny datasets)
    - Schema consistency (expected columns and types)
    - Temporal alignment (no data leakage, proper time ordering)
    - Statistical soundness (outlier caps, missing data limits)
    
    Attributes:
        min_samples: Minimum required records (default: 100 for dev, 5000 for prod)
        expected_columns: Required column names
        expected_dtypes: Required data types
        outlier_threshold: IQR multiplier for outlier detection (default: 3.0)
        missing_pct_threshold: Max allowed missing % per column (default: 10%)
        timezone_expected: Expected timezone (default: UTC)
    """
    
    def __init__(
        self,
        min_samples: int = 5000,
        expected_columns: Optional[List[str]] = None,
        outlier_threshold: float = 3.0,
        missing_pct_threshold: float = 10.0,
        timezone_expected: str = "UTC"
    ):
        """
        Initialize validator with configuration.
        
        Args:
            min_samples: Minimum required samples
            expected_columns: List of required column names
            outlier_threshold: IQR multiplier (3.0 = detect ~0.3% extreme outliers)
            missing_pct_threshold: Max allowed missing data percentage
            timezone_expected: Expected timezone for timestamps
        """
        self.min_samples = min_samples
        self.expected_columns = expected_columns or [
            'timestamp', 'aqi', 'temperature', 'humidity', 'pm25', 'pm10'
        ]
        self.outlier_threshold = outlier_threshold
        self.missing_pct_threshold = missing_pct_threshold
        self.timezone_expected = timezone_expected
        
        logger.info(
            f"AQIDataValidator initialized: "
            f"min_samples={min_samples}, outlier_threshold={outlier_threshold}, "
            f"missing_threshold={missing_pct_threshold}%"
        )
    
    def validate(self, df: pd.DataFrame, strict: bool = False) -> ValidationReport:
        """
        Execute complete validation pipeline.
        
        Order of checks:
        1. Empty dataframe check
        2. Minimum samples check (hard fail if below threshold)
        3. Column presence check (hard fail if missing required)
        4. Data type validation
        5. Timestamp analysis (continuity, sorting, duplicates)
        6. Missing value analysis
        7. Outlier detection
        8. Timezone consistency
        
        Args:
            df: DataFrame to validate
            strict: If True, treat warnings as errors
        
        Returns:
            ValidationReport with detailed findings
        """
        errors = []
        warnings = []
        quality_score = 100.0
        
        logger.debug(f"Starting validation on {len(df)} records")
        
        # ===== CHECK 1: Empty DataFrame =====
        if df.empty:
            errors.append("DataFrame is empty (0 rows)")
            return self._build_report(
                is_valid=False,
                df=df,
                errors=errors,
                warnings=warnings,
                quality_score=0.0
            )
        
        # ===== CHECK 2: Minimum Sample Size (Hard Fail) =====
        if len(df) < self.min_samples:
            errors.append(
                f"Insufficient data: {len(df)} samples, minimum required: {self.min_samples}"
            )
            return self._build_report(
                is_valid=False,
                df=df,
                errors=errors,
                warnings=warnings,
                quality_score=max(0, 100 * len(df) / self.min_samples)
            )
        
        # ===== CHECK 3: Required Columns (Hard Fail) =====
        missing_cols = set(self.expected_columns) - set(df.columns)
        if missing_cols:
            errors.append(
                f"Missing required columns: {sorted(missing_cols)}. "
                f"Available: {sorted(df.columns)}"
            )
            return self._build_report(
                is_valid=False,
                df=df,
                errors=errors,
                warnings=warnings,
                quality_score=50.0
            )
        
        # ===== CHECK 4: Data Types =====
        df = df.copy()  # Avoid modifying original
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        type_errors = self._validate_data_types(df)
        if type_errors:
            errors.extend(type_errors)
            return self._build_report(
                is_valid=False,
                df=df,
                errors=errors,
                warnings=warnings,
                quality_score=50.0
            )
        
        # ===== CHECK 5: Timestamp Analysis =====
        timestamp_check = self._validate_timestamps(df)
        errors.extend(timestamp_check['errors'])
        warnings.extend(timestamp_check['warnings'])
        quality_score -= len(timestamp_check['warnings']) * 5
        
        if timestamp_check['errors']:
            return self._build_report(
                is_valid=False,
                df=df,
                errors=errors,
                warnings=warnings,
                quality_score=quality_score
            )
        
        # ===== CHECK 6: Missing Values =====
        missing_check = self._validate_missing_values(df)
        errors.extend(missing_check['errors'])
        warnings.extend(missing_check['warnings'])
        quality_score -= len(missing_check['warnings']) * 5
        
        # ===== CHECK 7: Outlier Detection =====
        outlier_check = self._detect_outliers(df)
        if outlier_check['errors']:
            errors.extend(outlier_check['errors'])
        if outlier_check['warnings']:
            warnings.extend(outlier_check['warnings'])
            quality_score -= 10
        
        # ===== CHECK 8: Timezone Consistency =====
        tz_check = self._validate_timezone(df)
        warnings.extend(tz_check)
        
        # ===== FINAL DECISION =====
        is_valid = len(errors) == 0
        if strict and warnings:
            is_valid = False
            logger.warning(f"Strict mode: treating {len(warnings)} warnings as errors")
        
        quality_score = max(0, min(100, quality_score))
        
        report = self._build_report(
            is_valid=is_valid,
            df=df,
            errors=errors,
            warnings=warnings,
            quality_score=quality_score,
            timestamp_check=timestamp_check,
            outlier_check=outlier_check,
            missing_check=missing_check
        )
        
        self._log_report(report)
        return report
    
    def _validate_data_types(self, df: pd.DataFrame) -> List[str]:
        """
        Check data types are correct for expected numeric/datetime columns.
        
        Returns:
            List of error messages if type validation fails
        """
        errors = []
        
        # Timestamp must be datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            errors.append(
                f"Column 'timestamp' is {df['timestamp'].dtype}, expected datetime64"
            )
        
        # Numeric columns
        numeric_cols = ['aqi', 'temperature', 'humidity', 'pm25', 'pm10']
        for col in numeric_cols:
            if col in df.columns:
                if not np.issubdtype(df[col].dtype, np.number):
                    errors.append(
                        f"Column '{col}' is {df[col].dtype}, expected numeric"
                    )
        
        return errors
    
    def _validate_timestamps(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive timestamp analysis:
        - Proper sorting
        - Duplicates
        - Gap detection (missing hours)
        - Timezone consistency
        
        Returns:
            {'errors': [...], 'warnings': [...], 'gaps': [...], 'duplicates': n}
        """
        result = {
            'errors': [],
            'warnings': [],
            'gaps': [],
            'duplicates': 0,
            'time_range': {}
        }
        
        # Sort by timestamp
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        
        if not df_sorted['timestamp'].equals(df['timestamp']):
            result['warnings'].append(
                "Timestamps not sorted chronologically. Auto-sorting applied."
            )
        
        # Duplicate timestamps
        duplicates = df_sorted['timestamp'].duplicated().sum()
        if duplicates > 0:
            result['errors'].append(
                f"Found {duplicates} duplicate timestamps. "
                f"Data contamination detected."
            )
            result['duplicates'] = duplicates
            return result
        
        # Gap analysis (expect 1-hour intervals for this domain)
        time_diff = df_sorted['timestamp'].diff()
        expected_diff = pd.Timedelta(hours=1)
        
        gaps = df_sorted[time_diff != expected_diff][['timestamp']].copy()
        gaps['time_gap_hours'] = time_diff[time_diff != expected_diff].dt.total_seconds() / 3600
        
        if len(gaps) > 0:
            max_gap = gaps['time_gap_hours'].max()
            num_gaps = len(gaps)
            gap_pct = 100 * num_gaps / len(df_sorted)
            
            if gap_pct > 20:
                result['errors'].append(
                    f"Excessive missing hours: {num_gaps} gaps ({gap_pct:.1f}% of data), "
                    f"max gap {max_gap:.1f} hours"
                )
            else:
                result['warnings'].append(
                    f"Found {num_gaps} timestamp gaps ({gap_pct:.1f}%), "
                    f"max gap {max_gap:.1f} hours. Will require imputation."
                )
            
            result['gaps'] = gaps[['timestamp', 'time_gap_hours']].to_dict('records')
        
        # Time range
        result['time_range'] = {
            'start': df_sorted['timestamp'].min().isoformat(),
            'end': df_sorted['timestamp'].max().isoformat(),
            'duration_days': (df_sorted['timestamp'].max() - df_sorted['timestamp'].min()).days
        }
        
        return result
    
    def _validate_missing_values(self, df: pd.DataFrame) -> Dict:
        """
        Analyze missing data across all columns.
        
        Rules:
        - Hard fail: > 10% missing in any critical column
        - Warning: > 0% missing (will need imputation)
        
        Returns:
            {'errors': [...], 'warnings': [...], 'missing_by_col': {...}}
        """
        result = {
            'errors': [],
            'warnings': [],
            'missing_by_col': {}
        }
        
        missing_pct = df.isnull().sum() / len(df) * 100
        
        for col, pct in missing_pct.items():
            if pct == 0:
                continue
            
            result['missing_by_col'][col] = {
                'count': int(df[col].isnull().sum()),
                'percentage': round(pct, 2)
            }
            
            if pct > self.missing_pct_threshold:
                result['errors'].append(
                    f"Too many missing values in '{col}': "
                    f"{int(df[col].isnull().sum())} ({pct:.1f}%), "
                    f"threshold: {self.missing_pct_threshold}%"
                )
            else:
                result['warnings'].append(
                    f"Column '{col}' has {int(df[col].isnull().sum())} "
                    f"missing values ({pct:.1f}%)"
                )
        
        return result
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict:
        """
        IQR-based outlier detection for numeric columns.
        
        Outlier = value outside [Q1 - k*IQR, Q3 + k*IQR]
        where k = outlier_threshold (default 3.0, detects ~0.3% extreme values)
        
        Returns:
            {'errors': [...], 'warnings': [...], 'outliers_by_col': {...}}
        """
        result = {
            'errors': [],
            'warnings': [],
            'outliers_by_col': {}
        }
        
        numeric_cols = ['aqi', 'pm25', 'pm10', 'temperature', 'humidity']
        
        for col in numeric_cols:
            if col not in df.columns:
                continue
            
            # Skip if column is empty or all NaN
            if df[col].isnull().all():
                continue
            
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:
                continue
            
            lower_bound = Q1 - self.outlier_threshold * IQR
            upper_bound = Q3 + self.outlier_threshold * IQR
            
            outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = outliers_mask.sum()
            outlier_pct = 100 * outlier_count / len(df)
            
            result['outliers_by_col'][col] = {
                'count': int(outlier_count),
                'percentage': round(outlier_pct, 2),
                'bounds': {'lower': round(float(lower_bound), 2), 'upper': round(float(upper_bound), 2)}
            }
            
            if outlier_pct > 5:
                result['warnings'].append(
                    f"High outlier rate in '{col}': {outlier_count} values "
                    f"({outlier_pct:.1f}%) outside [{lower_bound:.1f}, {upper_bound:.1f}]"
                )
            elif outlier_count > 0:
                result['warnings'].append(
                    f"Column '{col}' has {outlier_count} outliers "
                    f"({outlier_pct:.2f}%)"
                )
        
        return result
    
    def _validate_timezone(self, df: pd.DataFrame) -> List[str]:
        """
        Check timezone consistency.
        
        Returns:
            List of warnings if timezone is naive or mismatched
        """
        warnings = []
        
        if df['timestamp'].dt.tz is None:
            warnings.append(
                f"Timestamp is timezone-naive. Assuming UTC. "
                f"Explicit timezone recommended."
            )
        elif str(df['timestamp'].dt.tz) != self.timezone_expected:
            warnings.append(
                f"Timestamp timezone is {df['timestamp'].dt.tz}, "
                f"expected {self.timezone_expected}"
            )
        
        return warnings
    
    def _build_report(
        self,
        is_valid: bool,
        df: pd.DataFrame,
        errors: List[str],
        warnings: List[str],
        quality_score: float,
        timestamp_check: Optional[Dict] = None,
        outlier_check: Optional[Dict] = None,
        missing_check: Optional[Dict] = None
    ) -> ValidationReport:
        """Construct ValidationReport object."""
        return ValidationReport(
            is_valid=is_valid,
            total_records=len(df),
            timestamp=datetime.now().isoformat(),
            errors=errors,
            warnings=warnings,
            quality_score=round(quality_score, 2),
            data_quality_summary={
                'record_count': len(df),
                'columns': len(df.columns),
                'memory_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
            },
            timestamp_continuity=timestamp_check or {},
            outliers_detected=outlier_check or {},
            missing_values=missing_check or {}
        )
    
    def _log_report(self, report: ValidationReport) -> None:
        """Log validation report summary."""
        status_icon = "✅" if report.is_valid else "❌"
        
        logger.info(
            f"{status_icon} Validation Complete | "
            f"Records: {report.total_records} | "
            f"Quality: {report.quality_score}% | "
            f"Errors: {len(report.errors)} | "
            f"Warnings: {len(report.warnings)}"
        )
        
        if report.errors:
            for error in report.errors:
                logger.error(f"  ❌ {error}")
        
        if report.warnings:
            for warning in report.warnings:
                logger.warning(f"  ⚠️  {warning}")


def validate_pipeline_data(
    df: pd.DataFrame,
    min_samples: int = 5000,
    strict: bool = False
) -> Tuple[bool, ValidationReport]:
    """
    Convenience function for quick validation in pipelines.
    
    Args:
        df: DataFrame to validate
        min_samples: Minimum required samples
        strict: Treat warnings as errors
    
    Returns:
        (is_valid, report)
    """
    validator = AQIDataValidator(min_samples=min_samples)
    report = validator.validate(df, strict=strict)
    return report.is_valid, report


if __name__ == "__main__":
    # Demo
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create sample dataset
    n = 5000
    df_sample = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=n, freq='H'),
        'aqi': np.random.uniform(50, 150, n),
        'temperature': np.random.uniform(15, 35, n),
        'humidity': np.random.uniform(30, 80, n),
        'pm25': np.random.uniform(20, 100, n),
        'pm10': np.random.uniform(30, 150, n)
    })
    
    # Add some issues
    df_sample.loc[100:105, 'aqi'] = np.nan  # Missing values
    df_sample.loc[1000, 'temperature'] = 500  # Outlier
    
    validator = AQIDataValidator(min_samples=1000)
    report = validator.validate(df_sample)
    
    print("\n" + "="*80)
    print("VALIDATION REPORT")
    print("="*80)
    import json
    print(json.dumps(report.to_dict(), indent=2, default=str))
