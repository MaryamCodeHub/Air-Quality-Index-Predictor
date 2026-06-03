#!/usr/bin/env python3
"""
Automates generation and preprocessing of baseline historical data.
This script:
1. Runs HistoricalBackfillManager to create realistic historical records.
2. Copies the generated backfill Parquet into data/raw/raw_aqi_data.parquet.
3. Automatically triggers python run.py ingest to run cleaning, feature
   engineering, Feast registration, Feast materialization, and model training.
"""

import os
import shutil
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_PROJECT_ROOT)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.helpers import load_config
from src.ingestion.historical_backfill import HistoricalBackfillManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("run_backfill")


def main():
    logger.info("Starting baseline backfill generation...")
    config = load_config("configs/config.yaml")
    
    # Define directories
    raw_dir = Path(config["paths"]["raw_data"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "raw_aqi_data.parquet"
    
    # Initialize and run backfill
    manager = HistoricalBackfillManager(
        data_dir=config["paths"].get("backfill_data", "data/backfill"),
        min_samples=1000,
        lookback_days=250,
        city="Islamabad"
    )
    
    # Force synthetic baseline for complete and consistent 6+ month history
    success, df, report = manager.run_backfill_pipeline(use_synthetic=True)
    if not success or df.empty:
        logger.error("Historical backfill failed.")
        return
        
    generated_path = report.get("parquet_path")
    if not generated_path or not os.path.exists(generated_path):
        logger.error(f"Generated backfill file not found at {generated_path}")
        return
        
    logger.info(f"Generated backfill dataset has {len(df)} rows.")
    
    # Copy file to data/raw/raw_aqi_data.parquet
    logger.info(f"Copying {generated_path} -> {raw_path}")
    shutil.copy(generated_path, raw_path)
    logger.info("Raw historical data successfully populated.")
    
    # Trigger ingest pipeline to clean and engineer features
    logger.info("Triggering end-to-end ingestion and Feast materialization...")
    import subprocess
    subprocess.run([sys.executable, "pipelines/run.py", "ingest"], check=True)
    logger.info("🎉 E2E backfill pipeline execution complete!")


if __name__ == "__main__":
    main()
