"""
AQI Intelligent Forecasting & Health Advisory System — Islamabad
=================================================================
Master CLI entry point for all system operations.
Real data only: AQICN API + Open-Meteo API (no synthetic generation).

Usage:
    python run.py ingest          Fetch + clean + engineer features
    python run.py features        Push engineered features to Hopsworks
    python run.py train           Train forecasting models
    python run.py explain         Generate SHAP explanations
    python run.py drift           Run drift detection
    python run.py serve           Start FastAPI backend
    python run.py dashboard       Launch Streamlit dashboard
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta

from src.utils.helpers import load_config, ensure_directories
from src.utils.logger import setup_logger

logger = setup_logger("run")


def cmd_ingest(config):
    """Fetch AQI + weather data → clean → engineer features."""
    from src.ingestion.api_client import DataIngestionPipeline
    from src.ingestion.data_cleaner import run_cleaning_pipeline
    from src.ingestion.feature_engineer import run_feature_pipeline

    pipeline = DataIngestionPipeline(config)
    raw_df = pipeline.run()
    if raw_df.empty:
        logger.error("Ingestion returned no data. Check API keys and network.")
        return

    clean_df = run_cleaning_pipeline(config)
    if clean_df.empty:
        logger.error("Cleaning produced no data.")
        return

    featured_df = run_feature_pipeline(config, clean_df)
    logger.info(f"Pipeline complete: {len(featured_df)} processed records")


def cmd_features(config):
    """Push engineered features to Hopsworks feature store."""
    import pandas as pd
    from src.feature_store import HopsworksConnector
    
    # Load processed data
    proc_path = os.path.join(config["paths"]["processed_data"], "processed_aqi_data.parquet")
    if not os.path.exists(proc_path):
        logger.error("No processed data. Run 'python run.py ingest' first.")
        return
    
    df = pd.read_parquet(proc_path)
    
    # Add city entity for Hopsworks
    df["city"] = config.get("city", {}).get("id", "islamabad")
    
    # Ensure timestamp is datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    logger.info(f"Pushing {len(df)} feature rows to Hopsworks")
    
    # Initialize Hopsworks connector
    connector = HopsworksConnector(config)
    status = connector.get_feature_store_status()
    
    if not status["connected"]:
        logger.warning(
            "Hopsworks not connected (HOPSWORKS_API_KEY not set in .env). "
            "Features are stored locally only (Parquet). "
            "To enable Hopsworks: "
            "1. Create account at hopsworks.ai\n"
            "2. Create project 'aqi_forecasting'\n"
            "3. Add API key to .env as HOPSWORKS_API_KEY"
        )
        return
    
    # Push features to Hopsworks (24h feature group)
    success = connector.push_features(
        df=df,
        feature_group_name="aqi_features_24h",
        version=1,
        primary_key=["city", "timestamp"],
        event_time="timestamp",
    )
    
    if success:
        logger.info(f"✓ Features successfully pushed to Hopsworks")
    else:
        logger.error("Failed to push features to Hopsworks")


def cmd_materialize(config):
    """Push features to Hopsworks (alias for cmd_features for backward compatibility)."""
    cmd_features(config)


def cmd_train(config):
    """Train Ridge, RF, XGBoost models for 24h/48h/72h forecasting."""
    from src.training.trainer import train_all_models
    results = train_all_models(config)
    if not results:
        logger.error("Training failed — see logs above.")


def cmd_explain(config):
    """Generate SHAP explanations for best models."""
    from src.intelligence.explainability import ExplainabilityEngine
    engine = ExplainabilityEngine(config)
    results = engine.explain_all_horizons()
    if results:
        logger.info(f"Generated explanations for {len(results)} horizons")
    else:
        logger.warning("No explanations generated. Train models first.")


def cmd_drift(config):
    """Run drift detection: compare training vs recent data distributions."""
    import pandas as pd
    from src.intelligence.drift_detector import DriftDetector

    proc_path = os.path.join(config["paths"]["processed_data"], "processed_aqi_data.parquet")
    if not os.path.exists(proc_path):
        logger.error("No processed data. Run ingestion first.")
        return

    df = pd.read_parquet(proc_path)
    if len(df) < 20:
        logger.error("Not enough data for drift detection.")
        return

    # Split: first 80% = reference, last 20% = current
    split = int(len(df) * 0.8)
    ref = df.iloc[:split]
    cur = df.iloc[split:]

    detector = DriftDetector(config)
    report = detector.detect(ref, cur)
    logger.info(f"Drift detection complete: status={report['status']}")


def cmd_serve(config):
    """Start the FastAPI backend server."""
    import uvicorn
    logger.info("Starting FastAPI server on http://localhost:8000")
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)


def cmd_dashboard(config):
    """Launch the Streamlit dashboard."""
    logger.info("Launching Streamlit dashboard …")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/dashboard/app.py"])


# ============================================================
# CLI Parser
# ============================================================

COMMANDS = {
    "ingest": cmd_ingest,
    "features": cmd_features,
    "materialize": cmd_materialize,
    "train": cmd_train,
    "explain": cmd_explain,
    "drift": cmd_drift,
    "serve": cmd_serve,
    "dashboard": cmd_dashboard,
}


def main():
    parser = argparse.ArgumentParser(
        description="AQI Intelligent Forecasting — Islamabad",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  ingest        Fetch live AQICN + weather data, clean, engineer features
  features      Push engineered features to Hopsworks feature store
  materialize   Alias for features (backward compatibility)
  train         Train forecasting models (Ridge, RF, XGBoost × 3 horizons)
  explain       Generate SHAP model explanations
  drift         Run distribution drift detection
  serve         Start FastAPI backend server (:8000)
  dashboard     Launch Streamlit dashboard

Examples:
  python run.py ingest                      # Fetch real data from APIs
  python run.py features                    # Push to Hopsworks
  python run.py train                       # Train 9 models
  python run.py serve & python run.py dashboard  # Both services
        """,
    )
    parser.add_argument("command", choices=COMMANDS.keys())
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.error(f"Config not found: {args.config}")
        sys.exit(1)

    ensure_directories(config)

    logger.info(f"{'=' * 60}")
    logger.info(f"COMMAND: {args.command.upper()} | City: Islamabad")
    logger.info(f"{'=' * 60}")

    COMMANDS[args.command](config)


if __name__ == "__main__":
    main()
