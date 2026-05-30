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
    """Push engineered features to Feast (primary) and Hopsworks (optional)."""
    import pandas as pd
    from src.feature_store.feast_integration import FeastIntegration
    
    # Load processed data
    proc_path = os.path.join(config["paths"]["processed_data"], "processed_aqi_data.parquet")
    if not os.path.exists(proc_path):
        logger.error("No processed data. Run 'python run.py ingest' first.")
        return
    
    df = pd.read_parquet(proc_path)
    logger.info(f"Loaded {len(df)} processed records for feature store")
    
    # ===== FEAST INTEGRATION (Primary) =====
    try:
        logger.info("▶ Pushing features to Feast feature store...")
        feast = FeastIntegration(repo_path="feature_store")
        
        # Split raw vs engineered features
        raw_cols = {"timestamp", "aqi", "pm25", "pm10", "temp", "humidity", "wind_speed"}
        raw_df = df[[col for col in df.columns if col in raw_cols]]
        
        # Ingest and register all features
        feast_result = feast.ingest_and_register(raw_df=raw_df, engineered_df=df)
        logger.info(f"✅ Feast integration complete")
        logger.info(f"   - Raw features registered: {feast_result['step_1_raw_registration']}")
        logger.info(f"   - Engineered features registered: {feast_result['step_2_engineered_registration']}")
        logger.info(f"   - Materialization: {feast_result['step_3_materialization']}")
        
        # Export feature manifest for governance
        feast.export_feature_manifest()
        
    except Exception as e:
        logger.error(f"Feast integration error: {e}")
        return
    
    # ===== HOPSWORKS INTEGRATION (Optional) =====
    try:
        from src.feature_store.hopsworks_connector import HopsworksConnector
        
        logger.info("▶ Checking Hopsworks connection...")
        connector = HopsworksConnector(config)
        status = connector.get_feature_store_status()
        
        if status["connected"]:
            # Add city entity for Hopsworks
            df["city"] = config.get("city", {}).get("id", "islamabad")
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Push to Hopsworks
            success = connector.push_features(
                df=df,
                feature_group_name="aqi_features_24h",
                version=1,
                primary_key=["city", "timestamp"],
                event_time="timestamp",
            )
            
            if success:
                logger.info(f"✅ Features successfully pushed to Hopsworks")
            else:
                logger.warning("Hopsworks push returned false, but Feast succeeded")
        else:
            logger.info("ℹ Hopsworks not available (optional). Feast primary store active.")
    
    except Exception as e:
        logger.warning(f"Hopsworks integration skipped: {e}. Feast is primary.")


def cmd_materialize(config):
    """Push features to Feast (primary) and optionally Hopsworks."""
    cmd_features(config)


def cmd_feast(config):
    """Feast-specific operations: status, manifest, health check."""
    from src.feature_store.feast_integration import FeastIntegration
    
    feast = FeastIntegration(repo_path="feature_store")
    
    logger.info("▶ Feast Feature Store Operations")
    logger.info("=" * 60)
    
    # Health check
    health = feast.health_check()
    logger.info(f"Health Status: {health['overall_status'].upper()}")
    logger.info(f"  - Store: {health['store_health']['status']}")
    logger.info(f"  - Registered feature views: {health['registry_stats']['num_registered_views']}")
    logger.info(f"  - Materialized: {health['latest_materialization']}")
    
    # Feature statistics
    logger.info("\n▶ Feature Registry Statistics")
    logger.info(f"  - Total features: {health['registry_stats']['total_features']}")
    for category, count in health['registry_stats']['by_category'].items():
        logger.info(f"    {category}: {count}")
    
    # Feature quality report
    quality = feast.feature_quality_report()
    if quality.get("status") != "no_data":
        logger.info("\n▶ Feature Quality Report")
        logger.info(f"  - Total features: {quality['total_features']}")
        logger.info(f"  - Records: {quality['total_records']}")
        
        missing = quality['quality_checks']['missing_values']
        if missing['columns_with_missing']:
            logger.warning(f"  ⚠ Columns with missing values: {len(missing['columns_with_missing'])}")
        logger.info(f"  - Max missing %: {missing['max_missing_pct']:.2f}%")
        
        freshness = quality['quality_checks']['freshness']
        if freshness.get('is_fresh'):
            logger.info(f"  ✓ Data is fresh (age: {freshness.get('age_hours', 0):.1f}h)")
        else:
            logger.warning(f"  ⚠ Data is stale (age: {freshness.get('age_hours', 0):.1f}h)")
    
    logger.info("\nℹ Feast is now your primary feature store (local Parquet backend)")
    logger.info("  - For cloud deployment: Feast supports S3, GCS, Azure, Snowflake, BigQuery")
    logger.info("  - Feature retrieval: FeastIntegration.get_features_for_training()")
    logger.info("  - Inference: FeastIntegration.get_features_for_inference()")



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
    "feast": cmd_feast,
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
  features      Push engineered features to Feast (primary) + Hopsworks (optional)
  materialize   Alias for features (backward compatibility)
  feast         Feast feature store status, quality reports, health checks
  train         Train forecasting models (Ridge, RF, XGBoost × 3 horizons)
  explain       Generate SHAP model explanations
  drift         Run distribution drift detection
  serve         Start FastAPI backend server (:8000)
  dashboard     Launch Streamlit dashboard

Examples:
  python run.py ingest                      # Fetch real data from APIs
  python run.py features                    # Push to Feast + optional Hopsworks
  python run.py feast                       # Feature store status & health
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
