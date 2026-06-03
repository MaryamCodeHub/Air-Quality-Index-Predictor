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
from pathlib import Path

# Ensure project root is cwd and on sys.path (CLI may be invoked from any directory)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_PROJECT_ROOT)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.helpers import load_config, ensure_directories
from src.utils.logger import setup_logger

logger = setup_logger("run")


def cmd_ingest(config):
    """Fetch AQI + weather data → clean → engineer features."""
    from src.ingestion.api_client import DataIngestionPipeline
    from src.ingestion.data_cleaner import run_cleaning_pipeline
    from src.feature_engineering.feature_engineer import run_feature_pipeline

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

    # ===== FEAST AUTO-MATERIALIZATION =====
    try:
        logger.info("▶ Auto-materializing features to Feast feature store...")
        from src.feature_store.feast_integration import FeastIntegration
        feast = FeastIntegration(repo_path="feature_store")
        
        # Split raw vs engineered features
        raw_cols = {"timestamp", "aqi", "pm25", "pm10", "temp", "humidity", "wind_speed"}
        raw_df = featured_df[[col for col in featured_df.columns if col in raw_cols]]
        
        # Ingest and register all features (runs apply + materialize)
        feast.ingest_and_register(raw_df=raw_df, engineered_df=featured_df)
        logger.info("✅ Feast auto-materialization complete")
    except Exception as e:
        logger.error(f"Feast auto-materialization failed: {e}")

    # ===== AUTO-TRAINING =====
    try:
        min_samples = config["training"].get("min_samples_required", 50)
        if len(featured_df) >= min_samples:
            logger.info(f"▶ Sufficient data available ({len(featured_df)} >= {min_samples}). Auto-triggering model training...")
            cmd_train(config)
        else:
            logger.info(f"ℹ Skipping auto-training: {len(featured_df)} records (need >= {min_samples})")
    except Exception as e:
        logger.error(f"Auto-training failed: {e}")


def cmd_features(config):
    """Push engineered features to Feast (primary store)."""
    import pandas as pd
    from src.feature_store.feast_integration import FeastIntegration
    
    # Load processed data
    proc_path = os.path.join(config["paths"]["processed_data"], "processed_aqi_data.parquet")
    if not os.path.exists(proc_path):
        logger.error("No processed data. Run 'python run.py ingest' first.")
        return
    
    df = pd.read_parquet(proc_path)
    logger.info(f"Loaded {len(df)} processed records for feature store")
    
    # ===== FEAST INTEGRATION (Primary Feature Store) =====
    try:
        logger.info("▶ Pushing features to Feast feature store (primary)...")
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
        logger.info("ℹ Features successfully pushed to Feast")
        
    except Exception as e:
        logger.error(f"Feast integration error: {e}")
        logger.info("ℹ Parquet fallback will be used for training/inference")
        return


def cmd_materialize(config):
    """Materialize features to Feast (alias for features command)."""
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
  python pipelines/run.py ingest
  python pipelines/run.py features
  python pipelines/run.py train
        """,
    )
    parser.add_argument("command", choices=COMMANDS.keys())
    parser.add_argument("--config", default="configs/config.yaml")
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
