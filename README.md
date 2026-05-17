# AQI Intelligent Forecasting & Health Advisory System

A **production-grade MLOps system** for air quality forecasting, combining:
- **Feast** feature store (Parquet offline / SQLite online)
- **Multi-model forecasting** (Ridge, Random Forest, XGBoost) at 24h/48h/72h horizons
- **SHAP explainability** for model transparency
- **Distribution drift detection** with automated alerting
- **Health advisory engine** with AQI-based recommendations
- **FastAPI** REST backend + **Streamlit** interactive dashboard

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
Edit `.env` (or config/config.yaml) and set:
- **AQICN API key** — free at https://aqicn.org/data-platform/token/
*(Note: Weather features are fetched from Open-Meteo, which is free and requires no API key)*

### 3. Run Data Ingestion
```bash
python run.py ingest
```

### 4. Train Models
```bash
python run.py train
```

### 5. Start API Server
```bash
python run.py serve
```

### 6. Launch Dashboard
```bash
python run.py dashboard
```

## Project Structure
```
AQI/
├── config/config.yaml           # All system settings
├── data/
│   ├── raw/                     # Raw API data (Parquet)
│   ├── processed/               # Cleaned features (Parquet)
│   ├── feast/                   # Feast offline store
│   └── cache/                   # API response cache
├── feature_store/               # Feast definitions
├── src/
│   ├── ingestion/               # API clients, cleaning, features
│   ├── training/                # Model training pipeline
│   ├── intelligence/            # SHAP, drift, health, alerts
│   ├── api/                     # FastAPI backend
│   ├── dashboard/               # Streamlit UI
│   └── utils/                   # Logging, helpers
├── models/                      # Saved models + metadata
├── logs/                        # Structured log files
├── plots/                       # SHAP plots, charts
├── tests/                       # Test suite
├── run.py                       # CLI entry point
└── requirements.txt
```

## Commands
| Command | Description |
|---------|-------------|
| `python run.py ingest` | Fetch + clean + engineer features |
| `python run.py train` | Train forecasting models |
| `python run.py explain` | Generate SHAP explanations |
| `python run.py drift` | Run drift detection |
| `python run.py materialize` | Sync Feast feature store |
| `python run.py serve` | Start FastAPI backend |
| `python run.py dashboard` | Launch Streamlit dashboard |

## Tech Stack
Python · Feast · FastAPI · Streamlit · Scikit-learn · XGBoost · SHAP · Pandas · Plotly · SQLite · Parquet
