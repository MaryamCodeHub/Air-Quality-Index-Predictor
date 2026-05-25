"""
FastAPI Application — Islamabad AQI System
============================================
Main application entry point with CORS and route registration.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router

app = FastAPI(
    title="AQI Intelligent Forecasting — Islamabad",
    description=(
        "Production-grade AQI forecasting, health advisory, and MLOps system "
        "for Islamabad, Pakistan. Powered by AQICN + Open-Meteo data, "
        "Feast feature store, and XGBoost/RF/Ridge models."
    ),
    version="1.0.0",
)

# CORS — allow Streamlit dashboard to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router, prefix="/api/v1", tags=["AQI Forecasting"])


@app.get("/")
def root():
    return {
        "service": "AQI Intelligent Forecasting — Islamabad",
        "version": "1.0.0",
        "docs": "/docs",
    }
