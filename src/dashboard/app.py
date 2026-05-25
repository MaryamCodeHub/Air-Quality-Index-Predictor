"""
Streamlit Dashboard — Islamabad AQI Forecasting System
========================================================
Interactive UI for AQI monitoring, forecasting, and health advisories.
Decoupled version calling the FastAPI serving layer.

Run backend first: python run.py serve
Run: streamlit run src/dashboard/app.py
"""

import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import requests

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.helpers import load_config

# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="AQI Forecasting — Islamabad",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CONFIG = load_config()
API_BASE_URL = "http://localhost:8000/api/v1"

# ============================================================
# Custom CSS
# ============================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        padding: 0.5rem 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .aqi-good { background: linear-gradient(135deg, #11998e, #38ef7d); }
    .aqi-moderate { background: linear-gradient(135deg, #F2994A, #F2C94C); }
    .aqi-unhealthy-sg { background: linear-gradient(135deg, #eb3349, #f45c43); }
    .aqi-unhealthy { background: linear-gradient(135deg, #c0392b, #e74c3c); }
    .aqi-very-unhealthy { background: linear-gradient(135deg, #6c3483, #8e44ad); }
    .aqi-hazardous { background: linear-gradient(135deg, #641E16, #922B21); }
    .drift-ok { color: #27ae60; font-weight: bold; }
    .drift-warn { color: #f39c12; font-weight: bold; }
    .drift-bad { color: #e74c3c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# API Helper Functions
# ============================================================

def api_get(endpoint: str):
    """Safely make a GET request to the FastAPI serving layer."""
    try:
        resp = requests.get(f"{API_BASE_URL}/{endpoint}", timeout=10)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 404:
            st.warning(f"Endpoint '{endpoint}' returned 404: Not Found.")
            return None
        else:
            st.error(f"Error {resp.status_code} from API: {resp.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Connection Error: Cannot connect to the FastAPI backend. Ensure you run `python run.py serve` first.")
        return None
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        return None


def api_post(endpoint: str, json_data: dict = None):
    """Safely make a POST request to the FastAPI serving layer."""
    try:
        resp = requests.post(f"{API_BASE_URL}/{endpoint}", json=json_data, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"Error {resp.status_code} from API: {resp.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Connection Error: Cannot connect to the FastAPI backend. Ensure you run `python run.py serve` first.")
        return None
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        return None


def get_aqi_css_class(level: str) -> str:
    mapping = {
        "Good": "aqi-good",
        "Moderate": "aqi-moderate",
        "Unhealthy for Sensitive Groups": "aqi-unhealthy-sg",
        "Unhealthy": "aqi-unhealthy",
        "Very Unhealthy": "aqi-very-unhealthy",
        "Hazardous": "aqi-hazardous",
    }
    return mapping.get(level, "metric-card")


def fetch_history_dfs():
    """Fetch raw and processed historical records from the API."""
    data = api_get("history")
    if data:
        raw_df = pd.DataFrame(data.get("raw", []))
        proc_df = pd.DataFrame(data.get("processed", []))
        return raw_df, proc_df
    return pd.DataFrame(), pd.DataFrame()


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/Flag_of_Pakistan.svg/200px-Flag_of_Pakistan.svg.png", width=60)
    st.title("🌫️ AQI System")
    st.markdown("**Islamabad, Pakistan**")
    st.divider()

    section = st.radio(
        "Navigate",
        ["📊 Dashboard", "🔮 Forecast", "📈 Model Metrics", "🔍 SHAP Explanations",
         "📉 Drift Status", "🏥 Health Advisory", "⚙️ Retrain"],
    )
    st.divider()
    st.caption("AQI Intelligent Forecasting v1.0")


# ============================================================
# Section: Dashboard
# ============================================================

if section == "📊 Dashboard":
    st.markdown('<div class="main-header">🌫️ Islamabad Air Quality Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time AQI monitoring powered by AQICN & Open-Meteo</div>', unsafe_allow_html=True)

    # Call /health-advice for latest AQI & health guidance
    advisory_data = api_get("health-advice")
    raw_df, _ = fetch_history_dfs()

    if advisory_data and not raw_df.empty:
        latest = raw_df.iloc[-1]

        # Top metrics row
        cols = st.columns(4)
        with cols[0]:
            css = get_aqi_css_class(advisory_data.get("level", "Good"))
            st.markdown(f'<div class="metric-card {css}"><h2>{advisory_data.get("current_aqi", "N/A")}</h2><p>Current AQI</p></div>', unsafe_allow_html=True)
        with cols[1]:
            st.metric("🌡️ Temperature", f"{latest.get('temperature', 'N/A')}°C")
        with cols[2]:
            st.metric("💧 Humidity", f"{latest.get('humidity', 'N/A')}%")
        with cols[3]:
            st.metric("💨 Wind Speed", f"{latest.get('wind_speed', 'N/A')} m/s")

        st.info(f"**{advisory_data.get('level')}**: {advisory_data.get('advice')}")

        # AQI History chart
        if "timestamp" in raw_df.columns and "aqi" in raw_df.columns:
            raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], errors="coerce")
            chart_df = raw_df.dropna(subset=["timestamp", "aqi"]).sort_values("timestamp")
            if not chart_df.empty:
                fig = px.line(chart_df, x="timestamp", y="aqi",
                              title="AQI History — Islamabad",
                              labels={"aqi": "AQI", "timestamp": "Time"})
                fig.update_layout(hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

        # Pollutant breakdown
        pollutants = ["pm25", "pm10", "o3", "no2", "so2", "co"]
        poll_vals = {p: latest.get(p) for p in pollutants if latest.get(p) is not None}
        if poll_vals:
            fig = go.Figure(data=[go.Bar(x=list(poll_vals.keys()), y=list(poll_vals.values()),
                                         marker_color=["#e74c3c", "#e67e22", "#3498db", "#9b59b6", "#1abc9c", "#95a5a6"])])
            fig.update_layout(title="Latest Pollutant Readings", yaxis_title="Sub-Index")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Ensure the API backend is running (`python run.py serve`) and raw data has been ingested (`python run.py ingest`).")


# ============================================================
# Section: Forecast
# ============================================================

elif section == "🔮 Forecast":
    st.markdown('<div class="main-header">🔮 AQI Forecast — Islamabad</div>', unsafe_allow_html=True)

    horizons = CONFIG["training"]["forecast_horizons"]
    predictions = {}

    for h in horizons:
        pred_data = api_post("predict", json_data={"horizon": h})
        if pred_data:
            predictions[h] = pred_data

    if predictions:
        cols = st.columns(len(predictions))
        for i, (h, pred_val) in enumerate(predictions.items()):
            aqi = pred_val.get("predicted_aqi", 0.0)
            advisory = pred_val.get("health_advisory", {})
            level = advisory.get("level", "Good")
            with cols[i]:
                css = get_aqi_css_class(level)
                st.markdown(f'<div class="metric-card {css}"><h2>{aqi:.0f}</h2><p>{h}h Forecast</p></div>', unsafe_allow_html=True)
                st.caption(level)

        # Forecast chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[f"{h}h" for h in predictions.keys()],
                                  y=[p.get("predicted_aqi", 0.0) for p in predictions.values()],
                                  mode="lines+markers", name="Predicted AQI",
                                  line=dict(width=3, color="#667eea")))
        fig.update_layout(title="AQI Forecast Trend", yaxis_title="AQI", xaxis_title="Horizon")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Could not fetch predictions. Ensure models are trained and API server is running.")


# ============================================================
# Section: Model Metrics
# ============================================================

elif section == "📈 Model Metrics":
    st.markdown('<div class="main-header">📈 Model Performance Comparison</div>', unsafe_allow_html=True)

    metrics_data = api_get("metrics")

    if metrics_data and metrics_data.get("models"):
        rows = []
        for m in metrics_data["models"]:
            rows.append({
                "Model": m["model_name"],
                "Horizon": f"{m['horizon_hours']}h",
                "RMSE": round(m["metrics"]["rmse"], 4),
                "MAE": round(m["metrics"]["mae"], 4),
                "R²": round(m["metrics"]["r2"], 4),
                "Trained": m["trained_at"][:19],
            })
        metrics_df = pd.DataFrame(rows)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        # Visual comparison
        fig = px.bar(metrics_df, x="Model", y="RMSE", color="Horizon", barmode="group",
                     title="RMSE Comparison Across Models & Horizons")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No metrics available. Make sure models have been trained.")


# ============================================================
# Section: SHAP Explanations
# ============================================================

elif section == "🔍 SHAP Explanations":
    st.markdown('<div class="main-header">🔍 Model Explainability (SHAP)</div>', unsafe_allow_html=True)

    plots_dir = CONFIG["paths"]["plots"]
    horizons = CONFIG["training"]["forecast_horizons"]

    found = False
    for h in horizons:
        summary = os.path.join(plots_dir, f"shap_summary_{h}h.png")
        bar = os.path.join(plots_dir, f"shap_bar_{h}h.png")

        if os.path.exists(summary) or os.path.exists(bar):
            found = True
            st.subheader(f"{h}h Forecast Horizon")
            cols = st.columns(2)
            if os.path.exists(summary):
                with cols[0]:
                    st.image(summary, caption=f"SHAP Summary — {h}h")
            if os.path.exists(bar):
                with cols[1]:
                    st.image(bar, caption=f"Feature Importance — {h}h")
            st.divider()

    if not found:
        st.warning("No SHAP plots found. Run `python run.py explain` to generate them.")


# ============================================================
# Section: Drift Status
# ============================================================

elif section == "📉 Drift Status":
    st.markdown('<div class="main-header">📉 Feature Drift Monitor</div>', unsafe_allow_html=True)

    drift_data = api_get("drift-status")
    drift_history = api_get("drift-history")

    if drift_data and drift_data.get("status") != "no_data":
        status = drift_data.get("status", "unknown")

        if status == "no_drift":
            st.success("✅ No significant drift detected")
        elif status == "moderate_drift":
            st.warning(f"⚠️ Moderate drift — {drift_data.get('drifted_features', 0)} features drifted")
        elif status == "high_drift":
            st.error(f"🚨 High drift — {drift_data.get('drifted_features', 0)} features drifted. Consider retraining.")
        else:
            st.info(f"Status: {status}")

        st.metric("Drift Ratio", f"{drift_data.get('drift_ratio', 0.0):.1%}")
        st.metric("Features Checked", drift_data.get("total_features", 0))

        if drift_data.get("drifted_feature_names"):
            st.write("**Drifted features:**", ", ".join(drift_data["drifted_feature_names"]))

        # Drift history chart
        if drift_history and len(drift_history) > 1:
            hist_df = pd.DataFrame(drift_history)
            hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"], errors="coerce")
            fig = px.line(hist_df, x="timestamp", y="drift_ratio",
                          title="Drift Ratio Over Time", markers=True)
            fig.add_hline(y=0.3, line_dash="dash", line_color="red", annotation_text="High drift threshold")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No drift history available. Run `python run.py drift` on the server.")


# ============================================================
# Section: Health Advisory
# ============================================================

elif section == "🏥 Health Advisory":
    st.markdown('<div class="main-header">🏥 Health Advisory — Islamabad</div>', unsafe_allow_html=True)

    advisory_data = api_get("health-advice")

    if advisory_data:
        color_map = {"green": "#27ae60", "yellow": "#f1c40f", "orange": "#e67e22",
                     "red": "#e74c3c", "purple": "#8e44ad", "maroon": "#641E16", "gray": "#95a5a6"}
        bg = color_map.get(advisory_data.get("color"), "#333")

        st.markdown(f"""
        <div style="background: {bg}; color: white; padding: 2rem; border-radius: 16px; text-align: center; margin: 1rem 0;">
            <h1 style="margin:0;">AQI: {advisory_data.get('current_aqi') or 'N/A'}</h1>
            <h2 style="margin:0.5rem 0;">{advisory_data.get('level')}</h2>
            <p style="font-size: 1.1rem;">{advisory_data.get('advice')}</p>
        </div>
        """, unsafe_allow_html=True)

        # Show all advisory levels for reference
        st.subheader("📋 AQI Reference Table")
        for level in CONFIG["health_advisory"]["levels"]:
            lo, hi = level["range"]
            st.markdown(f"- **{lo}–{hi}** — {level['level']}: {level['advice']}")
    else:
        st.warning("No AQI data available. Make sure the backend is active.")


# ============================================================
# Section: Retrain
# ============================================================

elif section == "⚙️ Retrain":
    st.markdown('<div class="main-header">⚙️ Model Retraining</div>', unsafe_allow_html=True)

    st.info("Click below to retrain all models with the latest data.")

    if st.button("🔄 Retrain Models", type="primary", use_container_width=True):
        with st.spinner("Training models — this may take a minute …"):
            retrain_result = api_post("retrain")
            if retrain_result and retrain_result.get("status") == "success":
                st.success(f"✅ {retrain_result.get('message')}")
            else:
                st.error("Retraining failed. Check backend logs.")
