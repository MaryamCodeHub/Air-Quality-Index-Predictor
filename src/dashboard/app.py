"""
Streamlit Dashboard — Islamabad AQI Forecasting System
========================================================
Interactive UI for AQI monitoring, forecasting, health advisories, model metrics,
SHAP explainability, drift monitoring, and retraining.

Run backend first:
    python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000

Run dashboard:
    python -m streamlit run src/dashboard/app.py
"""

import os
import sys
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

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

# Prefer 127.0.0.1 for local testing. It avoids browser/system confusion with 0.0.0.0.
API_BASE_URL = os.getenv("AQI_API_BASE_URL", "http://127.0.0.1:8000/api/v1")


# ============================================================
# Custom CSS
# ============================================================

st.markdown(
    """
<style>
    /* Main page headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #F8FAFC;
        text-align: center;
        padding: 0.4rem 0 0.2rem 0;
        letter-spacing: -0.5px;
    }

    .sub-header {
        text-align: center;
        color: #CBD5E1;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    /* Cards */
    .metric-card {
        padding: 2rem 1.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 12px 32px rgba(0,0,0,0.28);
        border: 1px solid rgba(255,255,255,0.14);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        min-height: 170px;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 18px 45px rgba(0,0,0,0.36);
    }

    .metric-card h2 {
        font-size: 3rem;
        font-weight: 800;
        margin: 0 0 0.6rem 0;
    }

    .metric-card p {
        font-size: 1.05rem;
        font-weight: 600;
        margin: 0;
        opacity: 0.96;
    }

    /* AQI category colors */
    .aqi-good {
        background: linear-gradient(135deg, #11998e, #38ef7d);
    }

    .aqi-moderate {
        background: linear-gradient(135deg, #F2994A, #F2C94C);
    }

    .aqi-unhealthy-sg {
        background: linear-gradient(135deg, #F7971E, #FFD200);
    }

    .aqi-unhealthy {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
    }

    .aqi-very-unhealthy {
        background: linear-gradient(135deg, #6c3483, #8e44ad);
    }

    .aqi-hazardous {
        background: linear-gradient(135deg, #641E16, #922B21);
    }

    /* Status styles */
    .drift-ok {
        color: #27ae60;
        font-weight: bold;
    }

    .drift-warn {
        color: #f39c12;
        font-weight: bold;
    }

    .drift-bad {
        color: #e74c3c;
        font-weight: bold;
    }

    /* Sidebar polish */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] h1 {
        font-size: 1.6rem;
        font-weight: 800;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Utility Helpers
# ============================================================

def safe_float(value: Any) -> Optional[float]:
    """Convert a value to float only if it is valid and non-null."""
    try:
        converted = float(value)
        if pd.notna(converted):
            return converted
    except (TypeError, ValueError):
        return None
    return None


def format_number(value: Any, decimals: int = 1) -> str:
    """Format numbers safely for dashboard display."""
    numeric = safe_float(value)
    if numeric is None:
        return "N/A"

    if abs(numeric - round(numeric)) < 1e-9:
        return str(int(round(numeric)))

    return f"{numeric:.{decimals}f}"


def format_datetime(value: Any) -> str:
    """Format datetime-like values safely."""
    if value is None or pd.isna(value):
        return "N/A"
    try:
        return str(value)[:19]
    except Exception:
        return "N/A"


# ============================================================
# API Helper Functions
# ============================================================

def api_get(endpoint: str):
    """Safely make a GET request to the FastAPI serving layer."""
    try:
        resp = requests.get(f"{API_BASE_URL}/{endpoint}", timeout=60)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 404:
            st.warning(f"Endpoint '{endpoint}' returned 404: Not Found.")
            return None

        st.error(f"Error {resp.status_code} from API: {resp.text}")
        return None

    except requests.exceptions.ConnectionError:
        st.error(
            "Connection Error: Cannot connect to the FastAPI backend. "
            "Run `python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000` first."
        )
        return None
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        return None


def api_post(endpoint: str, json_data: dict = None):
    """Safely make a POST request to the FastAPI serving layer."""
    try:
        # Retraining takes longer than normal prediction requests.
        timeout_seconds = 180 if endpoint == "retrain" else 60

        resp = requests.post(
            f"{API_BASE_URL}/{endpoint}",
            json=json_data,
            timeout=timeout_seconds,
        )

        if resp.status_code == 200:
            return resp.json()

        st.error(f"Error {resp.status_code} from API: {resp.text}")
        return None

    except requests.exceptions.ConnectionError:
        st.error(
            "Connection Error: Cannot connect to the FastAPI backend. "
            "Run `python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000` first."
        )
        return None
    except requests.exceptions.ReadTimeout:
        st.error(
            "Request timed out while retraining. The backend may still be training. "
            "Check the backend terminal logs, then refresh the dashboard."
        )
        return None
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        return None


def get_aqi_css_class(level: str) -> str:
    """Map AQI health level to CSS class."""
    mapping = {
        "Good": "aqi-good",
        "Moderate": "aqi-moderate",
        "Unhealthy for Sensitive Groups": "aqi-unhealthy-sg",
        "Unhealthy": "aqi-unhealthy",
        "Very Unhealthy": "aqi-very-unhealthy",
        "Hazardous": "aqi-hazardous",
    }
    return mapping.get(level, "metric-card")


def fetch_history_dfs() -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    st.markdown(
        """
        <div style="
            width: 60px;
            height: 60px;
            border-radius: 20px;
            background: linear-gradient(135deg, #11998e, #38ef7d);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
            font-weight: 800;
            color: white;
            margin-bottom: 0.8rem;
            box-shadow: 0 8px 22px rgba(0,0,0,0.25);
        ">
            AQI
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.title("AQI System")
    st.markdown("**Islamabad, Pakistan**")
    st.divider()

    section = st.radio(
        "Navigate",
        [
            "Dashboard",
            "Forecast",
            "Model Metrics",
            "SHAP Explanations",
            "Drift Status",
            "Health Advisory",
            "Retrain",
        ],
    )

    st.divider()
    st.caption("AQI Intelligent Forecasting v1.0")


# ============================================================
# Section: Dashboard
# ============================================================

if section == "Dashboard":
    st.markdown(
        '<div class="main-header">Islamabad Air Quality Dashboard</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Real-time AQI monitoring powered by AQICN and Open-Meteo</div>',
        unsafe_allow_html=True,
    )

    current_data = api_get("current")
    raw_df, _ = fetch_history_dfs()

    if current_data:
        cols = st.columns(4)

        with cols[0]:
            css = get_aqi_css_class(current_data.get("category", "Good"))
            current_aqi = format_number(current_data.get("current_aqi"), decimals=1)
            st.markdown(
                f"""
                <div class="metric-card {css}">
                    <h2>{current_aqi}</h2>
                    <p>Current AQI</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with cols[1]:
            st.metric("Temperature", f"{format_number(current_data.get('temperature'), 1)}°C")

        with cols[2]:
            st.metric("Humidity", f"{format_number(current_data.get('humidity'), 1)}%")

        with cols[3]:
            st.metric("Wind Speed", f"{format_number(current_data.get('wind_speed'), 1)} m/s")

        st.caption(
            f"Source: {current_data.get('aqi_source', 'AQICN')} + "
            f"{current_data.get('weather_source', 'Open-Meteo')}"
        )
        st.caption(f"Last updated: {format_datetime(current_data.get('last_updated'))}")

        if current_data.get("is_stale"):
            st.warning("AQI data may be stale. Latest live fetch failed.")

        st.info(f"**{current_data.get('category', 'Unknown')}**")

        # AQI History chart
        if "timestamp" in raw_df.columns and "aqi" in raw_df.columns:
            raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], errors="coerce")
            raw_df["aqi"] = pd.to_numeric(raw_df["aqi"], errors="coerce")
            chart_df = raw_df.dropna(subset=["timestamp", "aqi"]).sort_values("timestamp")

            if not chart_df.empty:
                fig = px.line(
                    chart_df,
                    x="timestamp",
                    y="aqi",
                    title="AQI History — Islamabad",
                    labels={"aqi": "AQI", "timestamp": "Time"},
                )
                fig.update_layout(hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("AQI history is not available for plotting.")

        # Pollutant breakdown
        # The latest row can sometimes contain missing pollutant values while
        # AQI and weather are available. For a better dashboard experience,
        # use the latest historical row that has at least one valid pollutant.
        pollutants = ["pm25", "pm10", "o3", "no2", "so2", "co"]
        available_pollutants = [p for p in pollutants if p in raw_df.columns]

        if available_pollutants:
            pollutant_df = raw_df.copy()

            for pollutant in available_pollutants:
                pollutant_df[pollutant] = pd.to_numeric(
                    pollutant_df[pollutant],
                    errors="coerce",
                )

            pollutant_df = pollutant_df.dropna(
                subset=available_pollutants,
                how="all",
            )

            if not pollutant_df.empty:
                latest_pollutant_row = pollutant_df.iloc[-1]

                poll_vals: Dict[str, float] = {}
                for pollutant in available_pollutants:
                    value = safe_float(latest_pollutant_row.get(pollutant))
                    if value is not None:
                        poll_vals[pollutant.upper()] = value

                if poll_vals:
                    pollutant_colors = [
                        "#e74c3c",
                        "#e67e22",
                        "#3498db",
                        "#9b59b6",
                        "#1abc9c",
                        "#95a5a6",
                    ]

                    fig = go.Figure(
                        data=[
                            go.Bar(
                                x=list(poll_vals.keys()),
                                y=list(poll_vals.values()),
                                marker_color=pollutant_colors[: len(poll_vals)],
                                text=[f"{v:.1f}" for v in poll_vals.values()],
                                textposition="auto",
                            )
                        ]
                    )

                    max_val = max(poll_vals.values()) if poll_vals else 1
                    y_upper = max(max_val * 1.25, 1)

                    fig.update_layout(
                        title="Latest Available Pollutant Readings",
                        yaxis_title="Pollutant Value / Sub-Index",
                        xaxis_title="Pollutant",
                        bargap=0.35,
                        showlegend=False,
                    )
                    fig.update_yaxes(range=[0, y_upper])

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Pollutant readings are not available in the latest records.")
            else:
                st.info("Pollutant readings are not available in the current dataset.")
        else:
            st.info("Pollutant columns are not available in the dataset.")

    else:
        st.warning(
            "Ensure the API backend is running and raw data has been ingested."
        )


# ============================================================
# Section: Forecast
# ============================================================

elif section == "Forecast":
    st.markdown(
        '<div class="main-header">AQI Forecast — Islamabad</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">24-hour, 48-hour, and 72-hour AQI prediction using trained ML models</div>',
        unsafe_allow_html=True,
    )

    horizons = CONFIG["training"]["forecast_horizons"]
    predictions = {}

    for h in horizons:
        pred_data = api_post("predict", json_data={"horizon": h})
        if pred_data:
            predictions[h] = pred_data

    if predictions:
        cols = st.columns(len(predictions))

        for i, (h, pred_val) in enumerate(predictions.items()):
            aqi = safe_float(pred_val.get("predicted_aqi")) or 0.0
            advisory = pred_val.get("health_advisory", {})
            level = advisory.get("level", "Good")
            css = get_aqi_css_class(level)

            with cols[i]:
                st.markdown(
                    f"""
                    <div class="metric-card {css}">
                        <h2>{aqi:.0f}</h2>
                        <p>{h}h Forecast</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(level)

        # Forecast chart
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[f"{h}h" for h in predictions.keys()],
                y=[safe_float(p.get("predicted_aqi")) or 0.0 for p in predictions.values()],
                mode="lines+markers",
                name="Predicted AQI",
                line=dict(width=3, color="#667eea"),
            )
        )
        fig.update_layout(
            title="AQI Forecast Trend",
            yaxis_title="AQI",
            xaxis_title="Horizon",
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning(
            "Could not fetch predictions. Ensure models are trained and API server is running."
        )


# ============================================================
# Section: Model Metrics
# ============================================================

elif section == "Model Metrics":
    st.markdown(
        '<div class="main-header">Model Performance Comparison</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">RMSE, MAE, and R² comparison across forecast horizons</div>',
        unsafe_allow_html=True,
    )

    metrics_data = api_get("metrics")

    if metrics_data and metrics_data.get("models"):
        rows = []

        for m in metrics_data["models"]:
            metrics = m.get("metrics", {})
            rows.append(
                {
                    "Model": m.get("model_name", "unknown"),
                    "Horizon": f"{m.get('horizon_hours', 'N/A')}h",
                    "RMSE": round(safe_float(metrics.get("rmse")) or 0.0, 4),
                    "MAE": round(safe_float(metrics.get("mae")) or 0.0, 4),
                    "R²": round(safe_float(metrics.get("r2")) or 0.0, 4),
                    "Trained": format_datetime(m.get("trained_at")),
                }
            )

        metrics_df = pd.DataFrame(rows)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        fig = px.bar(
            metrics_df,
            x="Model",
            y="RMSE",
            color="Horizon",
            barmode="group",
            title="RMSE Comparison Across Models and Horizons",
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No metrics available. Make sure models have been trained.")


# ============================================================
# Section: SHAP Explanations
# ============================================================

elif section == "SHAP Explanations":
    st.markdown(
        '<div class="main-header">Model Explainability</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">SHAP-based feature importance for trained AQI models</div>',
        unsafe_allow_html=True,
    )

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
        st.warning("No SHAP plots found. Run `python pipelines/run.py explain` to generate them.")


# ============================================================
# Section: Drift Status
# ============================================================

elif section == "Drift Status":
    st.markdown(
        '<div class="main-header">Feature Drift Monitor</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Monitoring changes in feature distributions over time</div>',
        unsafe_allow_html=True,
    )

    drift_data = api_get("drift-status")
    drift_history = api_get("drift-history")

    if drift_data and drift_data.get("status") != "no_data":
        status = drift_data.get("status", "unknown")

        if status == "no_drift":
            st.success("No significant drift detected")
        elif status == "moderate_drift":
            st.warning(
                f"Moderate drift — {drift_data.get('drifted_features', 0)} features drifted"
            )
        elif status == "high_drift":
            st.error(
                f"High drift — {drift_data.get('drifted_features', 0)} features drifted. Consider retraining."
            )
        else:
            st.info(f"Status: {status}")

        cols = st.columns(2)
        with cols[0]:
            st.metric("Drift Ratio", f"{drift_data.get('drift_ratio', 0.0):.1%}")
        with cols[1]:
            st.metric("Features Checked", drift_data.get("total_features", 0))

        if drift_data.get("drifted_feature_names"):
            st.write(
                "**Drifted features:**",
                ", ".join(drift_data["drifted_feature_names"]),
            )

        if drift_history and len(drift_history) > 1:
            hist_df = pd.DataFrame(drift_history)
            hist_df["timestamp"] = pd.to_datetime(
                hist_df["timestamp"],
                errors="coerce",
            )
            hist_df["drift_ratio"] = pd.to_numeric(hist_df["drift_ratio"], errors="coerce")
            hist_df = hist_df.dropna(subset=["timestamp", "drift_ratio"])

            if not hist_df.empty:
                fig = px.line(
                    hist_df,
                    x="timestamp",
                    y="drift_ratio",
                    title="Drift Ratio Over Time",
                    markers=True,
                )
                fig.add_hline(
                    y=0.3,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="High drift threshold",
                )
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No drift history available. Run `python pipelines/run.py drift` on the server.")


# ============================================================
# Section: Health Advisory
# ============================================================

elif section == "Health Advisory":
    st.markdown(
        '<div class="main-header">Health Advisory — Islamabad</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">AQI-based health recommendations for outdoor activity</div>',
        unsafe_allow_html=True,
    )

    advisory_data = api_get("health-advice")

    if advisory_data:
        color_map = {
            "green": "#27ae60",
            "yellow": "#f1c40f",
            "orange": "#e67e22",
            "red": "#e74c3c",
            "purple": "#8e44ad",
            "maroon": "#641E16",
            "gray": "#95a5a6",
        }
        bg = color_map.get(advisory_data.get("color"), "#333")

        st.markdown(
            f"""
            <div style="background: {bg}; color: white; padding: 2rem; border-radius: 16px; text-align: center; margin: 1rem 0; box-shadow: 0 12px 32px rgba(0,0,0,0.25);">
                <h1 style="margin:0;">AQI: {format_number(advisory_data.get('current_aqi'), 1)}</h1>
                <h2 style="margin:0.5rem 0;">{advisory_data.get('level')}</h2>
                <p style="font-size: 1.1rem;">{advisory_data.get('advice')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("AQI Reference Table")

        for level in CONFIG["health_advisory"]["levels"]:
            lo, hi = level["range"]
            st.markdown(
                f"- **{lo}–{hi}** — {level['level']}: {level['advice']}"
            )

    else:
        st.warning("No AQI data available. Make sure the backend is active.")


# ============================================================
# Section: Retrain
# ============================================================

elif section == "Retrain":
    st.markdown(
        '<div class="main-header">Model Retraining</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Trigger model retraining with the latest available AQI data</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "This will retrain all forecasting models using the latest available AQI features. "
        "It may take 1–3 minutes."
    )

    if st.button("Retrain Models", type="primary", use_container_width=True):
        with st.spinner("Retraining models. Please wait, this may take 1–3 minutes ..."):
            retrain_result = api_post("retrain")

            if retrain_result and retrain_result.get("status") == "success":
                st.success(retrain_result.get("message"))
                st.info(
                    "Retraining is complete. To refresh SHAP visualizations, run "
                    "`python pipelines/run.py explain` and reload this dashboard."
                )
            else:
                st.error("Retraining failed or timed out. Check backend logs.")
