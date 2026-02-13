"""
LabPulse AI — Streamlit Dashboard
==================================
Predictive reagent demand management powered by RKI wastewater surveillance.
Built for Ganzimmun Diagnostics (Limbach Group).

Features:
- Multi-pathogen overview with sparklines
- Per-pathogen deep dive with correlation + burndown charts
- AI-powered insights via local Ollama
- ML forecasting (Prophet) with simple fallback
- PDF report export
- CSV upload for real lab data
- Regional hotspot map (Bundesland)
- Automated alerts (webhook / email)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime

from data_engine import (
    fetch_rki_raw,
    fetch_rki_wastewater,
    fetch_all_pathogens,
    get_available_pathogens,
    generate_lab_volume,
    build_forecast,
    AVG_REVENUE_PER_TEST,
    PATHOGEN_REAGENT_MAP,
    PATHOGEN_GROUPS,
    PATHOGEN_SCALE,
)
from modules.ollama_client import get_client as get_ollama_client
from modules.pdf_export import generate_report as generate_pdf_report
from modules.lab_data_merger import validate_csv, merge_with_synthetic, get_data_quality_summary
from modules.regional_aggregation import (
    aggregate_by_bundesland,
    create_scatter_map_data,
    BUNDESLAND_COORDS,
    GERMANY_CENTER,
)
from modules.ml_forecaster import LabVolumeForecaster
from modules.alert_engine import AlertManager, AlertRule
from modules.external_data import (
    fetch_grippeweb,
    fetch_are_konsultation,
    fetch_trends_for_pathogen,
    get_trends_limitations,
    PATHOGEN_SEARCH_TERMS,
)
from modules.signal_fusion import fuse_all_signals, SIGNAL_CONFIG

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LabPulse AI — Ganzimmun Edition",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design System CSS ────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg-primary: #0b0e17;
        --bg-card: #111827;
        --border-subtle: rgba(255,255,255,0.06);
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent-orange: #f77f00;
        --accent-blue: #3b82f6;
        --accent-green: #22c55e;
        --accent-red: #ef4444;
        --radius: 16px;
        --radius-sm: 10px;
    }

    .block-container {
        padding: 2rem 2.5rem 1rem 2.5rem;
        max-width: 1400px;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    div[data-testid="stMetric"] {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius);
        padding: 1.25rem 1.4rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3), 0 8px 24px rgba(0,0,0,0.15);
        transition: border-color 0.2s ease, transform 0.15s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(247, 127, 0, 0.3);
        transform: translateY(-1px);
    }
    div[data-testid="stMetric"] label {
        color: var(--text-muted) !important;
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 1.5rem !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
        font-size: 0.75rem !important;
    }

    .section-header {
        color: var(--text-primary);
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding-bottom: 0.6rem;
        margin: 2.5rem 0 1.2rem 0;
        border-bottom: none;
        position: relative;
    }
    .section-header::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0;
        width: 40px; height: 3px;
        background: var(--accent-orange);
        border-radius: 2px;
    }

    [data-testid="stSidebar"] {
        background: #0f1219;
        border-right: 1px solid var(--border-subtle);
    }
    [data-testid="stSidebar"] .stMarkdown h1 {
        color: var(--text-primary);
        font-size: 1.15rem;
        font-weight: 700;
    }

    .sidebar-label {
        color: var(--text-muted);
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.6rem;
    }

    .alert-critical {
        background: linear-gradient(135deg, rgba(239,68,68,0.12) 0%, rgba(239,68,68,0.06) 100%);
        border: 1px solid rgba(239,68,68,0.25);
        border-left: 4px solid var(--accent-red);
        border-radius: var(--radius-sm);
        padding: 1rem 1.25rem;
        margin: 0.75rem 0 1.5rem 0;
        color: #fca5a5;
        font-size: 0.88rem;
        line-height: 1.6;
    }
    .alert-info {
        background: linear-gradient(135deg, rgba(59,130,246,0.1) 0%, rgba(59,130,246,0.04) 100%);
        border: 1px solid rgba(59,130,246,0.2);
        border-left: 4px solid var(--accent-blue);
        border-radius: var(--radius-sm);
        padding: 1rem 1.25rem;
        margin: 0.75rem 0 1.5rem 0;
        color: #93c5fd;
        font-size: 0.88rem;
        line-height: 1.6;
    }
    .alert-ai {
        background: linear-gradient(135deg, rgba(168,85,247,0.1) 0%, rgba(168,85,247,0.04) 100%);
        border: 1px solid rgba(168,85,247,0.2);
        border-left: 4px solid #a855f7;
        border-radius: var(--radius-sm);
        padding: 1rem 1.25rem;
        margin: 0.75rem 0 1.5rem 0;
        color: #d8b4fe;
        font-size: 0.88rem;
        line-height: 1.6;
    }

    .chart-container {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius);
        padding: 0.5rem;
        margin-bottom: 0.5rem;
    }

    [data-testid="stDataFrame"] > div {
        border-radius: var(--radius-sm);
        border: 1px solid var(--border-subtle);
    }

    .footer-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 0 0.5rem 0;
        border-top: 1px solid var(--border-subtle);
        margin-top: 2rem;
        color: var(--text-muted);
        font-size: 0.72rem;
    }
    .footer-bar a { color: var(--accent-orange); text-decoration: none; }

    .app-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding-bottom: 0.5rem;
    }
    .app-header-icon {
        width: 44px; height: 44px;
        background: linear-gradient(135deg, var(--accent-orange), #e06c00);
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.4rem; flex-shrink: 0;
    }
    .app-header-text h1 {
        margin: 0; font-size: 1.35rem; font-weight: 700;
        color: var(--text-primary); letter-spacing: -0.02em; line-height: 1.2;
    }
    .app-header-text p {
        margin: 0; font-size: 0.82rem; color: var(--text-muted);
    }

    .sidebar-pill {
        display: inline-block;
        background: rgba(247,127,0,0.12);
        color: var(--accent-orange);
        font-size: 0.65rem; font-weight: 600;
        padding: 0.2rem 0.6rem; border-radius: 99px;
        letter-spacing: 0.04em; margin-left: 0.4rem;
    }

    .sparkline-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius);
        padding: 1rem 1.2rem;
        transition: border-color 0.2s ease, transform 0.15s ease;
        cursor: pointer;
    }
    .sparkline-card:hover {
        border-color: rgba(247, 127, 0, 0.3);
        transform: translateY(-2px);
    }
    .sparkline-title {
        color: var(--text-muted);
        font-size: 0.7rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.08em;
    }
    .sparkline-value {
        color: var(--text-primary);
        font-size: 1.3rem; font-weight: 700;
        margin: 0.3rem 0;
    }
    .sparkline-trend-up { color: var(--accent-red); font-size: 0.8rem; }
    .sparkline-trend-down { color: var(--accent-green); font-size: 0.8rem; }
    .sparkline-trend-flat { color: var(--text-muted); font-size: 0.8rem; }

    .confidence-badge {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 99px;
        font-size: 0.65rem;
        font-weight: 600;
    }
    .confidence-high { background: rgba(34,197,94,0.15); color: #22c55e; }
    .confidence-medium { background: rgba(247,127,0,0.15); color: #f77f00; }
    .confidence-low { background: rgba(239,68,68,0.15); color: #ef4444; }

    .data-quality {
        display: flex; gap: 0.5rem; align-items: center;
        padding: 0.4rem 0; font-size: 0.75rem;
    }
    .dq-real { color: var(--accent-green); }
    .dq-synthetic { color: var(--text-muted); }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Data Loading (cached) ───────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Lade RKI-Abwasserdaten …")
def load_raw():
    return fetch_rki_raw()


# ── Session State Defaults ───────────────────────────────────────────────────
if "uploaded_lab_data" not in st.session_state:
    st.session_state.uploaded_lab_data = None
if "alert_webhook_url" not in st.session_state:
    st.session_state.alert_webhook_url = ""

# Per-kit inventory — initialize with defaults
# Unique kits (some pathogens share the same kit)
UNIQUE_KITS = {
    "SARS-CoV-2 PCR Kit": {"cost": 45, "pathogens": ["SARS-CoV-2"], "default_stock": 5000, "unit": "Tests", "lieferzeit_tage": 5},
    "Influenza A/B PCR Panel": {"cost": 38, "pathogens": ["Influenza A", "Influenza B", "Influenza (gesamt)"], "default_stock": 3000, "unit": "Tests", "lieferzeit_tage": 3},
    "RSV PCR Kit": {"cost": 42, "pathogens": ["RSV"], "default_stock": 2000, "unit": "Tests", "lieferzeit_tage": 4},
}

if "kit_inventory" not in st.session_state:
    st.session_state.kit_inventory = {
        kit_name: info["default_stock"] for kit_name, info in UNIQUE_KITS.items()
    }

def get_kit_for_pathogen(pathogen: str) -> str:
    """Return the kit name for a given pathogen."""
    for kit_name, info in UNIQUE_KITS.items():
        if pathogen in info["pathogens"]:
            return kit_name
    return "SARS-CoV-2 PCR Kit"  # fallback

def get_stock_for_pathogen(pathogen: str) -> int:
    """Return current stock for the kit associated with a pathogen."""
    kit = get_kit_for_pathogen(pathogen)
    return st.session_state.kit_inventory.get(kit, 5000)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '# 🧬 LabPulse AI <span class="sidebar-pill">v2.0</span>',
        unsafe_allow_html=True,
    )
    st.caption("Ganzimmun Diagnostics · Limbach Group")

    st.markdown("")
    st.markdown('<div class="sidebar-label">Pathogen</div>', unsafe_allow_html=True)

    raw_df = load_raw()
    pathogens = get_available_pathogens(raw_df)
    selected_pathogen = st.selectbox(
        "Pathogen waehlen",
        pathogens,
        index=0,
        label_visibility="collapsed",
    )

    reagent_info = PATHOGEN_REAGENT_MAP.get(selected_pathogen, {})
    if reagent_info:
        st.caption(
            f"Kit: {reagent_info.get('test_name', '–')} · "
            f"EUR {reagent_info.get('cost_per_test', 45)}/Test"
        )

    st.markdown("")
    st.markdown('<div class="sidebar-label">Prognose</div>', unsafe_allow_html=True)
    forecast_horizon = st.slider(
        "Horizont (Tage)", min_value=7, max_value=21, value=14, step=7
    )
    safety_buffer = st.slider(
        "Sicherheitspuffer (%)", min_value=0, max_value=30, value=10, step=5
    )
    use_ml = st.toggle("ML-Prognose (Prophet)", value=False, help="Prophet Time-Series statt einfacher 14-Tage-Verschiebung")

    st.markdown("")
    st.markdown('<div class="sidebar-label">Simulation</div>', unsafe_allow_html=True)
    scenario_uplift = st.slider(
        "Viruslast-Uplift (%)",
        min_value=0, max_value=50, value=0, step=5,
        help="Simuliert einen ploetzlichen Anstieg.",
    )

    # Data import
    with st.expander("Datenimport & Einstellungen", expanded=False):
        st.markdown('<div class="sidebar-label">Reale Labordaten</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "CSV hochladen (date, test_volume)",
            type=["csv"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            is_valid, msg, real_df = validate_csv(uploaded_file)
            if is_valid:
                st.session_state.uploaded_lab_data = real_df
                st.success(msg)
            else:
                st.error(msg)

        st.markdown('<div class="sidebar-label">Benachrichtigungen</div>', unsafe_allow_html=True)
        alert_webhook = st.text_input(
            "Webhook-URL (Slack/Teams)",
            value=st.session_state.alert_webhook_url,
            placeholder="https://hooks.slack.com/...",
            label_visibility="collapsed",
        )
        st.session_state.alert_webhook_url = alert_webhook

    st.markdown("")
    refresh = st.button("Daten neu laden", use_container_width=True, type="secondary")
    if refresh:
        st.cache_data.clear()
        raw_df = load_raw()

    st.markdown("")
    st.caption(f"Letzter Sync: {datetime.now().strftime('%H:%M')} · RKI AMELAG")


# ── Load Pathogen-Specific Data ──────────────────────────────────────────────
stock_on_hand = get_stock_for_pathogen(selected_pathogen)
wastewater_df = fetch_rki_wastewater(raw_df, pathogen=selected_pathogen)
lab_df = generate_lab_volume(wastewater_df, lag_days=14, pathogen=selected_pathogen)

# Merge with real data if uploaded
if st.session_state.uploaded_lab_data is not None:
    lab_df = merge_with_synthetic(lab_df, st.session_state.uploaded_lab_data, selected_pathogen)

# ML Forecast
ml_model_info = None
if use_ml:
    forecaster = LabVolumeForecaster(lab_df, wastewater_df, selected_pathogen)
    forecaster.train()
    ml_model_info = forecaster.model_info
    ml_forecast = forecaster.forecast(periods=forecast_horizon)

# Standard forecast (always computed for table/KPIs)
forecast_df, kpis = build_forecast(
    lab_df,
    horizon_days=forecast_horizon,
    safety_buffer_pct=safety_buffer / 100,
    stock_on_hand=stock_on_hand,
    scenario_uplift_pct=scenario_uplift / 100,
    pathogen=selected_pathogen,
)

# Revenue column name
rev_col = [c for c in forecast_df.columns if "Revenue" in c]
rev_col_name = rev_col[0] if rev_col else "Est. Revenue"


# ── Alerts Evaluation ────────────────────────────────────────────────────────
alert_mgr = AlertManager()
triggered_alerts = alert_mgr.evaluate_all(kpis, ml_model_info)

if triggered_alerts and alert_webhook:
    alert_mgr.send_webhook(alert_webhook, triggered_alerts, selected_pathogen)


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="app-header">
        <div class="app-header-icon">🧬</div>
        <div class="app-header-text">
            <h1>LabPulse AI</h1>
            <p>Reagenz-Bedarfsprognose · RKI Abwassersurveillance · Ganzimmun Diagnostics</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_overview, tab_detail, tab_regional, tab_trends = st.tabs([
    "Uebersicht", "Pathogen-Analyse", "Regionale Analyse", "Google Trends"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW (Inventory Dashboard + Multi-Pathogen Cards)
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:

    # ── Inventory Dashboard ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">Bestandsverwaltung — Testkits</div>', unsafe_allow_html=True)
    st.caption("Aktuelle Lagerbestaende pro Testkit eingeben. Die Prognose passt sich automatisch an.")

    kit_cols = st.columns(len(UNIQUE_KITS), gap="medium")
    for idx, (kit_name, kit_info) in enumerate(UNIQUE_KITS.items()):
        with kit_cols[idx]:
            current_stock = st.session_state.kit_inventory.get(kit_name, kit_info["default_stock"])

            # Compute demand for this kit across all its pathogens
            kit_demand_7d = 0
            for p in kit_info["pathogens"]:
                try:
                    ww_tmp = fetch_rki_wastewater(raw_df, pathogen=p)
                    lab_tmp = generate_lab_volume(ww_tmp, lag_days=14, pathogen=p)
                    _, p_kpis_tmp = build_forecast(lab_tmp, pathogen=p, stock_on_hand=current_stock)
                    kit_demand_7d += p_kpis_tmp.get("predicted_tests_7d", 0)
                except Exception:
                    pass

            # Stock health
            days_coverage = int(current_stock / (kit_demand_7d / 7)) if kit_demand_7d > 0 else 99
            if days_coverage >= 14:
                stock_color = "#22c55e"
                stock_label = f"{days_coverage}d Reichweite"
            elif days_coverage >= 7:
                stock_color = "#f77f00"
                stock_label = f"{days_coverage}d Reichweite"
            else:
                stock_color = "#ef4444"
                stock_label = f"KRITISCH — {days_coverage}d"

            st.markdown(
                f'<div class="sparkline-card">'
                f'<div class="sparkline-title">{kit_name}</div>'
                f'<div style="color: var(--text-muted); font-size: 0.68rem; margin: 0.2rem 0;">'
                f'EUR {kit_info["cost"]}/Test · Lieferzeit: {kit_info["lieferzeit_tage"]} Tage'
                f'</div>'
                f'<div style="color: {stock_color}; font-size: 0.78rem; font-weight: 600; margin: 0.3rem 0;">'
                f'{stock_label}</div>'
                f'<div style="color: var(--text-muted); font-size: 0.68rem;">'
                f'Bedarf 7d: {kit_demand_7d:,} · Pathogene: {", ".join(kit_info["pathogens"])}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            new_stock = st.number_input(
                f"Bestand {kit_name}",
                min_value=0,
                value=current_stock,
                step=500,
                key=f"stock_{kit_name}",
                label_visibility="collapsed",
            )
            st.session_state.kit_inventory[kit_name] = new_stock

    st.markdown("")

    # ── Pathogen Overview Cards ───────────────────────────────────────────────
    st.markdown('<div class="section-header">Pathogen-Uebersicht</div>', unsafe_allow_html=True)

    overview_cols = st.columns(len(pathogens), gap="medium")

    for i, pathogen_name in enumerate(pathogens):
        with overview_cols[i]:
            try:
                p_stock = get_stock_for_pathogen(pathogen_name)
                ww = fetch_rki_wastewater(raw_df, pathogen=pathogen_name)
                lab = generate_lab_volume(ww, lag_days=14, pathogen=pathogen_name)
                _, p_kpis = build_forecast(lab, pathogen=pathogen_name, stock_on_hand=p_stock)

                trend = p_kpis.get("trend_pct", 0)
                if trend > 5:
                    trend_class = "sparkline-trend-up"
                    trend_icon = "↑"
                elif trend < -5:
                    trend_class = "sparkline-trend-down"
                    trend_icon = "↓"
                else:
                    trend_class = "sparkline-trend-flat"
                    trend_icon = "→"

                status = p_kpis.get("reagent_status", "N/A")
                pred_7d = p_kpis.get("predicted_tests_7d", 0)
                risk = p_kpis.get("risk_eur", 0)

                risk_html = ""
                if risk > 0:
                    risk_html = f'<div style="color: #ef4444; font-size: 0.7rem; margin-top: 0.2rem;">Risk: EUR {risk:,.0f}</div>'

                st.markdown(
                    f'<div class="sparkline-card">'
                    f'<div class="sparkline-title">{pathogen_name}</div>'
                    f'<div class="sparkline-value">{pred_7d:,}</div>'
                    f'<div style="color: var(--text-muted); font-size: 0.65rem;">Tests progn. (7d)</div>'
                    f'<div class="{trend_class}">{trend_icon} {trend:+.1f}% WoW</div>'
                    f'<div style="color: var(--text-muted); font-size: 0.7rem; margin-top: 0.3rem;">{status}</div>'
                    f'{risk_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Mini sparkline chart
                recent = lab.tail(14)
                if not recent.empty:
                    fig_spark = go.Figure(
                        go.Scatter(
                            x=recent["date"], y=recent["order_volume"],
                            mode="lines",
                            line=dict(color="#f77f00", width=2),
                            fill="tozeroy",
                            fillcolor="rgba(247,127,0,0.08)",
                        )
                    )
                    fig_spark.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=80,
                        margin=dict(l=0, r=0, t=0, b=0),
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_spark, use_container_width=True)

            except Exception as exc:
                st.caption(f"{pathogen_name}: Fehler ({exc})")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: PATHOGEN DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab_detail:

    # ── Alert Banners ────────────────────────────────────────────────────────
    if kpis.get("risk_eur", 0) > 0:
        stockout = kpis.get("stockout_day")
        stockout_str = (
            f" Lager erschoepft bis <strong>{stockout.strftime('%d. %b %Y')}</strong>."
            if stockout else ""
        )
        st.markdown(
            f'<div class="alert-critical">'
            f"<strong>KRITISCH</strong> — {selected_pathogen}: Revenue at Risk "
            f"<strong>EUR {kpis['risk_eur']:,.0f}</strong> "
            f"in den naechsten {forecast_horizon} Tagen.{stockout_str} "
            f"Sofortige Nachbestellung von {kpis.get('test_name', 'Reagenzien')} empfohlen."
            f"</div>",
            unsafe_allow_html=True,
        )
    elif scenario_uplift > 0:
        st.markdown(
            f'<div class="alert-info">'
            f"<strong>Simulation aktiv</strong> — {selected_pathogen}-Szenario mit "
            f"+{scenario_uplift}% Viruslast-Uplift."
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Data Quality Badge ───────────────────────────────────────────────────
    if st.session_state.uploaded_lab_data is not None:
        dq = get_data_quality_summary(lab_df)
        st.markdown(
            f'<div class="data-quality">'
            f'<span class="dq-real">● {dq.get("real_pct", 0)}% Realdaten</span>'
            f'<span class="dq-synthetic">● {dq.get("synthetic_pct", 100)}% Synthetisch</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── ML Model Badge ───────────────────────────────────────────────────────
    if use_ml and ml_model_info:
        conf = ml_model_info.get("confidence_score", 0)
        if conf >= 70:
            conf_class = "confidence-high"
        elif conf >= 50:
            conf_class = "confidence-medium"
        else:
            conf_class = "confidence-low"
        st.markdown(
            f'<span class="confidence-badge {conf_class}">'
            f'ML-Modell: {ml_model_info["model_type"]} · Konfidenz: {conf:.0f}%'
            f'</span>',
            unsafe_allow_html=True,
        )

    # ── Section 1: KPI Dashboard ─────────────────────────────────────────────
    st.markdown('<div class="section-header">Kennzahlen</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5, gap="medium")
    with c1:
        st.metric("Progn. Tests (7 d)", f"{kpis['predicted_tests_7d']:,}")
    with c2:
        st.metric("Umsatzprognose (7 d)", f"EUR {kpis['revenue_forecast_7d']:,.0f}")
    with c3:
        st.metric(
            "Reagenzstatus", kpis["reagent_status"],
            delta="Bestand OK" if "Optimal" in kpis["reagent_status"] else "Stockout-Risiko!",
            delta_color="normal" if "Optimal" in kpis["reagent_status"] else "inverse",
        )
    with c4:
        risk_val = kpis.get("risk_eur", 0)
        st.metric(
            "Revenue at Risk", f"EUR {risk_val:,.0f}",
            delta="Kein Risiko" if risk_val == 0 else "Handlungsbedarf!",
            delta_color="off" if risk_val == 0 else "inverse",
        )
    with c5:
        st.metric(
            "Woche-zu-Woche", f"{kpis['trend_pct']:+.1f}%",
            delta=f"{kpis['trend_pct']:+.1f}%",
            delta_color="normal" if kpis["trend_pct"] >= 0 else "inverse",
        )

    # ── Section 1b: Multi-Signal Confidence ─────────────────────────────────
    st.markdown('<div class="section-header">Signal-Konfidenz</div>', unsafe_allow_html=True)
    st.caption("Automatische Fusion aller Datenquellen — je mehr Signale uebereinstimmen, desto hoeher die Prognose-Konfidenz.")

    # Fetch surveillance data for fusion
    with st.spinner("Signale analysieren …"):
        _gw_fusion = fetch_grippeweb(erkrankung="ARE", region="Bundesweit")
        _are_fusion = fetch_are_konsultation(bundesland="Bundesweit")

    composite = fuse_all_signals(
        wastewater_df=wastewater_df,
        grippeweb_df=_gw_fusion,
        are_df=_are_fusion,
        trends_df=None,  # trends loaded lazily in own tab
    )

    # Confidence bar + signal indicators
    conf = composite.confidence_pct
    if conf >= 70:
        conf_color = "#22c55e"
        conf_label_de = "HOCH"
    elif conf >= 50:
        conf_color = "#f77f00"
        conf_label_de = "MITTEL"
    else:
        conf_color = "#ef4444"
        conf_label_de = "NIEDRIG"

    direction_icons = {"rising": "↑", "falling": "↓", "flat": "→", "mixed": "↔"}
    direction_labels = {"rising": "steigend", "falling": "fallend", "flat": "stabil", "mixed": "uneinheitlich"}

    # Main confidence display
    conf_col1, conf_col2 = st.columns([1, 2], gap="medium")

    with conf_col1:
        st.markdown(
            f'<div class="sparkline-card" style="text-align:center; padding:1.2rem;">'
            f'<div style="font-size:2.2rem; font-weight:700; color:{conf_color};">{conf:.0f}%</div>'
            f'<div style="font-size:0.75rem; font-weight:600; color:{conf_color}; margin:0.2rem 0;">'
            f'Konfidenz: {conf_label_de}</div>'
            f'<div style="font-size:0.7rem; color:var(--text-muted);">'
            f'Richtung: {direction_icons.get(composite.direction, "?")} '
            f'{direction_labels.get(composite.direction, composite.direction)} '
            f'({composite.weighted_trend:+.1f}%)</div>'
            f'<div style="font-size:0.65rem; color:var(--text-muted); margin-top:0.3rem;">'
            f'Signal-Uebereinstimmung: {composite.agreement_pct:.0f}%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with conf_col2:
        # Per-signal status bars
        signal_html = '<div class="sparkline-card" style="padding:1rem;">'
        for sig in composite.signals:
            cfg = SIGNAL_CONFIG.get(sig.name, {})
            icon = cfg.get("icon", "")
            label = cfg.get("label", sig.name)
            color = cfg.get("color", "#94a3b8")

            if not sig.available:
                status_html = '<span style="color:#64748b;">nicht verfuegbar</span>'
                bar_width = 0
            else:
                dir_icon = direction_icons.get(sig.direction, "?")
                dir_label = direction_labels.get(sig.direction, sig.direction)
                status_html = (
                    f'<span style="color:{color};">{dir_icon} {dir_label} ({sig.magnitude:+.1f}%)</span>'
                )
                bar_width = min(100, max(5, abs(sig.magnitude) * 2))

            signal_html += (
                f'<div style="display:flex; align-items:center; gap:0.6rem; margin:0.4rem 0;">'
                f'<span style="width:1.2rem; text-align:center;">{icon}</span>'
                f'<span style="width:10rem; font-size:0.72rem; color:var(--text-muted);">{label}</span>'
                f'<div style="flex:1; height:6px; background:rgba(255,255,255,0.05); border-radius:3px; overflow:hidden;">'
                f'<div style="width:{bar_width}%; height:100%; background:{color}; border-radius:3px;"></div>'
                f'</div>'
                f'<span style="font-size:0.72rem; min-width:8rem;">{status_html}</span>'
                f'</div>'
            )
        signal_html += '</div>'
        st.markdown(signal_html, unsafe_allow_html=True)

    # Narrative expander
    with st.expander("Detailanalyse anzeigen", expanded=False):
        for line in composite.narrative_de.split("\n"):
            if line.strip():
                st.caption(line)

    # ── Section 2: Correlation Chart ─────────────────────────────────────────
    st.markdown(
        f'<div class="section-header">{selected_pathogen} — Abwasser vs. Laborvolumen</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Die Viruslast im Abwasser (blau) korreliert mit dem Testaufkommen im Labor (orange) "
        "mit ca. 14 Tagen Verzoegerung. Pinch-to-Zoom oder Mausrad zum Reinzoomen."
    )

    today = pd.Timestamp(datetime.today()).normalize()
    chart_start = today - pd.Timedelta(days=120)

    ww_chart = wastewater_df[wastewater_df["date"] >= chart_start].copy()
    lab_chart = lab_df[lab_df["date"] >= chart_start].copy()
    lab_actuals = lab_chart[lab_chart["date"] <= today]

    lab_forecast_chart = forecast_df.rename(
        columns={"Date": "date", "Predicted Volume": "order_volume"}
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Wastewater raw (faint)
    fig.add_trace(
        go.Scatter(
            x=ww_chart["date"], y=ww_chart["virus_load"],
            name="Viruslast (taeglich)", showlegend=False,
            line=dict(color="#3b82f6", width=0.5), opacity=0.15,
            fill="tozeroy", fillcolor="rgba(59,130,246,0.03)",
            hovertemplate="%{x|%d %b}: %{y:,.0f} Kopien/L<extra></extra>",
        ),
        secondary_y=False,
    )

    # Wastewater 7d average
    ww_smoothed = ww_chart.copy()
    ww_smoothed["vl_7d"] = ww_smoothed["virus_load"].rolling(7, min_periods=1).mean()
    fig.add_trace(
        go.Scatter(
            x=ww_smoothed["date"], y=ww_smoothed["vl_7d"],
            name="Viruslast (7d Ø)",
            line=dict(color="#3b82f6", width=2),
            hovertemplate="%{x|%d %b}: %{y:,.0f} Ø Kopien/L<extra></extra>",
        ),
        secondary_y=False,
    )

    # Lab raw (faint)
    fig.add_trace(
        go.Scatter(
            x=lab_actuals["date"], y=lab_actuals["order_volume"],
            name="Labortests (taeglich)", showlegend=False,
            line=dict(color="#f77f00", width=1), opacity=0.2,
            hovertemplate="%{x|%d %b}: %{y:,.0f} Tests<extra></extra>",
        ),
        secondary_y=True,
    )

    # Lab 7d average
    lab_smoothed = lab_actuals.copy()
    lab_smoothed["vol_7d"] = lab_smoothed["order_volume"].rolling(7, min_periods=1).mean()
    fig.add_trace(
        go.Scatter(
            x=lab_smoothed["date"], y=lab_smoothed["vol_7d"],
            name="Labortests (7d Ø)",
            line=dict(color="#f77f00", width=2.5),
            hovertemplate="%{x|%d %b}: %{y:,.0f} Ø/Tag<extra></extra>",
        ),
        secondary_y=True,
    )

    # ML Forecast overlay (with confidence band)
    if use_ml and ml_model_info and "ml_forecast" in dir():
        try:
            ml_fc = ml_forecast
            fig.add_trace(
                go.Scatter(
                    x=ml_fc["date"], y=ml_fc["upper"],
                    name="ML Upper", showlegend=False,
                    line=dict(width=0), mode="lines",
                ),
                secondary_y=True,
            )
            fig.add_trace(
                go.Scatter(
                    x=ml_fc["date"], y=ml_fc["lower"],
                    name="ML-Konfidenzband",
                    fill="tonexty", fillcolor="rgba(168,85,247,0.1)",
                    line=dict(width=0), mode="lines",
                ),
                secondary_y=True,
            )
            fig.add_trace(
                go.Scatter(
                    x=ml_fc["date"], y=ml_fc["predicted"],
                    name="ML-Prognose",
                    line=dict(color="#a855f7", width=2.5, dash="dash"),
                    hovertemplate="%{x|%d %b}: %{y:,.0f} ML-Prognose<extra></extra>",
                ),
                secondary_y=True,
            )
        except Exception:
            pass

    # Standard forecast (dashed)
    if not lab_forecast_chart.empty:
        vol_col = "order_volume"
        if not lab_actuals.empty:
            connector = pd.DataFrame(
                {"date": [lab_actuals["date"].iloc[-1]], vol_col: [lab_actuals["order_volume"].iloc[-1]]}
            )
            fc_plot = pd.concat([connector, lab_forecast_chart], ignore_index=True)
        else:
            fc_plot = lab_forecast_chart

        label = f"Prognose (+{scenario_uplift}%)" if scenario_uplift > 0 else "Prognose"
        fig.add_trace(
            go.Scatter(
                x=fc_plot["date"], y=fc_plot[vol_col], name=label,
                line=dict(color="#f77f00", width=2.5, dash="dot"),
                hovertemplate="%{x|%d %b}: %{y:,.0f} prognostiziert<extra></extra>",
            ),
            secondary_y=True,
        )

    # TODAY marker
    today_str = today.strftime("%Y-%m-%d")
    fig.add_shape(
        type="line", x0=today_str, x1=today_str, y0=0, y1=1, yref="paper",
        line=dict(color="rgba(239,68,68,0.5)", width=1.5, dash="dot"),
    )
    fig.add_annotation(
        x=today_str, y=1.06, yref="paper", text="HEUTE", showarrow=False,
        font=dict(size=9, color="#ef4444", family="Inter"),
        bgcolor="rgba(239,68,68,0.1)", borderpad=3,
        bordercolor="rgba(239,68,68,0.2)", borderwidth=1,
    )

    # Lag annotation
    fig.add_annotation(
        x=(today - pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
        y=0.93, yref="paper", text="14-Tage Lag", showarrow=False,
        font=dict(size=9, color="#64748b", family="Inter"),
        bgcolor="rgba(15,20,35,0.85)", borderpad=4,
        bordercolor="rgba(255,255,255,0.08)", borderwidth=1,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.5)",
        height=420,
        margin=dict(l=0, r=0, t=35, b=0),
        legend=dict(
            orientation="h", y=1.12, x=0.5, xanchor="center",
            font=dict(size=10, color="#94a3b8", family="Inter"),
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1e293b", bordercolor="#334155", font_size=12, font_family="Inter"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.03)", dtick="M1", tickformat="%b '%y", tickfont=dict(size=10, color="#64748b"))
    fig.update_yaxes(title_text="Viruslast", secondary_y=False, showgrid=False, title_font=dict(color="#3b82f6", size=11), tickfont=dict(size=9, color="#64748b"))
    fig.update_yaxes(title_text="Tests / Tag", secondary_y=True, gridcolor="rgba(255,255,255,0.04)", title_font=dict(color="#f77f00", size=11), tickfont=dict(size=9, color="#64748b"))

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(
        fig, use_container_width=True, key="correlation_chart",
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "displaylogo": False,
        },
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Store fig for PDF export
    correlation_fig_for_pdf = fig

    # ── Section 2b: Surveillance-Signale (GrippeWeb + ARE) ────────────────────
    st.markdown(
        '<div class="section-header">Ergaenzende Surveillance-Signale</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Diese Datenquellen ergaenzen die Abwasserdaten: GrippeWeb erfasst Selbstmeldungen aus der Bevoelkerung, "
        "ARE-Konsultationen messen die Arztbesuche. Steigende Werte hier → mehr Testaufkommen in 1-2 Wochen."
    )

    surv_col1, surv_col2 = st.columns(2, gap="medium")

    with surv_col1:
        st.caption("GrippeWeb — Bevoelkerungsbasierte ARE/ILI-Inzidenz (freiwillige Meldungen)")
        gw_type = st.radio(
            "Typ", ["ARE", "ILI"], horizontal=True, key="gw_type",
            help="ARE = Akute Atemwegserkrankung, ILI = Influenza-like Illness",
        )
        with st.spinner("Lade GrippeWeb …"):
            gw_df = fetch_grippeweb(erkrankung=gw_type, region="Bundesweit")

        if not gw_df.empty:
            gw_recent = gw_df[gw_df["date"] >= (today - pd.Timedelta(days=730))].copy()
            fig_gw = go.Figure()
            fig_gw.add_trace(
                go.Scatter(
                    x=gw_recent["date"], y=gw_recent["incidence"],
                    name=f"GrippeWeb {gw_type}",
                    fill="tozeroy",
                    line=dict(color="#8b5cf6", width=2),
                    fillcolor="rgba(139,92,246,0.08)",
                    hovertemplate="%{x|%d %b %Y}: %{y:,.1f} / 100.000<extra></extra>",
                )
            )
            fig_gw.add_shape(
                type="line", x0=today_str, x1=today_str, y0=0, y1=1, yref="paper",
                line=dict(color="rgba(239,68,68,0.4)", width=1, dash="dot"),
            )
            fig_gw.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17,24,39,0.5)",
                height=250, margin=dict(l=0, r=0, t=5, b=0),
                yaxis_title="Inzidenz / 100.000",
                yaxis_title_font=dict(size=10, color="#64748b"),
                showlegend=False, hovermode="x unified",
            )
            fig_gw.update_xaxes(gridcolor="rgba(255,255,255,0.03)", tickfont=dict(size=9, color="#64748b"))
            fig_gw.update_yaxes(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=9, color="#64748b"))

            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig_gw, use_container_width=True, key="gw_detail")
            st.markdown("</div>", unsafe_allow_html=True)

            if len(gw_recent) >= 2:
                latest_gw = gw_recent["incidence"].iloc[-1]
                prev_gw = gw_recent["incidence"].iloc[-2]
                chg_gw = ((latest_gw - prev_gw) / prev_gw * 100) if prev_gw > 0 else 0
                st.caption(f"Aktuell: {latest_gw:,.1f} / 100.000 ({chg_gw:+.1f}% vs. Vorwoche)")
        else:
            st.caption("GrippeWeb nicht verfuegbar.")

    with surv_col2:
        st.caption("ARE-Konsultationsinzidenz — Praxen-Sentinel (Arztbesuche)")
        with st.spinner("Lade ARE …"):
            are_df = fetch_are_konsultation(bundesland="Bundesweit")

        if not are_df.empty:
            are_recent = are_df[are_df["date"] >= (today - pd.Timedelta(days=730))].copy()
            fig_are = go.Figure()
            fig_are.add_trace(
                go.Scatter(
                    x=are_recent["date"], y=are_recent["consultation_incidence"],
                    name="ARE Konsultationen",
                    fill="tozeroy",
                    line=dict(color="#06b6d4", width=2),
                    fillcolor="rgba(6,182,212,0.08)",
                    hovertemplate="%{x|%d %b %Y}: %{y:,.0f} / 100.000<extra></extra>",
                )
            )
            fig_are.add_shape(
                type="line", x0=today_str, x1=today_str, y0=0, y1=1, yref="paper",
                line=dict(color="rgba(239,68,68,0.4)", width=1, dash="dot"),
            )
            fig_are.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17,24,39,0.5)",
                height=250, margin=dict(l=0, r=0, t=5, b=0),
                yaxis_title="Konsultationen / 100.000",
                yaxis_title_font=dict(size=10, color="#64748b"),
                showlegend=False, hovermode="x unified",
            )
            fig_are.update_xaxes(gridcolor="rgba(255,255,255,0.03)", tickfont=dict(size=9, color="#64748b"))
            fig_are.update_yaxes(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=9, color="#64748b"))

            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig_are, use_container_width=True, key="are_detail")
            st.markdown("</div>", unsafe_allow_html=True)

            if len(are_recent) >= 2:
                latest_are = are_recent["consultation_incidence"].iloc[-1]
                prev_are = are_recent["consultation_incidence"].iloc[-2]
                chg_are = ((latest_are - prev_are) / prev_are * 100) if prev_are > 0 else 0
                st.caption(f"Aktuell: {latest_are:,.0f} / 100.000 ({chg_are:+.1f}% vs. Vorwoche)")
        else:
            st.caption("ARE-Daten nicht verfuegbar.")

    # ── Section 3: Burndown + Table ──────────────────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="medium")

    burndown_fig_for_pdf = None

    with col_left:
        st.markdown('<div class="section-header">Reagenz-Bestandsprognose</div>', unsafe_allow_html=True)
        kit_name_current = get_kit_for_pathogen(selected_pathogen)
        st.caption(f"Kit: {kit_name_current} · Aktueller Bestand: {stock_on_hand:,} · Bestand oben in der Uebersicht aendern.")

        remaining = kpis.get("remaining_stock", [])
        if remaining:
            burndown_dates = forecast_df["Date"].values
            fig_burn = go.Figure()

            fig_burn.add_trace(
                go.Scatter(
                    x=burndown_dates, y=remaining, name="Restbestand",
                    fill="tozeroy", line=dict(color="#22c55e", width=2),
                    fillcolor="rgba(34,197,94,0.1)",
                    hovertemplate="%{x|%d %b}: %{y:,.0f} Einheiten<extra></extra>",
                )
            )

            cum_demand = np.cumsum(forecast_df["Predicted Volume"].values)
            fig_burn.add_trace(
                go.Scatter(
                    x=burndown_dates, y=cum_demand, name="Kumulativer Bedarf",
                    line=dict(color="#ef4444", width=1.5, dash="dot"),
                    hovertemplate="%{x|%d %b}: %{y:,.0f} kumulativ<extra></extra>",
                )
            )

            fig_burn.add_hline(
                y=stock_on_hand, line_dash="longdash", line_color="rgba(148,163,184,0.3)",
                annotation_text=f"Bestand: {stock_on_hand:,}", annotation_position="top left",
                annotation_font_color="#64748b", annotation_font_size=10,
            )

            stockout_day = kpis.get("stockout_day")
            if stockout_day is not None:
                so_str = pd.Timestamp(stockout_day).strftime("%Y-%m-%d")
                fig_burn.add_shape(
                    type="line", x0=so_str, x1=so_str, y0=0, y1=1, yref="paper",
                    line=dict(color="rgba(239,68,68,0.5)", width=1.5, dash="dash"),
                )
                fig_burn.add_annotation(
                    x=so_str, y=1.06, yref="paper", text="STOCKOUT", showarrow=False,
                    font=dict(size=9, color="#ef4444", family="Inter"),
                    bgcolor="rgba(239,68,68,0.1)", borderpad=3,
                    bordercolor="rgba(239,68,68,0.2)", borderwidth=1,
                )

            fig_burn.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17,24,39,0.5)",
                height=320, margin=dict(l=0, r=0, t=20, b=0),
                legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center", font=dict(size=10, color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
                hovermode="x unified",
                hoverlabel=dict(bgcolor="#1e293b", bordercolor="#334155", font_size=12),
                yaxis_title="Einheiten", yaxis_title_font=dict(size=11, color="#64748b"),
            )
            fig_burn.update_xaxes(gridcolor="rgba(255,255,255,0.03)", tickfont=dict(size=9, color="#64748b"))
            fig_burn.update_yaxes(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=9, color="#64748b"))

            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig_burn, use_container_width=True, key="burndown_chart")
            st.markdown("</div>", unsafe_allow_html=True)

            burndown_fig_for_pdf = fig_burn

    with col_right:
        st.markdown('<div class="section-header">Bestellempfehlungen</div>', unsafe_allow_html=True)

        def highlight_orders(row):
            if row["Reagent Order"] > 0:
                return ["background-color: rgba(239,68,68,0.1); color: #fca5a5; font-weight: 600;"] * len(row)
            return ["color: #94a3b8;"] * len(row)

        styled = forecast_df.style.apply(highlight_orders, axis=1).format(
            {
                "Predicted Volume": "{:,.0f}",
                "Reagent Order": "{:,.0f}",
                rev_col_name: "EUR {:,.0f}",
            }
        )

        st.dataframe(styled, use_container_width=True, hide_index=True, height=380)

    # ── Section 4: AI Insights ───────────────────────────────────────────────
    st.markdown('<div class="section-header">KI-Analyse</div>', unsafe_allow_html=True)

    ollama = get_ollama_client()
    ollama_ok = ollama.health_check()

    if ollama_ok:
        with st.spinner("KI generiert Analyse …"):
            ai_insight = ollama.generate_insight(kpis, selected_pathogen)
    else:
        ai_insight = ollama._fallback_insight(kpis, selected_pathogen)

    ai_source = "Ollama" if ollama_ok else "Regelbasiert"
    st.markdown(
        f'<div class="alert-ai">'
        f'<strong>KI-Einschaetzung</strong> <span style="opacity:0.5;">({ai_source})</span><br><br>'
        f'{ai_insight}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── PDF Export ───────────────────────────────────────────────────────────
    st.markdown("")
    col_pdf, col_space = st.columns([1, 3])
    with col_pdf:
        if st.button("PDF-Bericht herunterladen", use_container_width=True, type="primary"):
            with st.spinner("PDF wird generiert …"):
                try:
                    ai_commentary = ""
                    if ollama_ok:
                        ai_commentary = ollama.generate_pdf_commentary(kpis, selected_pathogen)
                    else:
                        ai_commentary = ai_insight

                    pdf_bytes = generate_pdf_report(
                        kpis=kpis,
                        forecast_df=forecast_df,
                        correlation_fig=correlation_fig_for_pdf,
                        burndown_fig=burndown_fig_for_pdf,
                        ai_commentary=ai_commentary,
                        pathogen=selected_pathogen,
                    )

                    filename = f"LabPulse_{selected_pathogen}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                    st.download_button(
                        label="PDF herunterladen",
                        data=pdf_bytes,
                        file_name=filename,
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.error(f"PDF-Generierung fehlgeschlagen: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: REGIONAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_regional:
    st.markdown('<div class="section-header">Regionale Viruslast-Analyse</div>', unsafe_allow_html=True)

    if raw_df.empty or "bundesland" not in raw_df.columns:
        st.info("Keine regionalen Daten verfuegbar. RKI-Daten enthalten keine Bundesland-Spalte.")
    else:
        pathogen_types = PATHOGEN_GROUPS.get(selected_pathogen, [selected_pathogen])
        regional_df = aggregate_by_bundesland(raw_df, pathogen_types, days_back=30)

        if regional_df.empty:
            st.info(f"Keine regionalen Daten fuer {selected_pathogen} in den letzten 30 Tagen.")
        else:
            # Scatter map (works without GeoJSON)
            map_df = create_scatter_map_data(regional_df)

            if not map_df.empty:
                col_map, col_table = st.columns([2, 1], gap="medium")

                with col_map:
                    # Bubble chart (offline — no external topojson needed)
                    # Size = avg virus load, Color = trend %
                    vl_min = map_df["avg_virus_load"].min()
                    vl_max = map_df["avg_virus_load"].max()
                    if vl_max > vl_min:
                        map_df["bubble_size"] = 12 + (map_df["avg_virus_load"] - vl_min) / (vl_max - vl_min) * 40
                    else:
                        map_df["bubble_size"] = 25

                    fig_map = go.Figure()
                    fig_map.add_trace(
                        go.Scatter(
                            x=map_df["lon"],
                            y=map_df["lat"],
                            mode="markers+text",
                            marker=dict(
                                size=map_df["bubble_size"],
                                color=map_df["trend_pct"],
                                colorscale=[[0, "#22c55e"], [0.5, "#f77f00"], [1, "#ef4444"]],
                                colorbar=dict(
                                    title=dict(text="Trend %", font=dict(color="#94a3b8")),
                                    tickfont=dict(color="#94a3b8"),
                                ),
                                opacity=0.85,
                                line=dict(width=1, color="rgba(255,255,255,0.15)"),
                            ),
                            text=map_df["bundesland"],
                            textposition="top center",
                            textfont=dict(size=9, color="#94a3b8"),
                            hovertemplate=(
                                "<b>%{text}</b><br>"
                                "Ø Viruslast: %{customdata[0]:,.0f}<br>"
                                "Trend: %{customdata[1]:+.1f}%<br>"
                                "Standorte: %{customdata[2]}"
                                "<extra></extra>"
                            ),
                            customdata=map_df[["avg_virus_load", "trend_pct", "site_count"]].values,
                        )
                    )
                    fig_map.update_layout(
                        template="plotly_dark",
                        height=550,
                        margin=dict(l=20, r=20, t=20, b=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(17,24,39,0.5)",
                        xaxis=dict(
                            title="Laengengrad",
                            range=[5.5, 15.5],
                            showgrid=True,
                            gridcolor="rgba(255,255,255,0.04)",
                            tickfont=dict(size=9, color="#64748b"),
                            title_font=dict(size=10, color="#64748b"),
                        ),
                        yaxis=dict(
                            title="Breitengrad",
                            range=[47.0, 55.5],
                            showgrid=True,
                            gridcolor="rgba(255,255,255,0.04)",
                            scaleanchor="x",
                            scaleratio=1.5,
                            tickfont=dict(size=9, color="#64748b"),
                            title_font=dict(size=10, color="#64748b"),
                        ),
                        showlegend=False,
                    )

                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(fig_map, use_container_width=True, key="regional_map")
                    st.markdown("</div>", unsafe_allow_html=True)

                with col_table:
                    st.markdown(
                        f'<div class="section-header">Top-Regionen ({selected_pathogen})</div>',
                        unsafe_allow_html=True,
                    )

                    display_regional = regional_df[["bundesland", "avg_virus_load", "trend_pct", "site_count"]].copy()
                    display_regional.columns = ["Bundesland", "Ø Viruslast", "Trend %", "Standorte"]
                    display_regional["Ø Viruslast"] = display_regional["Ø Viruslast"].map(lambda x: f"{x:,.0f}")
                    display_regional["Trend %"] = display_regional["Trend %"].map(lambda x: f"{x:+.1f}%")

                    st.dataframe(
                        display_regional,
                        use_container_width=True,
                        hide_index=True,
                        height=450,
                    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: GOOGLE TRENDS
# ══════════════════════════════════════════════════════════════════════════════
with tab_trends:
    # ── Google Trends ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Google Trends — Suchinteresse (echte Daten)</div>', unsafe_allow_html=True)

    # Suggestions per pathogen
    suggested_terms = PATHOGEN_SEARCH_TERMS.get(selected_pathogen, [])

    # All available suggestions across all pathogens (for the pill buttons)
    all_suggestions = sorted(set(
        term for terms_list in PATHOGEN_SEARCH_TERMS.values() for term in terms_list
    ))
    # Add some extra general terms
    extra_suggestions = [
        "Fieber", "Husten", "Schnupfen", "Halsschmerzen", "Atemnot",
        "Krankschreibung", "Arzt Termin", "Notaufnahme", "Apotheke",
        "Impfung", "Maske", "PCR Test", "Antigentest",
    ]
    all_suggestions = sorted(set(all_suggestions + extra_suggestions))

    # Session state for custom terms
    if "trends_custom_terms" not in st.session_state:
        st.session_state.trends_custom_terms = ", ".join(suggested_terms[:3])

    trends_col_input, trends_col_time = st.columns([3, 1], gap="medium")

    with trends_col_input:
        custom_input = st.text_input(
            "Suchbegriffe (kommagetrennt, max. 5)",
            value=st.session_state.trends_custom_terms,
            placeholder="z.B. Corona Test, Grippe Symptome, Fieber",
            key="trends_input",
            help="Eigene Suchbegriffe eingeben oder aus den Vorschlaegen waehlen.",
        )
        st.session_state.trends_custom_terms = custom_input

    with trends_col_time:
        trends_timeframe = st.selectbox(
            "Zeitraum",
            ["today 3-m", "today 12-m", "today 5-y"],
            index=1,
            format_func=lambda x: {"today 3-m": "3 Monate", "today 12-m": "12 Monate", "today 5-y": "5 Jahre"}[x],
            key="trends_timeframe",
        )

    # Suggestion pills
    st.markdown(
        '<div style="display:flex; flex-wrap:wrap; gap:0.4rem; margin:0.5rem 0 1rem 0;">',
        unsafe_allow_html=True,
    )

    # Show pathogen-specific suggestions first, then general
    display_suggestions = suggested_terms + [s for s in extra_suggestions if s not in suggested_terms]

    pill_cols = st.columns(min(len(display_suggestions[:12]), 12))
    for i, suggestion in enumerate(display_suggestions[:12]):
        with pill_cols[i % len(pill_cols)]:
            if st.button(
                suggestion,
                key=f"pill_{suggestion}",
                use_container_width=True,
                type="secondary",
            ):
                # Add to current terms
                current = [t.strip() for t in st.session_state.trends_custom_terms.split(",") if t.strip()]
                if suggestion not in current:
                    current.append(suggestion)
                    # Keep max 5
                    current = current[-5:]
                st.session_state.trends_custom_terms = ", ".join(current)
                st.rerun()

    # Parse user input into list
    user_terms = [t.strip() for t in custom_input.split(",") if t.strip()][:5]

    if user_terms:
        st.caption(
            f"Aktive Begriffe: " +
            ", ".join(f'**{t}**' for t in user_terms) +
            f" · Region: Deutschland · Werte: relativ (0-100)"
        )

        with st.spinner("Lade Google Trends …"):
            from modules.external_data import fetch_google_trends
            trends_df = fetch_google_trends(user_terms, timeframe=trends_timeframe, geo="DE")

        if not trends_df.empty:
            fig_trends = go.Figure()

            term_cols = [c for c in trends_df.columns if c != "date"]
            colors = ["#f77f00", "#3b82f6", "#22c55e", "#a855f7", "#ef4444"]

            for i, col in enumerate(term_cols):
                fig_trends.add_trace(
                    go.Scatter(
                        x=trends_df["date"], y=trends_df[col],
                        name=col,
                        line=dict(color=colors[i % len(colors)], width=2),
                        hovertemplate=f"{col}: " + "%{y}<extra></extra>",
                    )
                )

            fig_trends.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17,24,39,0.5)",
                height=350, margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title="Relatives Suchinteresse (0-100)",
                yaxis_title_font=dict(size=10, color="#64748b"),
                legend=dict(
                    orientation="h", y=1.12, x=0.5, xanchor="center",
                    font=dict(size=9, color="#94a3b8"), bgcolor="rgba(0,0,0,0)",
                ),
                hovermode="x unified",
                hoverlabel=dict(bgcolor="#1e293b", bordercolor="#334155", font_size=12),
            )
            fig_trends.update_xaxes(gridcolor="rgba(255,255,255,0.03)", tickfont=dict(size=9, color="#64748b"))
            fig_trends.update_yaxes(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=9, color="#64748b"))

            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig_trends, use_container_width=True, key="trends_chart")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning(
                "Google Trends hat keine Daten zurueckgegeben. "
                "Moeglich: Rate-Limit erreicht (~10-15 Abfragen/Stunde) oder Begriffe zu nischig."
            )

        # Limitations expander
        with st.expander("Hinweis: Google Trends Einschraenkungen"):
            for lim in get_trends_limitations():
                st.caption(f"• {lim}")
    else:
        st.info("Suchbegriffe eingeben um Google Trends zu laden.")


# ── Footer ───────────────────────────────────────────────────────────────────
cost = kpis.get("cost_per_test", AVG_REVENUE_PER_TEST)
model_str = f" · ML: {ml_model_info['model_type']} ({ml_model_info['confidence_score']:.0f}%)" if ml_model_info else ""
st.markdown(
    f"""
    <div class="footer-bar">
        <span>LabPulse AI v2.0 · {selected_pathogen} · {kpis.get('test_name', '')} · EUR {cost}/Test{model_str}</span>
        <span>Daten: <a href="https://github.com/robert-koch-institut/Abwassersurveillance_AMELAG" target="_blank">
            RKI AMELAG
        </a> · {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
    </div>
    """,
    unsafe_allow_html=True,
)
