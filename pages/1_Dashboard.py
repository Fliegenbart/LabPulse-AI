"""
LabPulse AI — Dashboard (Zen Mode)
====================================
Progressive Disclosure: Chart first, everything else on demand.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from data_engine import (
    fetch_rki_raw, fetch_rki_wastewater, get_available_pathogens,
    generate_lab_volume, build_forecast, AVG_REVENUE_PER_TEST,
    PATHOGEN_REAGENT_MAP, PATHOGEN_GROUPS, PATHOGEN_SCALE,
)
from modules.ollama_client import get_client as get_ollama_client
from modules.pdf_export import generate_report as generate_pdf_report
from modules.lab_data_merger import validate_csv, merge_with_synthetic, get_data_quality_summary
from modules.regional_aggregation import (
    aggregate_by_bundesland, create_scatter_map_data,
    BUNDESLAND_COORDS, GERMANY_CENTER,
)
from modules.ml_forecaster import LabVolumeForecaster
from modules.alert_engine import AlertManager
from modules.external_data import (
    fetch_grippeweb, fetch_are_konsultation, fetch_google_trends,
    get_trends_limitations, PATHOGEN_SEARCH_TERMS,
)
from modules.signal_fusion import fuse_all_signals, SIGNAL_CONFIG

# ─────────────────────────────────────────────────────────────────────────────
# ZEN MODE CSS — 3 colors, 3 sizes, maximum breathing room
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,300;400;600&family=Azeret+Mono:wght@400;600&display=swap');

:root {
    --bg: #212131;
    --surface: #2c2c40;
    --border: rgba(212,149,106,0.14);
    --text: #e8e3db;
    --dim: #908a80;
    --accent: #d4956a;
    --radius: 16px;
    --font: 'Bricolage Grotesque', sans-serif;
    --mono: 'Azeret Mono', monospace;
}

/* 3 font sizes only */
.zen-lg { font-size: 1.5rem; font-weight: 600; color: var(--text); font-family: var(--mono); letter-spacing: -0.02em; }
.zen-body { font-size: 0.85rem; color: var(--text); font-family: var(--font); line-height: 1.7; }
.zen-sm { font-size: 0.72rem; color: var(--dim); font-family: var(--font); }

.stApp { background: var(--bg); }
.stApp::before { display: none; }

.block-container {
    padding: 3rem 3.75rem 1.5rem 3.75rem;
    max-width: 1280px;
    font-family: var(--font);
}
html, body, [class*="css"] { font-family: var(--font); }

/* ── Header ── */
.zen-header {
    display: flex; align-items: baseline; gap: 0.8rem;
    padding-bottom: 0.5rem;
    margin-bottom: 2.5rem;
    border-bottom: 1px solid var(--border);
}
.zen-header h1 {
    margin: 0; font-size: 1.1rem; font-weight: 600;
    font-family: var(--font); color: var(--text);
    letter-spacing: 0.02em;
}
.zen-header span {
    font-family: var(--mono); font-size: 0.72rem;
    color: var(--accent); font-weight: 600;
}

/* ── KPI strip ── */
.zen-kpi-strip {
    display: flex; gap: 3rem; align-items: baseline;
    margin-bottom: 3rem;
}
.zen-kpi { text-align: left; }
.zen-kpi .val { font-family: var(--mono); font-size: 1.5rem; font-weight: 600; color: var(--text); }
.zen-kpi .lbl { font-size: 0.72rem; color: var(--dim); margin-top: 0.15rem; }
.zen-kpi .delta { font-family: var(--mono); font-size: 0.72rem; }
.zen-kpi .delta.warn { color: var(--accent); }
.zen-kpi .delta.ok { color: #5eead4; }

/* ── ML Toggle ── */
.zen-ml-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1.4rem 1.8rem;
    margin-bottom: 2.5rem;
    background: linear-gradient(135deg, var(--surface), rgba(212,149,106,0.04));
    border: 1.5px solid rgba(212,149,106,0.15);
    border-radius: var(--radius);
}
.zen-ml-row .label { font-size: 1.1rem; color: var(--text); font-weight: 600; }
.zen-ml-row .sub { font-size: 0.78rem; color: var(--dim); margin-top: 0.2rem; }
/* Bigger toggle switch */
[data-testid="stCheckbox"]:has(input[aria-label="ML-Prognose aktivieren"]) label {
    transform: scale(1.5); transform-origin: left center;
    padding: 0.3rem 0;
}
[data-testid="stCheckbox"]:has(input[aria-label="ML-Prognose aktivieren"]) span[data-testid="stCheckboxLabel"] {
    font-size: 0.9rem !important; font-weight: 600 !important; color: var(--accent) !important;
}

/* ── Alert ── */
.zen-alert {
    padding: 0.8rem 1.2rem;
    margin-bottom: 2rem;
    border-radius: var(--radius);
    border-left: 3px solid #fb7185;
    background: rgba(251,113,133,0.04);
    color: #fda4af;
    font-size: 0.85rem; line-height: 1.6;
}

/* ── Chart wrapper ── */
.zen-chart {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.5rem;
    margin-bottom: 3rem;
}

/* ── Cards (expander content) ── */
div[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    margin-bottom: 1.5rem;
}
div[data-testid="stExpander"] summary {
    font-family: var(--font) !important;
    font-size: 0.85rem !important;
    color: var(--dim) !important;
    font-weight: 600 !important;
}

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
}
div[data-testid="stMetric"] label { color: var(--dim) !important; font-family: var(--font) !important; font-size: 0.72rem !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: var(--text) !important; font-family: var(--mono) !important; font-size: 1.2rem !important; }
div[data-testid="stMetric"] div[data-testid="stMetricDelta"] { font-family: var(--mono) !important; font-size: 0.72rem !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid var(--border); margin-top: 1.5rem; }
.stTabs [data-baseweb="tab"] { font-family: var(--font); font-size: 0.85rem; font-weight: 500; padding: 0.75rem 1.5rem; color: var(--dim); border-bottom: 2px solid transparent; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom-color: var(--accent) !important; }

/* ── Data table ── */
[data-testid="stDataFrame"] > div { border-radius: var(--radius); border: 1px solid var(--border); }

/* ── Buttons ── */
.stButton > button {
    font-family: var(--font); font-weight: 600;
    border-radius: 8px; border: 1px solid var(--border);
    background: var(--surface); color: var(--text);
}
.stButton > button:hover { border-color: var(--accent); }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: var(--bg); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] .stMarkdown h1 { color: var(--accent); font-family: var(--font); font-size: 1rem; font-weight: 600; }
.sidebar-label { color: var(--dim); font-size: 0.62rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 0.5rem; }
.sidebar-pill { display: inline-block; background: rgba(212,149,106,0.1); color: var(--accent); font-family: var(--mono); font-size: 0.6rem; font-weight: 600; padding: 0.15rem 0.5rem; border-radius: 99px; margin-left: 0.4rem; }

/* ── Footer ── */
.zen-foot { padding: 2rem 0 0.5rem 0; border-top: 1px solid var(--border); margin-top: 4rem; color: var(--dim); font-family: var(--mono); font-size: 0.68rem; display: flex; justify-content: space-between; }
.zen-foot a { color: var(--accent); text-decoration: none; }

/* ── Hide chrome ── */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
</style>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Lade RKI-Daten …")
def load_raw():
    return fetch_rki_raw()

# Session State
if "ml_enabled" not in st.session_state:
    st.session_state.ml_enabled = False
if "uploaded_lab_data" not in st.session_state:
    st.session_state.uploaded_lab_data = None
if "alert_webhook_url" not in st.session_state:
    st.session_state.alert_webhook_url = ""

UNIQUE_KITS = {
    "SARS-CoV-2 PCR Kit": {"cost": 45, "pathogens": ["SARS-CoV-2"], "default_stock": 5000, "unit": "Tests", "lieferzeit_tage": 5},
    "Influenza A/B PCR Panel": {"cost": 38, "pathogens": ["Influenza A", "Influenza B", "Influenza (gesamt)"], "default_stock": 3000, "unit": "Tests", "lieferzeit_tage": 3},
    "RSV PCR Kit": {"cost": 42, "pathogens": ["RSV"], "default_stock": 2000, "unit": "Tests", "lieferzeit_tage": 4},
}
if "kit_inventory" not in st.session_state:
    st.session_state.kit_inventory = {k: v["default_stock"] for k, v in UNIQUE_KITS.items()}

def get_kit_for_pathogen(p):
    for k, v in UNIQUE_KITS.items():
        if p in v["pathogens"]:
            return k
    return "SARS-CoV-2 PCR Kit"

def get_stock_for_pathogen(p):
    return st.session_state.kit_inventory.get(get_kit_for_pathogen(p), 5000)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Settings drawer (all secondary controls)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('# LabPulse AI <span class="sidebar-pill">v2</span>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="sidebar-label">Pathogen</div>', unsafe_allow_html=True)
    raw_df = load_raw()
    pathogens = get_available_pathogens(raw_df)
    selected_pathogen = st.selectbox("Pathogen", pathogens, index=0, label_visibility="collapsed")

    st.markdown("")
    st.markdown('<div class="sidebar-label">Prognose</div>', unsafe_allow_html=True)
    forecast_horizon = st.slider("Horizont (Tage)", 7, 21, 14, 7)
    safety_buffer = st.slider("Sicherheitspuffer %", 0, 30, 10, 5)

    st.markdown("")
    st.markdown('<div class="sidebar-label">Simulation</div>', unsafe_allow_html=True)
    scenario_uplift = st.slider("Viruslast-Uplift %", 0, 50, 0, 5, help="Simuliert ploetzlichen Anstieg")

    with st.expander("Erweitert"):
        st.markdown('<div class="sidebar-label">Labordaten-Import</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("CSV hochladen", type=["csv"], label_visibility="collapsed")
        if uploaded_file:
            is_valid, msg, real_df = validate_csv(uploaded_file)
            if is_valid:
                st.session_state.uploaded_lab_data = real_df
                st.success(msg)
            else:
                st.error(msg)

        st.markdown('<div class="sidebar-label">Webhook</div>', unsafe_allow_html=True)
        alert_webhook = st.text_input("URL", value=st.session_state.alert_webhook_url, placeholder="https://hooks.slack.com/...", label_visibility="collapsed")
        st.session_state.alert_webhook_url = alert_webhook

        st.markdown("")
        # Kit inventory
        st.markdown('<div class="sidebar-label">Bestand</div>', unsafe_allow_html=True)
        for kit_name, kit_info in UNIQUE_KITS.items():
            current = st.session_state.kit_inventory.get(kit_name, kit_info["default_stock"])
            new_val = st.number_input(kit_name, min_value=0, value=current, step=500, key=f"stock_{kit_name}", label_visibility="visible")
            st.session_state.kit_inventory[kit_name] = new_val

    st.markdown("")
    if st.button("Daten neu laden", use_container_width=True, type="secondary"):
        st.cache_data.clear()
        raw_df = load_raw()

    st.caption(f"Sync: {datetime.now().strftime('%H:%M')} · RKI AMELAG")


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE
# ─────────────────────────────────────────────────────────────────────────────
use_ml = st.session_state.ml_enabled
stock_on_hand = get_stock_for_pathogen(selected_pathogen)
wastewater_df = fetch_rki_wastewater(raw_df, pathogen=selected_pathogen)
lab_df = generate_lab_volume(wastewater_df, lag_days=14, pathogen=selected_pathogen)

if st.session_state.uploaded_lab_data is not None:
    lab_df = merge_with_synthetic(lab_df, st.session_state.uploaded_lab_data, selected_pathogen)

ml_model_info = None
if use_ml:
    forecaster = LabVolumeForecaster(lab_df, wastewater_df, selected_pathogen)
    forecaster.train()
    ml_model_info = forecaster.model_info
    ml_forecast = forecaster.forecast(periods=forecast_horizon)

forecast_df, kpis = build_forecast(
    lab_df, horizon_days=forecast_horizon, safety_buffer_pct=safety_buffer / 100,
    stock_on_hand=stock_on_hand, scenario_uplift_pct=scenario_uplift / 100,
    pathogen=selected_pathogen,
)
rev_col = [c for c in forecast_df.columns if "Revenue" in c]
rev_col_name = rev_col[0] if rev_col else "Est. Revenue"

# Alerts
alert_mgr = AlertManager()
triggered_alerts = alert_mgr.evaluate_all(kpis, ml_model_info)
if triggered_alerts and st.session_state.alert_webhook_url:
    alert_mgr.send_webhook(st.session_state.alert_webhook_url, triggered_alerts, selected_pathogen)

# Signal fusion (computed once, shown on demand)
with st.spinner(""):
    _gw = fetch_grippeweb(erkrankung="ARE", region="Bundesweit")
    _are = fetch_are_konsultation(bundesland="Bundesweit")
    try:
        _tt = PATHOGEN_SEARCH_TERMS.get(selected_pathogen, [])
        _trends = fetch_google_trends(_tt, timeframe="today 3-m", geo="DE") if _tt else None
    except Exception:
        _trends = None
composite = fuse_all_signals(wastewater_df=wastewater_df, grippeweb_df=_gw, are_df=_are, trends_df=_trends)


# ═════════════════════════════════════════════════════════════════════════════
# VIEW — Zen Mode: One screen, one action
# ═════════════════════════════════════════════════════════════════════════════

# ── Header (minimal) ─────────────────────────────────────────────────────────
st.markdown(
    f'<div class="zen-header">'
    f'<h1>LabPulse AI</h1>'
    f'<span>{selected_pathogen}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── Alert (only when critical — correct progressive disclosure) ──────────────
risk_val = kpis.get("risk_eur", 0)
if risk_val > 0:
    stockout = kpis.get("stockout_day")
    so_str = f" Lager reicht bis {stockout.strftime('%d. %b')}." if stockout else ""
    st.markdown(
        f'<div class="zen-alert">'
        f'<strong>Revenue at Risk: EUR {risk_val:,.0f}</strong> in {forecast_horizon} Tagen.{so_str}'
        f'</div>', unsafe_allow_html=True,
    )

# ── KPI strip (2 numbers — that's it) ───────────────────────────────────────
trend = kpis["trend_pct"]
trend_cls = "warn" if abs(trend) > 5 else "ok"
conf = composite.confidence_pct
conf_cls = "ok" if conf >= 60 else "warn"

st.markdown(
    f'<div class="zen-kpi-strip">'
    f'<div class="zen-kpi"><div class="val">{kpis["predicted_tests_7d"]:,}</div><div class="lbl">Tests progn. 7d</div></div>'
    f'<div class="zen-kpi"><div class="val">EUR {kpis["revenue_forecast_7d"]:,.0f}</div><div class="lbl">Umsatz 7d</div></div>'
    f'<div class="zen-kpi"><div class="val delta {trend_cls}">{trend:+.1f}%</div><div class="lbl">Trend WoW</div></div>'
    f'<div class="zen-kpi"><div class="val delta {conf_cls}">{conf:.0f}%</div><div class="lbl">Signal-Konfidenz</div></div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── ML Toggle (the ONE primary action) ──────────────────────────────────────
_ml_label = "🧠 ML-Prognose aktiv" if use_ml else "🧠 ML-Prognose"
_ml_sub = f"Prophet · {ml_model_info.get('confidence_score', 0):.0f}% Konfidenz" if (use_ml and ml_model_info) else "Prophet Time-Series aktivieren"
st.markdown(f'<div class="zen-ml-row"><div><div class="label">{_ml_label}</div><div class="sub">{_ml_sub}</div></div></div>', unsafe_allow_html=True)
_toggled = st.toggle("ML-Prognose aktivieren", value=st.session_state.ml_enabled, key="ml_zen")
if _toggled != st.session_state.ml_enabled:
    st.session_state.ml_enabled = _toggled
    st.rerun()


# ── THE CHART (hero — the product IS this chart) ────────────────────────────
today = pd.Timestamp(datetime.today()).normalize()
today_str = today.strftime("%Y-%m-%d")
chart_start = today - pd.Timedelta(days=120)

ww_chart = wastewater_df[wastewater_df["date"] >= chart_start].copy()
lab_chart = lab_df[lab_df["date"] >= chart_start].copy()
lab_actuals = lab_chart[lab_chart["date"] <= today]

lab_forecast_chart = forecast_df.rename(columns={"Date": "date", "Predicted Volume": "order_volume"})

fig = make_subplots(specs=[[{"secondary_y": True}]])

# Wastewater 7d
ww_s = ww_chart.copy()
ww_s["vl_7d"] = ww_s["virus_load"].rolling(7, min_periods=1).mean()
fig.add_trace(go.Scatter(
    x=ww_s["date"], y=ww_s["vl_7d"], name="Viruslast (7d Ø)",
    line=dict(color="#5eead4", width=2),
    hovertemplate="%{x|%d %b}: %{y:,.0f} Kopien/L<extra></extra>",
), secondary_y=False)

# Lab 7d
lab_s = lab_actuals.copy()
lab_s["vol_7d"] = lab_s["order_volume"].rolling(7, min_periods=1).mean()
fig.add_trace(go.Scatter(
    x=lab_s["date"], y=lab_s["vol_7d"], name="Labortests (7d Ø)",
    line=dict(color="#d4956a", width=2.5),
    hovertemplate="%{x|%d %b}: %{y:,.0f} Ø/Tag<extra></extra>",
), secondary_y=True)

# ML Forecast
if use_ml and ml_model_info and "ml_forecast" in dir():
    try:
        ml_fc = ml_forecast
        fig.add_trace(go.Scatter(x=ml_fc["date"], y=ml_fc["upper"], showlegend=False, line=dict(width=0), mode="lines"), secondary_y=True)
        fig.add_trace(go.Scatter(
            x=ml_fc["date"], y=ml_fc["lower"], name="ML-Konfidenzband",
            fill="tonexty", fillcolor="rgba(167,139,250,0.12)",
            line=dict(width=0.5, color="rgba(167,139,250,0.2)", dash="dot"), mode="lines",
        ), secondary_y=True)
        fig.add_trace(go.Scatter(
            x=ml_fc["date"], y=ml_fc["predicted"], name="ML-Prognose",
            line=dict(color="#a78bfa", width=3, dash="dash"),
            mode="lines+markers", marker=dict(size=3, color="#a78bfa"),
            hovertemplate="<b>ML</b> %{x|%d %b}: <b>%{y:,.0f}</b><extra></extra>",
        ), secondary_y=True)
    except Exception:
        pass

# Standard forecast
if not lab_forecast_chart.empty:
    vol_col = "order_volume"
    if not lab_actuals.empty:
        connector = pd.DataFrame({"date": [lab_actuals["date"].iloc[-1]], vol_col: [lab_actuals["order_volume"].iloc[-1]]})
        fc_plot = pd.concat([connector, lab_forecast_chart], ignore_index=True)
    else:
        fc_plot = lab_forecast_chart
    fig.add_trace(go.Scatter(
        x=fc_plot["date"], y=fc_plot[vol_col], name="Prognose",
        line=dict(color="#d4956a", width=2, dash="dot"),
        hovertemplate="%{x|%d %b}: %{y:,.0f}<extra></extra>",
    ), secondary_y=True)

# Forecast zone
_fc_end = today + pd.Timedelta(days=forecast_horizon)
fig.add_vrect(x0=today_str, x1=_fc_end.strftime("%Y-%m-%d"), fillcolor="rgba(167,139,250,0.03)", layer="below", line_width=0)

# Today line
fig.add_shape(type="line", x0=today_str, x1=today_str, y0=0, y1=1, yref="paper", line=dict(color="rgba(212,149,106,0.3)", width=1, dash="dot"))
fig.add_annotation(x=today_str, y=1.04, yref="paper", text="heute", showarrow=False, font=dict(size=9, color="#d4956a", family="Bricolage Grotesque"), bgcolor="rgba(212,149,106,0.08)", borderpad=3)

# Auto-zoom when ML active
_xrange = None
if use_ml and ml_model_info:
    _xrange = [(today - pd.Timedelta(days=21)).strftime("%Y-%m-%d"), (today + pd.Timedelta(days=forecast_horizon + 3)).strftime("%Y-%m-%d")]

fig.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(44,44,64,0.5)",
    height=520, margin=dict(l=0, r=0, t=40, b=10),
    legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center", font=dict(size=10, color="#524e48", family="Bricolage Grotesque"), bgcolor="rgba(0,0,0,0)"),
    hovermode="x unified", hoverlabel=dict(bgcolor="#2c2c40", bordercolor="rgba(212,149,106,0.1)", font_size=12, font_family="Bricolage Grotesque"),
    **({"xaxis_range": _xrange} if _xrange else {}),
)
fig.update_xaxes(
    gridcolor="rgba(255,255,255,0.02)", dtick="M1", tickformat="%b '%y",
    tickfont=dict(size=10, color="#524e48"),
    rangeslider=dict(visible=True, thickness=0.035, bgcolor="rgba(14,14,26,0.5)", bordercolor="rgba(255,255,255,0.05)", borderwidth=1),
    rangeselector=dict(
        buttons=[
            dict(count=1, label="1M", step="month", stepmode="backward"),
            dict(count=3, label="3M", step="month", stepmode="backward"),
            dict(count=forecast_horizon, label="Prognose", step="day", stepmode="todate"),
            dict(step="all", label="Alles"),
        ],
        bgcolor="rgba(14,14,26,0.9)", activecolor="#d4956a",
        bordercolor="rgba(255,255,255,0.06)", borderwidth=1,
        font=dict(size=10, color="#d4cfc8", family="Bricolage Grotesque"),
        x=0, y=1.14,
    ),
)
fig.update_yaxes(title_text="Viruslast", secondary_y=False, showgrid=False, title_font=dict(color="#5eead4", size=10), tickfont=dict(size=9, color="#524e48"))
fig.update_yaxes(title_text="Tests/Tag", secondary_y=True, gridcolor="rgba(255,255,255,0.02)", title_font=dict(color="#d4956a", size=10), tickfont=dict(size=9, color="#524e48"))

st.markdown('<div class="zen-chart">', unsafe_allow_html=True)
st.plotly_chart(fig, use_container_width=True, key="main_chart", config={
    "scrollZoom": True, "displayModeBar": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "displaylogo": False,
    "toImageButtonOptions": dict(format="png", filename="labpulse_prognose", scale=2),
})
st.markdown("</div>", unsafe_allow_html=True)
correlation_fig_for_pdf = fig


# ═════════════════════════════════════════════════════════════════════════════
# PROGRESSIVE DISCLOSURE — Everything below is on-demand
# ═════════════════════════════════════════════════════════════════════════════

# ── KI-Analyse (expander) ────────────────────────────────────────────────────
ollama = get_ollama_client()
ollama_ok = ollama.health_check()
if ollama_ok:
    with st.spinner(""):
        ai_insight = ollama.generate_insight(kpis, selected_pathogen)
else:
    ai_insight = ollama._fallback_insight(kpis, selected_pathogen)

with st.expander(f"KI-Analyse — {selected_pathogen}", expanded=False):
    ai_source = "Ollama LLM" if ollama_ok else "Regelbasiert"
    st.caption(f"Engine: {ai_source}")
    st.markdown(f'<div class="zen-body">{ai_insight}</div>', unsafe_allow_html=True)


# ── Signal-Details (expander) ────────────────────────────────────────────────
with st.expander("Signal-Details", expanded=False):
    direction_icons = {"rising": "↑", "falling": "↓", "flat": "→", "mixed": "↔"}
    direction_labels = {"rising": "steigend", "falling": "fallend", "flat": "stabil", "mixed": "uneinheitlich"}
    st.caption(f"Konfidenz: {composite.confidence_pct:.0f}% · Richtung: {direction_labels.get(composite.direction, '?')} ({composite.weighted_trend:+.1f}%)")
    for sig in composite.signals:
        cfg = SIGNAL_CONFIG.get(sig.name, {})
        icon = cfg.get("icon", "")
        if sig.available:
            di = direction_icons.get(sig.direction, "?")
            st.caption(f"{icon} {sig.name}: {di} {sig.magnitude:+.1f}%")
        else:
            st.caption(f"{icon} {sig.name}: —")
    st.markdown("")
    for line in composite.narrative_de.split("\n"):
        if line.strip():
            st.caption(line)


# ── ML-Details (expander, only when active) ──────────────────────────────────
if use_ml and ml_model_info:
    with st.expander("ML-Modell Details", expanded=False):
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Modell", ml_model_info.get("model_type", "—"))
        mc2.metric("Konfidenz", f"{ml_model_info.get('confidence_score', 0):.0f}%")
        mc3.metric("Features", ml_model_info.get("n_features", "—"))
        mc4.metric("Training", f"{ml_model_info.get('training_days', '—')}d")


# ═════════════════════════════════════════════════════════════════════════════
# TABS — 3 instead of 5 (merged)
# ═════════════════════════════════════════════════════════════════════════════
tab_forecast, tab_data, tab_regional = st.tabs(["Prognose", "Signale & Trends", "Regional"])


# ── TAB: Prognose ────────────────────────────────────────────────────────────
with tab_forecast:
    burndown_fig_for_pdf = None
    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.caption("Reagenz-Burndown")
        remaining = kpis.get("remaining_stock", [])
        if remaining:
            bd = forecast_df["Date"].values
            fig_burn = go.Figure()
            fig_burn.add_trace(go.Scatter(x=bd, y=remaining, name="Restbestand", fill="tozeroy", line=dict(color="#5eead4", width=2), fillcolor="rgba(94,234,212,0.06)", hovertemplate="%{x|%d %b}: %{y:,.0f}<extra></extra>"))
            fig_burn.add_trace(go.Scatter(x=bd, y=np.cumsum(forecast_df["Predicted Volume"].values), name="Kum. Bedarf", line=dict(color="#ef4444", width=1.5, dash="dot"), hovertemplate="%{x|%d %b}: %{y:,.0f}<extra></extra>"))
            fig_burn.add_hline(y=stock_on_hand, line_dash="longdash", line_color="rgba(148,163,184,0.2)")

            stockout_day = kpis.get("stockout_day")
            if stockout_day:
                so_str = pd.Timestamp(stockout_day).strftime("%Y-%m-%d")
                fig_burn.add_shape(type="line", x0=so_str, x1=so_str, y0=0, y1=1, yref="paper", line=dict(color="rgba(239,68,68,0.4)", width=1, dash="dash"))

            fig_burn.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(44,44,64,0.5)", height=300, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center", font=dict(size=9, color="#524e48"), bgcolor="rgba(0,0,0,0)"), hovermode="x unified")
            fig_burn.update_xaxes(gridcolor="rgba(255,255,255,0.02)", tickfont=dict(size=9, color="#524e48"))
            fig_burn.update_yaxes(gridcolor="rgba(255,255,255,0.02)", tickfont=dict(size=9, color="#524e48"))
            st.plotly_chart(fig_burn, use_container_width=True, key="burndown")
            burndown_fig_for_pdf = fig_burn

    with col_r:
        st.caption("Bestellempfehlungen")
        def hl(row):
            if row["Reagent Order"] > 0:
                return ["background-color: rgba(239,68,68,0.08); color: #fca5a5; font-weight: 600;"] * len(row)
            return ["color: #8a857e;"] * len(row)
        styled = forecast_df.style.apply(hl, axis=1).format({"Predicted Volume": "{:,.0f}", "Reagent Order": "{:,.0f}", rev_col_name: "EUR {:,.0f}"})
        st.dataframe(styled, use_container_width=True, hide_index=True, height=350)

    # PDF export
    with st.expander("PDF-Export"):
        if st.button("PDF-Bericht generieren", use_container_width=True, type="primary"):
            with st.spinner("Generiere …"):
                try:
                    ai_commentary = ollama.generate_pdf_commentary(kpis, selected_pathogen) if ollama_ok else ai_insight
                    pdf_bytes = generate_pdf_report(kpis=kpis, forecast_df=forecast_df, correlation_fig=correlation_fig_for_pdf, burndown_fig=burndown_fig_for_pdf, ai_commentary=ai_commentary, pathogen=selected_pathogen)
                    st.download_button("Herunterladen", data=pdf_bytes, file_name=f"LabPulse_{selected_pathogen}_{datetime.now().strftime('%Y%m%d')}.pdf", mime="application/pdf", use_container_width=True)
                except Exception as exc:
                    st.error(f"Fehler: {exc}")


# ── TAB: Signale & Trends (merged from 3 old tabs) ──────────────────────────
with tab_data:
    st.caption("Surveillance-Signale — GrippeWeb & ARE")
    sc1, sc2 = st.columns(2, gap="large")

    with sc1:
        gw_type = st.radio("GrippeWeb", ["ARE", "ILI"], horizontal=True, key="gw_type")
        with st.spinner(""):
            gw_df = fetch_grippeweb(erkrankung=gw_type, region="Bundesweit")
        if not gw_df.empty:
            gw_r = gw_df[gw_df["date"] >= (today - pd.Timedelta(days=730))].copy()
            fig_gw = go.Figure()
            fig_gw.add_trace(go.Scatter(x=gw_r["date"], y=gw_r["incidence"], fill="tozeroy", line=dict(color="#a78bfa", width=2), fillcolor="rgba(167,139,250,0.05)", hovertemplate="%{x|%d %b}: %{y:,.1f}<extra></extra>"))
            fig_gw.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(44,44,64,0.5)", height=260, margin=dict(l=0, r=0, t=5, b=0), showlegend=False, hovermode="x unified")
            fig_gw.update_xaxes(gridcolor="rgba(255,255,255,0.02)", tickfont=dict(size=9, color="#524e48"))
            fig_gw.update_yaxes(gridcolor="rgba(255,255,255,0.02)", tickfont=dict(size=9, color="#524e48"))
            st.plotly_chart(fig_gw, use_container_width=True, key="gw")
        else:
            st.caption("Nicht verfuegbar.")

    with sc2:
        st.caption("ARE-Konsultationen")
        with st.spinner(""):
            are_df = fetch_are_konsultation(bundesland="Bundesweit")
        if not are_df.empty:
            ar = are_df[are_df["date"] >= (today - pd.Timedelta(days=730))].copy()
            fig_are = go.Figure()
            fig_are.add_trace(go.Scatter(x=ar["date"], y=ar["consultation_incidence"], fill="tozeroy", line=dict(color="#5eead4", width=2), fillcolor="rgba(94,234,212,0.05)", hovertemplate="%{x|%d %b}: %{y:,.0f}<extra></extra>"))
            fig_are.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(44,44,64,0.5)", height=260, margin=dict(l=0, r=0, t=5, b=0), showlegend=False, hovermode="x unified")
            fig_are.update_xaxes(gridcolor="rgba(255,255,255,0.02)", tickfont=dict(size=9, color="#524e48"))
            fig_are.update_yaxes(gridcolor="rgba(255,255,255,0.02)", tickfont=dict(size=9, color="#524e48"))
            st.plotly_chart(fig_are, use_container_width=True, key="are")
        else:
            st.caption("Nicht verfuegbar.")

    # Google Trends (collapsed by default)
    st.markdown("")
    with st.expander("Google Trends"):
        suggested_terms = PATHOGEN_SEARCH_TERMS.get(selected_pathogen, [])
        if "trends_custom_terms" not in st.session_state:
            st.session_state.trends_custom_terms = ", ".join(suggested_terms[:3])

        tc1, tc2 = st.columns([3, 1])
        with tc1:
            custom_input = st.text_input("Suchbegriffe (max 5)", value=st.session_state.trends_custom_terms, key="trends_input", label_visibility="collapsed", placeholder="z.B. Corona Test, Grippe")
            st.session_state.trends_custom_terms = custom_input
        with tc2:
            tf = st.selectbox("Zeitraum", ["today 3-m", "today 12-m", "today 5-y"], index=1, format_func=lambda x: {"today 3-m": "3M", "today 12-m": "12M", "today 5-y": "5J"}[x], key="trends_tf", label_visibility="collapsed")

        user_terms = [t.strip() for t in custom_input.split(",") if t.strip()][:5]
        if user_terms:
            with st.spinner(""):
                trends_df = fetch_google_trends(user_terms, timeframe=tf, geo="DE")
            if not trends_df.empty:
                fig_t = go.Figure()
                colors = ["#d4956a", "#5eead4", "#a78bfa", "#a3e635", "#fb7185"]
                for i, col in enumerate([c for c in trends_df.columns if c != "date"]):
                    fig_t.add_trace(go.Scatter(x=trends_df["date"], y=trends_df[col], name=col, line=dict(color=colors[i % 5], width=2)))
                fig_t.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(44,44,64,0.5)", height=280, margin=dict(l=0, r=0, t=5, b=0), legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center", font=dict(size=9, color="#524e48"), bgcolor="rgba(0,0,0,0)"), hovermode="x unified")
                fig_t.update_xaxes(gridcolor="rgba(255,255,255,0.02)", tickfont=dict(size=9, color="#524e48"))
                fig_t.update_yaxes(gridcolor="rgba(255,255,255,0.02)", tickfont=dict(size=9, color="#524e48"))
                st.plotly_chart(fig_t, use_container_width=True, key="trends")
            else:
                st.caption("Keine Daten (Rate-Limit oder zu nischig).")
        else:
            st.caption("Suchbegriffe eingeben.")


# ── TAB: Regional ────────────────────────────────────────────────────────────
with tab_regional:
    if raw_df.empty or "bundesland" not in raw_df.columns:
        st.caption("Keine regionalen Daten verfuegbar.")
    else:
        pathogen_types = PATHOGEN_GROUPS.get(selected_pathogen, [selected_pathogen])
        regional_df = aggregate_by_bundesland(raw_df, pathogen_types, days_back=30)

        if regional_df.empty:
            st.caption(f"Keine Daten fuer {selected_pathogen}.")
        else:
            map_df = create_scatter_map_data(regional_df)
            if not map_df.empty:
                col_m, col_t = st.columns([2, 1], gap="large")
                with col_m:
                    vl_min, vl_max = map_df["avg_virus_load"].min(), map_df["avg_virus_load"].max()
                    map_df["bs"] = 25 if vl_max <= vl_min else 12 + (map_df["avg_virus_load"] - vl_min) / (vl_max - vl_min) * 40
                    fig_m = go.Figure()
                    fig_m.add_trace(go.Scatter(
                        x=map_df["lon"], y=map_df["lat"], mode="markers+text",
                        marker=dict(size=map_df["bs"], color=map_df["trend_pct"], colorscale=[[0, "#5eead4"], [0.5, "#d4956a"], [1, "#ef4444"]], opacity=0.8, line=dict(width=1, color="rgba(255,255,255,0.1)")),
                        text=map_df["bundesland"], textposition="top center", textfont=dict(size=9, color="#524e48"),
                        hovertemplate="<b>%{text}</b><br>Ø %{customdata[0]:,.0f}<br>Trend %{customdata[1]:+.1f}%<extra></extra>",
                        customdata=map_df[["avg_virus_load", "trend_pct"]].values,
                    ))
                    fig_m.update_layout(template="plotly_dark", height=480, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(44,44,64,0.5)",
                        xaxis=dict(range=[5.5, 15.5], showgrid=True, gridcolor="rgba(255,255,255,0.02)", tickfont=dict(size=9, color="#524e48")),
                        yaxis=dict(range=[47, 55.5], showgrid=True, gridcolor="rgba(255,255,255,0.02)", scaleanchor="x", scaleratio=1.5, tickfont=dict(size=9, color="#524e48")),
                        showlegend=False)
                    st.plotly_chart(fig_m, use_container_width=True, key="map")

                with col_t:
                    st.caption(f"Top-Regionen · {selected_pathogen}")
                    disp = regional_df[["bundesland", "avg_virus_load", "trend_pct", "site_count"]].copy()
                    disp.columns = ["Land", "Ø Viruslast", "Trend", "Standorte"]
                    disp["Ø Viruslast"] = disp["Ø Viruslast"].map(lambda x: f"{x:,.0f}")
                    disp["Trend"] = disp["Trend"].map(lambda x: f"{x:+.1f}%")
                    st.dataframe(disp, use_container_width=True, hide_index=True, height=400)


# ── Footer ───────────────────────────────────────────────────────────────────
cost = kpis.get("cost_per_test", AVG_REVENUE_PER_TEST)
ml_str = f" · ML: {ml_model_info['model_type']} ({ml_model_info['confidence_score']:.0f}%)" if ml_model_info else ""
st.markdown(
    f'<div class="zen-foot">'
    f'<span>LabPulse AI v2 · {selected_pathogen} · EUR {cost}/Test{ml_str}</span>'
    f'<span><a href="https://github.com/robert-koch-institut/Abwassersurveillance_AMELAG" target="_blank">RKI AMELAG</a> · {datetime.now().strftime("%Y-%m-%d %H:%M")}</span>'
    f'</div>',
    unsafe_allow_html=True,
)
