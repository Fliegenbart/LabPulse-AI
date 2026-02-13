"""
LabPulse AI — Streamlit Dashboard
==================================
Predictive reagent demand management powered by RKI wastewater surveillance.
Built for Ganzimmun Diagnostics (Limbach Group) — Hackathon MVP.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from data_engine import (
    fetch_rki_wastewater,
    generate_lab_volume,
    build_forecast,
    AVG_REVENUE_PER_TEST,
)

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LabPulse AI — Ganzimmun Edition",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,.3);
    }
    div[data-testid="stMetric"] label {
        color: #8892b0 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #e6f1ff !important;
        font-size: 1.6rem !important;
        font-weight: 700;
    }

    .section-header {
        color: #ccd6f6;
        font-size: 1.15rem;
        font-weight: 600;
        border-bottom: 2px solid #f77f00;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
        letter-spacing: 0.02em;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a1a 0%, #1a1a2e 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h1 { color: #f77f00; font-size: 1.4rem; }

    .stAlert { font-weight: bold; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🧬 LabPulse AI")
    st.caption("Ganzimmun Edition · Limbach Group")
    st.divider()

    st.subheader("⚙️ Forecast Settings")
    forecast_horizon = st.slider(
        "Forecast Horizon (days)", min_value=7, max_value=21, value=14, step=7
    )
    safety_buffer = st.slider(
        "Safety Stock Buffer (%)", min_value=0, max_value=30, value=10, step=5
    )
    stock_on_hand = st.number_input(
        "Current Reagent Stock (units)", min_value=0, value=5000, step=500
    )

    st.divider()
    st.markdown("**🔬 Stress Test Scenario**")
    scenario_uplift = st.slider(
        "Virus Load Uplift (%)",
        min_value=0, max_value=50, value=0, step=5,
        help="Simulate a sudden spike in infection rates to stress-test supply chain resilience.",
    )

    st.divider()
    refresh = st.button("🔄  Refresh Data from RKI", use_container_width=True)
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M')}")
    st.caption("Data: RKI AMELAG Wastewater Surveillance")


# ── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Fetching RKI wastewater data …")
def load_data():
    ww = fetch_rki_wastewater()
    lab = generate_lab_volume(ww, lag_days=14)
    return ww, lab


if refresh:
    st.cache_data.clear()

wastewater_df, lab_df = load_data()

forecast_df, kpis = build_forecast(
    lab_df,
    horizon_days=forecast_horizon,
    safety_buffer_pct=safety_buffer / 100,
    stock_on_hand=stock_on_hand,
    scenario_uplift_pct=scenario_uplift / 100,
)

# ── Header & Alerts ──────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.5rem;">
        <span style="font-size:2rem;">🧬</span>
        <div>
            <span style="font-size:1.6rem;font-weight:700;color:#e6f1ff;">LabPulse AI</span>
            <span style="font-size:0.9rem;color:#8892b0;margin-left:0.6rem;">
                Predictive Supply Chain Control
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Critical alert banner
if kpis.get("risk_eur", 0) > 0:
    stockout = kpis.get("stockout_day")
    stockout_str = f" Stock depleted by **{stockout.strftime('%d %b %Y')}**." if stockout else ""
    st.error(
        f"🚨 **CRITICAL:** Revenue at Risk of **€{kpis['risk_eur']:,.0f}** "
        f"over the next {forecast_horizon} days.{stockout_str} Order immediately!"
    )
elif scenario_uplift > 0:
    st.info(f"ℹ️ **Simulation active:** +{scenario_uplift}% virus load uplift applied to forecast.")


# ── Section 1: Controller's Cockpit ─────────────────────────────────────────
st.markdown(
    '<div class="section-header">📊 Controller\'s Cockpit</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Predicted Tests (7 d)", f"{kpis['predicted_tests_7d']:,}")
with c2:
    st.metric("Revenue Forecast (7 d)", f"€{kpis['revenue_forecast_7d']:,.0f}")
with c3:
    st.metric(
        "Reagent Status",
        kpis["reagent_status"],
        delta="Stock OK" if "Optimal" in kpis["reagent_status"] else "Stockout Risk!",
        delta_color="normal" if "Optimal" in kpis["reagent_status"] else "inverse",
    )
with c4:
    risk_val = kpis.get("risk_eur", 0)
    st.metric(
        "Revenue at Risk",
        f"€{risk_val:,.0f}",
        delta="No risk" if risk_val == 0 else "Action needed!",
        delta_color="off" if risk_val == 0 else "inverse",
    )
with c5:
    st.metric(
        "Trend vs Last Week",
        f"{kpis['trend_pct']:+.1f} %",
        delta_color="normal" if kpis["trend_pct"] >= 0 else "inverse",
    )


# ── Section 2: Correlation Chart ────────────────────────────────────────────
st.markdown(
    '<div class="section-header">📈 Wastewater Signal → Lab Volume Correlation</div>',
    unsafe_allow_html=True,
)

today = pd.Timestamp(datetime.today()).normalize()
chart_start = today - pd.Timedelta(days=120)

ww_chart = wastewater_df[wastewater_df["date"] >= chart_start].copy()
lab_chart = lab_df[lab_df["date"] >= chart_start].copy()
lab_actuals = lab_chart[lab_chart["date"] <= today]

# Prepare forecast line for chart
lab_forecast_chart = forecast_df.rename(
    columns={"Date": "date", "Predicted Volume": "order_volume"}
)

fig = make_subplots(specs=[[{"secondary_y": True}]])

# 1. Wastewater (leading indicator)
fig.add_trace(
    go.Scatter(
        x=ww_chart["date"],
        y=ww_chart["virus_load"],
        name="Virus Load (RKI)",
        line=dict(color="#4a6fa5", width=2),
        opacity=0.6,
        hovertemplate="%{x|%d %b}: %{y:,.0f} copies/L<extra></extra>",
    ),
    secondary_y=False,
)

# 2. Lab actuals
fig.add_trace(
    go.Scatter(
        x=lab_actuals["date"],
        y=lab_actuals["order_volume"],
        name="Lab Tests (Actual)",
        line=dict(color="#f77f00", width=2.5),
        hovertemplate="%{x|%d %b}: %{y:,.0f} tests<extra></extra>",
    ),
    secondary_y=True,
)

# 3. Lab forecast (dashed, connected to last actual)
if not lab_forecast_chart.empty:
    vol_col = "order_volume"
    if not lab_actuals.empty:
        connector = pd.DataFrame(
            {
                "date": [lab_actuals["date"].iloc[-1]],
                vol_col: [lab_actuals["order_volume"].iloc[-1]],
            }
        )
        fc_plot = pd.concat([connector, lab_forecast_chart], ignore_index=True)
    else:
        fc_plot = lab_forecast_chart

    label = f"Forecast (+{scenario_uplift}%)" if scenario_uplift > 0 else "Forecast"
    fig.add_trace(
        go.Scatter(
            x=fc_plot["date"],
            y=fc_plot[vol_col],
            name=label,
            line=dict(color="#f77f00", width=2.5, dash="dot"),
            hovertemplate="%{x|%d %b}: %{y:,.0f} predicted<extra></extra>",
        ),
        secondary_y=True,
    )

# TODAY marker
fig.add_vline(
    x=today,
    line_width=2,
    line_dash="dash",
    line_color="#e74c3c",
    annotation_text="TODAY",
    annotation_position="top",
    annotation_font_color="#e74c3c",
    annotation_font_size=12,
)

# Lag annotation
fig.add_annotation(
    x=today - pd.Timedelta(days=7),
    y=0.95,
    yref="paper",
    text="← 14-day lag →",
    showarrow=False,
    font=dict(size=11, color="#8892b0"),
    bgcolor="rgba(26,26,46,0.8)",
    bordercolor="#0f3460",
    borderwidth=1,
    borderpad=4,
)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,10,26,0.6)",
    height=440,
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center", font=dict(size=11)),
    hovermode="x unified",
)
fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", dtick="M1", tickformat="%b %Y")
fig.update_yaxes(
    title_text="Virus Load (copies/L)",
    secondary_y=False,
    showgrid=False,
    title_font=dict(color="#4a6fa5"),
)
fig.update_yaxes(
    title_text="Daily Lab Tests",
    secondary_y=True,
    gridcolor="rgba(255,255,255,0.1)",
    title_font=dict(color="#f77f00"),
)

st.plotly_chart(fig, use_container_width=True)


# ── Section 2b: Stock Burndown Chart ────────────────────────────────────────
st.markdown(
    '<div class="section-header">🏭 Reagent Stock Burndown</div>',
    unsafe_allow_html=True,
)

remaining = kpis.get("remaining_stock", [])
if remaining:
    burndown_dates = forecast_df["Date"].values
    fig_burn = go.Figure()

    # Remaining stock area
    fig_burn.add_trace(
        go.Scatter(
            x=burndown_dates,
            y=remaining,
            name="Remaining Stock",
            fill="tozeroy",
            line=dict(color="#2ecc71", width=2),
            fillcolor="rgba(46,204,113,0.15)",
            hovertemplate="%{x|%d %b}: %{y:,.0f} units left<extra></extra>",
        )
    )

    # Cumulative demand line
    forecast_buffered = forecast_df["Predicted Volume"].values
    cum_demand = np.cumsum(forecast_buffered)
    fig_burn.add_trace(
        go.Scatter(
            x=burndown_dates,
            y=cum_demand,
            name="Cumulative Demand",
            line=dict(color="#e74c3c", width=2, dash="dot"),
            hovertemplate="%{x|%d %b}: %{y:,.0f} cumulative tests<extra></extra>",
        )
    )

    # Starting stock reference line
    fig_burn.add_hline(
        y=stock_on_hand,
        line_dash="longdash",
        line_color="#8892b0",
        annotation_text=f"Starting Stock: {stock_on_hand:,}",
        annotation_position="top left",
        annotation_font_color="#8892b0",
    )

    # Mark stockout day
    stockout_day = kpis.get("stockout_day")
    if stockout_day is not None:
        fig_burn.add_vline(
            x=stockout_day,
            line_width=2,
            line_dash="dash",
            line_color="#e74c3c",
            annotation_text="STOCKOUT",
            annotation_position="top",
            annotation_font_color="#e74c3c",
            annotation_font_size=11,
        )

    fig_burn.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,26,0.6)",
        height=300,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center", font=dict(size=11)),
        hovermode="x unified",
        yaxis_title="Units",
    )
    fig_burn.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig_burn.update_yaxes(gridcolor="rgba(255,255,255,0.1)")

    st.plotly_chart(fig_burn, use_container_width=True)


# ── Section 3: Actionable Insights Table ─────────────────────────────────────
st.markdown(
    '<div class="section-header">📋 Actionable Order Recommendation</div>',
    unsafe_allow_html=True,
)


def highlight_orders(row):
    if row["Reagent Order"] > 0:
        return [
            "background-color: rgba(231, 76, 60, 0.25); color: #ff8888; font-weight: bold;"
        ] * len(row)
    return [""] * len(row)


styled = forecast_df.style.apply(highlight_orders, axis=1).format(
    {
        "Predicted Volume": "{:,.0f}",
        "Reagent Order": "{:,.0f}",
        "Est. Revenue (€)": "€{:,.0f}",
    }
)

st.dataframe(
    styled,
    use_container_width=True,
    hide_index=True,
    height=min(400, 40 + 35 * len(forecast_df)),
)


# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
col_a, col_b = st.columns([3, 1])
with col_a:
    st.caption(
        "LabPulse AI · Hackathon MVP · "
        "Data: RKI AMELAG Abwassersurveillance · "
        f"Avg. Revenue/Test: €{AVG_REVENUE_PER_TEST}"
    )
with col_b:
    st.caption(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
