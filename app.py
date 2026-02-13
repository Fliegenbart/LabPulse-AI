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
    fetch_rki_raw,
    fetch_rki_wastewater,
    get_available_pathogens,
    generate_lab_volume,
    build_forecast,
    AVG_REVENUE_PER_TEST,
    PATHOGEN_REAGENT_MAP,
)

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

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Data Loading (cached) ───────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Fetching RKI wastewater data …")
def load_raw():
    return fetch_rki_raw()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '# 🧬 LabPulse AI <span class="sidebar-pill">BETA</span>',
        unsafe_allow_html=True,
    )
    st.caption("Ganzimmun Diagnostics · Limbach Group")

    st.markdown("")
    st.markdown('<div class="sidebar-label">Pathogen</div>', unsafe_allow_html=True)

    raw_df = load_raw()
    pathogens = get_available_pathogens(raw_df)
    selected_pathogen = st.selectbox(
        "Select Pathogen",
        pathogens,
        index=0,
        label_visibility="collapsed",
    )

    reagent_info = PATHOGEN_REAGENT_MAP.get(selected_pathogen, {})
    if reagent_info:
        st.caption(
            f"Kit: {reagent_info.get('test_name', '–')} · "
            f"€{reagent_info.get('cost_per_test', 45)}/test"
        )

    st.markdown("")
    st.markdown('<div class="sidebar-label">Forecast Configuration</div>', unsafe_allow_html=True)
    forecast_horizon = st.slider(
        "Forecast Horizon (days)", min_value=7, max_value=21, value=14, step=7
    )
    safety_buffer = st.slider(
        "Safety Stock Buffer (%)", min_value=0, max_value=30, value=10, step=5
    )
    stock_on_hand = st.number_input(
        "Current Reagent Stock (units)", min_value=0, value=5000, step=500
    )

    st.markdown("")
    st.markdown('<div class="sidebar-label">Stress Test</div>', unsafe_allow_html=True)
    scenario_uplift = st.slider(
        "Virus Load Uplift (%)",
        min_value=0, max_value=50, value=0, step=5,
        help="Simulate a sudden spike to stress-test supply chain resilience.",
    )

    st.markdown("")
    refresh = st.button("Refresh Data from RKI", use_container_width=True, type="secondary")
    if refresh:
        st.cache_data.clear()
        raw_df = load_raw()

    st.markdown("")
    st.caption(f"Last sync: {datetime.now().strftime('%H:%M')} · RKI AMELAG")


# ── Load pathogen-specific data ──────────────────────────────────────────────
wastewater_df = fetch_rki_wastewater(raw_df, pathogen=selected_pathogen)
lab_df = generate_lab_volume(wastewater_df, lag_days=14, pathogen=selected_pathogen)

forecast_df, kpis = build_forecast(
    lab_df,
    horizon_days=forecast_horizon,
    safety_buffer_pct=safety_buffer / 100,
    stock_on_hand=stock_on_hand,
    scenario_uplift_pct=scenario_uplift / 100,
    pathogen=selected_pathogen,
)

# Revenue column name (dynamic per pathogen)
rev_col = [c for c in forecast_df.columns if "Revenue" in c]
rev_col_name = rev_col[0] if rev_col else "Est. Revenue"


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="app-header">
        <div class="app-header-icon">🧬</div>
        <div class="app-header-text">
            <h1>LabPulse AI</h1>
            <p>Predictive Supply Chain Control · Powered by RKI Wastewater Surveillance</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Alert Banners ────────────────────────────────────────────────────────────
if kpis.get("risk_eur", 0) > 0:
    stockout = kpis.get("stockout_day")
    stockout_str = (
        f" Stock depleted by <strong>{stockout.strftime('%d %b %Y')}</strong>."
        if stockout else ""
    )
    st.markdown(
        f'<div class="alert-critical">'
        f"<strong>CRITICAL</strong> — {selected_pathogen}: Revenue at Risk "
        f"<strong>€{kpis['risk_eur']:,.0f}</strong> "
        f"over the next {forecast_horizon} days.{stockout_str} "
        f"Immediate reorder of {kpis.get('test_name', 'reagents')} recommended."
        f"</div>",
        unsafe_allow_html=True,
    )
elif scenario_uplift > 0:
    st.markdown(
        f'<div class="alert-info">'
        f"<strong>Simulation Active</strong> — {selected_pathogen} scenario with "
        f"+{scenario_uplift}% virus load uplift."
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Section 1: KPI Dashboard ────────────────────────────────────────────────
st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5, gap="medium")
with c1:
    st.metric("Predicted Tests (7 d)", f"{kpis['predicted_tests_7d']:,}")
with c2:
    st.metric("Revenue Forecast (7 d)", f"€{kpis['revenue_forecast_7d']:,.0f}")
with c3:
    st.metric(
        "Reagent Status", kpis["reagent_status"],
        delta="Stock OK" if "Optimal" in kpis["reagent_status"] else "Stockout Risk!",
        delta_color="normal" if "Optimal" in kpis["reagent_status"] else "inverse",
    )
with c4:
    risk_val = kpis.get("risk_eur", 0)
    st.metric(
        "Revenue at Risk", f"€{risk_val:,.0f}",
        delta="No risk" if risk_val == 0 else "Action needed!",
        delta_color="off" if risk_val == 0 else "inverse",
    )
with c5:
    st.metric(
        "Week-over-Week", f"{kpis['trend_pct']:+.1f}%",
        delta=f"{kpis['trend_pct']:+.1f}%",
        delta_color="normal" if kpis["trend_pct"] >= 0 else "inverse",
    )


# ── Section 2: Correlation Chart ────────────────────────────────────────────
st.markdown(
    f'<div class="section-header">{selected_pathogen} — Wastewater vs. Lab Volume</div>',
    unsafe_allow_html=True,
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
        name="Virus Load (Daily)", showlegend=False,
        line=dict(color="#3b82f6", width=0.5), opacity=0.15,
        fill="tozeroy", fillcolor="rgba(59,130,246,0.03)",
        hovertemplate="%{x|%d %b}: %{y:,.0f} copies/L<extra></extra>",
    ),
    secondary_y=False,
)

# Wastewater 7d average
ww_smoothed = ww_chart.copy()
ww_smoothed["vl_7d"] = ww_smoothed["virus_load"].rolling(7, min_periods=1).mean()
fig.add_trace(
    go.Scatter(
        x=ww_smoothed["date"], y=ww_smoothed["vl_7d"],
        name="Virus Load (7d Avg)",
        line=dict(color="#3b82f6", width=2),
        hovertemplate="%{x|%d %b}: %{y:,.0f} avg copies/L<extra></extra>",
    ),
    secondary_y=False,
)

# Lab raw (faint)
fig.add_trace(
    go.Scatter(
        x=lab_actuals["date"], y=lab_actuals["order_volume"],
        name="Lab Tests (Daily)", showlegend=False,
        line=dict(color="#f77f00", width=1), opacity=0.2,
        hovertemplate="%{x|%d %b}: %{y:,.0f} tests<extra></extra>",
    ),
    secondary_y=True,
)

# Lab 7d average
lab_smoothed = lab_actuals.copy()
lab_smoothed["vol_7d"] = lab_smoothed["order_volume"].rolling(7, min_periods=1).mean()
fig.add_trace(
    go.Scatter(
        x=lab_smoothed["date"], y=lab_smoothed["vol_7d"],
        name="Lab Tests (7d Avg)",
        line=dict(color="#f77f00", width=2.5),
        hovertemplate="%{x|%d %b}: %{y:,.0f} avg/day<extra></extra>",
    ),
    secondary_y=True,
)

# Forecast (dashed)
if not lab_forecast_chart.empty:
    vol_col = "order_volume"
    if not lab_actuals.empty:
        connector = pd.DataFrame(
            {"date": [lab_actuals["date"].iloc[-1]], vol_col: [lab_actuals["order_volume"].iloc[-1]]}
        )
        fc_plot = pd.concat([connector, lab_forecast_chart], ignore_index=True)
    else:
        fc_plot = lab_forecast_chart

    label = f"Forecast (+{scenario_uplift}%)" if scenario_uplift > 0 else "Forecast"
    fig.add_trace(
        go.Scatter(
            x=fc_plot["date"], y=fc_plot[vol_col], name=label,
            line=dict(color="#f77f00", width=2.5, dash="dot"),
            hovertemplate="%{x|%d %b}: %{y:,.0f} predicted<extra></extra>",
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
    x=today_str, y=1.06, yref="paper", text="TODAY", showarrow=False,
    font=dict(size=9, color="#ef4444", family="Inter"),
    bgcolor="rgba(239,68,68,0.1)", borderpad=3,
    bordercolor="rgba(239,68,68,0.2)", borderwidth=1,
)

# Lag annotation
fig.add_annotation(
    x=(today - pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
    y=0.93, yref="paper", text="14-day lag", showarrow=False,
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
fig.update_yaxes(title_text="Virus Load", secondary_y=False, showgrid=False, title_font=dict(color="#3b82f6", size=11), tickfont=dict(size=9, color="#64748b"))
fig.update_yaxes(title_text="Tests / Day", secondary_y=True, gridcolor="rgba(255,255,255,0.04)", title_font=dict(color="#f77f00", size=11), tickfont=dict(size=9, color="#64748b"))

st.markdown('<div class="chart-container">', unsafe_allow_html=True)
st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)


# ── Section 3: Burndown + Table side by side ─────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    st.markdown('<div class="section-header">Reagent Stock Burndown</div>', unsafe_allow_html=True)

    remaining = kpis.get("remaining_stock", [])
    if remaining:
        burndown_dates = forecast_df["Date"].values
        fig_burn = go.Figure()

        fig_burn.add_trace(
            go.Scatter(
                x=burndown_dates, y=remaining, name="Remaining Stock",
                fill="tozeroy", line=dict(color="#22c55e", width=2),
                fillcolor="rgba(34,197,94,0.1)",
                hovertemplate="%{x|%d %b}: %{y:,.0f} units<extra></extra>",
            )
        )

        cum_demand = np.cumsum(forecast_df["Predicted Volume"].values)
        fig_burn.add_trace(
            go.Scatter(
                x=burndown_dates, y=cum_demand, name="Cumulative Demand",
                line=dict(color="#ef4444", width=1.5, dash="dot"),
                hovertemplate="%{x|%d %b}: %{y:,.0f} cumulative<extra></extra>",
            )
        )

        fig_burn.add_hline(
            y=stock_on_hand, line_dash="longdash", line_color="rgba(148,163,184,0.3)",
            annotation_text=f"Stock: {stock_on_hand:,}", annotation_position="top left",
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
            yaxis_title="Units", yaxis_title_font=dict(size=11, color="#64748b"),
        )
        fig_burn.update_xaxes(gridcolor="rgba(255,255,255,0.03)", tickfont=dict(size=9, color="#64748b"))
        fig_burn.update_yaxes(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=9, color="#64748b"))

        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig_burn, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-header">Order Recommendations</div>', unsafe_allow_html=True)

    def highlight_orders(row):
        if row["Reagent Order"] > 0:
            return ["background-color: rgba(239,68,68,0.1); color: #fca5a5; font-weight: 600;"] * len(row)
        return ["color: #94a3b8;"] * len(row)

    styled = forecast_df.style.apply(highlight_orders, axis=1).format(
        {
            "Predicted Volume": "{:,.0f}",
            "Reagent Order": "{:,.0f}",
            rev_col_name: "€{:,.0f}",
        }
    )

    st.dataframe(styled, use_container_width=True, hide_index=True, height=380)


# ── Footer ───────────────────────────────────────────────────────────────────
cost = kpis.get("cost_per_test", AVG_REVENUE_PER_TEST)
st.markdown(
    f"""
    <div class="footer-bar">
        <span>LabPulse AI · {selected_pathogen} · {kpis.get('test_name', '')} · €{cost}/test</span>
        <span>Data: <a href="https://github.com/robert-koch-institut/Abwassersurveillance_AMELAG" target="_blank">
            RKI AMELAG
        </a> · {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
    </div>
    """,
    unsafe_allow_html=True,
)
