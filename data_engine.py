"""
LabPulse AI — Data Engine
=========================
Handles RKI wastewater data ingestion, DuckDB OLAP processing,
and synthetic lab volume generation with configurable time-lag correlation.
"""

import io
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
RKI_AMELAG_URL = (
    "https://raw.githubusercontent.com/robert-koch-institut/"
    "Abwassersurveillance_AMELAG/main/amelag_einzelstandorte.tsv"
)
LAB_SCALE_MIN = 500
LAB_SCALE_MAX = 2000
AVG_REVENUE_PER_TEST = 45  # €
DEFAULT_LAG_DAYS = 14
NOISE_FACTOR = 0.10  # ±10 %
REQUEST_TIMEOUT = 30  # seconds


# ── DuckDB Singleton ────────────────────────────────────────────────────────
_CON: Optional[duckdb.DuckDBPyConnection] = None


def _get_con() -> duckdb.DuckDBPyConnection:
    """Return a module-level in-memory DuckDB connection (lazy init)."""
    global _CON
    if _CON is None:
        _CON = duckdb.connect(database=":memory:")
    return _CON


# ── RKI Data Fetching ────────────────────────────────────────────────────────
def fetch_rki_wastewater(force_refresh: bool = False) -> pd.DataFrame:
    """
    Download RKI AMELAG TSV → DuckDB → aggregate national virus-load by date.

    Falls back to fully synthetic data if the download fails.
    """
    try:
        logger.info("Fetching RKI AMELAG data …")
        resp = requests.get(RKI_AMELAG_URL, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()

        con = _get_con()

        # Load TSV into DuckDB temp table
        raw_df = pd.read_csv(
            io.StringIO(resp.text),
            sep="\t",
            low_memory=False,
        )

        con.execute("DROP TABLE IF EXISTS amelag_raw")
        con.execute("CREATE TABLE amelag_raw AS SELECT * FROM raw_df")

        # Aggregate: national daily virus load
        agg_df = con.execute(
            """
            SELECT
                TRY_CAST(datum AS DATE)  AS date,
                SUM(TRY_CAST(viruslast AS DOUBLE)) AS virus_load
            FROM amelag_raw
            WHERE TRY_CAST(datum AS DATE) IS NOT NULL
              AND TRY_CAST(viruslast AS DOUBLE) IS NOT NULL
            GROUP BY TRY_CAST(datum AS DATE)
            ORDER BY date
            """
        ).fetchdf()

        agg_df["date"] = pd.to_datetime(agg_df["date"])
        agg_df = agg_df.dropna(subset=["virus_load"])
        agg_df = agg_df.sort_values("date").reset_index(drop=True)

        if agg_df.empty:
            raise ValueError("Aggregated RKI dataframe is empty after parsing.")

        logger.info("RKI data loaded: %d rows", len(agg_df))
        return agg_df

    except Exception as exc:
        logger.warning("RKI download failed (%s). Generating synthetic fallback.", exc)
        return _generate_synthetic_wastewater()


# ── Synthetic Fallback ───────────────────────────────────────────────────────
def _generate_synthetic_wastewater(days: int = 365) -> pd.DataFrame:
    """
    Generate a plausible wastewater virus-load curve (seasonal sinusoidal
    + noise) for demo/fallback purposes.
    """
    np.random.seed(42)
    end = datetime.today()
    dates = pd.date_range(end=end, periods=days, freq="D")

    # Seasonal wave (winter peak) + trend
    t = np.arange(days)
    base = 1e9 * (1.5 + np.sin(2 * np.pi * (t - 30) / 365))  # winter peak ~Jan
    noise = np.random.normal(1.0, 0.15, size=days)
    virus_load = np.abs(base * noise)

    return pd.DataFrame({"date": dates, "virus_load": virus_load})


# ── Lab Volume Synthesis ─────────────────────────────────────────────────────
def generate_lab_volume(
    wastewater_df: pd.DataFrame,
    lag_days: int = DEFAULT_LAG_DAYS,
) -> pd.DataFrame:
    """
    Derive synthetic lab test volume from wastewater signal:
      1. Shift forward by `lag_days`.
      2. Min-max normalize to [LAB_SCALE_MIN, LAB_SCALE_MAX].
      3. Add Gaussian noise (±10 %).
      4. Compute revenue.
    """
    np.random.seed(0)
    df = wastewater_df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Shift dates forward → lab sees cases `lag_days` later
    df["date"] = df["date"] + pd.Timedelta(days=lag_days)

    # Normalize virus_load → lab scale
    vl = df["virus_load"].values.astype(float)
    vl_min, vl_max = vl.min(), vl.max()
    if vl_max - vl_min == 0:
        scaled = np.full_like(vl, (LAB_SCALE_MIN + LAB_SCALE_MAX) / 2)
    else:
        scaled = LAB_SCALE_MIN + (vl - vl_min) / (vl_max - vl_min) * (
            LAB_SCALE_MAX - LAB_SCALE_MIN
        )

    # Add organic noise
    noise = np.random.normal(1.0, NOISE_FACTOR, size=len(scaled))
    order_volume = np.round(scaled * noise).astype(int)
    order_volume = np.clip(order_volume, LAB_SCALE_MIN, LAB_SCALE_MAX + 200)

    df["order_volume"] = order_volume
    df["revenue"] = df["order_volume"] * AVG_REVENUE_PER_TEST
    df = df[["date", "order_volume", "revenue"]].reset_index(drop=True)

    return df


# ── Forecast Helper ──────────────────────────────────────────────────────────
def build_forecast(
    lab_df: pd.DataFrame,
    horizon_days: int = 14,
    safety_buffer_pct: float = 0.10,
    stock_on_hand: int = 5000,
) -> Tuple[pd.DataFrame, dict]:
    """
    Build a simple forecast table and KPI dict.

    Returns
    -------
    forecast_df : DataFrame
        Date | Predicted Volume | Recommended Reagent Order | Est. Revenue
    kpis : dict
        predicted_tests_7d, revenue_forecast_7d, reagent_status, trend_pct
    """
    today = pd.Timestamp(datetime.today()).normalize()

    # Split actuals / future
    actuals = lab_df[lab_df["date"] <= today].copy()
    future = lab_df[lab_df["date"] > today].head(horizon_days).copy()

    # If we don't have enough future rows, extrapolate from last 14 days
    if len(future) < horizon_days:
        last_window = actuals.tail(14)
        if last_window.empty:
            last_window = lab_df.tail(14)
        mean_vol = int(last_window["order_volume"].mean())
        std_vol = int(last_window["order_volume"].std()) if len(last_window) > 1 else 50
        extra_dates = pd.date_range(
            start=today + timedelta(days=1),
            periods=horizon_days,
            freq="D",
        )
        np.random.seed(99)
        extra_vols = np.random.normal(mean_vol, std_vol, size=horizon_days).astype(int)
        extra_vols = np.clip(extra_vols, LAB_SCALE_MIN, LAB_SCALE_MAX + 200)
        future = pd.DataFrame(
            {
                "date": extra_dates,
                "order_volume": extra_vols,
                "revenue": extra_vols * AVG_REVENUE_PER_TEST,
            }
        )

    forecast = future.head(horizon_days).copy()
    forecast["buffered_volume"] = np.ceil(
        forecast["order_volume"] * (1 + safety_buffer_pct)
    ).astype(int)

    cumulative = forecast["buffered_volume"].cumsum()
    reagent_order = np.maximum(cumulative - stock_on_hand, 0)
    reagent_order_daily = reagent_order.diff().fillna(reagent_order.iloc[0]).astype(int)
    reagent_order_daily = np.maximum(reagent_order_daily, 0)
    forecast["recommended_order"] = reagent_order_daily
    forecast["est_revenue"] = forecast["order_volume"] * AVG_REVENUE_PER_TEST

    # ── KPIs ──
    pred_7d = int(forecast.head(7)["order_volume"].sum())
    rev_7d = pred_7d * AVG_REVENUE_PER_TEST
    total_demand = int(forecast["buffered_volume"].sum())
    reagent_status = "Optimal ✅" if stock_on_hand >= total_demand else "Critical ⚠️"

    # Trend: compare last 7 vs previous 7 actuals
    last_7 = actuals.tail(7)["order_volume"].sum()
    prev_7 = actuals.tail(14).head(7)["order_volume"].sum()
    trend_pct = ((last_7 - prev_7) / prev_7 * 100) if prev_7 else 0.0

    kpis = {
        "predicted_tests_7d": pred_7d,
        "revenue_forecast_7d": rev_7d,
        "reagent_status": reagent_status,
        "trend_pct": round(trend_pct, 1),
        "stock_on_hand": stock_on_hand,
        "total_demand": total_demand,
    }

    display_cols = {
        "date": "Date",
        "order_volume": "Predicted Volume",
        "recommended_order": "Reagent Order",
        "est_revenue": "Est. Revenue (€)",
    }
    forecast_display = forecast[list(display_cols.keys())].rename(columns=display_cols)

    return forecast_display, kpis
