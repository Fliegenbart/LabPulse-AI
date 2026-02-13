"""
LabPulse AI — Data Engine
=========================
Handles RKI wastewater data ingestion (per pathogen), DuckDB OLAP processing,
and synthetic lab volume generation with configurable time-lag correlation.
"""

import io
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

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

# Pathogen display names and grouping
PATHOGEN_GROUPS: Dict[str, List[str]] = {
    "SARS-CoV-2": ["SARS-CoV-2"],
    "Influenza A": ["Influenza A"],
    "Influenza B": ["Influenza B"],
    "Influenza (gesamt)": ["Influenza A", "Influenza B", "Influenza A+B"],
    "RSV": ["RSV A", "RSV B", "RSV A+B", "RSV A/B"],
}

# Reagent mapping: which pathogen drives which test kit
PATHOGEN_REAGENT_MAP = {
    "SARS-CoV-2": {"test_name": "SARS-CoV-2 PCR Kit", "cost_per_test": 45},
    "Influenza A": {"test_name": "Influenza A/B PCR Panel", "cost_per_test": 38},
    "Influenza B": {"test_name": "Influenza A/B PCR Panel", "cost_per_test": 38},
    "Influenza (gesamt)": {"test_name": "Influenza A/B PCR Panel", "cost_per_test": 38},
    "RSV": {"test_name": "RSV PCR Kit", "cost_per_test": 42},
}

# Lab scale ranges per pathogen (different test volumes)
PATHOGEN_SCALE = {
    "SARS-CoV-2": (400, 2000),
    "Influenza A": (200, 1200),
    "Influenza B": (100, 800),
    "Influenza (gesamt)": (300, 1500),
    "RSV": (150, 900),
}


# ── DuckDB Singleton ────────────────────────────────────────────────────────
_CON: Optional[duckdb.DuckDBPyConnection] = None


def _get_con() -> duckdb.DuckDBPyConnection:
    """Return a module-level in-memory DuckDB connection (lazy init)."""
    global _CON
    if _CON is None:
        _CON = duckdb.connect(database=":memory:")
    return _CON


# ── RKI Data Fetching ────────────────────────────────────────────────────────
def fetch_rki_raw() -> pd.DataFrame:
    """
    Download the full RKI AMELAG TSV and cache in DuckDB.
    Returns raw DataFrame with all columns.
    """
    try:
        logger.info("Fetching RKI AMELAG data …")
        resp = requests.get(RKI_AMELAG_URL, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()

        raw_df = pd.read_csv(io.StringIO(resp.text), sep="\t", low_memory=False)

        con = _get_con()
        con.execute("DROP TABLE IF EXISTS amelag_raw")
        con.execute("CREATE TABLE amelag_raw AS SELECT * FROM raw_df")

        logger.info("RKI raw data loaded: %d rows", len(raw_df))
        return raw_df

    except Exception as exc:
        logger.warning("RKI download failed (%s).", exc)
        return pd.DataFrame()


def get_available_pathogens(raw_df: pd.DataFrame) -> List[str]:
    """Return list of pathogen group names available in the data."""
    if raw_df.empty or "typ" not in raw_df.columns:
        return list(PATHOGEN_GROUPS.keys())

    available_types = set(raw_df["typ"].dropna().unique())
    result = []
    for group_name, type_list in PATHOGEN_GROUPS.items():
        if any(t in available_types for t in type_list):
            result.append(group_name)
    return result


def fetch_rki_wastewater(
    raw_df: Optional[pd.DataFrame] = None,
    pathogen: str = "SARS-CoV-2",
) -> pd.DataFrame:
    """
    Aggregate national virus-load by date for a specific pathogen group.
    Falls back to synthetic data if download/parsing fails.
    """
    try:
        if raw_df is None or raw_df.empty:
            raw_df = fetch_rki_raw()

        if raw_df.empty:
            raise ValueError("No raw data available")

        con = _get_con()

        # Get the typ values for this pathogen group
        type_values = PATHOGEN_GROUPS.get(pathogen, [pathogen])
        type_list_sql = ", ".join(f"'{t}'" for t in type_values)

        agg_df = con.execute(
            f"""
            SELECT
                TRY_CAST(datum AS DATE) AS date,
                SUM(TRY_CAST(viruslast AS DOUBLE)) AS virus_load
            FROM amelag_raw
            WHERE TRY_CAST(datum AS DATE) IS NOT NULL
              AND TRY_CAST(viruslast AS DOUBLE) IS NOT NULL
              AND typ IN ({type_list_sql})
            GROUP BY TRY_CAST(datum AS DATE)
            ORDER BY date
            """
        ).fetchdf()

        agg_df["date"] = pd.to_datetime(agg_df["date"])
        agg_df = agg_df.dropna(subset=["virus_load"])
        agg_df = agg_df.sort_values("date").reset_index(drop=True)
        agg_df["pathogen"] = pathogen

        if agg_df.empty:
            raise ValueError(f"No data for pathogen: {pathogen}")

        logger.info("RKI %s data: %d rows", pathogen, len(agg_df))
        return agg_df

    except Exception as exc:
        logger.warning("RKI fetch for %s failed (%s). Using synthetic.", pathogen, exc)
        df = _generate_synthetic_wastewater()
        df["pathogen"] = pathogen
        return df


def fetch_all_pathogens(raw_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Fetch aggregated wastewater data for ALL pathogen groups. Returns combined DF."""
    if raw_df is None or raw_df.empty:
        raw_df = fetch_rki_raw()

    frames = []
    for pathogen in PATHOGEN_GROUPS:
        df = fetch_rki_wastewater(raw_df, pathogen)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


# ── Synthetic Fallback ───────────────────────────────────────────────────────
def _generate_synthetic_wastewater(days: int = 365) -> pd.DataFrame:
    """Generate a plausible wastewater curve for demo/fallback."""
    np.random.seed(42)
    end = datetime.today()
    dates = pd.date_range(end=end, periods=days, freq="D")

    t = np.arange(days)
    base = 1e9 * (1.5 + np.sin(2 * np.pi * (t - 30) / 365))
    noise = np.random.normal(1.0, 0.15, size=days)
    virus_load = np.abs(base * noise)

    return pd.DataFrame({"date": dates, "virus_load": virus_load})


# ── Lab Volume Synthesis ─────────────────────────────────────────────────────
def generate_lab_volume(
    wastewater_df: pd.DataFrame,
    lag_days: int = DEFAULT_LAG_DAYS,
    pathogen: str = "SARS-CoV-2",
) -> pd.DataFrame:
    """
    Derive synthetic lab test volume from wastewater signal with WEEKLY SEASONALITY.
    Scale and revenue-per-test are pathogen-specific.
    """
    np.random.seed(hash(pathogen) % 2**31)
    df = wastewater_df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Shift dates forward → lab sees cases `lag_days` later
    df["date"] = df["date"] + pd.Timedelta(days=lag_days)

    # Pathogen-specific scale
    scale_min, scale_max = PATHOGEN_SCALE.get(pathogen, (LAB_SCALE_MIN, LAB_SCALE_MAX))
    cost_per_test = PATHOGEN_REAGENT_MAP.get(pathogen, {}).get("cost_per_test", AVG_REVENUE_PER_TEST)

    # Normalize virus_load → lab scale
    vl = df["virus_load"].values.astype(float)
    vl_min, vl_max = vl.min(), vl.max()
    if vl_max - vl_min == 0:
        scaled = np.full_like(vl, (scale_min + scale_max) / 2)
    else:
        scaled = scale_min + (vl - vl_min) / (vl_max - vl_min) * (scale_max - scale_min)

    # Weekly Seasonality (0=Monday … 6=Sunday)
    day_of_week = df["date"].dt.dayofweek
    seasonality = np.ones(len(df))
    seasonality = np.where(day_of_week == 0, 1.4, seasonality)   # Monday backlog
    seasonality = np.where(day_of_week == 4, 0.8, seasonality)   # Friday taper
    seasonality = np.where(day_of_week >= 5, 0.1, seasonality)   # Weekend skeleton crew
    scaled = scaled * seasonality

    # Add organic noise
    noise = np.random.normal(1.0, NOISE_FACTOR, size=len(scaled))
    order_volume = np.round(scaled * noise).astype(int)
    order_volume = np.maximum(order_volume, 20)

    df["order_volume"] = order_volume
    df["revenue"] = df["order_volume"] * cost_per_test
    df["pathogen"] = pathogen
    df = df[["date", "order_volume", "revenue", "pathogen"]].reset_index(drop=True)

    return df


# ── Forecast Helper ──────────────────────────────────────────────────────────
def build_forecast(
    lab_df: pd.DataFrame,
    horizon_days: int = 14,
    safety_buffer_pct: float = 0.10,
    stock_on_hand: int = 5000,
    scenario_uplift_pct: float = 0.0,
    pathogen: str = "SARS-CoV-2",
) -> Tuple[pd.DataFrame, dict]:
    """
    Build forecast with scenario simulation and Revenue-at-Risk KPI.
    Pathogen-aware cost-per-test.
    """
    today = pd.Timestamp(datetime.today()).normalize()
    cost_per_test = PATHOGEN_REAGENT_MAP.get(pathogen, {}).get("cost_per_test", AVG_REVENUE_PER_TEST)
    test_name = PATHOGEN_REAGENT_MAP.get(pathogen, {}).get("test_name", "Generic PCR Kit")

    # Split actuals / future
    actuals = lab_df[lab_df["date"] <= today].copy()
    future = lab_df[lab_df["date"] > today].head(horizon_days).copy()

    # Extrapolate if needed
    if len(future) < horizon_days:
        last_window = actuals.tail(14)
        if last_window.empty:
            last_window = lab_df.tail(14)
        mean_vol = int(last_window["order_volume"].mean()) if not last_window.empty else 500
        std_vol = int(last_window["order_volume"].std()) if len(last_window) > 1 else 50
        extra_dates = pd.date_range(
            start=today + timedelta(days=1), periods=horizon_days, freq="D",
        )
        np.random.seed(99)
        extra_vols = np.random.normal(mean_vol, std_vol, size=horizon_days).astype(int)
        extra_vols = np.maximum(extra_vols, 20)
        future = pd.DataFrame({
            "date": extra_dates,
            "order_volume": extra_vols,
            "revenue": extra_vols * cost_per_test,
        })

    forecast = future.head(horizon_days).copy()

    # Scenario simulation
    if scenario_uplift_pct > 0:
        forecast["order_volume"] = (forecast["order_volume"] * (1 + scenario_uplift_pct)).astype(int)
        forecast["revenue"] = forecast["order_volume"] * cost_per_test

    forecast["buffered_volume"] = np.ceil(
        forecast["order_volume"] * (1 + safety_buffer_pct)
    ).astype(int)

    # Day-by-day stock drawdown
    orders = []
    remaining_stock = []
    current_stock = stock_on_hand
    stockout_day = None
    for i, demand in enumerate(forecast["buffered_volume"]):
        if current_stock < demand:
            needed = demand - current_stock
            orders.append(needed)
            if stockout_day is None:
                stockout_day = forecast["date"].iloc[i]
            current_stock = 0
        else:
            orders.append(0)
            current_stock -= demand
        remaining_stock.append(current_stock)

    forecast["recommended_order"] = orders
    forecast["remaining_stock"] = remaining_stock
    forecast["est_revenue"] = forecast["order_volume"] * cost_per_test

    # KPIs
    pred_7d = int(forecast.head(7)["order_volume"].sum())
    rev_7d = pred_7d * cost_per_test
    total_demand_horizon = int(forecast["buffered_volume"].sum())

    shortage = max(0, total_demand_horizon - stock_on_hand)
    risk_eur = shortage * cost_per_test

    reagent_status = "Optimal ✅" if stock_on_hand >= total_demand_horizon else "Critical ⚠️"

    last_7 = actuals.tail(7)["order_volume"].sum()
    prev_7 = actuals.tail(14).head(7)["order_volume"].sum()
    trend_pct = ((last_7 - prev_7) / prev_7 * 100) if prev_7 else 0.0

    kpis = {
        "predicted_tests_7d": pred_7d,
        "revenue_forecast_7d": rev_7d,
        "reagent_status": reagent_status,
        "trend_pct": round(trend_pct, 1),
        "stock_on_hand": stock_on_hand,
        "total_demand": total_demand_horizon,
        "risk_eur": risk_eur,
        "stockout_day": stockout_day,
        "remaining_stock": remaining_stock,
        "cost_per_test": cost_per_test,
        "test_name": test_name,
        "pathogen": pathogen,
    }

    display_cols = {
        "date": "Date",
        "order_volume": "Predicted Volume",
        "recommended_order": "Reagent Order",
        "est_revenue": f"Est. Revenue (€{cost_per_test}/test)",
    }
    forecast_display = forecast[list(display_cols.keys())].rename(columns=display_cols)

    return forecast_display, kpis
