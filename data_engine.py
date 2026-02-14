"""
LabPulse AI — Data Engine
=========================
Handles RKI wastewater data ingestion (per pathogen), DuckDB OLAP processing,
and synthetic lab volume generation with configurable time-lag correlation.
"""

import io
import logging
from datetime import datetime, timedelta
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
RKI_DATASET_URLS = {
    "rki_amelag_einzelstandorte": (
        "https://raw.githubusercontent.com/robert-koch-institut/"
        "Abwassersurveillance_AMELAG/main/amelag_einzelstandorte.tsv"
    ),
    "rki_amelag_aggregated": (
        "https://raw.githubusercontent.com/robert-koch-institut/"
        "Abwassersurveillance_AMELAG/main/amelag_aggregierte_kurve.tsv"
    ),
    "rki_covid_7tage_inzidenz": (
        "https://raw.githubusercontent.com/robert-koch-institut/"
        "COVID-19_7-Tage-Inzidenz_in_Deutschland/main/COVID-19-Faelle_7-Tage-Inzidenz_Deutschland.csv"
    ),
    "rki_grippeweb_wochenbericht": (
        "https://raw.githubusercontent.com/robert-koch-institut/"
        "GrippeWeb_Daten_des_Wochenberichts/main/GrippeWeb_Daten_des_Wochenberichts.tsv"
    ),
    "rki_are_konsultationsinzidenz": (
        "https://raw.githubusercontent.com/robert-koch-institut/"
        "ARE-Konsultationsinzidenz/main/ARE-Konsultationsinzidenz.tsv"
    ),
    "rki_influenzafaelle": (
        "https://raw.githubusercontent.com/robert-koch-institut/"
        "Influenzafaelle_in_Deutschland/main/IfSG_Influenzafaelle.tsv"
    ),
    "rki_polio_wa": (
        "https://raw.githubusercontent.com/robert-koch-institut/"
        "Polioviren_im_Abwasser-PIA/main/Polioviren_im_Abwasser.csv"
    ),
}
RKI_DEFAULT_DATASET = "rki_amelag_einzelstandorte"

SIGNAL_SOURCES = OrderedDict(
    {
        "rki_amelag_einzelstandorte": {"label": "RKI AMELAG – Einzelstandorte"},
        "rki_amelag_aggregated": {"label": "RKI AMELAG – Aggregiert"},
        "rki_covid_7tage_inzidenz": {"label": "RKI COVID-19 7-Tage-Inzidenz"},
        "rki_grippeweb_wochenbericht": {"label": "RKI GrippeWeb (Wochenbericht)"},
        "rki_are_konsultationsinzidenz": {"label": "RKI ARE-Konsultationsinzidenz"},
        "rki_influenzafaelle": {"label": "RKI Influenzafälle"},
        "rki_polio_wa": {"label": "RKI Polioviren im Abwasser-PIA"},
    }
)
AMELAG_DATASET_IDS = {
    "rki_amelag_einzelstandorte",
    "rki_amelag_aggregated",
}
LAB_SCALE_MIN = 500
LAB_SCALE_MAX = 2000
AVG_REVENUE_PER_TEST = 45  # €
DEFAULT_LAG_DAYS = 14
NOISE_FACTOR = 0.10  # ±10 %
REQUEST_TIMEOUT = 12  # seconds

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

_NON_AMELAG_DEFAULT_PATHOGENS = {
    "rki_covid_7tage_inzidenz": ["COVID-19"],
    "rki_grippeweb_wochenbericht": ["Grippe"],
    "rki_are_konsultationsinzidenz": ["ARE"],
    "rki_influenzafaelle": ["Influenza"],
    "rki_polio_wa": ["Poliovirus"],
}
AMELAG_PATHOGEN_DEFAULT = ["SARS-CoV-2", "Influenza A", "Influenza B", "RSV", "Influenza (gesamt)"]


# ── DuckDB Singleton ────────────────────────────────────────────────────────
_CON: Optional[duckdb.DuckDBPyConnection] = None
_RAW_CACHE: Dict[str, pd.DataFrame] = {}


def _normalize_date_input(value: str) -> Optional[pd.Timestamp]:
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.to_datetime(value, errors="coerce")
    text = str(value).strip()
    if not text or text.lower() in {"na", "nan", "none"}:
        return None
    if len(text) >= 8 and "W" in text:
        normalized = text.replace("_", "-").replace(" ", "-")
        try:
            if "W" in normalized and len(normalized) >= 7:
                year = int(normalized[:4])
                week = int(normalized.split("W")[1][:2])
                return pd.to_datetime(datetime.fromisocalendar(year, week, 1))
        except Exception:
            return None
    return pd.to_datetime(text, errors="coerce")


def _to_numeric_series(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def get_signal_source_options() -> List[str]:
    """Return human-readable source labels."""
    return [spec["label"] for spec in SIGNAL_SOURCES.values()]


def get_signal_source_id_by_label(label: str) -> str:
    """Resolve source label to dataset id."""
    for source_id, spec in SIGNAL_SOURCES.items():
        if spec["label"] == label:
            return source_id
    return RKI_DEFAULT_DATASET


def get_signal_source_label(source_id: str) -> str:
    """Resolve source id to label."""
    return SIGNAL_SOURCES.get(source_id, {}).get("label", source_id)


def get_available_signal_sources() -> Dict[str, str]:
    """Return source id -> display label mapping."""
    return {source_id: spec["label"] for source_id, spec in SIGNAL_SOURCES.items()}


def _get_con() -> duckdb.DuckDBPyConnection:
    """Return a module-level in-memory DuckDB connection (lazy init)."""
    global _CON
    if _CON is None:
        _CON = duckdb.connect(database=":memory:")
    return _CON


# ── RKI Data Fetching ────────────────────────────────────────────────────────
def fetch_rki_raw() -> pd.DataFrame:
    """Download default RKI AMELAG dataset and cache in DuckDB."""
    return fetch_rki_raw_dataset(RKI_DEFAULT_DATASET)


def fetch_rki_raw_dataset(dataset_id: str = RKI_DEFAULT_DATASET) -> pd.DataFrame:
    """Download a specific RKI source and cache it in-memory."""
    if dataset_id not in RKI_DATASET_URLS:
        raise ValueError(f"Unknown dataset id: {dataset_id}")

    if dataset_id in _RAW_CACHE and not _RAW_CACHE[dataset_id].empty:
        return _RAW_CACHE[dataset_id]

    try:
        logger.info("Fetching RKI dataset (%s) …", dataset_id)
        resp = requests.get(RKI_DATASET_URLS[dataset_id], timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()

        url = RKI_DATASET_URLS[dataset_id].lower()
        sep = "\t" if url.endswith(".tsv") else ","
        raw_df = pd.read_csv(io.StringIO(resp.text), sep=sep, low_memory=False)

        if dataset_id in AMELAG_DATASET_IDS:
            con = _get_con()
            con.execute(f"DROP TABLE IF EXISTS {dataset_id}_raw")
            con.execute(f"CREATE TABLE {dataset_id}_raw AS SELECT * FROM raw_df")

        _RAW_CACHE[dataset_id] = raw_df
        logger.info("RKI %s data loaded: %d rows", dataset_id, len(raw_df))
        return raw_df
    except Exception as exc:
        logger.warning("RKI download for %s failed (%s).", dataset_id, exc)
        return pd.DataFrame()


def get_available_pathogens(raw_df: pd.DataFrame, dataset_id: str = RKI_DEFAULT_DATASET) -> List[str]:
    """Return list of pathogen names that can be selected for the dataset."""
    if dataset_id in AMELAG_DATASET_IDS:
        if raw_df.empty or "typ" not in raw_df.columns:
            return AMELAG_PATHOGEN_DEFAULT

        available_types = set(raw_df["typ"].dropna().unique())
        result: List[str] = []
        for group_name, type_list in PATHOGEN_GROUPS.items():
            if any(t in available_types for t in type_list):
                result.append(group_name)
        return result

    return _NON_AMELAG_DEFAULT_PATHOGENS.get(dataset_id, ["Gesamtsignal"])


def _extract_amelag_signal(raw_df: pd.DataFrame, pathogen: str) -> pd.DataFrame:
    if raw_df.empty or "typ" not in raw_df.columns or "datum" not in raw_df.columns:
        raise ValueError("AMELAG dataset missing expected columns")

    if "viruslast" not in raw_df.columns:
        raise ValueError("AMELAG dataset missing viruslast")

    type_values = PATHOGEN_GROUPS.get(pathogen, [pathogen])
    if not type_values:
        type_values = [pathogen]

    df = raw_df[raw_df["typ"].isin(type_values)].copy()
    if df.empty:
        raise ValueError(f"No data for pathogen: {pathogen}")

    df["date"] = pd.to_datetime(df["datum"], errors="coerce")
    df["virus_load"] = pd.to_numeric(df["viruslast"], errors="coerce")
    out = df.dropna(subset=["date", "virus_load"]).groupby("date", as_index=False)["virus_load"].sum()
    out["pathogen"] = pathogen
    return out.sort_values("date").reset_index(drop=True)


def _extract_covid_signal(raw_df: pd.DataFrame, pathogen: str = "COVID-19") -> pd.DataFrame:
    if raw_df.empty or "Meldedatum" not in raw_df.columns:
        raise ValueError("COVID dataset missing expected date column")

    df = raw_df.copy()
    if "Altersgruppe" in df.columns:
        all_mask = df["Altersgruppe"].astype(str).str.contains("00+", na=False)
        if all_mask.any():
            df = df[all_mask]

    if "Faelle_7-Tage" in df.columns:
        value_col = "Faelle_7-Tage"
    elif "Inzidenz_7-Tage" in df.columns:
        value_col = "Inzidenz_7-Tage"
    elif "Faelle_neu" in df.columns:
        value_col = "Faelle_neu"
    else:
        raise ValueError("COVID dataset missing value column")

    df["date"] = pd.to_datetime(df["Meldedatum"], errors="coerce")
    df["virus_load"] = _to_numeric_series(df[value_col])
    out = df.dropna(subset=["date", "virus_load"]).groupby("date", as_index=False)["virus_load"].sum()
    out["pathogen"] = pathogen
    return out.sort_values("date").reset_index(drop=True)


def _extract_weekly_signal(
    raw_df: pd.DataFrame,
    date_col: str,
    value_col: str,
    pathogen: str,
) -> pd.DataFrame:
    if raw_df.empty or date_col not in raw_df.columns or value_col not in raw_df.columns:
        raise ValueError("Weekly dataset missing required columns")

    df = raw_df.copy()
    df["date"] = df[date_col].apply(_normalize_date_input)
    df["virus_load"] = _to_numeric_series(df[value_col])
    out = df.dropna(subset=["date", "virus_load"]).groupby("date", as_index=False)["virus_load"].sum()
    out["pathogen"] = pathogen
    return out.sort_values("date").reset_index(drop=True)


def _extract_grippeweb_signal(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    if "Region" in df.columns:
        bundesweit = df["Region"].astype(str).str.contains("bundesweit|deutschland", case=False, na=False)
        if bundesweit.any():
            df = df[bundesweit]
    if "Altersgruppe" in df.columns:
        age_mask = df["Altersgruppe"].astype(str).str.contains("00+", na=False)
        if age_mask.any():
            df = df[age_mask]
    return _extract_weekly_signal(df, "Kalenderwoche", "Inzidenz", "Grippe")


def _extract_are_signal(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    if "Bundesland" in df.columns:
        bundesweit = df["Bundesland"].astype(str).str.contains("bundesweit", case=False, na=False)
        if bundesweit.any():
            df = df[bundesweit]
    return _extract_weekly_signal(df, "Kalenderwoche", "ARE_Konsultationsinzidenz", "ARE")


def _extract_influenzafaelle_signal(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    if "Region" in df.columns:
        deutschland = df["Region"].astype(str).str.contains("deutschland", case=False, na=False)
        if deutschland.any():
            df = df[deutschland]
    if "Altersgruppe" in df.columns:
        age_mask = df["Altersgruppe"].astype(str).str.contains("00+", na=False)
        if age_mask.any():
            df = df[age_mask]
    return _extract_weekly_signal(df, "Meldewoche", "Inzidenz", "Influenza")


def _extract_polio_signal(raw_df: pd.DataFrame) -> pd.DataFrame:
    if "Datum" not in raw_df.columns or "Virusisolate" not in raw_df.columns:
        raise ValueError("Polio dataset missing expected columns")
    df = raw_df.copy()
    df["date"] = df["Datum"].apply(_normalize_date_input)
    df["virus_load"] = _to_numeric_series(df["Virusisolate"])
    out = df.dropna(subset=["date", "virus_load"]).groupby("date", as_index=False)["virus_load"].sum()
    out["pathogen"] = "Poliovirus"
    return out.sort_values("date").reset_index(drop=True)


def fetch_rki_wastewater(
    raw_df: Optional[pd.DataFrame] = None,
    pathogen: str = "SARS-CoV-2",
    dataset_id: str = RKI_DEFAULT_DATASET,
) -> pd.DataFrame:
    """
    Aggregate a signal by date for a specific dataset/pathogen.
    Falls back to synthetic data if download/parsing fails.
    """
    try:
        if raw_df is None or raw_df.empty:
            raw_df = fetch_rki_raw_dataset(dataset_id)

        if raw_df.empty:
            raise ValueError("No raw data available")

        if dataset_id in AMELAG_DATASET_IDS:
            agg_df = _extract_amelag_signal(raw_df, pathogen)
        elif dataset_id == "rki_covid_7tage_inzidenz":
            agg_df = _extract_covid_signal(raw_df, "COVID-19")
        elif dataset_id == "rki_grippeweb_wochenbericht":
            agg_df = _extract_grippeweb_signal(raw_df)
        elif dataset_id == "rki_are_konsultationsinzidenz":
            agg_df = _extract_are_signal(raw_df)
        elif dataset_id == "rki_influenzafaelle":
            agg_df = _extract_influenzafaelle_signal(raw_df)
        elif dataset_id == "rki_polio_wa":
            agg_df = _extract_polio_signal(raw_df)
        else:
            raise ValueError(f"Unknown dataset id: {dataset_id}")

        agg_df["date"] = pd.to_datetime(agg_df["date"], errors="coerce")
        agg_df["virus_load"] = pd.to_numeric(agg_df["virus_load"], errors="coerce")
        agg_df = agg_df.dropna(subset=["date", "virus_load"])
        if agg_df.empty:
            raise ValueError("No parsed data rows")

        return agg_df.sort_values("date").reset_index(drop=True)

    except Exception as exc:
        logger.warning(
            "RKI fetch for dataset=%s pathogen=%s failed (%s). Using synthetic.",
            dataset_id,
            pathogen,
            exc,
        )
        df = _generate_synthetic_wastewater()
        df["pathogen"] = pathogen
        return df


def fetch_all_pathogens(
    raw_df: Optional[pd.DataFrame] = None,
    dataset_id: str = RKI_DEFAULT_DATASET,
) -> pd.DataFrame:
    """Fetch aggregated signals for all pathogen groups for a dataset."""
    if raw_df is None or raw_df.empty:
        raw_df = fetch_rki_raw_dataset(dataset_id)

    if dataset_id not in AMELAG_DATASET_IDS:
        raise ValueError("fetch_all_pathogens is only valid for AMELAG datasets")

    frames = []
    for pathogen in PATHOGEN_GROUPS:
        frames.append(fetch_rki_wastewater(raw_df, pathogen=pathogen, dataset_id=dataset_id))

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


def _prepare_lab_data_for_forecast(
    lab_df: pd.DataFrame,
    pathogen: str,
) -> Tuple[pd.DataFrame, int]:
    """
    Normalize lab history for deterministic forecast logic.

    Returns
    -------
    (prepared_df, cost_per_test)
    """
    if lab_df is None or lab_df.empty:
        return pd.DataFrame(), PATHOGEN_REAGENT_MAP.get(pathogen, {}).get("cost_per_test", AVG_REVENUE_PER_TEST)

    prepared = lab_df.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared["order_volume"] = pd.to_numeric(prepared["order_volume"], errors="coerce")
    prepared = prepared.dropna(subset=["date", "order_volume"]).sort_values("date")

    if prepared.empty:
        return pd.DataFrame(), PATHOGEN_REAGENT_MAP.get(pathogen, {}).get("cost_per_test", AVG_REVENUE_PER_TEST)

    cost_per_test = PATHOGEN_REAGENT_MAP.get(pathogen, {}).get(
        "cost_per_test", AVG_REVENUE_PER_TEST
    )
    if "revenue" not in prepared.columns:
        prepared["revenue"] = prepared["order_volume"] * cost_per_test

    prepared = prepared.groupby("date", as_index=False).agg(
        order_volume=("order_volume", "sum"),
        revenue=("revenue", "sum"),
    )

    if prepared.empty:
        return pd.DataFrame(), cost_per_test

    date_idx = pd.date_range(prepared["date"].min(), prepared["date"].max(), freq="D")
    prepared = prepared.set_index("date").reindex(date_idx)
    prepared.index.name = "date"

    prepared["order_volume"] = prepared["order_volume"].interpolate(limit_direction="both")
    if prepared["order_volume"].isna().all():
        prepared["order_volume"] = prepared["order_volume"].fillna(0)
    prepared["order_volume"] = prepared["order_volume"].clip(lower=0).round().astype(int)
    prepared["revenue"] = prepared["order_volume"] * cost_per_test
    prepared["pathogen"] = pathogen
    prepared = prepared.reset_index().rename(columns={"index": "date"})
    return prepared, cost_per_test


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
    prepared_df, cost_per_test = _prepare_lab_data_for_forecast(lab_df, pathogen)
    if prepared_df.empty:
        empty_dates = pd.date_range(today, periods=horizon_days, freq="D")
        forecast_df = pd.DataFrame({
            "Date": empty_dates,
            "Predicted Volume": [0] * horizon_days,
            "Reagent Order": [0] * horizon_days,
            f"Est. Revenue (€{cost_per_test}/test)": [0] * horizon_days,
        })
        return forecast_df, {
            "predicted_tests_7d": 0,
            "revenue_forecast_7d": 0,
            "reagent_status": "No data ❌",
            "trend_pct": 0.0,
            "stock_on_hand": stock_on_hand,
            "total_demand": 0,
            "risk_eur": 0,
            "stockout_day": None,
            "remaining_stock": [stock_on_hand] * horizon_days,
            "cost_per_test": cost_per_test,
            "test_name": PATHOGEN_REAGENT_MAP.get(pathogen, {}).get("test_name", "Generic PCR Kit"),
            "pathogen": pathogen,
        }

    lab_df = prepared_df
    test_name = PATHOGEN_REAGENT_MAP.get(pathogen, {}).get("test_name", "Generic PCR Kit")

    actuals = lab_df[lab_df["date"] <= today].copy()
    future = lab_df[lab_df["date"] > today].head(horizon_days).copy()

    if len(future) < horizon_days:
        last_window = actuals.tail(14)
        if last_window.empty:
            last_window = lab_df.tail(14)

        mean_vol = int(last_window["order_volume"].mean()) if not last_window.empty else 500
        std_vol = int(last_window["order_volume"].std()) if len(last_window) > 1 else 50
        if std_vol < 1:
            std_vol = 50

        extra_dates = pd.date_range(
            start=today + timedelta(days=1),
            periods=horizon_days,
            freq="D",
        )
        np.random.seed(99)
        extra_vols = np.random.normal(mean_vol, std_vol, size=horizon_days).astype(int)
        extra_vols = np.maximum(extra_vols, 20)

        future = pd.DataFrame({
            "date": extra_dates,
            "order_volume": extra_vols,
            "revenue": extra_vols * cost_per_test,
        })

    forecast = future.head(horizon_days).copy().sort_values("date")

    if scenario_uplift_pct > 0:
        forecast["order_volume"] = (forecast["order_volume"] * (1 + scenario_uplift_pct)).astype(int)
        forecast["revenue"] = forecast["order_volume"] * cost_per_test

    forecast["buffered_volume"] = np.ceil(
        forecast["order_volume"] * (1 + safety_buffer_pct)
    ).astype(int)

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
