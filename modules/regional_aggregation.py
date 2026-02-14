"""
LabPulse AI — Regional Aggregation
=====================================
Aggregates RKI wastewater data by Bundesland (German state) for
choropleth map visualization. Uses Plotly's built-in choropleth
(no folium/geopandas needed).
"""

import json
import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Bundesland abbreviation → full name (RKI AMELAG uses 2-letter codes)
BUNDESLAND_CODES = {
    "BW": "Baden-Württemberg",
    "BY": "Bayern",
    "BE": "Berlin",
    "BB": "Brandenburg",
    "HB": "Bremen",
    "HH": "Hamburg",
    "HE": "Hessen",
    "MV": "Mecklenburg-Vorpommern",
    "NI": "Niedersachsen",
    "NW": "Nordrhein-Westfalen",
    "RP": "Rheinland-Pfalz",
    "SL": "Saarland",
    "SN": "Sachsen",
    "ST": "Sachsen-Anhalt",
    "SH": "Schleswig-Holstein",
    "TH": "Thüringen",
}

# Also accept full names (identity mapping + common variants)
BUNDESLAND_NORMALIZE = {
    **{v: v for v in BUNDESLAND_CODES.values()},
    **{k: v for k, v in BUNDESLAND_CODES.items()},
    "Baden-Wuerttemberg": "Baden-Württemberg",
    "Thueringen": "Thüringen",
}

# Center coordinates for Germany map
GERMANY_CENTER = {"lat": 51.1657, "lon": 10.4515}


def aggregate_by_bundesland(
    raw_df: pd.DataFrame,
    pathogen_types: list[str],
    days_back: int = 30,
) -> pd.DataFrame:
    """
    Aggregate virus load by Bundesland for the last N days.

    Parameters
    ----------
    raw_df : pd.DataFrame — Full RKI AMELAG dataframe
    pathogen_types : list[str] — Typ values to filter (e.g. ["SARS-CoV-2"])
    days_back : int — How many days to look back

    Returns
    -------
    pd.DataFrame with columns: [bundesland, avg_virus_load, total_virus_load,
                                 site_count, trend_pct]
    """
    if raw_df.empty or "bundesland" not in raw_df.columns:
        logger.warning("No bundesland column in data")
        return pd.DataFrame()

    df = raw_df.copy()

    # Parse dates and virus load
    df["datum_parsed"] = pd.to_datetime(df.get("datum"), errors="coerce")
    df["vl_parsed"] = pd.to_numeric(df.get("viruslast"), errors="coerce")
    df = df.dropna(subset=["datum_parsed", "vl_parsed", "bundesland"])

    # Filter pathogen
    if "typ" in df.columns and pathogen_types:
        df = df[df["typ"].isin(pathogen_types)]

    if df.empty:
        return pd.DataFrame()

    # Date range
    latest_date = df["datum_parsed"].max()
    cutoff = latest_date - pd.Timedelta(days=days_back)
    cutoff_prev = cutoff - pd.Timedelta(days=days_back)

    current = df[df["datum_parsed"] >= cutoff].copy()
    previous = df[(df["datum_parsed"] >= cutoff_prev) & (df["datum_parsed"] < cutoff)].copy()

    # Normalize Bundesland names
    current["bundesland_clean"] = current["bundesland"].map(
        lambda x: BUNDESLAND_NORMALIZE.get(str(x).strip(), str(x).strip())
    )
    previous["bundesland_clean"] = previous["bundesland"].map(
        lambda x: BUNDESLAND_NORMALIZE.get(str(x).strip(), str(x).strip())
    )

    # Aggregate current period
    agg_current = (
        current.groupby("bundesland_clean")
        .agg(
            avg_virus_load=("vl_parsed", "mean"),
            total_virus_load=("vl_parsed", "sum"),
            site_count=("standort", "nunique") if "standort" in current.columns else ("vl_parsed", "count"),
        )
        .reset_index()
        .rename(columns={"bundesland_clean": "bundesland"})
    )

    # Aggregate previous period for trend
    agg_previous = (
        previous.groupby("bundesland_clean")
        .agg(avg_virus_load_prev=("vl_parsed", "mean"))
        .reset_index()
        .rename(columns={"bundesland_clean": "bundesland"})
    )

    # Merge and compute trend
    result = agg_current.merge(agg_previous, on="bundesland", how="left")
    result["trend_pct"] = (
        (result["avg_virus_load"] - result["avg_virus_load_prev"])
        / result["avg_virus_load_prev"]
        * 100
    ).fillna(0).round(1)

    result = result.drop(columns=["avg_virus_load_prev"], errors="ignore")
    result = result.sort_values("avg_virus_load", ascending=False).reset_index(drop=True)

    logger.info("Regional aggregation: %d Bundeslaender", len(result))
    return result


def get_geojson_path() -> str:
    """Return path to Germany states GeoJSON."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "germany_states.geojson")


def load_geojson() -> Optional[dict]:
    """Load Germany GeoJSON from local file or generate a remote URL fallback."""
    path = get_geojson_path()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    logger.warning("GeoJSON not found at %s — map will use scatter fallback", path)
    return None


# Bundesland center coordinates for scatter map fallback
BUNDESLAND_COORDS = {
    "Baden-Württemberg": {"lat": 48.66, "lon": 9.35},
    "Bayern": {"lat": 48.79, "lon": 11.50},
    "Berlin": {"lat": 52.52, "lon": 13.40},
    "Brandenburg": {"lat": 52.41, "lon": 12.53},
    "Bremen": {"lat": 53.08, "lon": 8.80},
    "Hamburg": {"lat": 53.55, "lon": 9.99},
    "Hessen": {"lat": 50.65, "lon": 9.16},
    "Mecklenburg-Vorpommern": {"lat": 53.61, "lon": 12.43},
    "Niedersachsen": {"lat": 52.64, "lon": 9.85},
    "Nordrhein-Westfalen": {"lat": 51.43, "lon": 7.66},
    "Rheinland-Pfalz": {"lat": 50.12, "lon": 7.31},
    "Saarland": {"lat": 49.40, "lon": 7.02},
    "Sachsen": {"lat": 51.10, "lon": 13.20},
    "Sachsen-Anhalt": {"lat": 51.95, "lon": 11.69},
    "Schleswig-Holstein": {"lat": 54.22, "lon": 9.70},
    "Thüringen": {"lat": 50.98, "lon": 11.03},
}


def create_scatter_map_data(regional_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich regional DataFrame with lat/lon for scatter map.
    Used as fallback when GeoJSON is not available.
    """
    if regional_df.empty:
        return regional_df

    df = regional_df.copy()
    df["lat"] = df["bundesland"].map(lambda x: BUNDESLAND_COORDS.get(x, {}).get("lat"))
    df["lon"] = df["bundesland"].map(lambda x: BUNDESLAND_COORDS.get(x, {}).get("lon"))
    df = df.dropna(subset=["lat", "lon"])
    return df
