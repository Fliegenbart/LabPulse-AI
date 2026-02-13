"""
LabPulse AI — External Data Sources
======================================
Integrates additional surveillance signals:
  1. GrippeWeb  — Population-based ARE/ILI incidence (weekly, voluntary reports)
  2. ARE-Konsultationsinzidenz — Practice sentinel consultation rates (weekly, per Bundesland)
  3. Google Trends — Search interest for health-related terms via pytrends

All data is returned as weekly time-series aligned to ISO calendar weeks.
"""

import io
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Data Source URLs ──────────────────────────────────────────────────────────
GRIPPEWEB_URL = (
    "https://raw.githubusercontent.com/robert-koch-institut/"
    "GrippeWeb_Daten_des_Wochenberichts/main/"
    "GrippeWeb_Daten_des_Wochenberichts.tsv"
)

ARE_KONSULTATION_URL = (
    "https://raw.githubusercontent.com/robert-koch-institut/"
    "ARE-Konsultationsinzidenz/main/"
    "ARE-Konsultationsinzidenz.tsv"
)

REQUEST_TIMEOUT = 30

# Mapping of pathogen names to Google Trends search terms (German)
PATHOGEN_SEARCH_TERMS: Dict[str, List[str]] = {
    "SARS-CoV-2": ["Corona Test", "Covid Symptome", "Corona Schnelltest"],
    "Influenza A": ["Grippe Symptome", "Grippeimpfung", "Grippe Test"],
    "Influenza B": ["Grippe Symptome", "Grippeimpfung"],
    "Influenza (gesamt)": ["Grippe Symptome", "Grippeimpfung", "Grippe Test"],
    "RSV": ["RSV Baby", "RSV Symptome", "Bronchiolitis"],
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. GrippeWeb (ARE/ILI Inzidenz)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_kw_to_date(kw_str: str) -> Optional[pd.Timestamp]:
    """Convert ISO week string 'YYYY-Www' to Monday date of that week."""
    try:
        # Format: "2024-W03"
        return pd.Timestamp.fromisocalendar(
            int(kw_str[:4]), int(kw_str.split("W")[1]), 1
        )
    except Exception:
        return None


def fetch_grippeweb(
    erkrankung: str = "ARE",
    region: str = "Bundesweit",
    altersgruppe: str = "00+",
) -> pd.DataFrame:
    """
    Fetch GrippeWeb weekly incidence data.

    Parameters
    ----------
    erkrankung : str — "ARE" (acute respiratory) or "ILI" (influenza-like)
    region : str — "Bundesweit", "Norden", "Sueden", "Osten", "Mitte"
    altersgruppe : str — "00+", "0-4", "5-14", "15-34", "35-59", "60+"

    Returns
    -------
    pd.DataFrame with columns: [date, week, incidence, erkrankung, region]
    """
    try:
        logger.info("Fetching GrippeWeb data …")
        resp = requests.get(GRIPPEWEB_URL, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()

        df = pd.read_csv(io.StringIO(resp.text), sep="\t", low_memory=False)
        logger.info("GrippeWeb raw: %d rows", len(df))

        # Filter
        mask = pd.Series([True] * len(df))
        if "Erkrankung" in df.columns:
            mask &= df["Erkrankung"] == erkrankung
        if "Region" in df.columns:
            mask &= df["Region"] == region
        if "Altersgruppe" in df.columns:
            mask &= df["Altersgruppe"] == altersgruppe

        df = df[mask].copy()

        # Parse calendar week to date
        if "Kalenderwoche" in df.columns:
            df["date"] = df["Kalenderwoche"].apply(_parse_kw_to_date)
        df = df.dropna(subset=["date"])

        # Parse incidence
        if "Inzidenz" in df.columns:
            df["incidence"] = pd.to_numeric(df["Inzidenz"], errors="coerce")
        df = df.dropna(subset=["incidence"])

        result = df[["date", "Kalenderwoche", "incidence"]].copy()
        result = result.rename(columns={"Kalenderwoche": "week"})
        result["erkrankung"] = erkrankung
        result["region"] = region
        result = result.sort_values("date").reset_index(drop=True)

        logger.info("GrippeWeb filtered: %d rows (%s, %s)", len(result), erkrankung, region)
        return result

    except Exception as exc:
        logger.warning("GrippeWeb fetch failed: %s", exc)
        return pd.DataFrame(columns=["date", "week", "incidence", "erkrankung", "region"])


# ══════════════════════════════════════════════════════════════════════════════
# 2. ARE-Konsultationsinzidenz (Practice Sentinel)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_are_konsultation(
    bundesland: str = "Bundesweit",
    altersgruppe: str = "00+",
) -> pd.DataFrame:
    """
    Fetch ARE consultation incidence (practice sentinel).

    Parameters
    ----------
    bundesland : str — "Bundesweit" or specific state name
    altersgruppe : str — "00+", "0-4", "5-14", "15-34", "35-59", "60+"

    Returns
    -------
    pd.DataFrame with columns: [date, week, consultation_incidence, bundesland]
    """
    try:
        logger.info("Fetching ARE-Konsultationsinzidenz …")
        resp = requests.get(ARE_KONSULTATION_URL, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()

        df = pd.read_csv(io.StringIO(resp.text), sep="\t", low_memory=False)
        logger.info("ARE raw: %d rows", len(df))

        # Filter
        mask = pd.Series([True] * len(df))
        if "Bundesland" in df.columns:
            mask &= df["Bundesland"] == bundesland
        if "Altersgruppe" in df.columns:
            mask &= df["Altersgruppe"] == altersgruppe

        df = df[mask].copy()

        # Parse calendar week
        if "Kalenderwoche" in df.columns:
            df["date"] = df["Kalenderwoche"].apply(_parse_kw_to_date)
        df = df.dropna(subset=["date"])

        # Parse incidence (may contain "NA" strings)
        if "ARE_Konsultationsinzidenz" in df.columns:
            df["consultation_incidence"] = pd.to_numeric(
                df["ARE_Konsultationsinzidenz"], errors="coerce"
            )
        df = df.dropna(subset=["consultation_incidence"])

        result = df[["date", "Kalenderwoche", "consultation_incidence"]].copy()
        result = result.rename(columns={"Kalenderwoche": "week"})
        result["bundesland"] = bundesland
        result = result.sort_values("date").reset_index(drop=True)

        logger.info("ARE filtered: %d rows (%s)", len(result), bundesland)
        return result

    except Exception as exc:
        logger.warning("ARE fetch failed: %s", exc)
        return pd.DataFrame(columns=["date", "week", "consultation_incidence", "bundesland"])


def fetch_are_by_bundesland(altersgruppe: str = "00+") -> pd.DataFrame:
    """Fetch ARE data for ALL Bundeslaender (for regional map overlay)."""
    try:
        resp = requests.get(ARE_KONSULTATION_URL, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), sep="\t", low_memory=False)

        if "Altersgruppe" in df.columns:
            df = df[df["Altersgruppe"] == altersgruppe]

        df["date"] = df["Kalenderwoche"].apply(_parse_kw_to_date)
        df["consultation_incidence"] = pd.to_numeric(
            df.get("ARE_Konsultationsinzidenz"), errors="coerce"
        )
        df = df.dropna(subset=["date", "consultation_incidence"])

        return df[["date", "Kalenderwoche", "Bundesland", "consultation_incidence"]].copy()

    except Exception as exc:
        logger.warning("ARE by Bundesland fetch failed: %s", exc)
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# 3. Google Trends (pytrends)
# ══════════════════════════════════════════════════════════════════════════════

# ── Limitations (Important!) ──────────────────────────────────────────────────
#
# pytrends / Google Trends API Limitations:
#
# 1. RATE LIMITING: Google aggressively rate-limits requests.
#    ~10-15 requests per hour before 429 errors. IP bans possible.
#    On a server without residential IP, bans come faster.
#
# 2. RELATIVE VALUES: Google Trends returns *relative* interest (0-100),
#    not absolute search volumes. Values are normalized to the peak
#    within the requested timeframe. Different timeframes → different scales.
#
# 3. SAMPLING: Data is sampled, not census. Small-volume terms have
#    high variance. Results may differ between identical requests.
#
# 4. WEEKLY GRANULARITY: For periods >90 days, data is weekly only.
#    Daily data only for 1-90 day windows.
#
# 5. GEO RESTRICTIONS: Germany ("DE") available, but state-level
#    data (DE-BY, DE-NW etc.) may be too sparse for niche medical terms.
#
# 6. NO OFFICIAL API: pytrends reverse-engineers the Google Trends
#    web interface. Google can break it anytime. Not production-stable.
#
# 7. NO HISTORICAL BACKFILL: Can only query data Google has indexed.
#    Very recent data (last 1-2 days) may not yet be available.
#
# 8. MAX 5 TERMS: Can compare at most 5 search terms per request.
#
# ──────────────────────────────────────────────────────────────────────────────

_PYTRENDS_AVAILABLE = None


def _check_pytrends() -> bool:
    """Check if pytrends is installed."""
    global _PYTRENDS_AVAILABLE
    if _PYTRENDS_AVAILABLE is None:
        try:
            from pytrends.request import TrendReq
            _PYTRENDS_AVAILABLE = True
        except ImportError:
            _PYTRENDS_AVAILABLE = False
    return _PYTRENDS_AVAILABLE


def fetch_google_trends(
    keywords: List[str],
    timeframe: str = "today 12-m",
    geo: str = "DE",
) -> pd.DataFrame:
    """
    Fetch Google Trends interest-over-time data.

    Parameters
    ----------
    keywords : list[str] — Up to 5 search terms
    timeframe : str — "today 12-m", "today 3-m", "2024-01-01 2024-12-31", etc.
    geo : str — Country code ("DE" for Germany)

    Returns
    -------
    pd.DataFrame with columns: [date] + one column per keyword (0-100 scale)
    Empty DataFrame if pytrends not installed or request fails.

    Notes
    -----
    - Values are RELATIVE (0-100), not absolute search volumes
    - Rate limited: ~10-15 requests/hour before 429 errors
    - Weekly granularity for timeframes > 90 days
    """
    if not _check_pytrends():
        logger.warning("pytrends not installed — Google Trends unavailable")
        return pd.DataFrame()

    # Max 5 keywords per request
    keywords = keywords[:5]

    try:
        from pytrends.request import TrendReq

        pytrends = TrendReq(hl="de-DE", tz=60, timeout=(10, 25))
        pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo=geo)

        df = pytrends.interest_over_time()

        if df.empty:
            logger.warning("Google Trends returned no data for: %s", keywords)
            return pd.DataFrame()

        # Drop isPartial column if present
        if "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])

        df = df.reset_index()
        df = df.rename(columns={"date": "date"})
        df["date"] = pd.to_datetime(df["date"])

        logger.info("Google Trends: %d rows for %s", len(df), keywords)
        return df

    except Exception as exc:
        logger.warning("Google Trends fetch failed: %s", exc)
        return pd.DataFrame()


def fetch_trends_for_pathogen(
    pathogen: str,
    timeframe: str = "today 12-m",
) -> pd.DataFrame:
    """
    Fetch Google Trends data for pathogen-specific search terms.

    Parameters
    ----------
    pathogen : str — Pathogen name from PATHOGEN_SEARCH_TERMS
    timeframe : str — Trends timeframe

    Returns
    -------
    pd.DataFrame with date + keyword columns + 'avg_interest' (mean of all terms)
    """
    terms = PATHOGEN_SEARCH_TERMS.get(pathogen)
    if not terms:
        logger.warning("No search terms defined for pathogen: %s", pathogen)
        return pd.DataFrame()

    df = fetch_google_trends(terms, timeframe=timeframe)

    if df.empty:
        return df

    # Add average interest across all terms
    term_cols = [c for c in df.columns if c != "date"]
    if term_cols:
        df["avg_interest"] = df[term_cols].mean(axis=1).round(1)

    return df


def get_trends_limitations() -> List[str]:
    """Return human-readable list of pytrends limitations (German)."""
    return [
        "Rate Limiting: Max. ~10-15 Abfragen/Stunde, danach IP-Sperre moeglich",
        "Relative Werte: Google gibt nur relative Werte (0-100) zurueck, keine absoluten Suchvolumina",
        "Sampling: Daten sind Stichproben, kleine Suchbegriffe haben hohe Varianz",
        "Woechentliche Granularitaet: Fuer Zeitraeume >90 Tage nur Wochendaten verfuegbar",
        "Keine offizielle API: pytrends reverse-engineered Google Trends — kann jederzeit brechen",
        "Max. 5 Suchbegriffe pro Abfrage vergleichbar",
        "Geo-Einschraenkung: Bundesland-Ebene oft zu duenn fuer medizinische Nischenbegriffe",
        "Aktualitaet: Letzte 1-2 Tage evtl. noch nicht indexiert",
    ]
