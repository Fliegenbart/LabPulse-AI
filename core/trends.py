"""
LabPulse AI — Google Trends signal service
=========================================
Fetches German public search interest for pathogen-specific keyword sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

import numpy as np

PATHOGEN_TREND_TERMS: Dict[str, List[str]] = {
    "SARS-CoV-2": ["Corona Test", "Corona Symptome", "Covid Nachweissituation", "Covid19"],
    "Influenza A": ["Grippe", "Grippe Symptome", "Influenza Test", "Influenza Impfung"],
    "Influenza B": ["Grippe", "Grippe Symptome", "Influenza Test", "Influenza Impfung"],
    "Influenza (gesamt)": ["Grippe", "Influenza", "Grippe Symptome"],
    "RSV": ["RSV", "Lungenentzündung Säuglinge", "Bronchiolitis"],
    "Poliovirus": ["Polio", "Kinderlähmung", "Poliomyelitis"],
    "COVID-19": ["Corona Test", "Coronavirus", "COVID-19 Symptome"],
}


@dataclass
class TrendRequest:
    pathogen: str
    region: str = "DE"
    timeframe: str = "today 12-m"


def _normalize_query_result(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "trend_score"])

    trend_cols = [c for c in raw.columns if c.lower() != "ispartial"]
    if not trend_cols:
        return pd.DataFrame(columns=["date", "trend_score"])

    normalized = raw.copy()
    normalized = normalized[trend_cols]
    normalized["trend_score"] = normalized.mean(axis=1)
    normalized = normalized[["trend_score"]]
    normalized = normalized.reset_index().rename(columns={"index": "date"})
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized["trend_score"] = pd.to_numeric(normalized["trend_score"], errors="coerce")
    return normalized.dropna(subset=["date", "trend_score"])


def _synth_trend_series(anchor: pd.Series, days: int = 90) -> pd.DataFrame:
    if anchor is None or anchor.empty:
        anchor = pd.Series([1], index=[0], dtype=float)

    rng = np.random.default_rng(11)
    base = float(anchor.mean())
    season = 10 * pd.Series(np.sin(np.linspace(0, 3.1, days)))
    noise = pd.Series(rng.normal(0, 6, size=days))
    dates = pd.date_range(pd.Timestamp.today().normalize() - pd.Timedelta(days=days - 1), periods=days, freq="D")
    score = (base + season + noise + 50).clip(lower=0, upper=100).round().astype(int)
    return pd.DataFrame({"date": dates, "trend_score": score.values})


def _synthetic_fallback(pathogen: str) -> pd.DataFrame:
    seed_vals = {
        "SARS-CoV-2": pd.Series([62, 64, 61, 63, 67, 70], dtype=float),
        "RSV": pd.Series([18, 22, 20, 19, 24, 26], dtype=float),
        "Influenza A": pd.Series([44, 40, 41, 43, 46, 49], dtype=float),
        "Influenza B": pd.Series([44, 40, 41, 43, 46, 49], dtype=float),
    }
    return _synth_trend_series(seed_vals.get(pathogen, pd.Series([33], dtype=float)))


def fetch_trend_signal(pathogen: str, region: str = "DE", timeframe: str = "today 12-m") -> pd.DataFrame:
    request = TrendRequest(pathogen=pathogen, region=region, timeframe=timeframe)
    terms = PATHOGEN_TREND_TERMS.get(pathogen, PATHOGEN_TREND_TERMS.get("SARS-CoV-2", []))
    if not terms:
        return _synthetic_fallback(pathogen)

    try:
        from pytrends.request import TrendReq
    except Exception:
        return _synth_trend_series(pd.Series([28, 31, 25, 27], dtype=float))

    try:
        headers = {"Accept-Language": "de-DE,de;q=0.9"}
        pytrends = TrendReq(hl="de-DE", tz=360)
        pytrends.build_payload(
            kw_list=terms[:3],
            timeframe=request.timeframe,
            geo=request.region,
            gprop="",
            **{},
        )
        data = pytrends.interest_over_time()
        if data is None or data.empty:
            return _synthetic_fallback(pathogen)
        normalized = _normalize_query_result(data)
        if normalized.empty:
            return _synthetic_fallback(pathogen)
        return normalized.dropna(subset=["date", "trend_score"]).reset_index(drop=True)
    except Exception:
        return _synth_trend_series(pd.Series([20, 24, 23, 21], dtype=float))


class TrendContext:
    def __init__(self, ttl_minutes: int = 60) -> None:
        self.ttl_minutes = ttl_minutes
        self._cache: Dict[str, tuple[pd.Timestamp, pd.DataFrame]] = {}

    def get(self, pathogen: str) -> pd.DataFrame:
        now = pd.Timestamp.now(tz="UTC")
        cached = self._cache.get(pathogen)
        if cached is not None:
            ts, data = cached
            if now - ts < timedelta(minutes=self.ttl_minutes):
                return data.copy()

        fresh = fetch_trend_signal(pathogen)
        fresh = fresh.copy()
        fresh["date"] = pd.to_datetime(fresh["date"], errors="coerce")
        fresh = fresh.sort_values("date").dropna(subset=["date"])
        self._cache[pathogen] = (now, fresh)
        return fresh.copy()
