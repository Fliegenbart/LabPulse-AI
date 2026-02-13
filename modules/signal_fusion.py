"""
LabPulse AI — Multi-Signal Fusion Engine
==========================================
Combines wastewater surveillance, GrippeWeb (population incidence),
and ARE consultation data into a composite confidence score.

Principle:
- Each signal provides an independent trend (rising/falling/flat)
- When signals AGREE → high confidence in forecast direction
- When signals DIVERGE → lower confidence, wider uncertainty
- Each signal has a weight based on reliability and lead-time

Signal characteristics:
  Wastewater (AMELAG)  — earliest signal (~14d lead), high reliability, direct viral measurement
  GrippeWeb            — population self-reports, ~7d lead over lab demand, moderate noise
  ARE-Konsultation     — doctor visit sentinel, ~3-5d lead, high clinical relevance
  Google Trends        — optional, noisy but captures public awareness spikes
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Signal Weights & Config ──────────────────────────────────────────────────

SIGNAL_CONFIG = {
    "wastewater": {
        "weight": 0.45,
        "label": "Abwasser (RKI AMELAG)",
        "lead_days": 14,
        "color": "#3b82f6",
        "icon": "🧪",
    },
    "grippeweb": {
        "weight": 0.25,
        "label": "GrippeWeb (Bevoelkerung)",
        "lead_days": 7,
        "color": "#8b5cf6",
        "icon": "👥",
    },
    "are_consultation": {
        "weight": 0.20,
        "label": "ARE-Konsultationen (Praxen)",
        "lead_days": 4,
        "color": "#06b6d4",
        "icon": "🏥",
    },
    "google_trends": {
        "weight": 0.10,
        "label": "Google Trends (Suchinteresse)",
        "lead_days": 2,
        "color": "#f77f00",
        "icon": "🔍",
    },
}


@dataclass
class SignalTrend:
    """Computed trend for one signal source."""
    name: str
    direction: str  # "rising", "falling", "flat"
    magnitude: float  # percent change (e.g. +15.3 or -8.2)
    confidence: float  # 0-1, how reliable is this signal's data
    data_points: int  # number of observations used
    available: bool = True


@dataclass
class CompositeScore:
    """Result of multi-signal fusion."""
    confidence_pct: float  # 0-100 overall confidence
    direction: str  # consensus direction: "rising", "falling", "mixed"
    agreement_pct: float  # 0-100 how much signals agree
    signals: List[SignalTrend] = field(default_factory=list)
    weighted_trend: float = 0.0  # weighted average trend magnitude
    narrative_de: str = ""  # German narrative summary


# ── Trend Computation ────────────────────────────────────────────────────────

def _compute_trend(series: pd.Series, window: int = 4) -> Tuple[str, float, float]:
    """
    Compute trend from a time series.

    Returns (direction, magnitude_pct, data_confidence)
    - direction: "rising" | "falling" | "flat"
    - magnitude_pct: percent change over window
    - data_confidence: 0-1 based on data quality
    """
    if series is None or len(series) < 2:
        return "flat", 0.0, 0.0

    clean = series.dropna()
    if len(clean) < 2:
        return "flat", 0.0, 0.0

    # Use last `window` points vs previous `window` points
    if len(clean) >= window * 2:
        recent = clean.iloc[-window:].mean()
        previous = clean.iloc[-window * 2:-window].mean()
    else:
        recent = clean.iloc[-1]
        previous = clean.iloc[0]

    if previous == 0:
        return "flat", 0.0, 0.5

    pct_change = ((recent - previous) / abs(previous)) * 100

    # Determine direction with a dead zone
    if pct_change > 5:
        direction = "rising"
    elif pct_change < -5:
        direction = "falling"
    else:
        direction = "flat"

    # Data confidence based on number of observations
    data_confidence = min(1.0, len(clean) / 20)

    return direction, round(pct_change, 1), round(data_confidence, 2)


# ── Signal Extractors ────────────────────────────────────────────────────────

def analyze_wastewater_signal(wastewater_df: pd.DataFrame) -> SignalTrend:
    """Extract trend from wastewater data."""
    if wastewater_df is None or wastewater_df.empty:
        return SignalTrend("wastewater", "flat", 0.0, 0.0, 0, available=False)

    # Use last 8 weeks of data
    cutoff = wastewater_df["date"].max() - pd.Timedelta(days=56)
    recent = wastewater_df[wastewater_df["date"] >= cutoff]

    # Weekly aggregation for smoother trend
    weekly = recent.set_index("date")["virus_load"].resample("W").mean()
    direction, magnitude, data_conf = _compute_trend(weekly, window=4)

    return SignalTrend(
        name="wastewater",
        direction=direction,
        magnitude=magnitude,
        confidence=data_conf,
        data_points=len(weekly),
    )


def analyze_grippeweb_signal(gw_df: pd.DataFrame) -> SignalTrend:
    """Extract trend from GrippeWeb incidence data."""
    if gw_df is None or gw_df.empty:
        return SignalTrend("grippeweb", "flat", 0.0, 0.0, 0, available=False)

    # Last 8 weeks
    cutoff = gw_df["date"].max() - pd.Timedelta(days=56)
    recent = gw_df[gw_df["date"] >= cutoff].copy()

    if "incidence" not in recent.columns:
        return SignalTrend("grippeweb", "flat", 0.0, 0.0, 0, available=False)

    direction, magnitude, data_conf = _compute_trend(recent["incidence"], window=4)

    return SignalTrend(
        name="grippeweb",
        direction=direction,
        magnitude=magnitude,
        confidence=data_conf,
        data_points=len(recent),
    )


def analyze_are_signal(are_df: pd.DataFrame) -> SignalTrend:
    """Extract trend from ARE consultation incidence data."""
    if are_df is None or are_df.empty:
        return SignalTrend("are_consultation", "flat", 0.0, 0.0, 0, available=False)

    cutoff = are_df["date"].max() - pd.Timedelta(days=56)
    recent = are_df[are_df["date"] >= cutoff].copy()

    col = "consultation_incidence" if "consultation_incidence" in recent.columns else "incidence"
    if col not in recent.columns:
        return SignalTrend("are_consultation", "flat", 0.0, 0.0, 0, available=False)

    direction, magnitude, data_conf = _compute_trend(recent[col], window=4)

    return SignalTrend(
        name="are_consultation",
        direction=direction,
        magnitude=magnitude,
        confidence=data_conf,
        data_points=len(recent),
    )


def analyze_trends_signal(trends_df: pd.DataFrame) -> SignalTrend:
    """Extract trend from Google Trends data (average across all terms)."""
    if trends_df is None or trends_df.empty:
        return SignalTrend("google_trends", "flat", 0.0, 0.0, 0, available=False)

    term_cols = [c for c in trends_df.columns if c != "date"]
    if not term_cols:
        return SignalTrend("google_trends", "flat", 0.0, 0.0, 0, available=False)

    # Average across all search terms
    avg_interest = trends_df[term_cols].mean(axis=1)
    direction, magnitude, data_conf = _compute_trend(avg_interest, window=4)

    # Google Trends is inherently noisier → reduce confidence
    data_conf *= 0.7

    return SignalTrend(
        name="google_trends",
        direction=direction,
        magnitude=magnitude,
        confidence=round(data_conf, 2),
        data_points=len(trends_df),
    )


# ── Fusion Engine ────────────────────────────────────────────────────────────

def compute_composite_score(
    signals: List[SignalTrend],
) -> CompositeScore:
    """
    Fuse multiple signal trends into one composite confidence score.

    Algorithm:
    1. Filter to available signals
    2. Compute weighted trend (direction + magnitude)
    3. Measure inter-signal agreement
    4. Derive confidence = base_confidence * agreement_bonus * data_quality
    """
    available = [s for s in signals if s.available and s.data_points >= 2]

    if not available:
        return CompositeScore(
            confidence_pct=20.0,
            direction="mixed",
            agreement_pct=0.0,
            signals=signals,
            weighted_trend=0.0,
            narrative_de="Keine ausreichenden Signaldaten verfuegbar.",
        )

    # ── 1. Weighted trend ────────────────────────────────────────────────
    total_weight = 0.0
    weighted_magnitude = 0.0
    direction_votes: Dict[str, float] = {"rising": 0, "falling": 0, "flat": 0}

    for sig in available:
        w = SIGNAL_CONFIG.get(sig.name, {}).get("weight", 0.1)
        effective_weight = w * sig.confidence
        total_weight += effective_weight
        weighted_magnitude += sig.magnitude * effective_weight
        direction_votes[sig.direction] += effective_weight

    if total_weight > 0:
        weighted_magnitude /= total_weight

    # ── 2. Consensus direction ───────────────────────────────────────────
    dominant_direction = max(direction_votes, key=direction_votes.get)
    dominant_weight = direction_votes[dominant_direction]

    if total_weight > 0:
        agreement_pct = (dominant_weight / total_weight) * 100
    else:
        agreement_pct = 0.0

    # If no clear majority, it's mixed (need >60% for consensus)
    if agreement_pct <= 60:
        consensus = "mixed"
    else:
        consensus = dominant_direction

    # ── 3. Confidence score ──────────────────────────────────────────────
    # Base: 30-50% depending on how many signals are available
    n_signals = len(available)
    base_confidence = 30 + (n_signals / 4) * 20  # max 50 with 4 signals

    # Agreement bonus: +0 to +35 based on agreement
    agreement_bonus = (agreement_pct / 100) * 35

    # Data quality bonus: average confidence across signals
    avg_data_quality = np.mean([s.confidence for s in available])
    quality_bonus = avg_data_quality * 15  # max 15

    confidence_pct = min(95, base_confidence + agreement_bonus + quality_bonus)

    # ── 4. Narrative ─────────────────────────────────────────────────────
    narrative = _build_narrative(available, consensus, agreement_pct, weighted_magnitude, confidence_pct)

    return CompositeScore(
        confidence_pct=round(confidence_pct, 1),
        direction=consensus,
        agreement_pct=round(agreement_pct, 1),
        signals=signals,
        weighted_trend=round(weighted_magnitude, 1),
        narrative_de=narrative,
    )


def _build_narrative(
    signals: List[SignalTrend],
    consensus: str,
    agreement: float,
    trend: float,
    confidence: float,
) -> str:
    """Build a German-language narrative summary of the signal fusion."""
    n = len(signals)
    signal_names = [SIGNAL_CONFIG.get(s.name, {}).get("label", s.name) for s in signals]

    direction_de = {
        "rising": "steigend",
        "falling": "fallend",
        "flat": "stabil",
        "mixed": "uneinheitlich",
    }

    parts = []

    # Opening
    parts.append(f"{n} von 4 Signalquellen verfuegbar ({', '.join(signal_names)}).")

    # Agreement
    if agreement >= 80:
        parts.append(
            f"Hohe Uebereinstimmung ({agreement:.0f}%): "
            f"Alle Signale zeigen {direction_de.get(consensus, consensus)} "
            f"({trend:+.1f}% gewichteter Trend)."
        )
    elif agreement >= 60:
        parts.append(
            f"Moderate Uebereinstimmung ({agreement:.0f}%): "
            f"Mehrheit zeigt {direction_de.get(consensus, consensus)} ({trend:+.1f}%)."
        )
    else:
        diverging = [s for s in signals if s.direction != consensus]
        div_names = [SIGNAL_CONFIG.get(s.name, {}).get("label", s.name) for s in diverging]
        parts.append(
            f"Niedrige Uebereinstimmung ({agreement:.0f}%): "
            f"Signale sind uneinheitlich. "
            f"Abweichend: {', '.join(div_names)}."
        )

    # Per-signal details
    for sig in signals:
        cfg = SIGNAL_CONFIG.get(sig.name, {})
        icon = cfg.get("icon", "")
        label = cfg.get("label", sig.name)
        parts.append(
            f"{icon} {label}: {direction_de.get(sig.direction, sig.direction)} "
            f"({sig.magnitude:+.1f}%, {sig.data_points} Datenpunkte)"
        )

    # Conclusion
    if confidence >= 70:
        parts.append(f"→ Prognose-Konfidenz: HOCH ({confidence:.0f}%). Bestellempfehlung belastbar.")
    elif confidence >= 50:
        parts.append(f"→ Prognose-Konfidenz: MITTEL ({confidence:.0f}%). Sicherheitspuffer empfohlen.")
    else:
        parts.append(f"→ Prognose-Konfidenz: NIEDRIG ({confidence:.0f}%). Erhoehter Puffer + manuelle Pruefung.")

    return "\n".join(parts)


# ── Convenience Function ─────────────────────────────────────────────────────

def fuse_all_signals(
    wastewater_df: Optional[pd.DataFrame] = None,
    grippeweb_df: Optional[pd.DataFrame] = None,
    are_df: Optional[pd.DataFrame] = None,
    trends_df: Optional[pd.DataFrame] = None,
) -> CompositeScore:
    """
    One-call function: analyze all signals and return composite score.
    Pass None for unavailable signals — they'll be marked as unavailable.
    """
    signals = [
        analyze_wastewater_signal(wastewater_df),
        analyze_grippeweb_signal(grippeweb_df),
        analyze_are_signal(are_df),
        analyze_trends_signal(trends_df),
    ]
    return compute_composite_score(signals)
