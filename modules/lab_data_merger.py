"""
LabPulse AI — Lab Data Merger (CSV Upload)
============================================
Handles user-uploaded CSV files with real lab test volumes.
Validates, parses, and merges with synthetic data.
"""

import io
import logging
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"date", "test_volume"}
OPTIONAL_COLUMNS = {"pathogen"}


def validate_csv(file_obj) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """
    Validate an uploaded CSV file.

    Returns
    -------
    (is_valid, message, dataframe_or_none)
    """
    try:
        if hasattr(file_obj, "read"):
            content = file_obj.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            file_obj.seek(0)
        else:
            content = file_obj

        df = pd.read_csv(io.StringIO(content))

        # Normalize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        # Check required columns
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            return (
                False,
                f"Fehlende Spalten: {', '.join(missing)}. "
                f"Erwartet: date, test_volume (und optional: pathogen).",
                None,
            )

        # Parse dates
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        invalid_dates = df["date"].isna().sum()
        if invalid_dates > 0:
            df = df.dropna(subset=["date"])
            if df.empty:
                return False, "Keine gueltigen Datumsangaben gefunden.", None

        # Validate test_volume
        df["test_volume"] = pd.to_numeric(df["test_volume"], errors="coerce")
        df = df.dropna(subset=["test_volume"])
        df["test_volume"] = df["test_volume"].astype(int)

        if df.empty:
            return False, "Keine gueltigen Datensaetze nach Validierung.", None

        # Sort by date
        df = df.sort_values("date").reset_index(drop=True)

        n_rows = len(df)
        date_range = f"{df['date'].min().strftime('%d.%m.%Y')} – {df['date'].max().strftime('%d.%m.%Y')}"

        return (
            True,
            f"{n_rows} Datensaetze geladen ({date_range}).",
            df,
        )

    except Exception as exc:
        logger.warning("CSV validation failed: %s", exc)
        return False, f"CSV-Parsing fehlgeschlagen: {str(exc)}", None


def merge_with_synthetic(
    synthetic_df: pd.DataFrame,
    real_df: pd.DataFrame,
    pathogen: str = "SARS-CoV-2",
) -> pd.DataFrame:
    """
    Merge real lab data into synthetic data.

    Strategy:
    - Where real data exists for a date, use real values.
    - Where only synthetic exists, keep synthetic.
    - Mark each row with data_source: 'real' or 'synthetic'.
    """
    # Normalize real_df columns
    real = real_df.copy()
    real["date"] = pd.to_datetime(real["date"])
    real = real.rename(columns={"test_volume": "order_volume"})

    # Filter by pathogen if column exists
    if "pathogen" in real.columns:
        real_filtered = real[real["pathogen"].str.lower() == pathogen.lower()]
        if not real_filtered.empty:
            real = real_filtered

    real["data_source"] = "real"
    real = real[["date", "order_volume", "data_source"]].copy()

    # Prepare synthetic
    synth = synthetic_df.copy()
    synth["data_source"] = "synthetic"

    # Merge: real data takes priority
    real_dates = set(real["date"].dt.date)
    synth_only = synth[~synth["date"].dt.date.isin(real_dates)].copy()

    merged = pd.concat([synth_only, real], ignore_index=True)
    merged = merged.sort_values("date").reset_index(drop=True)

    # Recalculate revenue if missing
    if "revenue" not in merged.columns:
        from data_engine import PATHOGEN_REAGENT_MAP, AVG_REVENUE_PER_TEST
        cost = PATHOGEN_REAGENT_MAP.get(pathogen, {}).get(
            "cost_per_test", AVG_REVENUE_PER_TEST
        )
        merged["revenue"] = merged["order_volume"] * cost

    if "pathogen" not in merged.columns:
        merged["pathogen"] = pathogen

    n_real = (merged["data_source"] == "real").sum()
    n_synth = (merged["data_source"] == "synthetic").sum()
    logger.info(
        "Merged data: %d real + %d synthetic = %d total",
        n_real, n_synth, len(merged),
    )

    return merged


def get_data_quality_summary(df: pd.DataFrame) -> dict:
    """Return data quality metrics for the merged dataset."""
    total = len(df)
    if total == 0:
        return {"real_pct": 0, "synthetic_pct": 100, "total_rows": 0}

    n_real = (df.get("data_source", pd.Series()) == "real").sum()
    return {
        "real_pct": round(n_real / total * 100, 1),
        "synthetic_pct": round((total - n_real) / total * 100, 1),
        "total_rows": total,
        "real_rows": int(n_real),
        "synthetic_rows": int(total - n_real),
    }
