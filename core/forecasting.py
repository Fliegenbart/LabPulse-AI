"""
LabPulse AI — Forecasting Core
==============================
Clean forecasting layer used by the NiceGUI app.

Includes:
* deterministic baseline forecast
* Prophet-based forecast with wastewater + trend regressors
* stock planning and KPI extraction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from data_engine import PATHOGEN_REAGENT_MAP, AVG_REVENUE_PER_TEST

try:
    from prophet import Prophet
except Exception:  # pragma: no cover - optional dependency guard
    Prophet = None


def _safe_numeric_series(values: pd.Series) -> pd.Series:
    if values is None or len(values) == 0:
        return pd.Series(dtype=float)
    return pd.to_numeric(values, errors="coerce").astype(float)


def _to_day_series(df: pd.DataFrame, date_col: str = "date", value_col: str = "order_volume") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "order_volume"])

    cleaned = df[[date_col, value_col]].copy()
    cleaned["date"] = pd.to_datetime(cleaned[date_col], errors="coerce")
    cleaned[value_col] = _safe_numeric_series(cleaned[value_col]).fillna(0.0)
    cleaned = cleaned.dropna(subset=["date"]).sort_values("date")

    if cleaned.empty:
        return pd.DataFrame(columns=["date", "order_volume"])

    daily = cleaned.groupby("date", as_index=False)[value_col].sum()
    if daily.empty:
        return pd.DataFrame(columns=["date", "order_volume"])
    return daily.reset_index(drop=True)


@dataclass
class ForecastOutcome:
    df: pd.DataFrame
    kpis: Dict[str, float | int | str]
    model_info: Dict[str, float | int | str | bool]


def _synth_series(periods: int, horizon_end: pd.Timestamp, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=650, scale=110, size=periods)
    cyclical = 70 * np.sin(np.linspace(0, np.pi * 2, periods))
    values = np.clip((base + cyclical).astype(float), 20.0, None)
    return pd.Series(np.round(values), index=pd.date_range(horizon_end + pd.Timedelta(days=1), periods=periods, freq="D"))


def _safe_pathogen_meta(pathogen: str) -> Tuple[str, float]:
    meta = PATHOGEN_REAGENT_MAP.get(pathogen, {})
    test_name = meta.get("test_name", "Generic PCR Kit")
    cost_per_test = float(meta.get("cost_per_test", AVG_REVENUE_PER_TEST))
    return test_name, cost_per_test


def _simple_forecast(
    history: pd.DataFrame,
    periods: int,
    uplift: float,
    seed: int = 42,
) -> pd.DataFrame:
    if history.empty:
        return _synth_series(periods, pd.Timestamp.today(), seed=seed).to_frame(name="predicted")

    window = history.tail(14)["order_volume"]
    trend = history.tail(14).set_index("date")["order_volume"].pct_change().fillna(0).mean()

    if window.empty:
        mean = 500.0
        std = 120.0
    else:
        mean = float(window.mean())
        std = float(window.std()) if len(window) > 1 else 90.0

    mean = max(20.0, mean * (1 + 0.35 * trend))
    std = max(20.0, std)

    rng = np.random.default_rng(seed)
    if len(history) >= 1:
        last_date = history["date"].max().normalize()
    else:
        last_date = pd.Timestamp.today().normalize()

    forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=periods, freq="D")
    predicted = rng.normal(mean, std / 2.0, size=periods)
    predicted = pd.Series(predicted, index=forecast_dates).round()
    predicted = predicted.clip(lower=20.0) * (1 + uplift)
    predicted = predicted.astype(int)

    low = (predicted * 0.75).clip(lower=0).round().astype(int)
    high = (predicted * 1.35).round().astype(int)

    return pd.DataFrame(
        {
            "date": forecast_dates,
            "predicted": predicted.astype(int),
            "lower": low.astype(int),
            "upper": high.astype(int),
        }
    )


def _prophet_forecast(
    history: pd.DataFrame,
    horizon: int,
    uplift: float,
    extra_features: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    if Prophet is None:
        return None

    if history.empty or len(history) < 45:
        return None

    train = history.copy()
    train = train.rename(columns={"date": "ds", "order_volume": "y"})
    train["ds"] = pd.to_datetime(train["ds"])
    train["y"] = _safe_numeric_series(train["y"]).clip(lower=0)
    train = train.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)
    if len(train) < 45:
        return None

    feature_df = extra_features.copy() if extra_features is not None and not extra_features.empty else None
    if feature_df is not None and not feature_df.empty:
        feature_df = feature_df.copy()
        if "date" not in feature_df.columns:
            feature_df = feature_df.copy()
            feature_df["date"] = pd.to_datetime(feature_df.index, errors="coerce")
        feature_df["date"] = pd.to_datetime(feature_df["date"], errors="coerce")
        feature_df = feature_df.sort_values("date")
        feature_df = feature_df.rename(columns={"trend": "trend_score"}) if "trend" in feature_df.columns else feature_df
        if "trend_score" in feature_df.columns:
            feature_df = feature_df[["date", "trend_score"]]
            feature_df["trend_score"] = _safe_numeric_series(feature_df["trend_score"]).fillna(0)
            train = train.merge(feature_df, on="date", how="left")
            train["trend_score"] = train["trend_score"].fillna(method="ffill").fillna(0)
        if "virus_load" in feature_df.columns:
            vv = feature_df[["date", "virus_load"]]
            vv["virus_load"] = _safe_numeric_series(vv["virus_load"]).fillna(0)
            train = train.merge(vv, on="date", how="left")
            train["virus_load"] = train["virus_load"].fillna(method="ffill").fillna(0)

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
    )
    if "trend_score" in train.columns:
        model.add_regressor("trend_score")
    if "virus_load" in train.columns:
        model.add_regressor("virus_load")

    try:
        model.fit(train)
    except Exception:
        return None

    future = model.make_future_dataframe(periods=horizon)
    future = future.copy()
    future["date"] = future["ds"]
    if feature_df is not None and not feature_df.empty:
        future = future.merge(feature_df, on="date", how="left")
    if "trend_score" in future.columns:
        future["trend_score"] = _safe_numeric_series(future["trend_score"]).fillna(method="ffill").fillna(0)
    if "virus_load" in future.columns:
        future["virus_load"] = _safe_numeric_series(future["virus_load"]).fillna(method="ffill").fillna(0)
    else:
        last_virus = None
        if not train.empty:
            last_virus = float(train["y"].tail(1).mean())
        future["virus_load"] = last_virus if last_virus is not None else 1.0

    pred = model.predict(future)
    pred = pred[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon)
    pred = pred.rename(
        columns={
            "ds": "date",
            "yhat": "predicted",
            "yhat_lower": "lower",
            "yhat_upper": "upper",
        }
    )
    pred["date"] = pd.to_datetime(pred["date"], errors="coerce")
    pred = pred.dropna(subset=["date"]).sort_values("date")
    pred["predicted"] = pred["predicted"].astype(float).clip(lower=20.0) * (1 + uplift)
    pred["lower"] = pred["lower"].astype(float).clip(lower=0) * (1 + uplift)
    pred["upper"] = pred["upper"].astype(float).clip(lower=pred["predicted"]) * (1 + uplift)
    pred["Predicted Volume"] = pred["predicted"].round().astype(int)
    pred["ML Lower"] = pred["lower"].round().astype(int)
    pred["ML Upper"] = pred["upper"].round().astype(int)
    return pred[["date", "predicted", "lower", "upper", "Predicted Volume", "ML Lower", "ML Upper"]]


def _build_stock_plan(
    forecast_volume: pd.Series,
    stock_on_hand: int,
    safety_buffer_pct: float,
) -> Tuple[pd.Series, pd.Series, int, int, float]:
    buffered = np.ceil(forecast_volume * (1 + safety_buffer_pct)).astype(float)
    orders = []
    remaining = []
    stock = float(stock_on_hand)
    stockout_day: Optional[pd.Timestamp] = None

    for idx, demand in enumerate(buffered.astype(int)):
        if demand <= 0:
            orders.append(0)
            remaining.append(int(stock))
            continue
        if stock < demand:
            needed = int(demand - stock)
            orders.append(needed)
            stock = 0.0
            if stockout_day is None:
                stockout_day = idx  # convert index later to date via caller
        else:
            orders.append(0)
            stock -= float(demand)
        remaining.append(int(stock))

    orders_series = pd.Series(orders, dtype=int)
    remaining_series = pd.Series(remaining, dtype=int)
    stockout_count = int(stockout_day) if stockout_day is not None else -1
    total_shortfall = int(max(0, orders_series.sum()))
    return orders_series, remaining_series, stockout_count, total_shortfall, float(np.sum(np.array(buffered)))


def _to_date_index(idx: int, history_end: pd.Timestamp) -> Optional[str]:
    if idx < 0:
        return None
    return (pd.to_datetime(history_end).normalize() + pd.Timedelta(days=idx + 1)).strftime("%Y-%m-%d")


def run_forecast_pipeline(
    history: pd.DataFrame,
    horizon_days: int,
    pathogen: str,
    use_prophet: bool,
    uplift_pct: float,
    stock_on_hand: int,
    safety_buffer_pct: float = 0.10,
    trend_df: Optional[pd.DataFrame] = None,
    wastewater_df: Optional[pd.DataFrame] = None,
) -> ForecastOutcome:
    horizon_days = int(max(7, min(28, horizon_days)))
    uplift = float(uplift_pct) / 100.0

    prepared_history = _to_day_series(history, date_col="date", value_col="order_volume")
    if prepared_history.empty:
        # fallback synthetic baseline
        prepared_history = pd.DataFrame({
            "date": pd.date_range(pd.Timestamp.today().normalize() - pd.Timedelta(days=365), periods=365, freq="D"),
            "order_volume": _synth_series(365, pd.Timestamp.today().normalize(), seed=12).round().astype(int),
        })

    merged_features = None
    if wastewater_df is not None and not wastewater_df.empty:
        ww = wastewater_df.copy()
        if "date" in ww.columns and "virus_load" in ww.columns:
            ww = ww[["date", "virus_load"]].copy()
            ww["date"] = pd.to_datetime(ww["date"], errors="coerce")
            ww["virus_load"] = _safe_numeric_series(ww["virus_load"]).fillna(0.0)
            merged_features = ww

    if trend_df is not None and not trend_df.empty:
        td = trend_df.copy()
        if "date" in td.columns and "trend_score" in td.columns:
            td["date"] = pd.to_datetime(td["date"], errors="coerce")
            td["trend_score"] = _safe_numeric_series(td["trend_score"]).fillna(0.0)
            if merged_features is None:
                merged_features = td[["date", "trend_score"]].copy()
            else:
                merged_features = pd.merge(
                    merged_features,
                    td[["date", "trend_score"]],
                    on="date",
                    how="outer",
                )
                merged_features["trend_score"] = _safe_numeric_series(merged_features["trend_score"]).fillna(0.0)

    if merged_features is not None and not merged_features.empty:
        merged_features = merged_features.sort_values("date").dropna(subset=["date"])

    model_name = "simple"
    confidence = 40.0
    model_payload: Dict[str, str | int | float | bool] = {
        "pathogen": pathogen,
        "use_prophet": bool(use_prophet),
        "model_version": "simple",
        "trained_rows": int(len(prepared_history)),
    }

    forecast_frame: Optional[pd.DataFrame] = None
    if use_prophet:
        forecast_frame = _prophet_forecast(
            prepared_history,
            horizon=horizon_days,
            uplift=uplift,
            extra_features=merged_features,
        )
        if forecast_frame is not None and not forecast_frame.empty:
            model_name = "prophet"
            confidence = 80.0
            model_payload["model_version"] = "prophet"
            model_payload["confidence"] = confidence

    if forecast_frame is None or forecast_frame.empty:
        forecast_frame = _simple_forecast(prepared_history, periods=horizon_days, uplift=uplift)
        model_name = "simple"
        confidence = 55.0
        model_payload["model_version"] = "simple"
        model_payload["confidence"] = confidence

    forecast_frame = forecast_frame.copy()
    forecast_frame["Predicted Volume"] = forecast_frame["Predicted Volume"] if "Predicted Volume" in forecast_frame.columns else forecast_frame["predicted"].round().astype(int)
    forecast_frame["Date"] = pd.to_datetime(forecast_frame["date"], errors="coerce")
    forecast_frame = forecast_frame.dropna(subset=["Date"]).sort_values("Date").head(horizon_days)

    orders, remaining, stockout_idx, shortfall, demand_total = _build_stock_plan(
        forecast_frame["Predicted Volume"].astype(float),
        stock_on_hand=stock_on_hand,
        safety_buffer_pct=safety_buffer_pct,
    )
    forecast_frame["Reagent Order"] = orders.astype(int)
    forecast_frame["remaining_stock"] = remaining.astype(int).clip(lower=0)

    test_name, cost_per_test = _safe_pathogen_meta(pathogen)
    forecast_frame["est_revenue"] = forecast_frame["Predicted Volume"] * cost_per_test
    forecast_frame[f"Est. Revenue (€{cost_per_test:.0f}/test)"] = forecast_frame["est_revenue"]

    forecast_frame["required_fte"] = forecast_frame["Predicted Volume"] / max(1, len(forecast_frame))
    forecast_frame["required_fte_rounded"] = forecast_frame["required_fte"].apply(
        lambda value: int(value) if float(value).is_integer() else int(value) + 1
    )

    last_7 = int(forecast_frame.head(7)["Predicted Volume"].sum())
    total_demand = int(np.ceil(forecast_frame["Predicted Volume"] * (1 + safety_buffer_pct)).astype(int).sum())
    revenue_7d = int(last_7 * cost_per_test)
    stockout_day = None
    if stockout_idx >= 0:
        stockout_day = forecast_frame.iloc[stockout_idx]["Date"]

    risk_eur = shortfall * cost_per_test
    prev_14 = prepared_history.tail(14)
    prev_7 = float(prepared_history.tail(7)["order_volume"].sum()) if not prepared_history.empty else 0.0
    prev_prev_7 = float(prepared_history.tail(14).head(7)["order_volume"].sum()) if len(prepared_history) >= 14 else 0.0
    trend_pct = 0.0 if prev_prev_7 <= 0 else float((prev_7 - prev_prev_7) / prev_prev_7 * 100)

    kpis: Dict[str, float | int | str] = {
        "predicted_tests_7d": int(last_7),
        "revenue_forecast_7d": revenue_7d,
        "trend_pct": round(trend_pct, 1),
        "stock_on_hand": int(stock_on_hand),
        "total_demand": total_demand,
        "risk_eur": float(risk_eur),
        "stockout_day": stockout_day,
        "remaining_stock": int(forecast_frame["remaining_stock"].iloc[-1]) if not forecast_frame.empty else int(stock_on_hand),
        "cost_per_test": cost_per_test,
        "test_name": test_name,
        "pathogen": pathogen,
        "recommended_order_sum": int(forecast_frame["Reagent Order"].sum()),
        "predicted_peak": int(forecast_frame["Predicted Volume"].max()),
        "ml_enabled": bool(use_prophet),
        "model": model_name,
    }

    model_payload["confidence"] = round(float(confidence), 1)
    model_payload["stock_out_count_days"] = int(stockout_idx + 1 if stockout_idx >= 0 else -1)
    model_payload["training_rows"] = int(len(prepared_history))
    model_payload["horizon_days"] = horizon_days

    forecast_display = forecast_frame[[
        "Date",
        "Predicted Volume",
        "Reagent Order",
        f"Est. Revenue (€{cost_per_test:.0f}/test)",
        "ML Lower" if "ML Lower" in forecast_frame.columns else "Predicted Volume",
        "ML Upper" if "ML Upper" in forecast_frame.columns else "Predicted Volume",
        "remaining_stock",
    ]].rename(columns={"remaining_stock": "Remaining Stock"})

    forecast_display = forecast_display.copy()
    if "ML Lower" not in forecast_display.columns:
        forecast_display["ML Lower"] = pd.Series([], dtype=int)
        forecast_display["ML Upper"] = pd.Series([], dtype=int)
    if "Date" in forecast_display.columns:
        forecast_display["Date"] = pd.to_datetime(forecast_display["Date"]).dt.date

    model_payload["success"] = True
    return ForecastOutcome(forecast_display, kpis, model_payload)
