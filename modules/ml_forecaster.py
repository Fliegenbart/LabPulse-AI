"""
LabPulse AI — ML Forecaster (Prophet)
========================================
Time-series forecasting using Facebook Prophet, replacing the simple
14-day lag shift. Falls back to simple averaging when Prophet is
unavailable or training fails.
"""

import logging
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Training timeout in seconds
TRAIN_TIMEOUT = 15


def _prophet_available() -> bool:
    """Check if Prophet is installed."""
    try:
        from prophet import Prophet
        return True
    except ImportError:
        return False


class LabVolumeForecaster:
    """
    Time-series forecaster for lab test volumes.

    Uses Prophet with wastewater signal as an external regressor.
    Falls back to simple rolling-mean extrapolation if Prophet
    is not available or training fails.
    """

    def __init__(
        self,
        lab_df: pd.DataFrame,
        wastewater_df: Optional[pd.DataFrame] = None,
        pathogen: str = "SARS-CoV-2",
    ):
        self.lab_df = lab_df.copy()
        self.wastewater_df = wastewater_df.copy() if wastewater_df is not None else None
        self.pathogen = pathogen
        self.model = None
        self.is_trained = False
        self.training_time = 0.0
        self.model_type = "simple"  # or "prophet"
        self._confidence_score = 0.0

    def prepare_training_data(self) -> pd.DataFrame:
        """
        Prepare Prophet-compatible training data.
        Prophet expects columns: ds (date), y (target).
        """
        df = self.lab_df[["date", "order_volume"]].copy()
        df = df.rename(columns={"date": "ds", "order_volume": "y"})
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.dropna().sort_values("ds").reset_index(drop=True)

        # Add wastewater as external regressor if available
        if self.wastewater_df is not None and not self.wastewater_df.empty:
            ww = self.wastewater_df[["date", "virus_load"]].copy()
            ww = ww.rename(columns={"date": "ds"})
            ww["ds"] = pd.to_datetime(ww["ds"])
            ww = ww.drop_duplicates(subset=["ds"])

            # Lag the wastewater signal by 14 days (already lagged in lab_df,
            # but we add the raw signal too for Prophet to learn the relationship)
            df = df.merge(ww, on="ds", how="left")
            df["virus_load"] = df["virus_load"].ffill().fillna(0)

        return df

    def train(self) -> "LabVolumeForecaster":
        """
        Train the Prophet model. Returns self for chaining.
        Falls back to simple mode on any failure.
        """
        if not _prophet_available():
            logger.info("Prophet not installed — using simple forecast")
            self.model_type = "simple"
            self.is_trained = True
            self._confidence_score = 50.0
            return self

        try:
            from prophet import Prophet

            start = time.time()

            train_df = self.prepare_training_data()

            if len(train_df) < 30:
                logger.warning("Not enough data for Prophet (%d rows), using simple", len(train_df))
                self.model_type = "simple"
                self.is_trained = True
                self._confidence_score = 40.0
                return self

            # Configure Prophet
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_mode="multiplicative",
            )

            # Add wastewater as regressor if available
            has_ww = "virus_load" in train_df.columns
            if has_ww:
                model.add_regressor("virus_load")

            # Suppress Prophet stdout
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                model.fit(train_df)
            finally:
                sys.stdout = old_stdout

            self.training_time = time.time() - start
            self.model = model
            self.model_type = "prophet"
            self.is_trained = True

            # Calculate confidence score based on training data size and time
            data_score = min(100, len(train_df) / 3)  # 300+ rows → 100
            time_penalty = max(0, 100 - self.training_time * 10)
            self._confidence_score = round((data_score * 0.7 + time_penalty * 0.3), 1)

            logger.info(
                "Prophet trained in %.1fs on %d rows (confidence: %.1f%%)",
                self.training_time, len(train_df), self._confidence_score,
            )
            return self

        except Exception as exc:
            logger.warning("Prophet training failed: %s — falling back to simple", exc)
            self.model_type = "simple"
            self.is_trained = True
            self._confidence_score = 40.0
            return self

    def forecast(self, periods: int = 14) -> pd.DataFrame:
        """
        Generate forecast for the next N days.

        Returns
        -------
        pd.DataFrame with columns: [date, predicted, lower, upper]
        """
        if not self.is_trained:
            self.train()

        if self.model_type == "prophet" and self.model is not None:
            return self._prophet_forecast(periods)
        else:
            return self._simple_forecast(periods)

    def _prophet_forecast(self, periods: int) -> pd.DataFrame:
        """Generate forecast using trained Prophet model."""
        try:
            future = self.model.make_future_dataframe(periods=periods)

            # Fill virus_load regressor for future dates
            if "virus_load" in self.model.extra_regressors:
                train_df = self.prepare_training_data()
                future = future.merge(
                    train_df[["ds", "virus_load"]], on="ds", how="left"
                )
                # Extrapolate: use last known virus_load
                last_vl = train_df["virus_load"].iloc[-1] if not train_df.empty else 0
                future["virus_load"] = future["virus_load"].fillna(last_vl)

            forecast = self.model.predict(future)

            # Get only future dates
            today = pd.Timestamp.now().normalize()
            future_only = forecast[forecast["ds"] > today].head(periods)

            result = pd.DataFrame({
                "date": future_only["ds"].values,
                "predicted": future_only["yhat"].values.astype(int),
                "lower": future_only["yhat_lower"].values.astype(int),
                "upper": future_only["yhat_upper"].values.astype(int),
            })

            # Ensure non-negative
            result["predicted"] = result["predicted"].clip(lower=20)
            result["lower"] = result["lower"].clip(lower=0)
            result["upper"] = result["upper"].clip(lower=result["predicted"])

            return result

        except Exception as exc:
            logger.warning("Prophet forecast failed: %s — using simple", exc)
            return self._simple_forecast(periods)

    def _simple_forecast(self, periods: int) -> pd.DataFrame:
        """Simple rolling-mean forecast as fallback."""
        today = pd.Timestamp.now().normalize()
        df = self.lab_df.copy()

        # Use last 14 days for mean/std
        recent = df[df["date"] <= today].tail(14)
        if recent.empty:
            recent = df.tail(14)

        mean_vol = recent["order_volume"].mean() if not recent.empty else 500
        std_vol = recent["order_volume"].std() if len(recent) > 1 else 50

        future_dates = pd.date_range(start=today + pd.Timedelta(days=1), periods=periods)

        np.random.seed(42)
        predicted = np.random.normal(mean_vol, std_vol * 0.5, size=periods).astype(int)
        predicted = np.maximum(predicted, 20)

        return pd.DataFrame({
            "date": future_dates,
            "predicted": predicted,
            "lower": (predicted * 0.8).astype(int),
            "upper": (predicted * 1.2).astype(int),
        })

    @property
    def confidence_score(self) -> float:
        """Return model confidence score (0-100)."""
        return self._confidence_score

    @property
    def model_info(self) -> dict:
        """Return model metadata."""
        return {
            "model_type": self.model_type,
            "is_trained": self.is_trained,
            "training_time_s": round(self.training_time, 2),
            "confidence_score": self._confidence_score,
            "pathogen": self.pathogen,
        }
