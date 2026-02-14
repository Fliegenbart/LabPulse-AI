import unittest
from datetime import datetime, timedelta

import pandas as pd

from data_engine import PATHOGEN_REAGENT_MAP, build_forecast


class DataEngineForecastTests(unittest.TestCase):
    def test_build_forecast_with_empty_data(self):
        lab_df = pd.DataFrame(columns=["date", "order_volume", "revenue", "pathogen"])
        forecast, kpis = build_forecast(
            lab_df=lab_df,
            horizon_days=7,
            safety_buffer_pct=0.1,
            stock_on_hand=500,
            scenario_uplift_pct=0,
            pathogen="SARS-CoV-2",
        )
        self.assertEqual(len(forecast), 7)
        self.assertEqual(kpis["predicted_tests_7d"], 0)
        self.assertEqual(kpis["reagent_status"], "No data ❌")

    def test_build_forecast_with_irregular_dates(self):
        dates = [datetime(2025, 1, 1) + timedelta(days=i * 2) for i in range(10)]
        cost = PATHOGEN_REAGENT_MAP["SARS-CoV-2"]["cost_per_test"]
        lab_df = pd.DataFrame({
            "date": dates,
            "order_volume": [100 + i * 3 for i in range(10)],
            "revenue": [(100 + i * 3) * cost for i in range(10)],
            "pathogen": ["SARS-CoV-2"] * 10,
        })

        forecast, kpis = build_forecast(
            lab_df=lab_df,
            horizon_days=7,
            safety_buffer_pct=0.1,
            stock_on_hand=10000,
            scenario_uplift_pct=0,
            pathogen="SARS-CoV-2",
        )

        self.assertEqual(len(forecast), 7)
        self.assertIn("Predicted Volume", forecast.columns)
        self.assertIn("predicted_tests_7d", kpis)
        self.assertGreaterEqual(kpis["predicted_tests_7d"], 0)


if __name__ == "__main__":
    unittest.main()
