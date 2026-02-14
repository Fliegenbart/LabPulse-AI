import unittest

import pandas as pd

from modules.signal_fusion import _compute_trend


class SignalFusionTests(unittest.TestCase):
    def test_compute_trend_rising(self):
        series = pd.Series([100, 110, 120, 130, 140, 150, 160, 170, 180, 190])
        direction, magnitude, conf = _compute_trend(series, window=2)
        self.assertEqual(direction, "rising")
        self.assertGreater(magnitude, 0)

    def test_compute_trend_flat(self):
        series = pd.Series([100, 101, 99, 100, 100, 101])
        direction, _, _ = _compute_trend(series, window=2)
        self.assertEqual(direction, "flat")


if __name__ == "__main__":
    unittest.main()
