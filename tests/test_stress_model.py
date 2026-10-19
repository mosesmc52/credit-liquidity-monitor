import unittest

import pandas as pd

from helpers import classify_stress_regime
from stress_model import compute_stress_indicator


class StressIndicatorTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2026-01-01", periods=80)
        self.hy = pd.Series(3.0 + pd.Series(range(80)).to_numpy() * 0.01, index=self.dates)
        self.bbb = pd.Series(1.2 + pd.Series(range(80)).to_numpy() * 0.003, index=self.dates)
        self.sofr = pd.Series(4.0, index=self.dates)
        self.effr = pd.Series(3.99, index=self.dates)
        self.rrp = pd.Series(100.0, index=self.dates)

    def calculate(self, **overrides):
        inputs = dict(hy_oas=self.hy, bbb_oas=self.bbb, sofr=self.sofr,
                      effr=self.effr, rrp_total=self.rrp)
        inputs.update(overrides)
        return compute_stress_indicator(**inputs)

    def test_optional_feed_is_omitted_and_weights_are_renormalized(self):
        result = self.calculate(rrp_total=pd.Series(dtype=float))
        self.assertTrue(pd.notna(result["stress_score"].iloc[-1]))
        self.assertLess(result["coverage"].iloc[-1], 1.0)

    def test_stale_required_feed_withholds_current_regime(self):
        as_of = self.dates[-1] + pd.offsets.BDay(5)
        result = self.calculate(as_of=as_of)
        self.assertTrue(pd.isna(result["stress_index"].iloc[-1]))
        self.assertEqual(classify_stress_regime(result["stress_index"].iloc[-1]), "UNKNOWN")

    def test_persistently_high_spreads_remain_visible(self):
        result = self.calculate(hy_oas=pd.Series(6.5, index=self.dates),
                                bbb_oas=pd.Series(3.0, index=self.dates))
        self.assertGreater(result["stress_index"].iloc[-1], 75.0)

    def test_persistent_funding_spread_remains_visible(self):
        result = self.calculate(sofr=pd.Series(4.15, index=self.dates),
                                effr=pd.Series(4.0, index=self.dates))
        self.assertGreater(result["stress_score"].iloc[-1],
                           self.calculate()["stress_score"].iloc[-1])

    def test_calm_inputs_can_produce_below_neutral_index(self):
        hy = pd.Series([3.5 - i * 0.01 for i in range(80)], index=self.dates)
        bbb = pd.Series([1.5 - i * 0.004 for i in range(80)], index=self.dates)
        result = self.calculate(hy_oas=hy, bbb_oas=bbb)
        self.assertLess(result["stress_index"].iloc[-1], 50.0)

    def test_future_observation_does_not_change_past_score(self):
        baseline = self.calculate()
        future = self.dates[-1] + pd.offsets.BDay(1)
        result = self.calculate(hy_oas=pd.concat([self.hy, pd.Series([10.0], index=[future])]))
        self.assertAlmostEqual(baseline["stress_score"].iloc[-1],
                               result.loc[self.dates[-1], "stress_score"])


if __name__ == "__main__":
    unittest.main()
