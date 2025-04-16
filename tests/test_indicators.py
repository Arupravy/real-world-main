# tests/test_indicators.py
import unittest
from src.price_engine.indicators.bollinger_bands import BollingerBands
from src.price_engine.indicators.mean_reversion import MeanReversion

class TestIndicators(unittest.TestCase):
    def test_bollinger_bands(self):
        data = [{"price": 100.0}, {"price": 101.0}, {"price": 102.0}, {"price": 103.0}, {"price": 104.0}]
        bb = BollingerBands(window=3, num_std=2)
        result = bb.calculate(data)
        self.assertIn("moving_avg", result)
        self.assertIn("upper_band", result)
        self.assertIn("lower_band", result)

    def test_mean_reversion(self):
        data = [{"price": 100.0}, {"price": 101.0}, {"price": 102.0}, {"price": 103.0}, {"price": 104.0}]
        mr = MeanReversion(window=3, threshold=1.5)
        result = mr.calculate(data)
        self.assertIn("overbought", result)
        self.assertIn("oversold", result)

if __name__ == "__main__":
    unittest.main()