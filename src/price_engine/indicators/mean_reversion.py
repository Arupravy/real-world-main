# src/price_engine/indicators/mean_reversion.py
from .base_indicator import BaseIndicator
import numpy as np

class MeanReversion(BaseIndicator):
    def __init__(self, window: int = 20, threshold: float = 2.0):
        """
        Initialize the Mean Reversion indicator.
        :param window: The window size for calculating the mean and standard deviation.
        :param threshold: The number of standard deviations to detect overbought/oversold conditions.
        """
        self.window = window
        self.threshold = threshold

    def calculate(self, data: list) -> dict:
        """
        Detect overbought/oversold conditions.
        :param data: List of price data (e.g., [{"price": 100.0}, {"price": 101.0}, ...]).
        :return: Dictionary with overbought/oversold signals.
        """
        prices = [entry["price"] for entry in data]
        if len(prices) < self.window:
            raise ValueError(f"Not enough data points. Required: {self.window}, Available: {len(prices)}")

        # Calculate mean and standard deviation
        mean = np.mean(prices[-self.window:])
        std_dev = np.std(prices[-self.window:])

        if std_dev == 0:
            # Avoid division by zero, return neutral signals
            return {
                "overbought": False,
                "oversold": False,
            }

        # Detect overbought/oversold conditions
        current_price = prices[-1]
        deviation = (current_price - mean) / std_dev

        return {
            "overbought": deviation > self.threshold,
            "oversold": deviation < -self.threshold,
        }
