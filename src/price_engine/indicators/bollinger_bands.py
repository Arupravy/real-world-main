# src/price_engine/indicators/bollinger_bands.py
from .base_indicator import BaseIndicator
import numpy as np

class BollingerBands(BaseIndicator):
    def __init__(self, window: int = 20, num_std: int = 2):
        """
        Initialize the Bollinger Bands indicator.
        :param window: The window size for the moving average and standard deviation.
        :param num_std: The number of standard deviations for the upper/lower bands.
        """
        self.window = window
        self.num_std = num_std

    def calculate(self, data: list) -> dict:
        """
        Calculate Bollinger Bands.
        :param data: List of price data (e.g., [{"price": 100.0}, {"price": 101.0}, ...]).
        :return: Dictionary with Bollinger Bands values.
        """
        prices = [entry["price"] for entry in data]
        if len(prices) < self.window:
            raise ValueError(f"Not enough data points. Required: {self.window}, Available: {len(prices)}")

        # Calculate moving average and standard deviation
        moving_avg = np.mean(prices[-self.window:])
        std_dev = np.std(prices[-self.window:])

        # Calculate upper and lower bands
        upper_band = moving_avg + (self.num_std * std_dev)
        lower_band = moving_avg - (self.num_std * std_dev)

        return {
            "moving_avg": moving_avg,
            "upper_band": upper_band,
            "lower_band": lower_band,
        }