# src/price_engine/indicators/base_indicator.py
from abc import ABC, abstractmethod

class BaseIndicator(ABC):
    """Base class for all indicators."""

    @abstractmethod
    def calculate(self, data: list) -> dict:
        """
        Calculate the indicator values based on the provided data.
        :param data: List of price data (e.g., [{"price": 100.0}, {"price": 101.0}, ...]).
        :return: Dictionary of indicator values (e.g., {"upper_band": 105.0, "lower_band": 95.0}).
        """
        pass