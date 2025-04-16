# src/price_engine/price_calculator.py
import numpy as np

class PriceCalculator:
    @staticmethod
    def calculate_weighted_average(prices: dict, weights: dict) -> float:
        valid_prices = {k: v for k, v in prices.items() if v is not None}
        if not valid_prices:
            raise ValueError("No valid prices available.")

        weighted_sum = sum(valid_prices[source] * weights[source] for source in valid_prices)
        total_weight = sum(weights[source] for source in valid_prices)
        return weighted_sum / total_weight

    @staticmethod
    def handle_outliers(prices: dict) -> dict:
        valid_prices = [v for v in prices.values() if v is not None]
        if not valid_prices:
            return prices

        mean = np.mean(valid_prices)
        std = np.std(valid_prices)
        return {k: v if v is None or abs(v - mean) <= 2 * std else None for k, v in prices.items()}