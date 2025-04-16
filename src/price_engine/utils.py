# src/price_engine/utils.py
import numpy as np

def handle_outliers(prices: dict) -> dict:
    """
    Handle outliers in the prices dictionary by excluding prices that are more than
    2 standard deviations away from the mean.
    """
    valid_prices = [v for v in prices.values() if v is not None]
    if not valid_prices:
        return prices

    mean = np.mean(valid_prices)
    std = np.std(valid_prices)
    return {k: v if v is None or abs(v - mean) <= 2 * std else None for k, v in prices.items()}