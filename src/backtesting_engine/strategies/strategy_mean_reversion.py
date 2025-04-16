# src/backtesting_engine/strategies/strategy_mean_reversion.py
from price_engine.indicators.mean_reversion import MeanReversion
import numpy as np

mr = MeanReversion(window=20, threshold=1.2)

def sma(prices, window):
    if len(prices) < window:
        return None
    return np.mean(prices[-window:])

def slope(prices, window):
    if len(prices) < window:
        return 0
    x = np.arange(window)
    y = np.array(prices[-window:])
    # Linear regression to calculate slope
    A = np.vstack([x, np.ones(len(x))]).T
    m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return m

def ema(prices, window):
    if len(prices) < window:
        return None
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a = np.convolve(prices[-window:], weights, mode='valid')
    return a[-1] if len(a) > 0 else None

def detect_trend(data_window, short_window=20, long_window=50, slope_threshold=0.003):
    prices = [candle["price"] for candle in data_window]

    if len(prices) < long_window:
        return False, "sideways"

    # Calculate EMAs
    short_ema = ema(prices, short_window)
    long_ema = ema(prices, long_window)
    trend_slope = slope(prices, long_window)

    # Basic sanity check
    if short_ema is None or long_ema is None:
        return False, "sideways"

    # Trend logic
    if abs(trend_slope) < slope_threshold:
        return False, "sideways"
    elif short_ema > long_ema and trend_slope > slope_threshold:
        return True, "up"
    elif short_ema < long_ema and trend_slope < -slope_threshold:
        return True, "down"

    return False, "sideways"


def strategy_mean_reversion(data_window: list, current_position: str = None) -> str:
    latest_price = data_window[-1]['price']
    is_trending, trend_direction = detect_trend(data_window)

    # print(f"Price: {latest_price:.2f}, Trend: {trend_direction}, In Trend Mode: {is_trending}, Position: {current_position}")

    # Trend-following logic
    if is_trending:
        if trend_direction == "up":
            if current_position != "long":
                return "buy"  # enter long
        elif trend_direction == "down":
            if current_position != "short":
                return "sell"  # enter short
        else:
            # Exit position if trend direction changes
            if current_position == "long" and trend_direction != "up":
                return "sell"
            elif current_position == "short" and trend_direction != "down":
                return "buy"

    # Mean Reversion logic (only if not trending)
    if not is_trending:
        result = mr.calculate(data_window)
       # print(f"[MR] Price: {latest_price:.2f} | Oversold: {result['oversold']} | Overbought: {result['overbought']}")
        if result["oversold"] and current_position != "long":
            return "buy"
        elif result["overbought"] and current_position != "short":
            return "sell"

    return None