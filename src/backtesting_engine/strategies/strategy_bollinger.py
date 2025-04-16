# C:\real-world-main\src\backtesting_engine\strategies\strategy_bollinger.py

from price_engine.indicators.bollinger_bands import BollingerBands
from ta.momentum import RSIIndicator
import pandas as pd

bb = BollingerBands(window=20, num_std=2)

def strategy_bollinger(data_window: list, current_position: str = None) -> str:
    """
    Improved strategy:
    - Buy when price < lower band AND RSI < 40 (oversold-ish), only if not already in long
    - Sell when price > upper band AND RSI > 60 (overbought-ish), only if currently in long
    """

    if len(data_window) < 20:
        return None  # not enough data yet

    df = pd.DataFrame(data_window)
    current_price = df["price"].iloc[-1]
    timestamp = df["timestamp"].iloc[-1] if "timestamp" in df.columns else "Unknown"

    # Bollinger Bands
    result = bb.calculate(data_window)

    # RSI calculation
    rsi = RSIIndicator(close=df["price"], window=14).rsi().iloc[-1]

    # Logging for debug
    #print(f"[{timestamp}] Price: {current_price:.2f}, RSI: {rsi:.2f}, Lower: {result['lower_band']:.2f}, Upper: {result['upper_band']:.2f}, Position: {current_position}")

    # Buy signal (relaxed condition)
    if current_position != "long" and current_price < result["lower_band"] and rsi < 40:
        #print(f"→ Buy signal triggered")
        return "buy"

    # Sell signal (relaxed condition)
    elif current_position == "long" and current_price > result["upper_band"] and rsi > 60:
        #print(f"→ Sell signal triggered")
        return "sell"

    return None
