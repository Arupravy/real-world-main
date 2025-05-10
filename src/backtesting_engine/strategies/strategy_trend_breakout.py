# src/backtesting_engine/strategies/strategy_trend_breakout.py

import numpy as np

def ema(prices, window):
    if len(prices) < window:
        return None
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    return np.convolve(prices[-window:], weights, mode='valid')[-1]

def atr(data_window, period=14):
    if len(data_window) < period + 1:
        return None
    tr_list = []
    for i in range(1, period+1):
        curr = data_window[-i]
        prev = data_window[-i-1]
        tr = max(curr['high']-curr['low'],
                 abs(curr['high']-prev['close']),
                 abs(curr['low']-prev['close']))
        tr_list.append(tr)
    return np.mean(tr_list)

def detect_trend(prices, short_window=5, long_window=20):
    if len(prices) < long_window:
        return False, 'sideways'
    closes = [c['close'] for c in prices]
    short_ema = ema(closes, short_window)
    long_ema = ema(closes, long_window)
    if short_ema is None or long_ema is None:
        return False, 'sideways'
    if short_ema > long_ema:
        return True, 'up'
    elif short_ema < long_ema:
        return True, 'down'
    return False, 'sideways'

def strategy_trend_breakout(data_window: list, current_position: str = None) -> str:
    if len(data_window) < 50:
        return None
    latest = data_window[-1]
    latest_price = latest['close']
    atr_value = atr(data_window, 14)
    if not atr_value:
        return None
    # Entry-only breakout logic
    is_trending, direction = detect_trend(data_window[-50:])
    prev_close = data_window[-2]['close']
    if is_trending and direction == 'up':
        if (latest_price - prev_close) > 1.0 * atr_value:
            return 'buy'
    if is_trending and direction == 'down':
        if (prev_close - latest_price) > 1.0 * atr_value:
            return 'sell'
    return None
