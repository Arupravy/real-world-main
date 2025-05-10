# strategy_trendpullback.py  v2 – tuned for stocks 2020-2025
from dataclasses import dataclass
from typing import List, Optional, Literal
import pandas as pd
from ta.trend import SMAIndicator
from ta.volatility import AverageTrueRange

Position = Optional[Literal["long", "short"]]

@dataclass
class CFG:
    sma_trend: int = 150   # <- was 200
    sma_entry: int = 30    # <- was 50
    atr_window: int = 20
    trail_mult: float = 6.0  # <- was 4.0

class LongTrend:
    def __init__(self, cfg: CFG = CFG()):
        self.cfg = cfg
        self.pos: Position = None
        self.highest: Optional[float] = None
        self.trail  : Optional[float] = None

    # ------------------------------------------------------------------ #
    def _ind(self, df: pd.DataFrame):
        c = df["price"]
        return (
            c.iloc[-1],
            SMAIndicator(c, self.cfg.sma_trend).sma_indicator().iloc[-1],
            SMAIndicator(c, self.cfg.sma_entry).sma_indicator().iloc[-1],
            AverageTrueRange(df["high"], df["low"], c,
                             self.cfg.atr_window).average_true_range().iloc[-1]
        )

    def generate(self, window: List[dict], current_position: Position = None):
        if len(window) < self.cfg.sma_trend:        # make sure trend SMA exists
            return None

        df = pd.DataFrame(window)
        price, sma_t, sma_e, atr = self._ind(df)

        # ---------- manage open long ----------------------------------- #
        if current_position == "long":
            self.highest = max(self.highest, price)
            self.trail   = self.highest - self.cfg.trail_mult * atr
            if price < self.trail or price < sma_t:
                self.highest = self.trail = None
                return "sell"
            return None

        # ---------- look for new long ---------------------------------- #
        in_uptrend = price > sma_t
        pullback   = price <= sma_e * 1.02        # within 2 % of entry SMA
        if in_uptrend and pullback:
            self.highest = price
            self.trail   = price - self.cfg.trail_mult * atr
            return "buy"
        return None

# wrapper
_cfg      = CFG()
_strategy = LongTrend(_cfg)

def strategy_trendpullback(data_window: List[dict], current_position: Position = None):
    return _strategy.generate(data_window, current_position)
