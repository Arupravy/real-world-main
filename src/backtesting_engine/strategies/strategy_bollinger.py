# src\backtesting_engine\strategies\strategy_bollinger.py
"""
Improved Bollinger-Band strategy with trade-throttle controls.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List, Literal
import numpy as np, pandas as pd
from price_engine.indicators.bollinger_bands import BollingerBands
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

Position = Optional[Literal["long", "short"]]


@dataclass
class StrategyConfig:
    # Bollinger parameters
    bb_window: int = 20
    bb_std: float = 2
    # RSI
    rsi_window: int = 14
    # Trend filter
    short_ema: int = 20
    long_ema: int = 50
    slope_threshold: float = 0.002
    # ATR & risk
    atr_window: int = 14
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 3.0
    trail_atr_mult: float = 1.0
    sl_pct: float = 0.03          # fallback stop / target when ATR not present
    tp_pct: float = 0.06
    # ───── trade-throttle additions ────────────────────────────────────
    min_bw: float = 0.04          # 4 % of price
    cooldown_bars: int = 7        # wait one trading week after every exit


class BollingerStrategy:
    def __init__(self, config: Optional[StrategyConfig] = None):
        self.cfg = config or StrategyConfig()
        self.bb = BollingerBands(self.cfg.bb_window, self.cfg.bb_std)
        self.prev_entry_price: Optional[float] = None
        self.sl_price: Optional[float] = None
        self.tp_price: Optional[float] = None
        self.position: Position = None
        self.cooldown_left: int = 0            # ← new

    # ---------- helpers -------------------------------------------------- #
    @staticmethod
    def _ema(series: pd.Series, window: int) -> float:
        return series.ewm(span=window, adjust=False).mean().iloc[-1]

    @staticmethod
    def _slope(series: pd.Series, window: int) -> float:
        y, x = series.iloc[-window:], np.arange(window)
        m, _ = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)[0]
        return m

    def _detect_trend(self, price: pd.Series) -> Tuple[bool, str]:
        if len(price) < self.cfg.long_ema:
            return False, "sideways"
        s_ema, l_ema = self._ema(price, self.cfg.short_ema), self._ema(price, self.cfg.long_ema)
        slp = self._slope(price, self.cfg.long_ema)
        if abs(slp) < self.cfg.slope_threshold:
            return False, "sideways"
        if s_ema > l_ema and slp > 0:
            return True, "up"
        if s_ema < l_ema and slp < 0:
            return True, "down"
        return False, "sideways"

    # ---------- risk helpers -------------------------------------------- #
    def _update_exit_levels(self, atr: Optional[float]):
        if atr is not None:
            sl_off, tp_off = self.cfg.sl_atr_mult * atr, self.cfg.tp_atr_mult * atr
        else:
            sl_off, tp_off = self.cfg.sl_pct * self.prev_entry_price, self.cfg.tp_pct * self.prev_entry_price
        if self.position == "long":
            self.sl_price, self.tp_price = self.prev_entry_price - sl_off, self.prev_entry_price + tp_off
        else:
            self.sl_price, self.tp_price = self.prev_entry_price + sl_off, self.prev_entry_price - tp_off

    def _trail_stop(self, px: float, atr: float):
        if self.position == "long" and px - self.prev_entry_price > self.cfg.trail_atr_mult * atr:
            self.sl_price = max(self.sl_price, px - self.cfg.sl_atr_mult * atr)
        elif self.position == "short" and self.prev_entry_price - px > self.cfg.trail_atr_mult * atr:
            self.sl_price = min(self.sl_price, px + self.cfg.sl_atr_mult * atr)

    # ---------- main ----------------------------------------------------- #
    def generate_signal(self, data_window: List[dict], current_position: Position = None) -> Optional[str]:
        if len(data_window) < max(self.cfg.long_ema, self.cfg.bb_window, self.cfg.atr_window):
            return None

        df = pd.DataFrame(data_window)
        price = df["price"]
        cur   = price.iloc[-1]

        # optional ATR
        atr = None
        if {"high", "low"}.issubset(df.columns):
            atr = AverageTrueRange(df["high"], df["low"], price, self.cfg.atr_window).average_true_range().iloc[-1]

        # band stats
        bb = self.bb.calculate(data_window)
        upper, lower = bb["upper_band"], bb["lower_band"]
        mid = (upper + lower) / 2
        band_width = (upper - lower) / cur

        # ---------- THROTTLE: width & cooldown -------------------------- #
        if band_width < self.cfg.min_bw or self.cooldown_left > 0:
            self.cooldown_left = max(0, self.cooldown_left - 1)
            return None
        # ---------------------------------------------------------------- #

        # trend & RSI
        trending, trend_dir = self._detect_trend(price)
        rsi = RSIIndicator(price, window=self.cfg.rsi_window).rsi().iloc[-1]

        # ---------- manage open trade -------------------------------- #
        if self.position:
            if atr is not None:
                self._trail_stop(cur, atr)
            if (self.position == "long" and cur <= self.sl_price) or (self.position == "short" and cur >= self.sl_price):
                sig = "sell" if self.position == "long" else "buy"
                self._reset_state()
                return sig
            if (self.position == "long" and cur >= self.tp_price) or (self.position == "short" and cur <= self.tp_price):
                sig = "sell" if self.position == "long" else "buy"
                self._reset_state()
                return sig
            if (self.position == "long" and cur < mid) or (self.position == "short" and cur > mid):
                sig = "sell" if self.position == "long" else "buy"
                self._reset_state()
                return sig
            return None

        # ---------- open new trade ----------------------------------- #
        if not trending:
            if cur < lower and rsi < 40:
                self._open_trade(cur, "long", atr)
                return "buy"
            if cur > upper and rsi > 60:
                self._open_trade(cur, "short", atr)
                return "sell"
        else:
            if trend_dir == "up" and cur > mid and rsi > 55:
                self._open_trade(cur, "long", atr)
                return "buy"
            if trend_dir == "down" and cur < mid and rsi < 45:
                self._open_trade(cur, "short", atr)
                return "sell"
        return None

    # ---------- state helpers ---------------------------------------- #
    def _open_trade(self, price: float, side: Position, atr: Optional[float]):
        self.position = side
        self.prev_entry_price = price
        self._update_exit_levels(atr)

    def _reset_state(self):
        self.position = None
        self.prev_entry_price = self.sl_price = self.tp_price = None
        self.cooldown_left = self.cfg.cooldown_bars      # ← start wait


# backward-compat wrapper
_cfg = StrategyConfig()
_strategy = BollingerStrategy(_cfg)

def strategy_bollinger(data_window: List[dict], current_position: Position = None) -> Optional[str]:
    return _strategy.generate_signal(data_window, current_position)
