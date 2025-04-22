# src/backtesting_engine/strategies/real_time_strategy.py
import numpy as np
from collections import defaultdict, deque
from typing import Dict, Optional

class RealTimeStrategy:
    def __init__(self):
        # Strategy parameters (tunable)
        self.risk_per_trade = 0.01  # 1% of capital per trade
        self.min_reward_ratio = 2.0  # 1:2 risk-reward minimum
        self.trend_confirmation_bars = 50
        self.max_trade_duration = 14400  # 4 hours in seconds
        
        # State tracking
        self.price_history = defaultdict(deque)
        self.last_signal = defaultdict(lambda: None)

    def update_data(self, symbol: str, price: float, timestamp: float):
        """Update price history for each symbol"""
        self.price_history[symbol].append(price)
        # Keep reasonable history size
        if len(self.price_history[symbol]) > 1000:
            self.price_history[symbol].popleft()

    def calculate_indicators(self, symbol: str) -> Dict:
        """Calculate all technical indicators"""
        prices = list(self.price_history[symbol])
        if len(prices) < self.trend_confirmation_bars:
            return {}
            
        # Moving Averages
        ema_20 = self._ema(prices, 20)
        ema_50 = self._ema(prices, 50)
        
        # Volatility (ATR)
        atr = self._calculate_atr(prices, 14)
        
        # Trend Detection
        trend_strength = self._calculate_trend_strength(prices)
        
        # Mean Reversion Bands
        mean = np.mean(prices[-20:])
        std = np.std(prices[-20:])
        
        return {
            'price': prices[-1],
            'ema_20': ema_20,
            'ema_50': ema_50,
            'atr': atr,
            'trend': trend_strength,
            'upper_band': mean + (std * 2),
            'lower_band': mean - (std * 2)
        }

    def generate_signal(self, symbol: str, current_position: Optional[str]) -> Dict:
        """
        Generate trading signals with risk parameters
        Returns: {
            'signal': 'buy'/'sell'/None,
            'stop_loss': float,
            'take_profit': float,
            'size': float
        }
        """
        if symbol not in self.price_history or len(self.price_history[symbol]) < 50:
            return {'signal': None}
            
        indicators = self.calculate_indicators(symbol)
        price = indicators['price']
        atr = indicators['atr']
        
        # Initialize default response
        signal = {
            'signal': None,
            'stop_loss': None,
            'take_profit': None,
            'size': None
        }
        
        # Trend Filter
        in_uptrend = indicators['trend'] > 0.5
        in_downtrend = indicators['trend'] < -0.5
        
        # Mean Reversion Signals (only in non-trending markets)
        if not in_uptrend and not in_downtrend:
            if price < indicators['lower_band']:
                signal.update({
                    'signal': 'buy',
                    'stop_loss': price - (atr * 1.5),
                    'take_profit': price + (atr * 3.0),
                    'size': self.risk_per_trade / (atr * 1.5)
                })
            elif price > indicators['upper_band']:
                signal.update({
                    'signal': 'sell',
                    'stop_loss': price + (atr * 1.5),
                    'take_profit': price - (atr * 3.0),
                    'size': self.risk_per_trade / (atr * 1.5)
                })
        
        # Position Management
        if current_position == 'long':
            if in_downtrend or price >= signal.get('take_profit', float('inf')):
                signal['signal'] = 'sell'
        elif current_position == 'short':
            if in_uptrend or price <= signal.get('take_profit', -float('inf')):
                signal['signal'] = 'buy'
                
        return signal

    # Technical Indicator Calculations ----------------------------
    
    def _ema(self, prices, window):
        if len(prices) < window:
            return None
        weights = np.exp(np.linspace(-1., 0., window))
        weights /= weights.sum()
        return np.convolve(prices[-window:], weights, mode='valid')[-1]
    
    def _calculate_atr(self, prices, period):
        if len(prices) < period:
            return 0
        true_ranges = []
        for i in range(1, len(prices)):
            high_low = abs(prices[i] - prices[i-1])
            true_ranges.append(high_low)
        return np.mean(true_ranges[-period:])
    
    def _calculate_trend_strength(self, prices):
        """Returns -1 (strong downtrend) to +1 (strong uptrend)"""
        if len(prices) < 50:
            return 0
            
        # Slope analysis
        x = np.arange(50)
        y = np.array(prices[-50:])
        slope = np.polyfit(x, y, 1)[0]
        
        # EMA alignment
        ema_20 = self._ema(prices, 20)
        ema_50 = self._ema(prices, 50)
        
        # Normalize to [-1, 1] range
        trend_strength = slope * 1000  # Scale slope
        if ema_20 and ema_50:
            trend_strength *= (1 if ema_20 > ema_50 else -1)
        return np.clip(trend_strength, -1, 1)