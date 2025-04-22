# C:\real-world-main\src\backtesting_engine\strategies\strategy_crypto_momentum.py

from price_engine.indicators.mean_reversion import MeanReversion
import numpy as np
from collections import deque, defaultdict

# Crypto-specific parameters
CRYPTO_VOLATILITY_WINDOW = 14  # Typical period for crypto volatility measurement
MIN_VOLATILITY_THRESHOLD = 0.01  # Minimum volatility to trade (1%)
MAX_VOLATILITY_THRESHOLD = 0.05  # Maximum volatility to trade (5%)
TREND_CONFIRMATION_PERIOD = 3  # Number of confirmations needed
MOMENTUM_WINDOW = 10  # Short-term momentum window
VOLUME_SPIKE_MULTIPLIER = 2.0  # Threshold for volume spikes

class CryptoMomentumStrategy:
    def __init__(self):
        self.trend_confirmation = defaultdict(int)
        self.volume_history = defaultdict(lambda: deque(maxlen=20))
        self.symbol = None  # Track current symbol being analyzed
        
    def calculate_volatility(self, prices):
        """Calculate normalized volatility for crypto markets"""
        if len(prices) < 2:
            return 0
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns) * np.sqrt(365)  # Annualized volatility

    def detect_volume_spike(self, current_volume):
        """Detect unusual volume activity typical in crypto markets"""
        if len(self.volume_history[self.symbol]) < 5:
            return False
        avg_volume = np.mean(self.volume_history[self.symbol])
        return current_volume > avg_volume * VOLUME_SPIKE_MULTIPLIER

    def get_momentum_strength(self, prices):
        """Calculate momentum strength with smoothing"""
        if len(prices) < MOMENTUM_WINDOW:
            return 0
        x = np.arange(MOMENTUM_WINDOW)
        y = np.array(prices[-MOMENTUM_WINDOW:])
        m, _ = np.linalg.lstsq(x.reshape(-1,1), y, rcond=None)[0]
        return m / np.mean(prices)  # Normalized momentum

    def strategy_crypto_momentum(self, data_window: list, current_position: str = None) -> str:
        """
        Crypto-optimized momentum strategy using same interface variables
        data_window: List of dictionaries with 'price' and other metrics
        current_position: Current position state ('long', 'short', or None)
        Returns: 'buy', 'sell', or None
        """
        if len(data_window) < 5:  # Minimum data requirement
            return None

        # Extract prices and volumes (assuming volume is available in data)
        prices = [d['price'] for d in data_window]
        volumes = [d.get('volume', 1) for d in data_window]
        latest_price = prices[-1]
        
        # Get symbol from the data window if available
        self.symbol = data_window[-1].get('symbol', 'default')
        
        # Update volume history
        self.volume_history[self.symbol].append(volumes[-1])
        
        # Calculate market conditions
        volatility = self.calculate_volatility(prices[-5:])
        momentum = self.get_momentum_strength(prices)
        volume_spike = self.detect_volume_spike(volumes[-1])
        
        # Filter conditions - don't trade in extreme volatility
        if volatility < MIN_VOLATILITY_THRESHOLD or volatility > MAX_VOLATILITY_THRESHOLD:
            return None

        # Strong momentum with volume confirmation
        if momentum > 0.005 and volume_spike:  # Positive momentum threshold
            if current_position != 'long':
                self.trend_confirmation[self.symbol] = self.trend_confirmation.get(self.symbol, 0) + 1
                if self.trend_confirmation[self.symbol] >= TREND_CONFIRMATION_PERIOD:
                    return 'buy'
        
        # Negative momentum with volume confirmation
        elif momentum < -0.005 and volume_spike:  # Negative momentum threshold
            if current_position != 'short':
                self.trend_confirmation[self.symbol] = self.trend_confirmation.get(self.symbol, 0) - 1
                if self.trend_confirmation[self.symbol] <= -TREND_CONFIRMATION_PERIOD:
                    return 'sell'
        
        # Exit conditions
        if current_position == 'long' and momentum < -0.002:
            return 'sell'
        elif current_position == 'short' and momentum > 0.002:
            return 'buy'
            
        return None

# Global instance
crypto_momentum = CryptoMomentumStrategy()

def strategy_mean_reversion(data_window: list, current_position: str = None) -> str:
    """
    Wrapper function with same signature as original
    Now routes to our crypto momentum strategy
    """
    return crypto_momentum.strategy_crypto_momentum(data_window, current_position)