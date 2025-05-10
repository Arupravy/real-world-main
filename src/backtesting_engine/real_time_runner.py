# src/backtesting_engine/real_time_runner.py

import time
import threading
from datetime import datetime
from collections import defaultdict
from .strategies.strategy_trend_breakout import (
    strategy_trend_breakout as strategy_mean_reversion,
    atr
)

class RealTimeTrader:
    def __init__(self, capital, runtime):
        self.runtime = runtime
        self.initial_capital = capital
        self.cash_balance = capital             # Realized PnL stored here (initial + realized)
        self.positions = {}
        self.data = defaultdict(list)
        self.logs = []
        self.pnl_timeline = []
        self.start_time = time.time()
        self.lock = threading.Lock()

        self.last_logged_action = defaultdict(lambda: None)
        self.last_logged_price = defaultdict(lambda: None)
        self.last_log_time = defaultdict(lambda: 0)

        # Execution controls
        self.cooldown_seconds = 5       # prevent overtrading
        self.min_price_change = 0.05    # price threshold for logs
        self.is_active = True
        self.current_candle = defaultdict(lambda: None)

        # Risk management
        self.risk_per_trade = 0.02       # 2% of equity per trade
        self.atr_sl_multiplier = 1.5     # SL at 1.5×ATR
        self.atr_tp_multiplier = 3.0     # TP at 3×ATR

    def on_price_update(self, symbol, price):
        if not self.is_active:
            return

        with self.lock:
            price = float(price)
            if price <= 0:
                return
            now = time.time()
            timestamp = datetime.now()

            # Build or update 1-second candle
            candle = self.current_candle[symbol]
            if not candle or (timestamp - candle['timestamp']).seconds >= 1:
                if candle:
                    self.data[symbol].append(candle)
                    if len(self.data[symbol]) > 1000:
                        self.data[symbol] = self.data[symbol][-1000:]
                self.current_candle[symbol] = {
                    'timestamp': timestamp,
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price
                }
            else:
                candle['high'] = max(candle['high'], price)
                candle['low']  = min(candle['low'], price)
                candle['close'] = price

            # SL/TP exit
            if symbol in self.positions:
                pos = self.positions[symbol]
                sl, tp, side = pos['stop_loss_price'], pos['take_profit_price'], pos['side']
                hit_sl = (side == 'long' and price <= sl) or (side == 'short' and price >= sl)
                hit_tp = (side == 'long' and price >= tp) or (side == 'short' and price <= tp)
                if hit_sl or hit_tp:
                    reason = 'TP' if hit_tp else 'SL'
                    self.logs.append(f"[{timestamp.strftime('%H:%M:%S')}] {symbol}: {price:.2f} ➤ Exit {reason}")
                    self.exit_position(symbol, price)
                    self.last_logged_action[symbol] = None
                    self.last_logged_price[symbol]  = price
                    self.last_log_time[symbol]      = now
                    return

            # Need history
            if len(self.data[symbol]) < 50:
                return

            # Signal
            current_side = self.positions.get(symbol, {}).get('side')
            action = strategy_mean_reversion(self.data[symbol], current_side)

            # Cooldown
            last_act = self.last_logged_action[symbol]
            last_tm  = self.last_log_time[symbol]
            if action and action != last_act and (now - last_tm) >= self.cooldown_seconds:
                self.logs.append(f"[{timestamp.strftime('%H:%M:%S')}] {symbol}: {price:.2f} ➤ Action: {action.upper()}")
                if action == 'buy':
                    if current_side == 'short':
                        self.exit_position(symbol, price)
                    if self.positions.get(symbol, {}).get('side') != 'long':
                        self.enter_position(symbol, 'long', price)
                else:
                    if current_side == 'long':
                        self.exit_position(symbol, price)
                    if self.positions.get(symbol, {}).get('side') != 'short':
                        self.enter_position(symbol, 'short', price)

                self.last_logged_action[symbol] = action
                self.last_logged_price[symbol]  = price
                self.last_log_time[symbol]      = now

            # Record PnL
            unreal = self.calculate_unrealized_pnl()
            total_val = self.cash_balance + unreal
            self.pnl_timeline.append({
                'timestamp': timestamp.isoformat(),
                'portfolio_value': total_val
            })

            # Stop after runtime
            if now - self.start_time > self.runtime:
                self.is_active = False

    def get_current_position(self, symbol):
        return self.positions.get(symbol, {}).get('side')

    def enter_position(self, symbol, side, price):
        if symbol in self.positions:
            return
        total_equity = self.cash_balance + self.calculate_unrealized_pnl()
        atr_value = atr(self.data[symbol][-15:], period=14)
        if not atr_value or atr_value <= 0:
            return

        stop_dist = atr_value * self.atr_sl_multiplier
        tp_dist   = atr_value * self.atr_tp_multiplier
        stop_price = price - stop_dist if side == 'long' else price + stop_dist
        tp_price   = price + tp_dist   if side == 'long' else price - tp_dist

        risk_amount = total_equity * self.risk_per_trade
        size = risk_amount / stop_dist

        self.positions[symbol] = {
            'side': side,
            'size': size,
            'entry_price': price,
            'stop_loss_price': stop_price,
            'take_profit_price': tp_price
        }

    def exit_position(self, symbol, price):
        pos = self.positions.pop(symbol, None)
        if not pos:
            return
        size  = pos['size']
        entry = pos['entry_price']
        side  = pos['side']
        pnl   = (price - entry) * size if side == 'long' else (entry - price) * size
        self.cash_balance += pnl

    def calculate_unrealized_pnl(self):
        total = 0
        for sym, pos in self.positions.items():
            current = (self.current_candle[sym]['close'] if self.current_candle[sym]
                       else self.data[sym][-1]['close'])
            entry = pos['entry_price']
            size  = pos['size']
            if pos['side'] == 'long':
                total += (current - entry) * size
            else:
                total += (entry - current) * size
        return total

    def get_trade_count(self):
        return len([l for l in self.logs if 'Action' in l])

    def get_portfolio_summary(self):
        val = self.cash_balance + self.calculate_unrealized_pnl()
        return {
            'initial_capital': self.initial_capital,
            'cash_balance': self.cash_balance,
            'unrealized_pnl': self.calculate_unrealized_pnl(),
            'final_pnl': val - self.initial_capital,
            'final_portfolio_value': val,
            'position_count': len(self.positions)
        }

    def get_positions(self):
        with self.lock:
            return self.positions.copy()

    def get_logs(self):
        with self.lock:
            return list(self.logs)

    def get_price_data(self):
        with self.lock:
            return dict(self.data)

    def get_pnl_data(self):
        with self.lock:
            return list(self.pnl_timeline)

    def reset(self):
        with self.lock:
            self.data.clear()
            self.logs.clear()
            self.pnl_timeline.clear()
            self.positions.clear()
            self.cash_balance = self.initial_capital
            self.start_time = time.time()
            self.is_active = True
            self.last_logged_action.clear()
            self.last_logged_price.clear()
            self.last_log_time.clear()

    def stop(self):
        with self.lock:
            self.is_active = False
            for sym in list(self.positions.keys()):
                price = (self.current_candle[sym]['close'] if self.current_candle[sym]
                         else self.data[sym][-1]['close'])
                self.exit_position(sym, price)

    def validate_state(self):
        with self.lock:
            elapsed = time.time() - self.start_time
            return {
                'is_active': self.is_active,
                'positions': len(self.positions),
                'last_update': elapsed
            }
