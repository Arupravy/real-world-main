# src/backtesting_engine/real_time_runner.py

import time
import threading
import csv
from datetime import datetime
from collections import defaultdict
from .strategies.strategy_mean_reversion import strategy_mean_reversion

class RealTimeTrader:
    def __init__(self, capital, runtime):
        self.runtime = runtime
        self.initial_capital = capital
        self.cash_balance = capital
        self.positions = {}
        self.data = defaultdict(list)
        self.logs = []
        self.pnl_timeline = []
        self.start_time = time.time()
        self.lock = threading.Lock()

        self.last_logged_action = defaultdict(lambda: None)
        self.last_logged_price = defaultdict(lambda: None)
        self.last_log_time = defaultdict(lambda: 0)

        self.cooldown_seconds = 1
        self.min_price_change = 0.05
        self.is_active = True

    def on_price_update(self, symbol, price):
        if not self.is_active:
            return

        with self.lock:
            price = float(price)
            timestamp = datetime.utcnow().isoformat()

            self.data[symbol].append({
                "timestamp": timestamp,
                "price": price
            })

            if len(self.data[symbol]) > 1000:
                self.data[symbol] = self.data[symbol][-1000:]

            action = strategy_mean_reversion(self.data[symbol], self.get_current_position(symbol))
            now = time.time()

            last_action = self.last_logged_action[symbol]
            last_price = self.last_logged_price[symbol]
            last_time = self.last_log_time[symbol]

            is_significant_price_move = abs(price - (last_price or 0)) >= self.min_price_change
            is_new_action = action != last_action
            is_cooldown_complete = (now - last_time) >= self.cooldown_seconds

            if is_new_action or is_significant_price_move or is_cooldown_complete:
                log_line = f"[{datetime.utcnow().strftime('%H:%M:%S')}] {symbol}: {price:.2f} ➤ Action: {action.upper()}"
                self.logs.append(log_line)

                if action == "buy":
                    self.enter_position(symbol, "long", price)
                elif action == "sell":
                    self.exit_position(symbol, price)

                self.last_logged_action[symbol] = action
                self.last_logged_price[symbol] = price
                self.last_log_time[symbol] = now

            self.pnl_timeline.append({
                "timestamp": timestamp,
                "portfolio_value": self.cash_balance + self.calculate_unrealized_pnl()
            })

            if now - self.start_time > self.runtime:
                self.is_active = False

    def get_current_position(self, symbol):
        return self.positions[symbol]["side"] if symbol in self.positions else None

    def enter_position(self, symbol, side, price):
        if symbol in self.positions:
            return

        allocation = 0.1 * self.cash_balance
        size = allocation / price

        self.positions[symbol] = {
            "side": side,
            "size": size,
            "entry_price": price
        }

        self.cash_balance -= allocation

    def exit_position(self, symbol, price):
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        pnl = 0

        if position["side"] == "long":
            pnl = (price - position["entry_price"]) * position["size"]

        self.cash_balance += (position["size"] * price) + pnl
        del self.positions[symbol]
    
    def get_trade_count(self):
        return len([log for log in self.logs if "Action" in log])

    def calculate_unrealized_pnl(self):
        total = 0
        for symbol, position in self.positions.items():
            if self.data[symbol]:
                latest_price = self.data[symbol][-1]['price']
                if position["side"] == "long":
                    total += (latest_price - position["entry_price"]) * position["size"]
        return total

    def get_portfolio_summary(self):
        final_balance = self.cash_balance + self.calculate_unrealized_pnl()
        net_pnl = final_balance - self.initial_capital
        return {
            "initial_capital": self.initial_capital,
            "cash_balance": self.cash_balance,
            "unrealized_pnl": self.calculate_unrealized_pnl(),
            "final_pnl": net_pnl,
            "final_portfolio_value": final_balance,
            "position_count": len(self.positions) 
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

    def is_running(self):
        return self.is_active
