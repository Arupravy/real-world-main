# C:\real-world-main\src\backtesting_engine\portfolio.py

class Portfolio:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # symbol -> quantity
        self.trade_log = []
        self.net_worth_history = []
        self.current_position = {}  # symbol -> "long" or "short" or None

    def buy(self, symbol: str, price: float, qty: int):
        cost = price * qty
        if self.cash >= cost:
            self.cash -= cost
            self.current_position[symbol] = "long"
            self.positions[symbol] = self.positions.get(symbol, 0) + qty
            self.trade_log.append({
                "action": "BUY",
                "symbol": symbol,
                "price": price,
                "qty": qty
            })
        else:
            print(f"[WARN] Not enough cash to buy {qty} of {symbol} at {price:.2f} | Available: {self.cash:.2f}")

    def sell(self, symbol: str, price: float, qty: int):
        held_qty = self.positions.get(symbol, 0)

        if held_qty > 0:
            # Selling from a long position
            if held_qty >= qty:
                self.positions[symbol] -= qty
            else:
                qty = held_qty
                self.positions[symbol] = 0
                print(f"[WARN] Partial sell: Only had {held_qty} of {symbol}, sold all.")
            
            self.cash += price * qty
            if self.positions[symbol] == 0:
                self.current_position[symbol] = None
            self.trade_log.append({
                "action": "SELL",
                "symbol": symbol,
                "price": price,
                "qty": qty,
                "note": "long position"
            })

        elif held_qty < 0:
             # Covering a short position
            cover_qty = min(abs(held_qty), qty)
            self.positions[symbol] += cover_qty
            self.cash -= price * cover_qty
            if self.positions[symbol] == 0:
                self.current_position[symbol] = None

            
            self.trade_log.append({
                "action": "BUY_TO_COVER",
                "symbol": symbol,
                "price": price,
                "qty": cover_qty,
                "note": "short position"
            })

        else:
            print(f"[WARN] No position to sell for {symbol}")


    def short(self, symbol: str, price: float, qty: int):
        # Borrow asset and sell at current price (cash increases)
        self.cash += price * qty
        self.positions[symbol] = self.positions.get(symbol, 0) - qty  # negative = short
        self.current_position[symbol] = "short"
        self.trade_log.append({
              "action": "SHORT",
              "symbol": symbol,
              "price": price,
              "qty": qty
        })

    def update_net_worth(self, current_prices: dict):
        """
        Handles both single float (for single symbol) or dict of symbol -> price
        """
        net_worth = self.cash

        for symbol, qty in self.positions.items():
            if isinstance(current_prices, dict):
                symbol_price = current_prices.get(symbol, 0.0)
            else:
                symbol_price = current_prices  # fallback if it's accidentally still passed as float

            net_worth += qty * symbol_price  # Applies same price to all holdings

        self.net_worth_history.append(net_worth)

    def calculate_pnl(self, current_prices: dict):
        realized_pnl = 0
        unrealized_pnl = 0

        for trade in self.trade_log:
            if trade["action"] in ["SELL", "BUY_TO_COVER"]:
                # Closing trade, calculate realized P&L
                symbol = trade["symbol"]
                qty = trade["qty"]
                price = trade["price"]

                # Find corresponding opening trade
                open_trades = [t for t in self.trade_log if t["symbol"] == symbol and
                            ((t["action"] == "BUY" and trade["action"] == "SELL") or
                                (t["action"] == "SHORT" and trade["action"] == "BUY_TO_COVER"))]

                if open_trades:
                    entry_price = open_trades[-1]["price"]
                    direction = 1 if trade["action"] == "SELL" else -1
                    realized_pnl += direction * qty * (price - entry_price)

        for symbol, qty in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                if qty > 0:
                    buy_trades = [t for t in self.trade_log if t["symbol"] == symbol and t["action"] == "BUY"]
                    if buy_trades:
                        entry_price = buy_trades[-1]["price"]
                        unrealized_pnl += qty * (current_price - entry_price)
                elif qty < 0:
                    short_trades = [t for t in self.trade_log if t["symbol"] == symbol and t["action"] == "SHORT"]
                    if short_trades:
                        entry_price = short_trades[-1]["price"]
                        unrealized_pnl += abs(qty) * (entry_price - current_price)

        return {
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": realized_pnl + unrealized_pnl
        }


    def get_final_net_worth(self):
        if not self.net_worth_history:
            return self.initial_capital
        return self.net_worth_history[-1]