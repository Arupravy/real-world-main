# C:\real-world-main\src\backtesting_engine\backtest_runner.py

import argparse
import json
import pandas as pd
from backtesting_engine.portfolio import Portfolio
from backtesting_engine.historical_data_loader import load_historical_data
from backtesting_engine.metrics import print_summary
from backtesting_engine.strategies.strategy_bollinger import strategy_bollinger
from backtesting_engine.strategies.strategy_mean_reversion import strategy_mean_reversion

def convert_to_indicator_format(row):
    return {"price": row["close"]}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run backtest using historical data.")
    parser.add_argument('--symbols', type=str, help="Comma-separated symbols (e.g., TSLA,AAPL)")
    parser.add_argument('--allocations', type=str, help="Comma-separated capital allocations in % (e.g., 40,30,30)")
    parser.add_argument('--start', type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument('--end', type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument('--asset_type', type=str, default='crypto', help="Asset type (e.g., crypto, stock)")
    parser.add_argument('--config', type=str, help="Optional config JSON file")
    parser.add_argument('--strategy', type=str, choices=['bollinger', 'mean_reversion'], default='bollinger', help="Strategy to run")
    return parser.parse_args()

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def run_backtest(symbol, start, end, asset_type, strategy_name, portfolio):
    df = load_historical_data(symbol, start, end, asset_type)
    data_for_indicators = []

    symbol_upper = symbol.upper()
    entry_price_map = {}

    for i in range(len(df)):
        row = df.iloc[i]
        data_for_indicators.append(convert_to_indicator_format(row))

        if len(data_for_indicators) < 50:
            continue

        price = row["close"]
        if symbol_upper not in portfolio.current_position:
            portfolio.current_position[symbol_upper] = None

        current_pos = portfolio.current_position.get(symbol_upper)
        current_qty = portfolio.positions.get(symbol_upper, 0)
        entry_price = entry_price_map.get(symbol_upper)

        # === TP/SL Logic ===
        if current_pos == "long" and entry_price:
            change_pct = ((price - entry_price) / entry_price) * 100
            if change_pct >= 20:
                portfolio.sell(symbol_upper, price, qty=current_qty)
                portfolio.current_position[symbol_upper] = None
                entry_price_map[symbol_upper] = None
                print(f"→ TAKE PROFIT LONG @ {price:.2f} (+{change_pct:.2f}%)")
                continue
            elif change_pct <= -7:
                portfolio.sell(symbol_upper, price, qty=current_qty)
                portfolio.current_position[symbol_upper] = None
                entry_price_map[symbol_upper] = None
                print(f"→ STOP LOSS LONG @ {price:.2f} ({change_pct:.2f}%)")
                continue

        if current_pos == "short" and entry_price:
            change_pct = ((entry_price - price) / entry_price) * 100
            if change_pct >= 20:
                portfolio.buy(symbol_upper, price, qty=abs(current_qty))
                portfolio.current_position[symbol_upper] = None
                entry_price_map[symbol_upper] = None
                print(f"→ TAKE PROFIT SHORT @ {price:.2f} (+{change_pct:.2f}%)")
                continue
            elif change_pct <= -7:
                portfolio.buy(symbol_upper, price, qty=abs(current_qty))
                portfolio.current_position[symbol_upper] = None
                entry_price_map[symbol_upper] = None
                print(f"→ STOP LOSS SHORT @ {price:.2f} ({change_pct:.2f}%)")
                continue

        # === Generate Strategy Signal ===
        if strategy_name == 'bollinger':
            signal = strategy_bollinger(data_for_indicators[-50:], current_position=portfolio.current_position)
        else:
            signal = strategy_mean_reversion(data_for_indicators[-50:], current_position=portfolio.current_position)
            print(f"[{symbol_upper}] Signal: {signal}, Current Pos: {current_pos}")

        # === Execute Signal ===
        if signal == "buy":
            if current_pos == "short":
                short_qty = abs(current_qty)
                if short_qty > 0:
                    portfolio.buy(symbol_upper, price, qty=short_qty)
                    print(f"→ BUY to cover short @ {price}")
                portfolio.current_position[symbol_upper] = None
                entry_price_map[symbol_upper] = None

            if portfolio.current_position[symbol_upper] is None:
                qty = int((portfolio.cash * 0.10) // price)
                if qty > 0:
                    portfolio.buy(symbol_upper, price, qty=qty)
                    portfolio.current_position[symbol_upper] = "long"
                    entry_price_map[symbol_upper] = price
                    print(f"→ NEW LONG @ {price:.2f}, Qty: {qty}")

        elif signal == "sell":
            if current_pos == "long":
                if current_qty > 0:
                    portfolio.sell(symbol_upper, price, qty=current_qty)
                    print(f"→ SELL to exit long @ {price}")
                portfolio.current_position[symbol_upper] = None
                entry_price_map[symbol_upper] = None

            if portfolio.current_position[symbol_upper] is None:
                qty = int((portfolio.cash * 0.10) // price)
                if qty > 0:
                    portfolio.sell(symbol_upper, price, qty=qty)
                    portfolio.current_position[symbol_upper] = "short"
                    entry_price_map[symbol_upper] = price
                    print(f"→ NEW SHORT @ {price:.2f}, Qty: {qty}")

        portfolio.update_net_worth({symbol_upper: price})

    print_summary(portfolio)
    buy_count = sum(1 for trade in portfolio.trade_log if trade["action"].lower() == "buy")
    sell_count = sum(1 for trade in portfolio.trade_log if trade["action"].lower() == "sell")

    return {
        "final_net_worth": portfolio.get_final_net_worth(),
        "buy_count": buy_count,
        "sell_count": sell_count,
        "total_trades": len(portfolio.trade_log),
    }


def main():
    args = parse_arguments()

    if args.config:
        config = load_config(args.config)
        symbols = config['symbols']
        allocations = config['allocations']
        start = config['start']
        end = config['end']
        asset_type = config.get('asset_type', 'crypto')
        strategy = config.get('strategy', 'bollinger')
    else:
        symbols = args.symbols.split(',')
        allocations = list(map(float, args.allocations.split(',')))
        start = args.start
        end = args.end
        asset_type = args.asset_type
        strategy = args.strategy

    if len(symbols) != len(allocations):
        raise ValueError("Number of symbols and allocations must match.")
    
    if round(sum(allocations), 2) != 100.0:
        raise ValueError("Allocations must sum to 100%")

    print(f"\nRunning Multi-Stock Backtest on: {symbols}")
    initial_capital = 1000000
    combined_portfolio = Portfolio(initial_capital=initial_capital)

    # Totals for summary
    total_net_worth = 0
    total_trades = 0
    total_buys = 0
    total_sells = 0

    for symbol, allocation in zip(symbols, allocations):
        print(f"\n=== Running backtest for {symbol.upper()} | Allocation: {allocation}% ===")

        capital = initial_capital * (allocation / 100)
        sub_portfolio = Portfolio(initial_capital=capital)

        result = run_backtest(symbol, start, end, asset_type, strategy, sub_portfolio)

        # ✅ Print per-symbol result
        print(f"{symbol.upper()} Final Net Worth: ${result['final_net_worth']:.2f}")

        # Aggregate portfolio
        combined_portfolio.cash += sub_portfolio.cash
        for sym, qty in sub_portfolio.positions.items():
            combined_portfolio.positions[sym] = combined_portfolio.positions.get(sym, 0) + qty
        combined_portfolio.trade_log.extend(sub_portfolio.trade_log)

        # Aggregate stats
        total_net_worth += result["final_net_worth"]
        total_trades += result["total_trades"]
        total_buys += result["buy_count"]
        total_sells += result["sell_count"]

    # Print final combined portfolio summary
    print("\n========== Combined Portfolio Summary ==========")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Net Worth: ${total_net_worth:,.2f}")
    print(f"Total Trades Executed: {total_trades}")
    print(f"  - Buys: {total_buys}")
    print(f"  - Sells: {total_sells}")
    total_return = ((total_net_worth - initial_capital) / initial_capital) * 100
    print(f"Total Return: {total_return:.2f}%")
    print("===============================================\n")

if __name__ == "__main__":
    main()
