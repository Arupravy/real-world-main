# src/backtesting_engine/backtest_runner.py
import argparse, json
import pandas as pd
from backtesting_engine.portfolio import Portfolio
from backtesting_engine.historical_data_loader import load_historical_data
from backtesting_engine.metrics import print_summary
from ta.volatility import AverageTrueRange
# strategies
from backtesting_engine.strategies.strategy_bollinger import strategy_bollinger
from backtesting_engine.strategies.strategy_mean_reversion import strategy_mean_reversion
from backtesting_engine.strategies.strategy_trendpullback import strategy_trendpullback

# ------------------------------------------------------------------------- #
# helper: convert DataFrame row to the dict format every strategy expects
def convert_to_indicator_format(row):
    return {"price": row["close"], "high": row["high"], "low": row["low"]}
# ------------------------------------------------------------------------- #

# minimum look-back bars each strategy needs
MIN_BARS = {
    "bollinger": 50,          # BB-20 + safety
    "mean_reversion": 50,     # MR-20 + safety
    "trendpullback": 200      # EMA-200
}

# ----------------------------- CLI --------------------------------------- #
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run backtest using historical data.")
    parser.add_argument("--symbols",     type=str, help="Comma-separated symbols (e.g. TSLA,AAPL)")
    parser.add_argument("--allocations", type=str, help="Comma-separated capital % (e.g. 60,40)")
    parser.add_argument("--start",       type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",         type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--asset_type",  type=str, default="crypto", help="Asset type (crypto, stock …)")
    parser.add_argument("--config",      type=str, help="Optional JSON config file")
    parser.add_argument("--strategy",
                        type=str,
                        choices=["bollinger", "mean_reversion", "trendpullback"],
                        default="bollinger")
    return parser.parse_args()

# ------------------------------------------------------------------
# helper — returns last ATR(20) in dollars, or None if not available
def latest_atr(indicator_window, atr_period=20):
    if len(indicator_window) < atr_period + 1:
        return None
    df_tmp = pd.DataFrame(indicator_window[-(atr_period + 1):])
    return AverageTrueRange(
        high=df_tmp["high"],
        low=df_tmp["low"],
        close=df_tmp["price"],
        window=atr_period
    ).average_true_range().iloc[-1]
# ------------------------------------------------------------------

def load_config(path):
    with open(path, "r") as f:
        return json.load(f)

# ------------------------------------------------------------------------- #
# core runner for ONE symbol
def run_backtest(symbol, start, end, asset_type, strategy_name, portfolio):
    df = load_historical_data(symbol, start, end, asset_type)
    data_for_indicators = []

    sym = symbol.upper()
    entry_price_map = {}

    min_bars = MIN_BARS[strategy_name]

    for _, row in df.iterrows():
        data_for_indicators.append(convert_to_indicator_format(row))
        if len(data_for_indicators) < min_bars:
            continue

        # shortcut vars
        price      = row["close"]
        current_pos = portfolio.current_position.get(sym)
        current_qty = portfolio.positions.get(sym, 0)
        entry_price = entry_price_map.get(sym)

        '''# -------------------- manual TP / SL --------------------------------
        if current_pos == "long" and entry_price:
            change = (price - entry_price) / entry_price * 100
            if change >= 20 or change <= -7:
                portfolio.sell(sym, price, qty=current_qty)
                portfolio.current_position[sym] = None
                entry_price_map[sym] = None
                tag = "TAKE PROFIT" if change >= 20 else "STOP LOSS"
                print(f"→ {tag} LONG @ {price:.2f} ({change:+.2f}%)")
                continue

        if current_pos == "short" and entry_price:
            change = (entry_price - price) / entry_price * 100
            if change >= 20 or change <= -7:
                portfolio.buy(sym, price, qty=abs(current_qty))
                portfolio.current_position[sym] = None
                entry_price_map[sym] = None
                tag = "TAKE PROFIT" if change >= 20 else "STOP LOSS"
                print(f"→ {tag} SHORT @ {price:.2f} ({change:+.2f}%)")
                continue
        # -------------------------------------------------------------------- '''

        # -------------------- strategy call ---------------------------------
        window = data_for_indicators[-min_bars:]

        if strategy_name == "bollinger":
            signal = strategy_bollinger(window, current_position=current_pos)
        elif strategy_name == "mean_reversion":
            signal = strategy_mean_reversion(window, current_position=current_pos)
        else:  # trendpullback
            signal = strategy_trendpullback(window, current_position=current_pos)
        # --------------------------------------------------------------------

        # -------------------- execute signal --------------------------------
        if signal == "buy":
            if current_pos == "short":                   # cover
                if abs(current_qty) > 0:
                    portfolio.buy(sym, price, qty=abs(current_qty))
                    print(f"→ BUY to cover short @ {price:.2f}")
                portfolio.current_position[sym] = None
                entry_price_map[sym] = None

            if portfolio.current_position.get(sym) is None:
                # --- risk-based sizing -------------------------------------
                atr = latest_atr(data_for_indicators)     # <-- get fresh ATR
                if atr:
                    risk_dollars = portfolio.cash * 0.02  # risk 2 % of sleeve
                    stop_dist    = atr * 2                # uses same 2×ATR stop assumption
                    qty = int(risk_dollars // stop_dist)
                else:
                    qty = int(portfolio.cash // price)    # fallback: all-in
                # -----------------------------------------------------------
                qty = int((portfolio.cash * 1) // price)  #all-in
                if qty > 0:
                    portfolio.buy(sym, price, qty=qty)
                    portfolio.current_position[sym] = "long"
                    entry_price_map[sym] = price
                    print(f"→ NEW LONG @ {price:.2f}, Qty: {qty}")

        elif signal == "sell":
            if current_pos == "long":                    # exit long
                if current_qty > 0:
                    portfolio.sell(sym, price, qty=current_qty)
                    print(f"→ SELL to exit long @ {price:.2f}")
                portfolio.current_position[sym] = None
                entry_price_map[sym] = None
            '''
            if portfolio.current_position.get(sym) is None:
                qty = int((portfolio.cash * 1) // price)
                if qty > 0:
                    portfolio.sell(sym, price, qty=qty)
                    portfolio.current_position[sym] = "short"
                    entry_price_map[sym] = price
                    print(f"→ NEW SHORT @ {price:.2f}, Qty: {qty}")
            '''
        # update P&L
        portfolio.update_net_worth({sym: price})

    # -------- per-symbol summary ------------------------------------------
    print_summary(portfolio)
    buys  = sum(1 for t in portfolio.trade_log if t["action"].lower() == "buy")
    sells = sum(1 for t in portfolio.trade_log if t["action"].lower() == "sell")
    return {
        "final_net_worth": portfolio.get_final_net_worth(),
        "buy_count": buys,
        "sell_count": sells,
        "total_trades": len(portfolio.trade_log)
    }

# ============================== main ===================================== #
def main():
    args = parse_arguments()
    if args.config:
        cfg       = load_config(args.config)
        symbols   = cfg["symbols"]
        allocs    = cfg["allocations"]
        start     = cfg["start"]
        end       = cfg["end"]
        a_type    = cfg.get("asset_type", "crypto")
        strategy  = cfg.get("strategy", "bollinger")
    else:
        symbols  = args.symbols.split(",")
        allocs   = list(map(float, args.allocations.split(",")))
        start    = args.start
        end      = args.end
        a_type   = args.asset_type
        strategy = args.strategy

    if len(symbols) != len(allocs):
        raise ValueError("Number of symbols and allocations must match.")
    if round(sum(allocs), 2) != 100.0:
        raise ValueError("Allocations must sum to 100%.")

    print(f"\nRunning Multi-Stock Backtest on: {symbols}")
    initial_capital = 1_000_000
    combined = Portfolio(initial_capital=initial_capital)

    total_net, total_trades, total_buys, total_sells = 0, 0, 0, 0

    for sym, alloc in zip(symbols, allocs):
        print(f"\n=== Running backtest for {sym.upper()} | Allocation: {alloc}% ===")
        sleeve_cap = initial_capital * (alloc / 100)
        sub = Portfolio(initial_capital=sleeve_cap)

        stats = run_backtest(sym, start, end, a_type, strategy, sub)
        print(f"{sym.upper()} Final Net Worth: ${stats['final_net_worth']:.2f}")

        # merge into combined portfolio (simplistic)
        combined.cash += sub.cash
        for s, q in sub.positions.items():
            combined.positions[s] = combined.positions.get(s, 0) + q
        combined.trade_log.extend(sub.trade_log)

        total_net   += stats["final_net_worth"]
        total_trades += stats["total_trades"]
        total_buys  += stats["buy_count"]
        total_sells += stats["sell_count"]

    print("\n========== Combined Portfolio Summary ==========")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Net Worth: ${total_net:,.2f}")
    print(f"Total Trades Executed: {total_trades}")
    print(f"  - Buys:  {total_buys}")
    print(f"  - Sells: {total_sells}")
    total_ret = (total_net - initial_capital) / initial_capital * 100
    print(f"Total Return: {total_ret:.2f}%")
    print("===============================================\n")

def run_backtest_for_ui(symbols, allocations, start, end, asset_type='crypto', strategy='bollinger', initial_capital=1000000):

    combined_portfolio = Portfolio(initial_capital=initial_capital)
    total_net_worth = 0
    total_trades = 0
    total_buys = 0
    total_sells = 0

    per_symbol_logs = []
    
    for symbol, allocation in zip(symbols, allocations):
        capital = initial_capital * (allocation / 100)
        sub_portfolio = Portfolio(initial_capital=capital)

        result = run_backtest(symbol, start, end, asset_type, strategy, sub_portfolio)

        combined_portfolio.cash += sub_portfolio.cash
        for sym, qty in sub_portfolio.positions.items():
            combined_portfolio.positions[sym] = combined_portfolio.positions.get(sym, 0) + qty
        combined_portfolio.trade_log.extend(sub_portfolio.trade_log)

        total_net_worth += result["final_net_worth"]
        total_trades += result["total_trades"]
        total_buys += result["buy_count"]
        total_sells += result["sell_count"]

        per_symbol_logs.append({
            "symbol": symbol.upper(),
            "final_net_worth": result["final_net_worth"],
            "trades": result["total_trades"],
            "buys": result["buy_count"],
            "sells": result["sell_count"]
        })

    total_return = ((total_net_worth - initial_capital) / initial_capital) * 100

    trade_log_df = pd.DataFrame(combined_portfolio.trade_log)
    pnl_chart_data = combined_portfolio.equity_curve if hasattr(combined_portfolio, 'equity_curve') else None

    return {
        "initial_capital": initial_capital,
        "final_net_worth": total_net_worth,
        "total_return": total_return,
        "total_trades": total_trades,
        "total_buys": total_buys,
        "total_sells": total_sells,
        "per_symbol_logs": per_symbol_logs,
        "trade_log_df": trade_log_df,
        "pnl_chart_data": pnl_chart_data
    }


if __name__ == "__main__":
    main()
