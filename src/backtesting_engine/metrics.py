# C:\real-world-main\src\backtesting_engine\metrics.py

def print_summary(portfolio):
    print("\n========== Backtest Summary ==========")
    print(f"Initial Capital: ${portfolio.initial_capital:,.2f}")
    print(f"Final Net Worth: ${portfolio.get_final_net_worth():,.2f}")
    print(f"Total Trades Executed: {len(portfolio.trade_log)}")
    
    buy_count = sum(1 for t in portfolio.trade_log if t["action"] == "BUY")
    sell_count = sum(1 for t in portfolio.trade_log if t["action"] == "SELL")
    
    print(f"  - Buys: {buy_count}")
    print(f"  - Sells: {sell_count}")
    
    total_return = ((portfolio.get_final_net_worth() - portfolio.initial_capital) / portfolio.initial_capital) * 100
    print(f"Total Return: {total_return:.2f}%")
    print("======================================\n")
