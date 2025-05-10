import streamlit as st
import pandas as pd
from backtesting_engine.backtest_runner import run_backtest_for_ui

def render_backtesting_ui(session_state):
    st.subheader("🔁 Backtest Historical Strategy")

    # === User Inputs ===
    symbols_input = st.text_input("Enter symbols (comma-separated)", value="BTCUSDT,ETHUSDT")
    allocations_input = st.text_input("Enter allocations (%) for each symbol", value="60,40")
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")
    strategy = st.selectbox("Select Strategy", options=["trendpullback", "bollinger", "mean_reversion"])

    if st.button("🚀 Run Backtest"):
        symbols = [s.strip().upper() for s in symbols_input.split(",")]
        allocations = list(map(float, allocations_input.split(",")))

        if len(symbols) != len(allocations):
            st.error("❌ Number of symbols must match number of allocations.")
            return
        if round(sum(allocations), 2) != 100.0:
            st.error("❌ Allocations must sum to 100%.")
            return

        st.info("Running backtest...")

        try:
            result = run_backtest_for_ui(
                symbols=symbols,
                allocations=allocations,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                strategy=strategy,
            )

            st.success("✅ Backtest Complete!")
            st.metric("Initial Capital", f"${result['initial_capital']:,}")
            st.metric("Final Net Worth", f"${result['final_net_worth']:.2f}")
            st.metric("Total Return", f"{result['total_return']:.2f}%")
            st.metric("Total Trades", result["total_trades"])

            # Store in session state
            session_state["backtest_history"].append(result)

            # Summary
            st.subheader("📊 Per-Symbol Performance")
            st.dataframe(result["per_symbol_logs"])
            st.subheader("🧾 Trade Log")
            st.dataframe(result["trade_log_df"])
            if result["pnl_chart_data"] is not None:
                st.line_chart(result["pnl_chart_data"])

        except Exception as e:
            st.error(f"❌ Error: {e}")
