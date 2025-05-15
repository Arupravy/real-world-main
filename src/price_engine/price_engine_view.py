# src/price_engine/price_engine_view.py

import streamlit as st
import time
from .aggregator import PriceAggregator
from .price_calculator import PriceCalculator
from .indicators.bollinger_bands import BollingerBands

def live_mode_view():
    st.subheader("📡 Live Price Engine")

    asset_type = st.selectbox("Select Asset Type", ["crypto", "stock"])
    symbol = st.selectbox("Select Symbol", ["BTCUSDT", "ETHUSDT", "AAPL", "GOOGL", "TSLA"])
    refresh_rate = st.slider("Refresh Interval (seconds)", 2, 30, 5)

        # Initialize only once
    if 'live_feed' not in st.session_state:
        st.session_state['live_feed'] = False
    if 'price_history' not in st.session_state:
        st.session_state['price_history'] = []

    if st.button("Start Live Feed"):
        st.session_state['live_feed'] = True

    if st.button("Stop Live Feed"):
        st.session_state['live_feed'] = False

    if st.session_state['live_feed']:
        aggregator = PriceAggregator(asset_type=asset_type, symbols=[symbol])
        calculator = PriceCalculator()
        indicator = BollingerBands()

        price_chart = st.empty()
        price_table = st.empty()
        indicator_box = st.empty()

        # ✅ Step 1: Fetch historical price bootstrapping
        if len(st.session_state['price_history']) < 20:
            from src.backtesting_engine.historical_data_loader import load_historical_data


            # Load historical prices (ensure your loader returns a list of floats or dicts with 'price')
            historical_data = load_historical_data(symbol, asset_type, limit=19)
            # Format to match history style
            st.session_state['price_history'] = [{"price": p} for p in historical_data]

        while st.session_state['live_feed']:
            prices = aggregator.get_all_prices_async(symbol)
            weights = {k: v["weight"] for k, v in aggregator.sources.items() if k in prices}
            cleaned_prices = calculator.handle_outliers(prices)

            try:
                weighted_avg = calculator.calculate_weighted_average(cleaned_prices, weights)
            except:
                weighted_avg = None

            if weighted_avg:
                st.session_state['price_history'].append({"price": weighted_avg})
                if len(st.session_state['price_history']) > 100:
                    st.session_state['price_history'].pop(0)

            price_table.markdown("### Price Table")
            price_table.dataframe({
                "Source": list(prices.keys()),
                "Raw Price": list(prices.values()),
                "Cleaned": [cleaned_prices.get(k) for k in prices],
            })

            if weighted_avg:
                try:
                    indicator_output = indicator.calculate(st.session_state['price_history'])
                except Exception as e:
                    indicator_output = {"Error": str(e)}
            else:
                indicator_output = {}

            indicator_box.markdown(f"""
                ### Summary
                - Weighted Average: **{weighted_avg:.2f}**
                - Bollinger Bands: `{indicator_output}`
            """)

            price_chart.line_chart([entry["price"] for entry in st.session_state['price_history']])
            time.sleep(refresh_rate)
