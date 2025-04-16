# C:\real-world-main\src\backtesting_engine\historical_data_loader.py

from price_engine.aggregator import PriceAggregator
import pandas as pd

def load_historical_data(symbol: str, start_date: str, end_date: str, asset_type="crypto") -> pd.DataFrame:
    """
    Fetch historical data for a symbol using the PriceAggregator.
    """
    try:
        aggregator = PriceAggregator(asset_type=asset_type, symbols=[symbol])
        df = aggregator.fetch_historical_data(symbol, start_date, end_date)
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        raise

