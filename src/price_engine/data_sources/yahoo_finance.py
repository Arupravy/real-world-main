# src/price_engine/data_sources/yahoo_finance.py
import yfinance as yf
from datetime import datetime

class YahooFinanceAPI:
    def __init__(self):
        self.source_name = "yahoo"

    def get_price(self, symbol: str) -> float:
        """Get current price from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d')
            if not data.empty:
                return float(data['Close'].iloc[-1])
        except Exception as e:
            print(f"Error fetching Yahoo Finance data: {e}")
        return None

    def get_historical_prices(self, symbol: str, from_date: str, to_date: str) -> list:
        """Get historical prices from Yahoo Finance."""
        try:
            data = yf.download(
                symbol,
                start=from_date,
                end=to_date,
                progress=False
            )
            return [
                {
                    "date": str(date),  # <-- renamed from "timestamp" to "date"
                    "symbol": symbol,
                    "source": self.source_name,
                    "price": float(row['Close'])
                }
                for date, row in data.iterrows()
            ]
        except Exception as e:
            print(f"Error fetching Yahoo Finance history: {e}")
            return []