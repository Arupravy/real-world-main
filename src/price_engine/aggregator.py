# src/price_engine/aggregator.py
from .data_sources.binance_api import BinanceAPI
from .data_sources.coingecko_api import CoinGeckoAPI
from .data_sources.coinbase_api import CoinbaseAPI
from .data_sources.yahoo_finance import YahooFinanceAPI
from .price_calculator import PriceCalculator
from .price_history import PriceHistory
from .data_sources.websocket_handler import BinanceWebSocketClient
import asyncio
import inspect
import pandas as pd


def run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        return asyncio.get_event_loop().run_until_complete(coro)

def is_async_callable(method):
    return inspect.iscoroutinefunction(method)


class PriceAggregator:
    def __init__(self, asset_type: str = "crypto", symbols=None):
        """
        Initialize with asset type (crypto/stock).
        Defaults to crypto for backward compatibility.
        """
        self.asset_type = asset_type.lower()
        self.symbols = symbols or []
        self.sources = self._initialize_sources()
        self.price_history = PriceHistory()

    def _initialize_sources(self):
        """Initialize data sources based on asset type."""
        if self.asset_type == "stock":
            return {
                "yahoo": {"handler": YahooFinanceAPI(), "weight": 1.0}
            }
        else:  # crypto
            return {
                "binance_ws": {"handler": BinanceWebSocketClient(self.symbols), "weight": 0.4},
                "binance": {"handler": BinanceAPI(), "weight": 0.4},
                "coingecko": {"handler": CoinGeckoAPI(), "weight": 0.3},
                "coinbase": {"handler": CoinbaseAPI(), "weight": 0.3}
            }

    def get_all_prices_async(self, symbol: str) -> dict:
        """
        Fetch prices from all sources asynchronously where supported.
        Falls back to synchronous calls if async is not available.
        """
        prices = {}
        for source_name, source_info in self.sources.items():
            handler = source_info["handler"]
            try:
                get_price = getattr(handler, "get_price", None)
                if not get_price:
                    continue

                if is_async_callable(get_price):
                    price = run_async(get_price(symbol))
                else:
                    price = get_price(symbol)

                if price is not None:
                    prices[source_name] = price
                    self.price_history.add_price(symbol, source_name, price)

            except Exception as e:
                print(f"Error fetching from {source_name}: {e}")
                prices[source_name] = None
        return prices

    def get_all_prices(self, symbol: str) -> dict:
        """Fetch prices from all available sources for the given symbol."""
        prices = {}
        for source_name, source_info in self.sources.items():
            try:
                if source_name == "coingecko" and self.asset_type == "crypto":
                    coin_id = self._get_coin_id(symbol)
                    price = source_info["handler"].get_price(coin_id)
                else:
                    price = source_info["handler"].get_price(symbol)

                if price is not None:
                    prices[source_name] = price
                    self.price_history.add_price(symbol, source_name, price)
            except Exception as e:
                print(f"Error fetching data from {source_name}: {e}")
                prices[source_name] = None
        return prices

    def get_historical_prices(self, symbol: str, from_date: str, to_date: str) -> list:
        """Get historical prices for the symbol."""
        if self.asset_type == "stock":
            yahoo = self.sources["yahoo"]["handler"]
            return yahoo.get_historical_prices(symbol, from_date, to_date)
        else:
            import requests
            from datetime import datetime

            try:
                from_ts = int(datetime.strptime(from_date, "%Y-%m-%d").timestamp() * 1000)
                to_ts = int(datetime.strptime(to_date, "%Y-%m-%d").timestamp() * 1000)

                url = "https://api.binance.com/api/v3/klines"
                params = {
                    "symbol": symbol.upper(),
                    "interval": "1d",
                    "startTime": from_ts,
                    "endTime": to_ts,
                    "limit": 1000
                }
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                prices = [
                    {
                        "date": datetime.fromtimestamp(candle[0] / 1000).strftime("%Y-%m-%d"),
                        "price": float(candle[4])  # Closing price
                    }
                    for candle in data
                ]
                return prices

            except Exception as e:
                print(f"Error fetching crypto historical prices from Binance: {e}")
                return []

    def fetch_historical_data(self, symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
        """
        Fetch historical price data in DataFrame format using internal logic.
        This wraps get_historical_prices into a DataFrame output.
        """
        try:
            raw_data = self.get_historical_prices(symbol, from_date, to_date)

            # Convert list of dicts to DataFrame
            df = pd.DataFrame(raw_data)

            # If it's crypto, we may only have 'date' and 'price'
            if "price" in df.columns:
                df.rename(columns={"price": "close", "date": "timestamp"}, inplace=True)
                df["open"] = df["close"]
                df["high"] = df["close"]
                df["low"] = df["close"]
                df["volume"] = 100  # Placeholder if you don’t have volume
            elif "close" not in df.columns:
                raise ValueError("Expected 'close' price in data.")

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)

            return df
        except Exception as e:
            print(f"Error in fetch_historical_data: {e}")
            return pd.DataFrame()




    def _get_coin_id(self, symbol: str) -> str:
        """Map trading symbols to CoinGecko coin IDs (for crypto only)."""
        coin_id_map = {
            "BTCUSDT": "bitcoin",
            "ETHUSDT": "ethereum",
            "BNBUSDT": "binancecoin",
            "XRPUSDT": "ripple",
            "SOLUSDT": "solana",
            "ADAUSDT": "cardano",
            "DOGEUSDT": "dogecoin",
            "DOTUSDT": "polkadot",
            "SHIBUSDT": "shiba-inu",
            "MATICUSDT": "matic-network",
        }
        return coin_id_map.get(symbol, symbol.lower())

    def get_price_history(self) -> list:
        """Return the price history."""
        return self.price_history.get_history()
