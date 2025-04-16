# src/price_engine/data_sources/coinbase_api.py
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class CoinbaseAPI:
    def __init__(self):
        self.base_url = "https://api.coinbase.com/v2/prices"
        self.session = self._create_session()

    def _create_session(self):
        """Create a requests session with retry logic."""
        session = requests.Session()
        retries = Retry(
            total=3,  # Number of retries
            backoff_factor=1,  # Delay between retries (1s, 2s, 4s, etc.)
            status_forcelist=[500, 502, 503, 504],  # Retry on these status codes
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def get_price(self, symbol: str) -> float:
        # Map symbol to Coinbase's format (e.g., BTC-USD)
        coinbase_symbol = self._map_symbol(symbol)
        url = f"{self.base_url}/{coinbase_symbol}/spot"
        try:
            response = self.session.get(url, timeout=5)  # Add a timeout
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()
            if "data" in data and "amount" in data["data"]:
                return float(data["data"]["amount"])
            else:
                raise Exception(f"Invalid response from Coinbase: {data}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch price from Coinbase: {e}")
            return None  # Return None if the API fails

    def _map_symbol(self, symbol: str) -> str:
        # Map common symbols to Coinbase's format
        symbol_map = {
            "BTCUSDT": "BTC-USD",
            "ETHUSDT": "ETH-USD",
        }
        return symbol_map.get(symbol, symbol)