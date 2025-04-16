# src/price_engine/data_sources/kraken_api.py
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class KrakenAPI:
    def __init__(self):
        self.base_url = "https://api.kraken.com/0/public"
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
        # Map symbol to Kraken's format
        kraken_symbol = self._map_symbol(symbol)
        url = f"{self.base_url}/Ticker"
        params = {"pair": kraken_symbol}
        try:
            response = self.session.get(url, params=params, timeout=5)  # Add a timeout
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()
            if "result" in data:
                result_key = next(iter(data["result"]))  # Get the first key in the result
                return float(data["result"][result_key]["c"][0])  # Last closed price
            else:
                raise Exception(f"Invalid response from Kraken: {data}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch price from Kraken: {e}")
            return None  # Return None if the API fails

    def _map_symbol(self, symbol: str) -> str:
        # Map common symbols to Kraken's format
        symbol_map = {
            "BTCUSDT": "XBTUSD",  # Kraken uses XBT for Bitcoin
            "ETHUSDT": "ETHUSD",  # Kraken uses ETH for Ethereum
        }
        return symbol_map.get(symbol, symbol)