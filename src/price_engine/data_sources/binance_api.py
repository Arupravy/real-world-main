import requests

class BinanceAPI:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"

    def get_price(self, symbol: str) -> float:
        url = f"{self.base_url}/ticker/price"
        params = {"symbol": symbol}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return float(response.json()["price"])
        else:
            raise Exception(f"Failed to fetch price from Binance: {response.text}")