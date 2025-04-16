# src/price_engine/api_handler.py 

from price_engine.data_sources.binance_api import BinanceAPI # adjust import path to match your structure

def fetch_price_from_api(symbol: str, asset_type: str) -> float:
    if asset_type.lower() == "crypto":
        api = BinanceAPI()
        return api.get_price(symbol)
    else:
        raise ValueError(f"Unsupported asset type for API mode: {asset_type}")

