import time
from src.price_engine.aggregator import PriceAggregator

if __name__ == "__main__":
    aggregator = PriceAggregator()
    while True:
        aggregator.fetch_and_store_prices("BTCUSDT")
        time.sleep(60)  # Fetch data every 60 seconds