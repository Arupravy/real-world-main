# src/price_engine/price_history.py
import json
from datetime import datetime
import os

class PriceHistory:
    def __init__(self, file_path: str = "prices.json"):
        self.file_path = file_path
        self.history = self._load_history()

    def _load_history(self) -> list:
        """Load historical prices from the file."""
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as file:
                return json.load(file)
        return []

    def _save_history(self):
        """Save historical prices to the file."""
        with open(self.file_path, "w") as file:
            json.dump(self.history, file, indent=4)

    def add_price(self, symbol: str, source: str, price: float):
        """Add a new price entry to the history."""
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "source": source,
            "price": price,
        }
        self.history.append(entry)
        self._save_history()

    def get_history(self) -> list:
        """Return the entire price history."""
        return self.history

    def clear_history(self):
        """Clear the price history."""
        self.history = []
        self._save_history()