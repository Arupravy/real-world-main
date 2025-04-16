import sqlite3  # or any other DB (e.g., PostgreSQL, MySQL)
from typing import List, Dict

class PriceDatabase:
    def __init__(self, db_path: str = "prices.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prices (
                id INTEGER PRIMARY KEY,
                source TEXT,
                symbol TEXT,
                price REAL,
                timestamp DATETIME
            )
        ''')
        self.conn.commit()

    def insert_price(self, source: str, symbol: str, price: float):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO prices (source, symbol, price, timestamp)
            VALUES (?, ?, ?, datetime('now'))
        ''', (source, symbol, price))
        self.conn.commit()

    def get_prices(self, symbol: str) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT source, price, timestamp FROM prices
            WHERE symbol = ?
            ORDER BY timestamp DESC
        ''', (symbol,))
        return cursor.fetchall()