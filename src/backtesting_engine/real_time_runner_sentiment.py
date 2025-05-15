# src/backtesting_engine/real_time_runner_sentiment.py
"""
A sentiment-gated version of RealTimeTrader: trades only when EMA sentiment crosses thresholds.
"""
import time
import threading
from datetime import datetime
from collections import defaultdict

# Base trader logic
from .real_time_runner import RealTimeTrader as BaseTrader
# Sentiment tools
from sentiment.improved_sentiment_module import EnhancedSentimentAnalyzer, SentimentTracker
from sentiment.news_fetcher import fetch_cryptopanic_news


class RealTimeSentimentTrader(BaseTrader):
    def __init__(
        self,
        capital: float,
        runtime: float,
        long_thresh: float = 0.20,
        short_thresh: float = -0.10,
        news_interval: float = 30.0,
        **kwargs
    ):
        super().__init__(capital=capital, runtime=runtime, **kwargs)
        # sentiment thresholds
        self.long_thresh = long_thresh
        self.short_thresh = short_thresh
        # sentiment engine + tracker
        self.analyzer = EnhancedSentimentAnalyzer(use_transformer=False)
        self.tracker = SentimentTracker(max_len=100, ema_alpha=0.3)
        # news-polling control (seconds)
        self.news_interval = news_interval
        self._last_news_time = 0.0

    def on_price_update(self, symbol: str, price: float):
        # always update candles/data and PnL timeline via base class
        super().on_price_update(symbol, price)

        now = time.time()
        # Poll news & update sentiment every self.news_interval seconds
        if now - self._last_news_time >= self.news_interval:
            self._last_news_time = now
            try:
                # fetch latest headlines for this symbol
                currency = symbol.replace("USDT", "")
                posts = fetch_cryptopanic_news(
                    filter="trending",
                    limit=3,
                    currencies=currency
                )
                for post in posts:
                    title = post.get("title", "")
                    tags  = [t.get("slug", "") for t in post.get("tags", [])]
                    score = self.analyzer.compute_sentiment(title, tags)
                    self.tracker.update(score, title)
            except Exception:
                pass

        # Only allow trading logic if EMA sentiment outside neutral zone
        ema = self.tracker.get_ema_sentiment() or 0.0
        if ema > self.long_thresh or ema < self.short_thresh:
            # call trading logic (entry/exit) via base class
            super().on_price_update(symbol, price)
        # else: skip new trades while neutral

    def get_sentiment_ema(self) -> float:
        """
        Expose current EMA sentiment for external use/logging.
        """
        return self.tracker.get_ema_sentiment() or 0.0

