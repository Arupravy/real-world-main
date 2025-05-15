# sentiment/sentiment_state.py
from collections import deque
from datetime import datetime

class SentimentTracker:
    def __init__(self, max_len=100):
        self.history = deque(maxlen=max_len)

    def update(self, sentiment_score: float, headline: str):
        self.history.append({
            "timestamp": datetime.now(),
            "headline": headline,
            "score": sentiment_score
        })

    def get_average_sentiment(self):
        if not self.history:
            return 0.0
        return sum(x["score"] for x in self.history) / len(self.history)

    def get_history(self):
        return list(self.history)
