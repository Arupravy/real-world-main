# sentiment/improved_sentiment_module.py

import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Optional transformer-based sentiment (requires `pip install transformers`)
try:
    from transformers import pipeline
    HAS_TRANSFORMER = True
except ImportError:
    HAS_TRANSFORMER = False

nltk.download('vader_lexicon', quiet=True)

class EnhancedSentimentAnalyzer:
    """
    Combines VADER, optional Transformer-based sentiment, tag-based weights, and keyword adjustments.
    """
    def __init__(self, use_transformer = False):
        self.vader = SentimentIntensityAnalyzer()
        self.tag_weights = {
            'negative': -0.6,
            'bearish': -0.4,
            'positive': 0.5,
            'bullish': 0.7,
            'important': 0.2
        }
        self.use_transformer = use_transformer and HAS_TRANSFORMER
        if self.use_transformer:
            self.transformer = pipeline("sentiment-analysis")

    def score_from_tags(self, tags):
        score = 0.0
        for tag in tags:
            tag_l = tag.lower()
            if tag_l in self.tag_weights:
                score += self.tag_weights[tag_l]
        return score

    def transformer_score(self, text: str) -> float:
        """
        Returns a score in [-1,1] based on Transformer model output.
        """
        result = self.transformer(text[:512])[0]  # truncate to model max length
        label = result['label'].upper()
        score = result.get('score', 0.0)
        if label == 'POSITIVE':
            return score
        elif label == 'NEGATIVE':
            return -score
        return 0.0

    def keyword_adjustment(self, text: str) -> float:
        text_l = text.lower()
        adjust = 0.0
        if 'crash' in text_l or 'plunge' in text_l or 'dump' in text_l:
            adjust -= 0.3
        if 'soar' in text_l or 'rally' in text_l or 'surge' in text_l or 'spike' in text_l:
            adjust += 0.3
        return adjust

    def compute_sentiment(self, text: str, tags: list[str] = None) -> float:
        """
        Compute a combined sentiment score.
        """
        vader_score = self.vader.polarity_scores(text)['compound']
        tag_score = self.score_from_tags(tags) if tags else 0.0
        transformer_sc = self.transformer_score(text) if self.use_transformer else 0.0
        keyword_adj = self.keyword_adjustment(text)
        # Weights for combining scores
        alpha, beta, gamma, delta = 0.6, 0.2, 0.1, 0.1
        combined = (
            alpha * vader_score +
            beta * tag_score +
            gamma * transformer_sc +
            delta * keyword_adj
        )
        # clamp to [-1,1]
        return max(min(combined, 1.0), -1.0)


# Updated tracker with exponential smoothing
from collections import deque
from datetime import datetime

class SentimentTracker:
    def __init__(self, max_len=100, ema_alpha: float = 0.3):
        self.history = deque(maxlen=max_len)
        self.ema_alpha = ema_alpha
        self.ema = None

    def update(self, sentiment_score: float, headline: str):
        timestamp = datetime.utcnow()
        self.history.append({
            "timestamp": timestamp,
            "headline": headline,
            "score": sentiment_score
        })
        if self.ema is None:
            self.ema = sentiment_score
        else:
            self.ema = self.ema_alpha * sentiment_score + (1 - self.ema_alpha) * self.ema

    def get_average_sentiment(self) -> float:
        if not self.history:
            return 0.0
        return sum(x["score"] for x in self.history) / len(self.history)

    def get_ema_sentiment(self) -> float:
        return self.ema if self.ema is not None else self.get_average_sentiment()

    def get_history(self):
        return list(self.history)
