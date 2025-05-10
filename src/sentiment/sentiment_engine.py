# sentiment/sentiment_engine.py
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon', quiet=True)
sid = SentimentIntensityAnalyzer()

def get_sentiment(text: str) -> float:
    if not text:
        return 0.0
    score = sid.polarity_scores(text)
    return score['compound']  # value between -1 (neg) to +1 (pos)
