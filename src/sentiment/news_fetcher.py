# sentiment/news_fetcher.py
import requests
import os
from dotenv import load_dotenv

load_dotenv()

MARKETAUX_API_KEY = os.getenv("MARKETAUX_API_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY")

def fetch_marketaux_news(symbols="BTC,ETH", limit=10):
    url = "https://api.marketaux.com/v1/news/all"
    params = {
        "api_token": MARKETAUX_API_KEY,
        "symbols": symbols,
        "filter_entities": "true",
        "language": "en",
        "limit": limit
    }
    response = requests.get(url, params=params)
    return response.json().get("data", [])

def fetch_polygon_news(ticker="X:BTCUSD"):
    url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&apiKey={POLYGON_API_KEY}"
    response = requests.get(url)
    return response.json().get("results", [])

def fetch_cryptopanic_news(filter="trending", public=True, limit=10, currencies="BTC,ETH"):
    url = "https://cryptopanic.com/api/v1/posts/"
    params = {
        "auth_token": CRYPTOPANIC_API_KEY,
        "filter":     filter,
        "public":     str(public).lower(),
        "currencies": currencies or ""
    }
    resp = requests.get(url, params=params, timeout=5)
    # if they send back a non-200 or an empty body, just return []
    if resp.status_code != 200 or not resp.text:
        return []
    try:
        data = resp.json()
    except ValueError:
        return []
    results = data.get("results")
    if not isinstance(results, list):
        return []
    return results[:limit]