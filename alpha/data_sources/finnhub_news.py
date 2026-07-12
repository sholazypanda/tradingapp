"""Finnhub headline ingestion, mirrors notebooks/node_transformer_sentiment_forecast.ipynb §4.1.

Kept here as the shared implementation both the notebook's training corpus
and this live service can use — see that notebook for the rationale on why
Finnhub over Alpha Vantage/Stocktwits for bulk headline pulls.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

from alpha.config import FINNHUB_API_KEY

FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/company-news"
SAFE_REQUESTS_PER_MIN = 55
DEFAULT_CACHE_DIR = Path("data/finnhub_news_cache")


def _load_cache(cache_file: Path) -> dict:
    return json.loads(cache_file.read_text()) if cache_file.exists() else {"_fetched_chunks": []}


def fetch_recent_headlines(ticker: str, days: int = 7, api_key: str | None = None) -> list[str]:
    """Lightweight live pull (no month-chunking/cache needed for a short window) —
    used for the day-of sentiment score, not for building the training corpus.
    """
    api_key = api_key or FINNHUB_API_KEY
    if not api_key:
        return []

    end = datetime.utcnow()
    start = end - timedelta(days=days)
    resp = requests.get(
        FINNHUB_NEWS_URL,
        params={"symbol": ticker, "from": start.strftime("%Y-%m-%d"), "to": end.strftime("%Y-%m-%d"), "token": api_key},
        timeout=15,
    )
    resp.raise_for_status()
    return [a.get("headline", "").strip() for a in resp.json() if a.get("headline")]
