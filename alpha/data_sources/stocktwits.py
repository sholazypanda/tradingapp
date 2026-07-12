"""Stocktwits crowd sentiment — public, unauthenticated symbol stream.

Recent/live messages only (not a historical archive — see tradingskills.md
§6.1's caveat on why this isn't used for training data). Users self-tag
messages Bullish/Bearish; we just tally those tags rather than running any
NLP over the message text.
"""

from __future__ import annotations

import requests

STREAM_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"


def crowd_sentiment(ticker: str) -> dict:
    """Returns {bullish_count, bearish_count, untagged_count, sample_size} from
    the most recent public messages for `ticker`. Returns zeros (not an error)
    if the endpoint is unreachable or rate-limited — this is a soft signal.
    """
    try:
        resp = requests.get(STREAM_URL.format(symbol=ticker), timeout=10)
        resp.raise_for_status()
        messages = resp.json().get("messages", [])
    except requests.RequestException:
        return {"bullish_count": 0, "bearish_count": 0, "untagged_count": 0, "sample_size": 0, "available": False}

    bullish = bearish = untagged = 0
    for msg in messages:
        sentiment = (msg.get("entities") or {}).get("sentiment")
        label = (sentiment or {}).get("basic")
        if label == "Bullish":
            bullish += 1
        elif label == "Bearish":
            bearish += 1
        else:
            untagged += 1

    return {
        "bullish_count": bullish, "bearish_count": bearish, "untagged_count": untagged,
        "sample_size": len(messages), "available": True,
    }
