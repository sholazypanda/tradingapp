"""Skill 8 — Sentiment, News & Backtesting (Sable).

Stocktwits crowd tally + recent Yahoo Finance headlines + a 6-month RSI
backtest, per tradingskills.md §8. FinBERT scoring of the Yahoo headlines
is optional (only runs if `transformers`/`torch` are installed) — falls
back to leaving headlines unscored rather than failing the whole skill.
"""

from __future__ import annotations

from alpha.data_sources import stocktwits
from alpha.data_sources.market_data import fetch_ohlcv_backtest


def _recent_yahoo_headlines(ticker: str, limit: int = 10) -> list[str]:
    import yfinance as yf

    try:
        news = yf.Ticker(ticker).news or []
    except Exception:
        return []
    return [item.get("title", "") for item in news[:limit] if item.get("title")]


def _score_headlines_if_possible(headlines: list[str]) -> float | None:
    if not headlines:
        return None
    try:
        from transformers import pipeline
    except ImportError:
        return None  # optional dependency not installed — leave unscored, not an error

    clf = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    results = clf(headlines, truncation=True)
    sign = {"positive": 1, "negative": -1, "neutral": 0}
    return sum(sign[r["label"].lower()] * r["score"] for r in results) / len(results)


def backtest_rsi_thresholds(ticker: str, oversold_candidates=(20, 25, 30, 35), overbought_candidates=(65, 70, 75, 80)) -> dict:
    """Skill 8's 6-month backtest: which RSI thresholds would have worked best
    for *this* ticker recently, rather than assuming fixed 30/70 universally.
    """
    df = fetch_ohlcv_backtest(ticker)
    forward_return = df["close"].shift(-5) / df["close"] - 1  # 5-day forward return per bar

    best = None
    for oversold in oversold_candidates:
        entries = df["rsi_14"] <= oversold
        if entries.sum() < 3:
            continue
        avg_fwd_return = float(forward_return[entries].mean())
        if best is None or avg_fwd_return > best["avg_forward_return_5d"]:
            best = {"oversold_threshold": oversold, "avg_forward_return_5d": avg_fwd_return, "n_signals": int(entries.sum())}

    worst = None
    for overbought in overbought_candidates:
        entries = df["rsi_14"] >= overbought
        if entries.sum() < 3:
            continue
        avg_fwd_return = float(forward_return[entries].mean())
        if worst is None or avg_fwd_return < worst["avg_forward_return_5d"]:
            worst = {"overbought_threshold": overbought, "avg_forward_return_5d": avg_fwd_return, "n_signals": int(entries.sum())}

    return {"best_oversold_entry": best, "best_overbought_exit": worst, "bars_analyzed": len(df)}


def analyze(ticker: str) -> dict:
    """Returns crowd sentiment, recent news (+ optional FinBERT score), and
    the 6-month RSI backtest — the three pieces of §8.
    """
    crowd = stocktwits.crowd_sentiment(ticker)
    headlines = _recent_yahoo_headlines(ticker)
    news_sentiment_score = _score_headlines_if_possible(headlines)

    contrarian_flag = (
        crowd.get("available") and crowd["bullish_count"] > 3 * max(crowd["bearish_count"], 1)
    )

    return {
        "crowd_sentiment": crowd,
        "contrarian_caution_flag": bool(contrarian_flag),
        "recent_headlines": headlines,
        "news_sentiment_score": news_sentiment_score,
        "rsi_backtest_6mo": backtest_rsi_thresholds(ticker),
    }
