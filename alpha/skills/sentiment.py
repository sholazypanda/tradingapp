"""Skill 8 — Sentiment, News & Backtesting (Sable).

Stocktwits crowd tally + recent headlines (Finnhub first, Yahoo Finance as
fallback) + a 6-month RSI backtest, per tradingskills.md §8. FinBERT scoring
of the headlines is optional (only runs if `transformers`/`torch` are
installed) — falls back to leaving headlines unscored rather than failing
the whole skill.
"""

from __future__ import annotations

from alpha.config import FINNHUB_API_KEY
from alpha.data_sources import finnhub_news, stocktwits
from alpha.data_sources.market_data import fetch_ohlcv_backtest


def _recent_yahoo_headlines(ticker: str, limit: int = 10) -> list[str]:
    import yfinance as yf

    try:
        news = yf.Ticker(ticker).news or []
    except Exception:
        return []
    return [item.get("title", "") for item in news[:limit] if item.get("title")]


def _recent_headlines(ticker: str, limit: int = 10) -> tuple[list[str], str]:
    """Prefers Finnhub (more reliable free tier — see the §6.1 notebook's
    comparison of Finnhub/Alpha Vantage/Stocktwits) when an API key is
    configured; falls back to Yahoo Finance's news endpoint otherwise or if
    Finnhub returns nothing. Returns (headlines, source) so the trace/report
    card can say where they actually came from, not just that they exist.
    """
    if FINNHUB_API_KEY:
        try:
            headlines = finnhub_news.fetch_recent_headlines(ticker, days=7)[:limit]
            if headlines:
                return headlines, "finnhub"
        except Exception:
            pass  # fall through to Yahoo rather than failing the whole skill

    return _recent_yahoo_headlines(ticker, limit), "yahoo_finance"


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


WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def weekday_seasonality(ticker: str, min_samples: int = 3) -> dict:
    """Day-of-week seasonality over the §12 6-month backtest window: for each
    weekday, the average same-day return and win rate. Classic "day-of-week
    effect" methodology (French, 1980) — not RSI-based, and not about
    specific calendar dates, just which weekday (Mon-Fri) this ticker has
    tended to be strong or weak on recently.

    "Best day to buy" = the weekday with the lowest average return (the
    ticker's typical relative dip during the week — cheapest average entry).
    "Best day to sell" = the weekday with the highest average return (the
    ticker's typical relative strength — best average day to realize gains).
    """
    df = fetch_ohlcv_backtest(ticker)
    weekday_idx = df.index.weekday  # 0=Monday ... 4=Friday (weekends aren't trading days anyway)

    by_day = {}
    for i, name in enumerate(WEEKDAY_NAMES):
        day_returns = df["return"][weekday_idx == i]
        if len(day_returns) < min_samples:
            by_day[name] = {"available": False, "n_days": int(len(day_returns))}
            continue
        by_day[name] = {
            "available": True,
            "avg_return_pct": round(float(day_returns.mean()) * 100, 3),
            "win_rate_pct": round(float((day_returns > 0).mean()) * 100, 1),
            "n_days": int(len(day_returns)),
        }

    available_days = {name: d for name, d in by_day.items() if d["available"]}
    best_buy = min(available_days, key=lambda n: available_days[n]["avg_return_pct"]) if available_days else None
    best_sell = max(available_days, key=lambda n: available_days[n]["avg_return_pct"]) if available_days else None

    return {
        "by_weekday": by_day,
        "best_day_to_buy": best_buy,
        "best_day_to_sell": best_sell,
        "bars_analyzed": len(df),
    }


def analyze(ticker: str) -> dict:
    """Returns crowd sentiment, recent news (+ optional FinBERT score), and
    the 6-month RSI backtest — the three pieces of §8.
    """
    crowd = stocktwits.crowd_sentiment(ticker)
    headlines, headline_source = _recent_headlines(ticker)
    news_sentiment_score = _score_headlines_if_possible(headlines)

    contrarian_flag = (
        crowd.get("available") and crowd["bullish_count"] > 3 * max(crowd["bearish_count"], 1)
    )

    return {
        "crowd_sentiment": crowd,
        "contrarian_caution_flag": bool(contrarian_flag),
        "recent_headlines": headlines,
        "headline_source": headline_source,
        "news_sentiment_score": news_sentiment_score,
        "rsi_backtest_6mo": backtest_rsi_thresholds(ticker),
        "weekday_seasonality": weekday_seasonality(ticker),
    }
