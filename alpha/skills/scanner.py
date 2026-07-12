"""Skill 9 — Real-Time Scanner (Wick), tradingskills.md §9.

Screens a ticker list for early spike setups: volume surge, price
acceleration vs 50/200-day MA, RSI breakout from oversold, MACD crossover,
plus a "coiling" (volatility contraction) flag for pre-breakout candidates.
Uses Skills 5/7/8 for the catalyst-flagging step per §9's cross-reference.
"""

from __future__ import annotations

import pandas as pd

from alpha.config import ToolBudget
from alpha.data_sources.market_data import fetch_ohlcv


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series]:
    macd_line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def scan_one(ticker: str) -> dict:
    df = fetch_ohlcv(ticker, bars=220)  # needs the 200-bar MA, wider than the §12 default on purpose

    close = df["close"]
    ma_50, ma_200 = close.rolling(50).mean(), close.rolling(200).mean()
    macd_line, macd_signal = _macd(close)

    volume_surge = float(df["rvol_30"].iloc[-1]) >= 1.5
    price_accel_vs_50ma = bool(close.iloc[-1] > ma_50.iloc[-1] > ma_200.iloc[-1]) if pd.notna(ma_200.iloc[-1]) else None
    rsi_breakout = bool(df["rsi_14"].iloc[-2] <= 30 < df["rsi_14"].iloc[-1])
    macd_crossover = bool(macd_line.iloc[-2] <= macd_signal.iloc[-2] and macd_line.iloc[-1] > macd_signal.iloc[-1])

    # "Coiling": recent realized vol well below its own 60-bar average = contraction
    vol_now, vol_avg = float(df["vol_20"].iloc[-1]), float(df["vol_20"].tail(60).mean())
    coiling = bool(vol_avg and vol_now < 0.6 * vol_avg)

    score = sum([volume_surge, bool(price_accel_vs_50ma), rsi_breakout, macd_crossover]) * 2.5
    if coiling:
        score += 2
    score = min(round(score, 1), 10.0)

    return {
        "ticker": ticker,
        "volume_surge": volume_surge,
        "price_accel_vs_50ma": price_accel_vs_50ma,
        "rsi_breakout_from_oversold": rsi_breakout,
        "macd_crossover": macd_crossover,
        "coiling": coiling,
        "continuation_probability_1_to_10": score,
        "last_close": float(close.iloc[-1]),
    }


def scan(tickers: list[str], budget: ToolBudget | None = None) -> list[dict]:
    """Screens each ticker, respecting the §12 'scan' task-type call budget
    (each ticker spends one budget unit — a full-market scan needs a bigger
    cap than a single report card, which TOOL_CALL_BUDGETS already reflects).
    """
    budget = budget or ToolBudget("scan")
    results = []
    for ticker in tickers:
        budget.spend("Wick", f"scan_one({ticker})", "fetch_ohlcv(bars=220) + MA/RSI/MACD/RVOL screen")
        try:
            results.append(scan_one(ticker))
        except ValueError as exc:
            results.append({"ticker": ticker, "error": str(exc)})

    results.sort(key=lambda r: r.get("continuation_probability_1_to_10", 0), reverse=True)
    return results
