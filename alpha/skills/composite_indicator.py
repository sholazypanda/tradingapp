"""Skill 5 — Composite Confirmation Indicator, ported from pinescript2.

Bollinger Bands + RSI + ATR + RVOL, read as confluence rather than any one
signal in isolation. See tradingskills.md §5.
"""

from __future__ import annotations

import pandas as pd


def analyze(df: pd.DataFrame, bb_length: int = 20, bb_mult: float = 2.0,
            rsi_overbought: int = 70, rsi_oversold: int = 30, rvol_threshold: float = 1.5) -> dict:
    """Returns the §5 output contract:
    {rsi, rsi_state, atr, rvol, rvol_spike, bb_pct, bb_signal}
    """
    close = df["close"]
    basis = close.rolling(bb_length).mean()
    dev = bb_mult * close.rolling(bb_length).std()
    upper, lower = basis + dev, basis - dev

    last_close = float(close.iloc[-1])
    last_upper, last_lower = float(upper.iloc[-1]), float(lower.iloc[-1])
    bb_pct = (last_close - last_lower) / (last_upper - last_lower) * 100 if last_upper != last_lower else 50.0

    prev_close = float(close.iloc[-2])
    bb_signal = "neutral"
    if prev_close <= last_lower <= last_close:
        bb_signal = "buy"  # crossed back above the lower band
    elif prev_close >= last_upper >= last_close:
        bb_signal = "sell"  # crossed back below the upper band

    rsi_val = float(df["rsi_14"].iloc[-1])
    rsi_state = "overbought" if rsi_val >= rsi_overbought else ("oversold" if rsi_val <= rsi_oversold else "neutral")

    rvol_val = float(df["rvol_30"].iloc[-1])

    return {
        "rsi": round(rsi_val, 2),
        "rsi_state": rsi_state,
        "atr": round(float(df["atr_14"].iloc[-1]), 4),
        "rvol": round(rvol_val, 2),
        "rvol_spike": bool(rvol_val >= rvol_threshold),
        "bb_pct": round(float(bb_pct), 1),
        "bb_signal": bb_signal,
    }
