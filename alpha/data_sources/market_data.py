"""OHLCV ingestion — yfinance, no API key. Enforces the §12 60-bar default.

Indicators like ATR(200) need real historical warmup to compute correctly,
so under the hood we pull extra history, compute everything on the full
series, then trim to the requested decision window. The caller only ever
sees `bars` rows — the warmup history never leaks into "how much history
did this decision use."
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha.config import DEFAULT_OHLCV_BARS, BACKTEST_MONTHS

_WARMUP_BARS = 220  # enough for ATR(200) to be fully warmed up


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = (-delta.clip(upper=0)).rolling(length).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    return _true_range(df).ewm(alpha=1 / length, min_periods=length, adjust=False).mean()


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=str.lower)
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"]).diff()
    df["vol_20"] = df["return"].rolling(20).std()
    df["rsi_14"] = _rsi(df["close"], 14)
    df["atr_14"] = _atr(df, 14)
    df["atr_200"] = _atr(df, 200)
    df["rvol_30"] = df["volume"] / df["volume"].rolling(30).mean()
    return df


def fetch_ohlcv(ticker: str, bars: int = DEFAULT_OHLCV_BARS, warmup: int = _WARMUP_BARS) -> pd.DataFrame:
    """Returns the last `bars` rows with technical columns computed using
    `warmup` extra bars of hidden history for correct indicator values.

    This is the default per-decision fetch (§12) — do not raise `bars` for
    routine report cards; use `fetch_ohlcv_backtest` for the explicit
    longer-history exception instead.
    """
    import yfinance as yf

    total_calendar_days = int((bars + warmup) * 1.6) + 10  # trading days -> calendar days, generously
    raw = yf.download(ticker, period=f"{total_calendar_days}d", progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker} — check the ticker symbol")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    enriched = _add_indicators(raw).dropna()
    if len(enriched) < bars:
        raise ValueError(
            f"Only {len(enriched)} bars available for {ticker} after indicator warmup, "
            f"need {bars} — try a more liquid/older ticker."
        )
    return enriched.tail(bars)


def fetch_ohlcv_backtest(ticker: str, months: int = BACKTEST_MONTHS) -> pd.DataFrame:
    """Skill 8's explicit, separate long-history budget — not the §12 default.

    Only requires the columns this backtest actually uses (return, rsi_14)
    to be non-NaN — `atr_200` needs 200 bars of warmup that a 6-month
    window won't have, but nothing here reads atr_200, so it's left as NaN
    rather than dropping ~90% of the window to satisfy an unused column.
    """
    import yfinance as yf

    raw = yf.download(ticker, period=f"{months * 21 + 20}d", progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker} — check the ticker symbol")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    enriched = _add_indicators(raw)
    return enriched.dropna(subset=["return", "log_return", "vol_20", "rsi_14"])
