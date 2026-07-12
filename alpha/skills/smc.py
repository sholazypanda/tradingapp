"""Skill 3 — Smart Money Concepts, ported from pinescript3 (LuxAlgo SMC).

Structure breaks (BOS/CHoCH), order blocks, equal highs/lows, fair value
gaps, and premium/discount/equilibrium zones. See tradingskills.md §3.

Note on lookback: the Pine original defaults to a 50-bar swing lookback
(needing ~101 bars to confirm one pivot), which doesn't fit inside the
§12 60-bar default decision window. `swing_lookback` here defaults to 10
instead so structure is still readable within budget — pass a larger
value when analyzing `fetch_ohlcv_backtest`'s longer window if you want
closer parity with the paper's default.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BULLISH = 1
BEARISH = -1


@dataclass
class OrderBlock:
    bar_time: pd.Timestamp
    high: float
    low: float
    bias: int
    mitigated: bool = False

    def to_dict(self) -> dict:
        return {
            "time": str(self.bar_time.date()), "high": self.high, "low": self.low,
            "bias": "bullish" if self.bias == BULLISH else "bearish", "mitigated": self.mitigated,
        }


@dataclass
class StructureBreak:
    bar_time: pd.Timestamp
    level: float
    bias: int
    kind: str  # "BOS" or "CHoCH"

    def to_dict(self) -> dict:
        return {
            "time": str(self.bar_time.date()), "level": self.level,
            "bias": "bullish" if self.bias == BULLISH else "bearish", "kind": self.kind,
        }


def _find_pivots(series_high: pd.Series, series_low: pd.Series, lookback: int) -> tuple[list[int], list[int]]:
    """Returns (pivot_high_indices, pivot_low_indices) confirmed with a
    `lookback`-bar lag on each side, matching ta.pivothigh/pivotlow.
    """
    n = len(series_high)
    pivot_highs, pivot_lows = [], []
    for i in range(lookback, n - lookback):
        window_h = series_high.iloc[i - lookback: i + lookback + 1]
        window_l = series_low.iloc[i - lookback: i + lookback + 1]
        if series_high.iloc[i] == window_h.max():
            pivot_highs.append(i)
        if series_low.iloc[i] == window_l.min():
            pivot_lows.append(i)
    return pivot_highs, pivot_lows


def _detect_structure(df: pd.DataFrame, lookback: int) -> tuple[list[StructureBreak], list[OrderBlock], int]:
    pivot_highs, pivot_lows = _find_pivots(df["high"], df["low"], lookback)

    breaks: list[StructureBreak] = []
    order_blocks: list[OrderBlock] = []
    trend_bias = 0

    active_high_idx: int | None = None  # most recent unbroken swing-high pivot
    active_low_idx: int | None = None
    high_broken = True
    low_broken = True

    pivot_high_set = set(pivot_highs)
    pivot_low_set = set(pivot_lows)

    for i in range(len(df)):
        if i in pivot_high_set:
            active_high_idx, high_broken = i, False
        if i in pivot_low_set:
            active_low_idx, low_broken = i, False

        close = df["close"].iloc[i]

        if active_high_idx is not None and not high_broken and close > df["high"].iloc[active_high_idx]:
            kind = "CHoCH" if trend_bias == BEARISH else "BOS"
            breaks.append(StructureBreak(df.index[i], df["high"].iloc[active_high_idx], BULLISH, kind))
            trend_bias = BULLISH
            high_broken = True
            segment = df.iloc[active_high_idx:i + 1]
            ob_idx = segment["low"].idxmin()
            order_blocks.append(OrderBlock(ob_idx, df.loc[ob_idx, "high"], df.loc[ob_idx, "low"], BULLISH))

        if active_low_idx is not None and not low_broken and close < df["low"].iloc[active_low_idx]:
            kind = "CHoCH" if trend_bias == BULLISH else "BOS"
            breaks.append(StructureBreak(df.index[i], df["low"].iloc[active_low_idx], BEARISH, kind))
            trend_bias = BEARISH
            low_broken = True
            segment = df.iloc[active_low_idx:i + 1]
            ob_idx = segment["high"].idxmax()
            order_blocks.append(OrderBlock(ob_idx, df.loc[ob_idx, "high"], df.loc[ob_idx, "low"], BEARISH))

    # Mitigation: check every bar after an order block formed
    for ob in order_blocks:
        after = df[df.index > ob.bar_time]
        if ob.bias == BULLISH and (after["low"] < ob.low).any():
            ob.mitigated = True
        elif ob.bias == BEARISH and (after["high"] > ob.high).any():
            ob.mitigated = True

    return breaks, order_blocks, trend_bias


def _equal_highs_lows(df: pd.DataFrame, lookback: int, threshold_atr_mult: float = 0.1) -> list[dict]:
    pivot_highs, pivot_lows = _find_pivots(df["high"], df["low"], lookback)
    atr_200 = df["atr_200"] if "atr_200" in df else pd.Series(np.nan, index=df.index)
    results = []

    for idx_list, col, label in [(pivot_highs, "high", "EQH"), (pivot_lows, "low", "EQL")]:
        for a, b in zip(idx_list, idx_list[1:]):
            level_a, level_b = df[col].iloc[a], df[col].iloc[b]
            atr_ref = atr_200.iloc[b]
            if pd.notna(atr_ref) and abs(level_a - level_b) < threshold_atr_mult * atr_ref:
                results.append({
                    "type": label, "time_a": str(df.index[a].date()), "time_b": str(df.index[b].date()),
                    "level": float((level_a + level_b) / 2),
                })
    return results


def _fair_value_gaps(df: pd.DataFrame) -> list[dict]:
    gaps = []
    for i in range(2, len(df)):
        atr_ref = df["atr_14"].iloc[i] if "atr_14" in df else None
        min_gap = 0.1 * atr_ref if pd.notna(atr_ref) else 0.0

        if df["low"].iloc[i] > df["high"].iloc[i - 2] and (df["low"].iloc[i] - df["high"].iloc[i - 2]) > min_gap:
            top, bottom = df["low"].iloc[i], df["high"].iloc[i - 2]
            filled = bool((df["low"].iloc[i + 1:] <= bottom).any())
            gaps.append({"bias": "bullish", "top": float(top), "bottom": float(bottom),
                         "time": str(df.index[i].date()), "filled": filled})

        if df["high"].iloc[i] < df["low"].iloc[i - 2] and (df["low"].iloc[i - 2] - df["high"].iloc[i]) > min_gap:
            top, bottom = df["low"].iloc[i - 2], df["high"].iloc[i]
            filled = bool((df["high"].iloc[i + 1:] >= top).any())
            gaps.append({"bias": "bearish", "top": float(top), "bottom": float(bottom),
                         "time": str(df.index[i].date()), "filled": filled})
    return gaps


def _range_zone(df: pd.DataFrame, lookback: int) -> str:
    pivot_highs, pivot_lows = _find_pivots(df["high"], df["low"], lookback)
    recent_top = df["high"].iloc[pivot_highs[-1]] if pivot_highs else df["high"].max()
    recent_bottom = df["low"].iloc[pivot_lows[-1]] if pivot_lows else df["low"].min()
    span = recent_top - recent_bottom
    if span <= 0:
        return "equilibrium"

    price = df["close"].iloc[-1]
    if price >= recent_top - 0.05 * span:
        return "premium"
    if price <= recent_bottom + 0.05 * span:
        return "discount"
    return "equilibrium"


def analyze(df: pd.DataFrame, swing_lookback: int = 10, internal_lookback: int = 5) -> dict:
    """Returns the §3 output contract:
    {trend_bias, structure_breaks[], order_blocks[], eqh_eql[], fvgs[], range_zone}
    """
    swing_breaks, swing_obs, swing_bias = _detect_structure(df, swing_lookback)
    internal_breaks, _internal_obs, _internal_bias = _detect_structure(df, internal_lookback)

    return {
        "trend_bias": "bullish" if swing_bias == BULLISH else ("bearish" if swing_bias == BEARISH else "neutral"),
        "structure_breaks": [b.to_dict() for b in swing_breaks] + [
            {**b.to_dict(), "scope": "internal"} for b in internal_breaks
        ],
        "order_blocks": [ob.to_dict() for ob in swing_obs],
        "eqh_eql": _equal_highs_lows(df, swing_lookback),
        "fvgs": _fair_value_gaps(df),
        "range_zone": _range_zone(df, swing_lookback),
    }
