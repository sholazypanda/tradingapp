"""Skill 4 — Liquidity Swings, ported from pinescript1 (LuxAlgo).

Pivot highs/lows with a touch-count/volume tally of how much subsequent
price action interacts with each level before it's eventually swept.
See tradingskills.md §4.
"""

from __future__ import annotations

import pandas as pd


def _pivot_zone(df: pd.DataFrame, i: int, length: int, side: str) -> tuple[float, float]:
    """Wick-extremity zone for the pivot bar at index i (matches the Pine
    default `area = "Wick Extremity"`)."""
    bar = df.iloc[i]
    if side == "high":
        return float(bar["high"]), float(max(bar["close"], bar["open"]))
    return float(min(bar["close"], bar["open"])), float(bar["low"])


def analyze(df: pd.DataFrame, length: int = 14) -> list[dict]:
    """Returns the §4 output contract:
    [{price, side: "high"|"low", touch_count, volume, crossed}]
    """
    levels: list[dict] = []
    n = len(df)

    for i in range(length, n - length):
        window_h = df["high"].iloc[i - length: i + length + 1]
        window_l = df["low"].iloc[i - length: i + length + 1]

        for side, is_pivot in [("high", df["high"].iloc[i] == window_h.max()),
                                ("low", df["low"].iloc[i] == window_l.min())]:
            if not is_pivot:
                continue

            top, bottom = _pivot_zone(df, i, length, side)
            touch_count, volume = 0, 0.0
            crossed = False

            for j in range(i + 1, n):
                bar = df.iloc[j]
                if bar["low"] < top and bar["high"] > bottom:
                    touch_count += 1
                    volume += float(bar["volume"])
                if side == "high" and bar["close"] > top:
                    crossed = True
                elif side == "low" and bar["close"] < bottom:
                    crossed = True

            levels.append({
                "price": float(bar_price(df, i, side)),
                "side": side,
                "touch_count": touch_count,
                "volume": volume,
                "crossed": crossed,
                "time": str(df.index[i].date()),
            })

    levels.sort(key=lambda lvl: lvl["price"], reverse=True)
    return levels


def bar_price(df: pd.DataFrame, i: int, side: str) -> float:
    return df["high"].iloc[i] if side == "high" else df["low"].iloc[i]
