"""The six analyst personas from tradingskills.md §1 — thin wrappers around
the skill modules, each returning a structured opinion Alpha can reconcile.

Budget/trace accounting happens at this layer (spend for the external data
pull, note for the evaluation that follows), not inside the skill modules
themselves — skills stay pure "given data, analyze it" functions; personas
are the orchestration boundary §12's tool-call budget is meant to bound,
and the trace log they build is a real record of what happened, not a
scripted UI value.
"""

from __future__ import annotations

import pandas as pd

from alpha.config import ToolBudget
from alpha.skills import (
    composite_indicator, fundamentals, liquidity_swings, prediction_market,
    prediction_stock, scanner, sentiment, smc,
)


def rook_structure(ticker: str, df: pd.DataFrame, budget: ToolBudget) -> dict:
    """Structure Analyst — §3 (SMC), §4 (Liquidity Swings)."""
    budget.note("Rook", "smc.analyze(df)")
    smc_result = smc.analyze(df)
    budget.note("Rook", "→ trend_bias / range_zone",
                f"trend_bias={smc_result['trend_bias']}, range_zone={smc_result['range_zone']}, "
                f"{len(smc_result['order_blocks'])} order block(s), {len(smc_result['fvgs'])} FVG(s)")

    budget.note("Rook", "liquidity_swings.analyze(df)")
    levels = liquidity_swings.analyze(df)
    uncrossed_nearby = [
        lvl for lvl in levels if not lvl["crossed"] and abs(lvl["price"] - df["close"].iloc[-1]) / df["close"].iloc[-1] < 0.03
    ]
    budget.note("Rook", "→ liquidity levels", f"{len(levels)} level(s), {len(uncrossed_nearby)} uncrossed within 3%")

    signal = {"bullish": "bullish", "bearish": "bearish", "neutral": "neutral"}[smc_result["trend_bias"]]
    rationale = (
        f"Swing structure is {smc_result['trend_bias']}, price sits in the {smc_result['range_zone']} zone"
        + (f", {len(uncrossed_nearby)} untouched liquidity level(s) within 3%." if uncrossed_nearby else ".")
    )
    budget.note("Rook", "evaluate → signal", f"{signal} — {rationale}")

    return {"persona": "Rook", "signal": signal, "rationale": rationale,
            "detail": {"smc": smc_result, "liquidity_levels": levels}}


def cortex_forecast(ticker: str, budget: ToolBudget) -> dict:
    """Quant Forecaster — §6.1 node-transformer + sentiment fusion."""
    budget.spend("Cortex", "prediction_stock.predict(ticker)", "loading checkpoint + running inference")
    result = prediction_stock.predict(ticker)

    if not result.get("available"):
        budget.note("Cortex", "evaluate → signal", f"unavailable — {result['reason']}")
        return {"persona": "Cortex", "signal": "unavailable", "rationale": result["reason"], "detail": result}

    ret = result["predicted_next_day_return"]
    signal = "bullish" if ret > 0 else ("bearish" if ret < 0 else "neutral")
    rationale = f"Model forecasts a {ret:+.3%} next-day return (learned signal, not deterministic)."
    budget.note("Cortex", "evaluate → signal", f"{signal} — {rationale}")
    return {"persona": "Cortex", "signal": signal, "rationale": rationale, "detail": result}


def vance_fundamentals(ticker: str, budget: ToolBudget) -> dict:
    """Fundamentals Analyst — §7 Finviz-style valuation/fundamentals."""
    budget.spend("Vance", "fundamentals.analyze(ticker)", "yfinance .info + .financials")
    result = fundamentals.analyze(ticker)
    trend = result["fundamentals_trend"]
    budget.note("Vance", "→ valuation/trend", f"P/E={result['valuation'].get('pe_ratio')}, "
                f"revenue_trend_available={trend.get('available')}")

    signal = "neutral"
    if trend.get("available"):
        growth = trend.get("revenue_growth_pct_over_period")
        margin = trend.get("margin_trend")
        if growth is not None and growth > 0 and margin != "deteriorating":
            signal = "bullish"
        elif growth is not None and growth < 0 and margin == "deteriorating":
            signal = "bearish"

    rationale = "Fundamentals unavailable." if not trend.get("available") else (
        f"Revenue {trend['revenue_growth_pct_over_period']:+.1f}% over the period, margins {trend['margin_trend']}."
    )
    budget.note("Vance", "evaluate → signal", f"{signal} — {rationale}")
    return {"persona": "Vance", "signal": signal, "rationale": rationale, "detail": result}


def sable_sentiment(ticker: str, budget: ToolBudget) -> dict:
    """Sentiment Analyst — §8 crowd + news + insider positioning."""
    budget.spend("Sable", "sentiment.analyze(ticker)", "Stocktwits + Yahoo news + 6mo RSI backtest")
    result = sentiment.analyze(ticker)
    budget.note("Sable", "→ crowd/news", f"crowd_available={result['crowd_sentiment']['available']}, "
                f"headlines={len(result['recent_headlines'])}, news_score={result['news_sentiment_score']}")
    seasonality = result["weekday_seasonality"]
    budget.note("Sable", "→ weekday seasonality (6mo)",
                f"best buy day={seasonality['best_day_to_buy']}, best sell day={seasonality['best_day_to_sell']}")

    signal = "neutral"
    if result["contrarian_caution_flag"]:
        signal = "caution"
    elif result["news_sentiment_score"] is not None:
        signal = "bullish" if result["news_sentiment_score"] > 0.1 else (
            "bearish" if result["news_sentiment_score"] < -0.1 else "neutral"
        )

    rationale = "Crowd sentiment lopsidedly bullish — contrarian caution." if signal == "caution" else (
        "No scored news/crowd signal available." if result["news_sentiment_score"] is None
        else f"News sentiment score {result['news_sentiment_score']:+.2f}."
    )
    budget.note("Sable", "evaluate → signal", f"{signal} — {rationale}")
    return {"persona": "Sable", "signal": signal, "rationale": rationale, "detail": result}


def ledger_risk(ticker: str, df: pd.DataFrame, budget: ToolBudget) -> dict:
    """Risk/Portfolio Manager — §5 composite indicator + §6.2 market-timing exposure."""
    budget.note("Ledger", "composite_indicator.analyze(df)")
    composite = composite_indicator.analyze(df)
    budget.note("Ledger", "→ BB/RSI/ATR/RVOL", f"RSI={composite['rsi']} ({composite['rsi_state']}), "
                f"RVOL={composite['rvol']}x, BB%={composite['bb_pct']}")

    budget.spend("Ledger", "prediction_market.predict()", "market-timing model inference")
    market_timing = prediction_market.predict()

    last_close = float(df["close"].iloc[-1])
    atr = composite["atr"]
    suggested_stop = round(last_close - 1.5 * atr, 2)

    exposure_note = "market-timing model not trained yet" if not market_timing.get("available") else (
        f"market-timing model says '{market_timing['timing_signal']}'"
    )
    rationale = f"RSI {composite['rsi']} ({composite['rsi_state']}), RVOL {composite['rvol']}x; {exposure_note}."
    budget.note("Ledger", "evaluate → stop/exposure", f"stop={suggested_stop}, {exposure_note}")

    return {
        "persona": "Ledger", "signal": composite["bb_signal"], "rationale": rationale,
        "detail": {
            "composite_indicator": composite, "market_timing": market_timing,
            "suggested_stop_loss": suggested_stop, "atr_based_stop_distance": round(1.5 * atr, 2),
        },
    }


def wick_scanner(ticker: str, budget: ToolBudget) -> dict:
    """Setup Scanner — §9 real-time spike/breakout detection."""
    budget.spend("Wick", "scanner.scan_one(ticker)", "fetch_ohlcv(bars=220) + MA/RSI/MACD/RVOL screen")
    result = scanner.scan_one(ticker)
    score = result["continuation_probability_1_to_10"]
    signal = "bullish" if score >= 5 else "neutral"
    rationale = f"Continuation probability {score}/10" + (" — coiling for a move." if result["coiling"] else ".")
    budget.note("Wick", "evaluate → signal", f"{signal} — {rationale}")
    return {"persona": "Wick", "signal": signal, "rationale": rationale, "detail": result}
