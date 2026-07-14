"""Alpha — chief strategist / orchestrator, tradingskills.md §1.

Consults all six personas, then reconciles their signals into one report
card (§10). Disagreement is surfaced explicitly rather than averaged away.
"""

from __future__ import annotations

from alpha.agents import personas
from alpha.config import ToolBudget
from alpha.data_sources.market_data import fetch_ohlcv

BULLISH_SIGNALS = {"bullish"}
BEARISH_SIGNALS = {"bearish"}


def _run_persona(persona_name: str, fn, *args) -> dict:
    """Isolates one persona's failure from the rest of the report card.

    Rook and Ledger reuse the already-fetched, already-validated shared `df`
    (see analyze_ticker below), so they aren't wrapped here — a failure
    there would be a real bug, not a data-availability gap, and the rest of
    the report card can't be built without their output anyway. Cortex,
    Vance, Sable, and Wick each do their own separate fetch (e.g. Wick's
    scanner needs 220 bars vs. the shared df's 60), so one of them lacking
    enough history for a given ticker shouldn't take down the whole card.
    """
    try:
        return fn(*args)
    except Exception as exc:
        return {
            "persona": persona_name, "signal": "unavailable",
            "rationale": f"{persona_name} failed: {exc}",
            "detail": {"available": False, "reason": str(exc)},
        }


def _reconcile(opinions: list[dict]) -> dict:
    bullish = [o for o in opinions if o["signal"] in BULLISH_SIGNALS]
    bearish = [o for o in opinions if o["signal"] in BEARISH_SIGNALS]
    caution = [o for o in opinions if o["signal"] == "caution"]

    net = len(bullish) - len(bearish)
    if net >= 4:
        rating = "Strong Buy"
    elif net >= 1:
        rating = "Buy"
    elif net <= -4:
        rating = "Strong Sell"
    elif net <= -1:
        rating = "Sell"
    else:
        rating = "Neutral"

    disagreement = bool(bullish and bearish)
    why_parts = []
    if disagreement:
        why_parts.append(
            f"{len(bullish)} bullish ({', '.join(o['persona'] for o in bullish)}) vs. "
            f"{len(bearish)} bearish ({', '.join(o['persona'] for o in bearish)}) — "
            "signals conflict, treat sizing cautiously."
        )
    if caution:
        why_parts.append(f"{', '.join(o['persona'] for o in caution)} flagged contrarian caution.")
    if not why_parts:
        why_parts.append(f"{len(bullish)} bullish, {len(bearish)} bearish, rest neutral/unavailable.")

    return {"signal": rating, "confidence_note": " ".join(why_parts), "disagreement": disagreement}


def analyze_ticker(ticker: str) -> dict:
    """Runs the full six-persona pipeline for one ticker and returns the
    §10 report-card data (rendering/formatting happens in alpha.reportcard).
    """
    budget = ToolBudget("report_card")
    budget.spend("Alpha", f"fetch_ohlcv({ticker}, bars=60)", "shared OHLCV panel for Rook/Ledger")
    df = fetch_ohlcv(ticker)

    opinions = [
        personas.rook_structure(ticker, df, budget),
        _run_persona("Cortex", personas.cortex_forecast, ticker, budget),
        _run_persona("Vance", personas.vance_fundamentals, ticker, budget),
        _run_persona("Sable", personas.sable_sentiment, ticker, budget),
        personas.ledger_risk(ticker, df, budget),
        _run_persona("Wick", personas.wick_scanner, ticker, budget),
    ]
    reconciled = _reconcile(opinions)
    budget.note("Alpha", "reconcile(opinions)", f"{reconciled['signal']} — {reconciled['confidence_note']}")

    ledger_detail = next(o for o in opinions if o["persona"] == "Ledger")["detail"]
    rook_detail = next(o for o in opinions if o["persona"] == "Rook")["detail"]
    cortex = next(o for o in opinions if o["persona"] == "Cortex")

    last_close = float(df["close"].iloc[-1])
    stop = ledger_detail["suggested_stop_loss"]
    stop_distance = last_close - stop
    targets = [round(last_close + 2 * stop_distance, 2), round(last_close + 3 * stop_distance, 2)]
    risk_reward = round((targets[0] - last_close) / stop_distance, 2) if stop_distance else None

    day_change_pct = round((df["close"].iloc[-1] / df["close"].iloc[-2] - 1) * 100, 2)

    return {
        "ticker": ticker,
        "signal": reconciled["signal"],
        "entry": last_close,
        "day_change_pct": day_change_pct,
        "stop_loss": stop,
        "targets": targets,
        "risk_reward": risk_reward,
        "confidence": reconciled["confidence_note"],
        "disagreement": reconciled["disagreement"],
        "key_levels": {
            "range_zone": rook_detail["smc"]["range_zone"],
            "order_blocks": rook_detail["smc"]["order_blocks"],
            "liquidity_levels": rook_detail["liquidity_levels"],
        },
        "model_forecast": cortex["detail"] if cortex["detail"].get("available") else {"available": False, "reason": cortex["rationale"]},
        "confluence_notes": [o["rationale"] for o in opinions],
        "opinions": opinions,
        "budget_used": f"{budget.calls_made}/{budget.cap}",
        "execution_log": budget.log,
        "price_series": [round(float(c), 2) for c in df["close"].tolist()],
    }
