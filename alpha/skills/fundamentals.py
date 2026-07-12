"""Skill 7 — Fundamental & Pattern Skills (Vance), Finviz-style.

Finviz has no official API (the popular wrappers scrape the site, a ToS
gray area) so this uses yfinance's `.info`/`.financials` for what it
actually has. Chart-pattern detection (wedges, triangles, H&S, double
top/bottom) and insider-trade filings have no equivalent free, structured
source here — see tradingskills.md §7 for the two paths worth pursuing
(SEC EDGAR Form 4 for insiders; a paid API like Financial Modeling Prep for
both) — so those fields report `available: False` rather than a fake value.
"""

from __future__ import annotations


def analyze(ticker: str) -> dict:
    import yfinance as yf

    info = yf.Ticker(ticker).info

    valuation = {
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "short_ratio": info.get("shortRatio"),
        "short_percent_of_float": info.get("shortPercentOfFloat"),
    }

    fundamentals_trend = _revenue_margin_trend(ticker)

    return {
        "valuation": valuation,
        "fundamentals_trend": fundamentals_trend,
        "chart_patterns": {
            "available": False,
            "reason": "No free structured pattern-detection source wired up — see tradingskills.md §7.",
        },
        "insider_activity": {
            "available": False,
            "reason": "No free structured insider-filing source wired up yet — SEC EDGAR Form 4 is the "
                       "free official option but needs its own parser; see tradingskills.md §7.",
        },
    }


def _revenue_margin_trend(ticker: str) -> dict:
    import yfinance as yf

    financials = yf.Ticker(ticker).financials  # annual, most-recent-first columns
    if financials is None or financials.empty or "Total Revenue" not in financials.index:
        return {"available": False, "reason": "yfinance returned no annual financials for this ticker."}

    revenue = financials.loc["Total Revenue"].sort_index().dropna()
    years = [str(c.year) for c in revenue.index]
    revenue_values = [float(v) for v in revenue.values]

    growth_pct = None
    if len(revenue_values) >= 2 and revenue_values[0]:
        growth_pct = round((revenue_values[-1] / revenue_values[0] - 1) * 100, 1)

    margin_trend = None
    if "Gross Profit" in financials.index:
        gross = financials.loc["Gross Profit"].reindex(revenue.index).dropna()
        common = gross.index.intersection(revenue.index)
        margins = [float(gross[d]) / float(revenue[d]) * 100 for d in common if revenue[d]]
        if len(margins) >= 2:
            margin_trend = "improving" if margins[-1] > margins[0] else "deteriorating"

    return {
        "available": True, "years": years, "revenue": revenue_values,
        "revenue_growth_pct_over_period": growth_pct, "margin_trend": margin_trend,
    }
