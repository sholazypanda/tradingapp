"""§10 delivery: a local, read-only Flask dashboard.

Lists watchlist tickers with their signal, and a per-ticker report-card
page. No write/execute routes exist — per §13, this app doesn't do that.
"""

from __future__ import annotations

import time

from flask import Flask, render_template, request

from alpha.agents.alpha import analyze_ticker
from alpha.config import DEFAULT_WATCHLIST
from alpha.reportcard import build_sparkline_svg, format_text
from alpha.skills import scanner

app = Flask(__name__)

_CACHE_TTL_SECONDS = 60
_cache: dict[str, tuple[float, dict]] = {}


def _cached_analyze(ticker: str) -> dict:
    now = time.time()
    cached = _cache.get(ticker)
    if cached and now - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]

    report = analyze_ticker(ticker)
    _cache[ticker] = (now, report)
    return report


@app.route("/")
def dashboard():
    watchlist = request.args.get("watchlist")
    tickers = [t.strip().upper() for t in watchlist.split(",")] if watchlist else DEFAULT_WATCHLIST

    reports, errors = [], []
    for ticker in tickers:
        try:
            reports.append(_cached_analyze(ticker))
        except Exception as exc:  # a bad ticker shouldn't take down the whole dashboard
            errors.append({"ticker": ticker, "error": str(exc)})

    return render_template("index.html", reports=reports, errors=errors, watchlist=",".join(tickers))


@app.route("/ticker/<ticker>")
def ticker_report(ticker: str):
    ticker = ticker.upper()
    report = _cached_analyze(ticker)
    sparkline_svg = build_sparkline_svg(report["price_series"])
    return render_template("report_card.html", report=report, plain_text=format_text(report), sparkline_svg=sparkline_svg)


@app.route("/scan")
def scan():
    watchlist = request.args.get("watchlist")
    tickers = [t.strip().upper() for t in watchlist.split(",")] if watchlist else DEFAULT_WATCHLIST
    results = scanner.scan(tickers)
    return render_template("scan.html", results=results, watchlist=",".join(tickers))


if __name__ == "__main__":
    app.run(debug=True, port=5050)
