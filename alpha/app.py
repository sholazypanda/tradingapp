"""§10 delivery: a local, read-only Flask dashboard.

Lists watchlist tickers with their signal, and a per-ticker report-card
page. No write/execute routes exist — per §13, this app doesn't do that.
"""

from __future__ import annotations

import concurrent.futures
import time

from flask import Flask, redirect, render_template, request, url_for

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


def _safe_analyze(ticker: str) -> tuple[dict | None, str | None]:
    try:
        return _cached_analyze(ticker), None
    except Exception as exc:  # a bad ticker shouldn't take down the whole dashboard
        return None, str(exc)


@app.route("/")
def dashboard():
    watchlist = request.args.get("watchlist")
    tickers = [t.strip().upper() for t in watchlist.split(",")] if watchlist else DEFAULT_WATCHLIST

    # Each ticker's 6-persona pipeline is I/O-bound (yfinance/Finnhub/Stocktwits
    # calls) — running them concurrently is the difference between ~2.5s/ticker
    # sequentially (55s for a 23-ticker watchlist) and one wall-clock pass.
    # analyze_ticker() creates its own ToolBudget per call, so there's no
    # shared mutable state across threads to worry about here.
    reports, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for ticker, (report, error) in zip(tickers, executor.map(_safe_analyze, tickers)):
            if report is not None:
                reports.append(report)
            else:
                errors.append({"ticker": ticker, "error": error})

    return render_template("index.html", reports=reports, errors=errors, watchlist=",".join(tickers))


@app.route("/go")
def go():
    """Jump-to-ticker search box: any symbol yfinance recognizes works here,
    not just watchlist entries — this isn't a lookup against a fixed list.
    """
    ticker = request.args.get("ticker", "").strip().upper()
    if not ticker:
        return redirect(url_for("dashboard"))
    return redirect(url_for("ticker_report", ticker=ticker))


@app.route("/ticker/<ticker>")
def ticker_report(ticker: str):
    ticker = ticker.upper()
    try:
        report = _cached_analyze(ticker)
    except Exception as exc:
        # The dashboard already isolates per-ticker failures (§ dashboard()
        # above) — this route needs the same treatment for direct navigation
        # (e.g. the /go search box) to a ticker that fails the initial
        # fetch_ohlcv call, which happens before any persona runs.
        return render_template("ticker_error.html", ticker=ticker, error=str(exc)), 404

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
