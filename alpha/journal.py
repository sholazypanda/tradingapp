"""§12 Journal — a rolling summary file, not raw entry history.

Keeps one JSON summary per ticker (last signal, when it changed, last
close) rather than feeding the agent its last N raw report cards. Call
`get_full_entry` only when you actually need to inspect one past decision.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from alpha.config import JOURNAL_DIR

SUMMARY_FILE = "summary.json"
ENTRIES_DIR = "entries"


def _summary_path() -> Path:
    return Path(JOURNAL_DIR) / SUMMARY_FILE


def _entries_dir() -> Path:
    return Path(JOURNAL_DIR) / ENTRIES_DIR


def load_summary() -> dict:
    path = _summary_path()
    return json.loads(path.read_text()) if path.exists() else {}


def record_report_card(report: dict) -> dict:
    """Updates the rolling summary for one ticker and archives the full
    report card as a dated full entry (only read back on demand — see
    `get_full_entry`). Returns {signal_changed, previous_signal}.
    """
    ticker = report["ticker"]
    summary = load_summary()
    previous = summary.get(ticker, {})
    signal_changed = previous.get("last_signal") != report["signal"]

    now = datetime.now(timezone.utc).isoformat()
    summary[ticker] = {
        "last_signal": report["signal"],
        "last_close": report["entry"],
        "last_checked_at": now,
        "signal_changed_at": now if signal_changed else previous.get("signal_changed_at", now),
    }

    Path(JOURNAL_DIR).mkdir(parents=True, exist_ok=True)
    _summary_path().write_text(json.dumps(summary, indent=2))

    _entries_dir().mkdir(parents=True, exist_ok=True)
    entry_path = _entries_dir() / f"{ticker}_{now.replace(':', '-')}.json"
    entry_path.write_text(json.dumps(report, indent=2, default=str))

    return {"signal_changed": signal_changed, "previous_signal": previous.get("last_signal")}


def get_full_entry(ticker: str, timestamp: str | None = None) -> dict | None:
    """Only pulls one full past entry on demand — never used for routine context."""
    entries = sorted(_entries_dir().glob(f"{ticker}_*.json"))
    if not entries:
        return None
    if timestamp is None:
        return json.loads(entries[-1].read_text())
    matches = [e for e in entries if timestamp in e.name]
    return json.loads(matches[-1].read_text()) if matches else None
