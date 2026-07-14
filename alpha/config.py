"""App-wide configuration and the §12 context/resource budgets.

Nothing here reads a secret's *value* into a log or a default — every
credential is read from the environment at call time and nothing is
hardcoded, per tradingskills.md §11's delivery requirement.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime


# --- §12 Context & Resource Budgets --------------------------------------

DEFAULT_OHLCV_BARS = 60
"""Default per-decision OHLCV window. Skills 3-5 and the Skill 6 forecaster's
per-sample lookback all read from this. Do not raise this for routine report
cards — widen explicitly (e.g. BACKTEST_MONTHS below) only for the specific
skills that need more history."""

BACKTEST_MONTHS = 6
"""Skill 8's explicit, separate long-history budget for backtesting."""

DEFAULT_WATCHLIST = [
    "AAPL", "GOOGL", "INTU", "MU", "NVDA", "SNDK", "AVGO", "WDC", "VRT", "LITE",
    "COHR", "AMD", "MRVL", "SKM", "BE", "IREN", "STX", "FLNC", "TSLA", "SPCX",
    "BLZE", "TQQQ", "QMCO", "DRAM",
]  # synced from TradingView watchlist "Watchlist" (list_id 316627301) via tv CLI
# DRAM (Roundhill Memory ETF) added manually — launched 2026-04-02, only ~69
# trading days of history so far; will show as unavailable on the dashboard
# until it has ~90 bars (needed for the 60-bar decision window + RVOL's
# 30-bar warmup), self-resolving in a few more weeks. Add it in TradingView
# too if you want it to survive the next `tv` CLI re-sync of this list.


class ToolBudgetExceeded(RuntimeError):
    """Raised when a run tries to exceed its task-type tool-call cap."""


# Caps are per task type, not one global number — a full-market scan
# legitimately needs more calls than a single report card (§12).
TOOL_CALL_BUDGETS = {
    "report_card": 20,   # one ticker: OHLCV + skills 3-9 + optional model inference
    "scan": 200,         # Wick screening a watchlist/universe
    "backtest": 50,      # Skill 8's 6-month replay
}


@dataclass
class ToolBudget:
    """In-memory call counter enforcing TOOL_CALL_BUDGETS for one run — and,
    since every `spend`/`note` call is timestamped, also the execution trace
    the UI's decision-log panel renders. This is a real record of what each
    persona actually did, not a scripted display value.

    Usage:
        budget = ToolBudget("report_card")
        budget.spend("Rook", "fetch_ohlcv(AAPL, bars=60)")       # counts against the cap
        budget.note("Rook", "smc.analyze()", "trend_bias=bullish")  # doesn't
    Raises ToolBudgetExceeded once the task-type cap is hit, so a scanner
    pass or multi-analyst Alpha run can't spiral into an unbounded loop.
    """

    task_type: str
    calls_made: int = field(default=0, init=False)
    log: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.task_type not in TOOL_CALL_BUDGETS:
            raise ValueError(f"Unknown task_type '{self.task_type}'; add it to TOOL_CALL_BUDGETS first.")

    @property
    def cap(self) -> int:
        return TOOL_CALL_BUDGETS[self.task_type]

    def _append(self, actor: str, action: str, result: str, kind: str) -> None:
        self.log.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "actor": actor, "action": action, "result": result, "kind": kind,
        })

    def spend(self, actor: str, action: str, result: str = "") -> None:
        """Records a real tool call/data fetch and counts it against the cap."""
        if self.calls_made >= self.cap:
            spent_actions = [e["action"] for e in self.log]
            raise ToolBudgetExceeded(
                f"'{self.task_type}' run hit its {self.cap}-call budget at step '{action}' "
                f"(spent so far: {spent_actions})"
            )
        self.calls_made += 1
        self._append(actor, action, result, kind="tool_call")

    def note(self, actor: str, action: str, result: str = "") -> None:
        """Records an evaluation/reasoning step (no external call) — doesn't
        count against the cap, but still shows up in the execution log."""
        self._append(actor, action, result, kind="evaluation")


# --- API credentials (env vars only — never hardcode) ---------------------

FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")

SMTP_HOST = os.environ.get("SMTP_HOST")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
DIGEST_RECIPIENT = os.environ.get("ALPHA_DIGEST_RECIPIENT", "panda.shobhika@gmail.com")

# Where Skill 6's trained model artifacts land after running the notebooks.
STOCK_MODEL_ARTIFACT_DIR = os.environ.get(
    "STOCK_MODEL_ARTIFACT_DIR", "notebooks/artifacts"
)
MARKET_MODEL_ARTIFACT_DIR = os.environ.get(
    "MARKET_MODEL_ARTIFACT_DIR", "notebooks/artifacts_market_timing"
)

JOURNAL_DIR = os.environ.get("ALPHA_JOURNAL_DIR", "journal")
