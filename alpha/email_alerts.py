"""§11 Daily Email Alerts — a digest, not a dump.

Only includes tickers whose signal changed since the last check (per the
journal summary), plus any new structure break / order-block touch / Wick
spike. One-way, informational only — see §13; nothing here can place an
order.

`send_daily_digest` defaults to `dry_run=True` and only actually calls
smtplib when a caller explicitly passes `dry_run=False` *and* SMTP env vars
are set — sending real email is something only the person running this
app should trigger themselves, not something this module does on its own.
"""

from __future__ import annotations

import smtplib
from email.message import EmailMessage

from alpha.agents.alpha import analyze_ticker
from alpha.config import DIGEST_RECIPIENT, SMTP_HOST, SMTP_PASSWORD, SMTP_PORT, SMTP_USERNAME
from alpha.journal import record_report_card
from alpha.reportcard import format_text


def build_digest(watchlist: list[str]) -> str:
    """Runs Alpha for each watchlist ticker, keeps only ones with a signal
    change (or no prior journal record), and formats the digest body.
    """
    sections = []
    for ticker in watchlist:
        report = analyze_ticker(ticker)
        journal_delta = record_report_card(report)
        if journal_delta["signal_changed"]:
            sections.append(format_text(report))

    if not sections:
        return "No signal changes today across the watchlist — nothing to report."
    return "\n\n---\n\n".join(sections)


def send_daily_digest(watchlist: list[str], recipient: str = DIGEST_RECIPIENT, dry_run: bool = True) -> dict:
    """Builds the digest; only sends via SMTP if dry_run=False and SMTP env
    vars are configured. Returns the digest body either way so the caller
    can inspect it before deciding to actually send.
    """
    body = build_digest(watchlist)

    if dry_run:
        return {"sent": False, "reason": "dry_run=True (default) — nothing was sent.", "body": body}

    if not all([SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD]):
        return {"sent": False, "reason": "SMTP_HOST/SMTP_USERNAME/SMTP_PASSWORD not set in the environment.", "body": body}

    msg = EmailMessage()
    msg["Subject"] = "Alpha — daily trading digest"
    msg["From"] = SMTP_USERNAME
    msg["To"] = recipient
    msg.set_content(body)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)

    return {"sent": True, "recipient": recipient, "body": body}
