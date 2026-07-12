"""§13 Safety Boundary — Execution.

Every skill in this app is analysis/notification only. This module exists
so that if broker or Polymarket integration is ever added, it has to go
through here first — and today, it always refuses, because no broker
integration exists yet and no automated flow should ever bypass a human
confirming the specific order in the app's own UI.
"""

from __future__ import annotations


class ExecutionNotAllowed(RuntimeError):
    """Raised by every execution entry point in this app, always."""


def execute_order(*args, human_confirmed: bool = False, **kwargs):
    """Placeholder for future broker/Polymarket order placement.

    Deliberately refuses unconditionally right now: no broker API is wired
    up (tradingskills.md §2 lists it as "future"), and even once one is,
    this function must require a human confirming the *specific* order in
    the app's UI immediately before the call — not a config flag, not a
    standing approval. `human_confirmed` is accepted only so future
    integration code has an obvious place to plumb that per-order gate
    through; passing it True does nothing today.
    """
    raise ExecutionNotAllowed(
        "Order execution is not implemented. Per tradingskills.md §13, every skill in this "
        "app is analysis/notification only — placing a real order requires the user to do it "
        "themselves, or an explicit human-confirmation step this app does not yet have."
    )
