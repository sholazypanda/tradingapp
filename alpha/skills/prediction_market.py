"""Skill 6.2 — Market-Level Transformer Return-Timing Model (Ledger).

Loads the checkpoint notebooks/market_transformer_return_timing.ipynb
exports for a given block size. No trained model exists until you run
that notebook, so this reports `available: False` with a clear reason
rather than fabricating a signal.

Live inference uses SPY's daily returns as a practical market-return proxy
— the notebook trains on the "proper" Fama-French daily market excess
return via pandas_datareader, but that data isn't intraday/live-updating,
so the running service substitutes a liquid market-tracking ETF instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from alpha.config import MARKET_MODEL_ARTIFACT_DIR

MARKET_PROXY_TICKER = "SPY"


def _artifact_paths(artifact_dir: str, block_size: int) -> dict[str, Path]:
    base = Path(artifact_dir)
    return {
        "config": base / "config.json",
        "weights": base / f"transformer_block{block_size}.pt",
        "calibration": base / f"calibration_block{block_size}.json",
    }


def is_trained(block_size: int = 5, artifact_dir: str = MARKET_MODEL_ARTIFACT_DIR) -> bool:
    return all(p.exists() for p in _artifact_paths(artifact_dir, block_size).values())


def predict(block_size: int = 5, artifact_dir: str = MARKET_MODEL_ARTIFACT_DIR) -> dict:
    """§6.2 output contract: {as_of_date, raw_forecast, calibrated_forecast, timing_signal}."""
    paths = _artifact_paths(artifact_dir, block_size)
    if not is_trained(block_size, artifact_dir):
        return {
            "available": False,
            "reason": (
                f"No trained block-{block_size} market-timing model found at {artifact_dir}. "
                "Run notebooks/market_transformer_return_timing.ipynb and export a checkpoint first."
            ),
        }

    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch not installed — see notebook §0 Requirements."}

    from alpha.models.market_transformer import CausalReturnTransformer
    from alpha.data_sources.market_data import fetch_ohlcv

    saved_cfg = json.loads(paths["config"].read_text())
    model = CausalReturnTransformer(
        block_size, saved_cfg["d_model"], saved_cfg["n_heads"], saved_cfg["n_layers"], saved_cfg["dropout"],
    )
    model.load_state_dict(torch.load(paths["weights"], map_location="cpu"))
    model.eval()
    calib = json.loads(paths["calibration"].read_text())

    market_df = fetch_ohlcv(MARKET_PROXY_TICKER, bars=block_size)
    recent_returns = market_df["return"].values.astype(np.float32)

    with torch.no_grad():
        x = torch.tensor(recent_returns, dtype=torch.float32).unsqueeze(0)
        raw = float(model(x)[0])

    calibrated = calib["coef"] * raw + calib["intercept"]
    return {
        "available": True,
        "as_of_date": str(market_df.index[-1].date()),
        "raw_forecast": raw,
        "calibrated_forecast": calibrated,
        "timing_signal": "long_market" if calibrated > 0 else "risk_free",
    }
