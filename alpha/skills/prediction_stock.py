"""Skill 6.1 — Node Transformer + BERT Sentiment Fusion (Cortex).

Loads the checkpoint notebooks/node_transformer_sentiment_forecast.ipynb
exports. The model needs the *entire* sector graph's recent data at once
(cross-sectional graph attention), so inference is naturally "predict for
every ticker in the trained graph," not a single-ticker call — `predict()`
below just picks one ticker's result out of that batch.

No trained model exists until you actually run the notebook, so this
reports `available: False` with a clear reason rather than fabricating a
number — that's a real, expected state, not an error to hide.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from alpha.config import STOCK_MODEL_ARTIFACT_DIR
from alpha.data_sources.market_data import fetch_ohlcv

FEATURE_COLS = ["return", "log_return", "vol_20", "rsi_14"]


def _artifact_paths(artifact_dir: str) -> dict[str, Path]:
    base = Path(artifact_dir)
    return {
        "config": base / "config.json",
        "weights": base / "node_transformer_forecaster.pt",
        "adjacency": base / "adjacency.pt",
    }


def is_trained(artifact_dir: str = STOCK_MODEL_ARTIFACT_DIR) -> bool:
    return all(p.exists() for p in _artifact_paths(artifact_dir).values())


def predict_all(artifact_dir: str = STOCK_MODEL_ARTIFACT_DIR) -> dict:
    """Runs the trained graph model for every ticker it was trained on.

    Returns {"available": False, "reason": ...} if no checkpoint exists yet,
    otherwise {"available": True, "predictions": {ticker: {...}}}.
    """
    paths = _artifact_paths(artifact_dir)
    if not is_trained(artifact_dir):
        return {
            "available": False,
            "reason": (
                f"No trained model found at {artifact_dir}. Run "
                "notebooks/node_transformer_sentiment_forecast.ipynb (needs a Finnhub "
                "API key for the headline corpus) and export a checkpoint first."
            ),
        }

    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch/transformers not installed — see notebook §0 Requirements."}

    from alpha.models.node_transformer import NodeTransformerForecaster

    saved_cfg = json.loads(paths["config"].read_text())
    tickers = saved_cfg["tickers"]
    lookback = saved_cfg["lookback"]

    model = NodeTransformerForecaster(
        n_nodes=len(tickers), in_dim=len(saved_cfg["feature_cols"]), lookback=lookback,
        d_model=saved_cfg["d_model"], n_heads=saved_cfg["n_heads"],
        n_temporal_layers=saved_cfg["n_temporal_layers"],
    )
    model.load_state_dict(torch.load(paths["weights"], map_location="cpu"))
    model.eval()
    adjacency = torch.load(paths["adjacency"], map_location="cpu")

    # Build the (lookback, N, F) panel from live data — sentiment defaults
    # neutral here; wire in alpha.data_sources.finnhub_news for real scores.
    per_ticker_features = []
    for ticker in tickers:
        df = fetch_ohlcv(ticker, bars=lookback)
        per_ticker_features.append(df[FEATURE_COLS].values)
    X = np.stack(per_ticker_features, axis=1)[-lookback:]  # (lookback, N, F)
    S = np.zeros((lookback, len(tickers), 1), dtype=np.float32)  # neutral sentiment placeholder

    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
        s = torch.tensor(S, dtype=torch.float32).unsqueeze(0)
        pred = model(x, s, adjacency).squeeze(0).numpy()

    predictions = {
        ticker: {"predicted_next_day_return": float(pred[i]), "model_confidence": "unvalidated"}
        for i, ticker in enumerate(tickers)
    }
    return {"available": True, "predictions": predictions}


def predict(ticker: str, artifact_dir: str = STOCK_MODEL_ARTIFACT_DIR) -> dict:
    """§6.1 output contract for one ticker: {ticker, predicted_next_day_return, model_confidence}."""
    batch = predict_all(artifact_dir)
    if not batch["available"]:
        return {"ticker": ticker, "available": False, "reason": batch["reason"]}

    if ticker not in batch["predictions"]:
        return {
            "ticker": ticker, "available": False,
            "reason": f"{ticker} isn't in the trained graph's ticker list — retrain including it.",
        }

    return {"ticker": ticker, "available": True, **batch["predictions"][ticker]}
