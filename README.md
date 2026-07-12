# Alpha — AI Trading Agent

A read-only Python trading-analysis app: six analyst personas (Rook, Cortex,
Vance, Sable, Ledger, Wick) each independently analyze a ticker, and an
**Alpha** orchestrator reconciles their signals — including disagreement —
into a single report card. Served by a local Flask dashboard. Two Jupyter
notebooks train supplementary ML forecasting models.

**Full spec**: [tradingskills.md](tradingskills.md) — every skill below is
numbered against a section there (§3, §5, §6.1, etc.); read that first if
you want the "why," not just the "how."

**Scope, up front**: this app never places a trade. Every skill is
analysis/notification only (§13). The email digest defaults to a dry run
and won't send anything until you explicitly turn that off.

---

## 1. Quick start — the Flask app

```bash
cd IdeaProjects/tradingapp
python3 -m venv .venv          # if .venv doesn't already exist
.venv/bin/pip install -r requirements.txt

# Optional but recommended (see §3 below for what needs it)
export FINNHUB_API_KEY="your-key-here"

.venv/bin/python3 -m flask --app alpha.app run --port 5050
```

Then open **http://localhost:5050**. Three pages:

| Route | What it shows |
|---|---|
| `/` | Dashboard — watchlist tickers, signal, entry/stop/targets. `?watchlist=AAPL,MSFT,...` to override the default list. |
| `/ticker/<SYMBOL>` | Full report card for one ticker: all 6 agent opinions, price sparkline, key levels, model forecast, weekday seasonality, and a live execution-trace log of every tool call each agent made. |
| `/scan` | Wick's real-time scanner across a ticker list — volume surge, MA acceleration, RSI breakout, MACD cross, coiling. |

If you're driving this from Claude Code, `.claude/launch.json` (in the
sibling `tradingview-mcp` project root — see note in §6) already has an
`alpha-flask` entry wired up for the preview tools.

---

## 2. Project layout

```
alpha/
  config.py            # env vars, §12 budgets (60-bar default, tool-call caps)
  app.py                # Flask routes
  agents/
    personas.py         # Rook, Cortex, Vance, Sable, Ledger, Wick
    alpha.py             # orchestrator: reconciles opinions into one report card
  skills/
    smc.py                    # §3 Smart Money Concepts (structure, order blocks, FVG, EQH/EQL)
    liquidity_swings.py       # §4 pivot levels / liquidity sweeps
    composite_indicator.py    # §5 Bollinger + RSI + ATR + RVOL
    prediction_stock.py       # §6.1 loads the node-transformer checkpoint
    prediction_market.py      # §6.2 loads the market-timing checkpoint
    fundamentals.py           # §7 valuation/revenue trend (yfinance)
    sentiment.py              # §8 crowd/news + RSI backtest + weekday seasonality
    scanner.py                # §9 real-time spike/breakout screen
  models/                # PyTorch model classes shared with the notebooks
  data_sources/          # yfinance, Finnhub, Stocktwits clients
  reportcard.py           # §10 report-card formatting + sparkline SVG
  journal.py               # §12 rolling summary file (not raw history)
  email_alerts.py           # §11 daily digest (dry-run by default)
  safety.py                  # §13 — execute_order() always refuses
  templates/, static/       # the dashboard's HTML/CSS

notebooks/
  node_transformer_sentiment_forecast.ipynb   # §6.1 per-stock forecaster
  market_transformer_return_timing.ipynb      # §6.2 market-timing forecaster

pinescript1/2/3    # the original TradingView indicators §3-5 were ported from
trading            # original design note this whole app is built from
tradingskills.md   # the full build spec
```

---

## 3. Environment variables

None are required just to run the Flask app on rule-based skills (§3-5, §9)
— those work with zero configuration. These unlock the rest:

| Variable | Unlocks | Get it from |
|---|---|---|
| `FINNHUB_API_KEY` | Sable's news headlines (§8) and the training corpus for the §6.1 notebook | Free at [finnhub.io/register](https://finnhub.io/register) |
| `ALPHA_VANTAGE_API_KEY` | Not yet wired into any skill — reserved | [alphavantage.co](https://www.alphavantage.co/support/#api-key) |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD` | Actually sending the §11 daily digest (still requires `dry_run=False` explicitly) | Your email provider |
| `ALPHA_DIGEST_RECIPIENT` | Overrides the default digest recipient | — |
| `STOCK_MODEL_ARTIFACT_DIR` | Where `prediction_stock.py` looks for a trained §6.1 checkpoint | defaults to `notebooks/artifacts` |
| `MARKET_MODEL_ARTIFACT_DIR` | Where `prediction_market.py` looks for a trained §6.2 checkpoint | defaults to `notebooks/artifacts_market_timing` |
| `ALPHA_JOURNAL_DIR` | Where the §12 journal summary/entries are written | defaults to `journal/` |

Set them in your shell before starting the app or Jupyter — `export
FINNHUB_API_KEY="..."` — or drop them in a local `.env` file (not committed;
see `.gitignore`) if you wire up `python-dotenv` yourself.

---

## 4. Using the prediction models (the notebooks)

Two separate models, at two different levels — see `tradingskills.md` §6
for the full explanation of why they're not the same thing:

### 6.1 — Per-stock forecaster (`node_transformer_sentiment_forecast.ipynb`)

Node-transformer + FinBERT sentiment fusion, predicts next-day return **per
ticker**. Feeds Cortex's persona.

```bash
pip install torch transformers yfinance pandas numpy scikit-learn requests
export FINNHUB_API_KEY="your-key-here"   # required — builds the headline training corpus
jupyter notebook notebooks/node_transformer_sentiment_forecast.ipynb
```

Run top to bottom, then uncomment and run the "Putting it together" cell
(§9) and the export cell (§10). This writes a checkpoint to
`notebooks/artifacts/` (config.json, `node_transformer_forecaster.pt`,
`adjacency.pt`). Once that exists, `alpha/skills/prediction_stock.py` picks
it up automatically — no restart of the Flask app needed beyond a normal
reload, since it loads the checkpoint fresh on each call.

**Status**: not yet trained in this environment (no `notebooks/artifacts/`
present) — Cortex will report `unavailable` with the reason until you run
this.

### 6.2 — Market-timing forecaster (`market_transformer_return_timing.ipynb`)

Causal Transformer predicting the next-day **aggregate market** return from
Fama-French daily excess returns, with post-ML regression recalibration.
Feeds Ledger's overall-exposure signal.

```bash
pip install torch pandas numpy scikit-learn pandas-datareader statsmodels
jupyter notebook notebooks/market_transformer_return_timing.ipynb
```

No API key needed (Ken French's data library is free/open). Run top to
bottom including §9 (Run end-to-end) and §10 (Export) — both are live code,
not commented out. This trains one model per `cfg.block_sizes` (5/20/60 by
default) and writes `notebooks/artifacts_market_timing/`
(`transformer_block{N}.pt`, `calibration_block{N}.json`, `config.json`).

**Status**: **already trained** in this environment — checkpoints for
block sizes 5, 20, and 60 exist in `notebooks/artifacts_market_timing/`.
Ledger will use them automatically.

`cfg.fast_mode = True` trades fidelity for runtime (shorter date range,
coarser refit cadence). Set it `False` and widen `start_date` for a fuller
replication of the paper — expect a much longer run.

### Retraining

Both models should be retrained periodically (regimes drift — see each
notebook's Caveats section). There's no scheduled retraining wired up;
re-running the notebook and re-exporting overwrites the existing checkpoint.

---

## 5. Known limitations

- **yfinance's `.news` endpoint** intermittently returns `401 Invalid Crumb`
  (a Yahoo-side auth quirk, not a bug here) — Sable's news-sentiment score
  falls back to unscored rather than crashing when this happens.
- **Finnhub free tier** is rate-limited (55 req/min used here, under the
  60/min cap) — building the full §6.1 training corpus across a large
  watchlist takes a while the first time; results are cached to
  `data/finnhub_news_cache/` so it's a one-time cost.
- **Weekday seasonality** (§8) uses a 6-month window — only ~20-27 samples
  per weekday, a tendency worth noting, not a statistically robust signal.
- **Chart patterns and insider-trade filings** (§7) have no free structured
  data source wired up; they report `available: false` rather than a
  fabricated value. See §7 in `tradingskills.md` for the paid/official
  options (Financial Modeling Prep, SEC EDGAR Form 4).
- **Fundamentals/Finviz-style skill (§7)** and **Stocktwits (§8)** both
  degrade gracefully to neutral/unavailable if their source is unreachable
  — they never block the rest of the report card.

---

## 6. TradingView MCP (separate project, optional)

`tradingskills.md` §2 lists TradingView (via the sibling `tradingview-mcp`
project's CDP bridge) as a possible data source for live ticker/OHLCV/Pine
indicator data. This app does **not** depend on it — all live data comes
from `yfinance`/Finnhub/Stocktwits/Ken French directly. The MCP server is
only relevant if you're driving TradingView Desktop itself from Claude
Code; see that project's own `SETUP_GUIDE.md`.
