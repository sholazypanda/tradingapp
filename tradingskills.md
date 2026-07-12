# Trading Skills Reference — "Alpha" AI Trading Agent

Consolidated skills spec for the Python trading app. Combines the agent
architecture described in the `trading` design note with the technical
analysis logic encoded in the three Pine Script indicators
(`pinescript1`, `pinescript2`, `pinescript3`).

This document is a **build spec / reference for the Python agent's logic**,
not investment advice.

---

## 1. Agent Architecture

**Alpha** — chief strategist / orchestrator. Takes a ticker (or "scan
everything") and current positions, delegates to the analyst sub-agents
below, then synthesizes their output into a single report card + trade plan.

Suggested analyst roster (6 sub-agents Alpha consults):

| Persona | Role | Skills used |
|---|---|---|
| **Rook** — Structure Analyst | Price action / market structure | §3 (SMC), §4 (Liquidity Swings) |
| **Cortex** — Quant Forecaster | Learned per-stock price forecast (node transformer + sentiment fusion) | §6.1 (Prediction Models) |
| **Vance** — Fundamentals Analyst | Revenue, margins, valuation | §7 (Finviz fundamentals) |
| **Sable** — Sentiment Analyst | Crowd + insider positioning | §8 (Stocktwits, insider trades, news) |
| **Ledger** — Risk/Portfolio Manager | Position sizing, hedges, stops, overall market exposure | §5 (Composite indicator), §6.2 (Market-timing model) |
| **Wick** — Setup Scanner | Real-time spike/breakout detection | §9 (Scanner skill) |

Alpha's job: reconcile disagreement between analysts (e.g., Rook says
bullish structure, Vance flags deteriorating margins) into one confidence
rating, not silently average them — the report card should say *why* when
signals conflict.

---

## 2. Data Source Integrations Required

| Source | Data pulled | Used by |
|---|---|---|
| TradingView (CDP, this repo's sibling MCP server) | live ticker, OHLCV, custom Pine indicator levels | Rook, Wick |
| Finviz (API/scrape) | support/resistance, chart pattern flags, insider buy/sell, short ratio, P/E, volume | Vance, Sable |
| Stocktwits | crowd sentiment score | Sable |
| Yahoo Finance | news, 6-month historical OHLCV for backtesting | Sable, Ledger |
| Broker API (future) | order execution, live position sync | Ledger (**never auto-executes trades — human confirms**) |
| Polymarket API (future) | prediction-market pricing for cross-reference | Alpha (macro context) |

---

## 3. Smart Money Concepts (from `pinescript3`)

Market-structure model: structure breaks, order blocks, imbalance, and
liquidity positioning.

- **Structure (BOS/CHoCH)**: swing structure (large lookback, default 50) sets
  trend bias; internal structure (default 5) gives short-term pullback
  confirmation. A break that continues the trend = BOS; a break that reverses
  it = CHoCH. Swing CHoCH is the primary reversal signal.
- **Order Blocks**: the last extreme candle before a structure break, stored
  as bullish/bearish zones; invalidated ("mitigated") once price trades back
  through them. Unmitigated OBs in the trend direction = candidate entries.
- **Equal Highs/Lows (EQH/EQL)**: two swing pivots within `0.1 * ATR(200)` of
  each other — marks obvious resting liquidity/stop clusters.
- **Fair Value Gaps (FVG)**: 3-bar imbalance (`low[0] > high[2]` or mirror)
  exceeding a threshold; acts as a magnet/pullback zone until filled.
- **Premium/Discount/Equilibrium zones**: position within the recent
  swing range — top 5% = premium (sell bias), bottom 5% = discount (buy
  bias), middle = equilibrium.

**Output contract**: `{trend_bias, structure_breaks[], order_blocks[], eqh_eql[], fvgs[], range_zone}`

---

## 4. Liquidity Swings (from `pinescript1`)

Tracks pivot highs/lows (`pivot_high/low(length, length)`, default 14) and how
much subsequent price/volume interacts with each level before it's confirmed
and before it's eventually swept (`close` trades through it).

**Output contract**: `[{price, side: "high"|"low", touch_count, volume, crossed: bool}]`

Use: price approaching an uncrossed, high-touch-count level = likely
reaction zone; a level just marked `crossed` = liquidity sweep, often a
reversal precursor — cross-reference with Skill 3's EQH/EQL and CHoCH.

---

## 5. Composite Confirmation Indicator (from `pinescript2`)

Bollinger Bands (20, 2.0) + RSI(14) + ATR(14, selectable smoothing) + RVOL
(volume vs. 30-bar average, spike ≥ 1.5x).

**Output contract**: `{rsi, rsi_state, atr, rvol, rvol_spike, bb_pct, bb_signal}`

Use:
- `bb_pct` (position within the bands, 0–100) + RSI extreme + RVOL spike
  together = stronger confluence than any single condition alone.
- `atr` sizes stops/targets (e.g. `stop = entry - 1.5*atr`) so risk scales
  with current volatility instead of a fixed %.

---

## 6. Prediction Models

Two learned forecasting models, both distinct from the rule-based Skill 5
composite indicator. They operate at different levels — one per-stock, one
market-wide — and feed different consumers.

### 6.1 Node Transformer + BERT Sentiment Fusion (per-stock forecast)

Owned by **Cortex** (Quant Forecaster).

**Source**: Al Ridhawi, Haj Ali, Al Osman, *"Stock Market Prediction Using
Node Transformer Architecture Integrated with BERT Sentiment Analysis"*
([arXiv:2603.05917](https://arxiv.org/abs/2603.05917)).

- Models the market as a graph — stocks as nodes, sector/supply-chain
  relationships as edges.
- **Node Transformer**: per-stock temporal attention over price/volume
  history, plus cross-sectional graph attention across related stocks.
- **BERT sentiment (FinBERT)**: sentiment extracted from headlines/social
  text, fused into the node embedding via attention rather than plain
  concatenation.
- **Output**: next-day price/return forecast per ticker + directional
  accuracy estimate.
- **Reported paper metrics** (as published, not independently verified):
  0.80% MAPE vs. 1.20% (ARIMA) / 1.00% (LSTM); sentiment fusion reduces
  error ~10% overall, ~25% around earnings; trained on 20 S&P 500 names,
  Jan 1982–Mar 2025.

**Notebook**: [`notebooks/node_transformer_sentiment_forecast.ipynb`](notebooks/node_transformer_sentiment_forecast.ipynb).

**Output contract**: `{ticker, predicted_next_day_return, model_confidence}`
— a *learned* signal; the report card (§10) should label it as a model
forecast and keep it separate from the deterministic technical signals in
§3–5, not blend the two silently.

**Caveats**: retrain periodically (regimes/sector relationships drift);
walk-forward validate only (a random train/test split leaks future data
into training); the sector-based graph here is a simplification of the
paper's fuller sector + supply-chain relationship graph.

### 6.2 Market-Level Transformer Return-Timing Model (aggregate market forecast)

Owned by **Ledger** (Risk/Portfolio Manager) — this is a market-wide
beta-exposure dial, not a per-ticker pick, so it feeds position sizing
(§5) rather than Cortex's stock-level report cards.

**Source**: Han, Huang, Huang, Zhou, *"Daily Market Return Prediction with
Transformer"* (SSRN, this version May 2026) — read directly from the PDF
the user provided (`ssrn-6835039.pdf`); the earlier fetch attempt was
blocked by Cloudflare bot-protection, since resolved.

**What it does**: predicts the next-day *aggregate* market excess return
(e.g. CRSP/Fama daily market excess return) purely from sequences of its
own lagged daily values — no sentiment, no cross-sectional graph. Compared
against Random Forest and a feedforward Neural Network as baselines, all
three sharing the same lagged-return input blocks (5, 20, or 60 days).

- **Architecture**: encoder-only Transformer (no decoder — this is
  forecasting, not seq2seq). Each day's return is embedded to `d_model`
  (32 or 64) + positional encoding, then `N` encoder layers (2 or 4) of
  multi-head self-attention (`Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V`) with a
  **causal mask** (lower-triangular — day *t* only attends to days ≤ *t*,
  never future days) + feed-forward block + residual/layer-norm. The final
  timestep's contextual embedding feeds a regression head for a single
  next-day scalar return prediction.
- **Baselines, same input blocks**: Random Forest (250/500 trees,
  max depth 1/3/5) and a feedforward NN (~4–5 layers, geometric pyramid
  sizing, ReLU, Adam).
- **Rolling estimation**: 5-year rolling windows (4 years train + 1 year
  for hyperparameter validation), predicting the following single year,
  then rolling forward one year at a time across the full sample
  (1926–2022 in the paper).
- **Post-ML regression recalibration** (the paper's key trick): the raw
  ML forecast predicts *direction* well but is badly scaled (slope < 1 in
  a Mincer-Zarnowitz regression of realized-on-forecast returns — i.e. it
  overstates how much expected returns actually move). Each year, refit an
  expanding-window OLS of realized returns on the raw forecast and use
  *that* rescaled output as the calibrated forecast. This single step is
  what turns a significant-but-useless-scale forecast into one with real
  out-of-sample R².
- **Reported paper results** (as published, not independently verified):
  raw Transformer forecasts significantly predict next-day returns
  (slope ≈ 0.7, t > 10) while a simple average of past returns does not
  (insignificant) — Random Forest (≈0.6) and NN (≈0.2) are directionally
  right but weaker. Post-ML out-of-sample R² for Transformer: 0.94%/1.03%/
  0.97% (5/20/60-day blocks) vs. ~0% for the recalibrated average-return
  benchmark. A "invest 100% market if forecast > 0, else 100% risk-free"
  timing strategy on the post-ML Transformer forecast delivers ~14%/yr,
  Sharpe ≈1.13–1.16, vs. buy-and-hold's 8.27%/yr, Sharpe 0.49. Predictive
  power is stronger in recessions, low-VIX regimes, high-sentiment
  periods, and around nonfarm payroll announcements.
- **Important nuance the paper stresses**: the learned attention weights
  come out fairly *flat* across layers — the authors argue the model's
  edge is **not** the attention mechanism cherry-picking specific lagged
  days, but the combination of the nonlinear feed-forward architecture and
  the *rolling re-estimation scheme*, which together let it learn a
  time-varying nonlinear transformation of past returns that a fixed
  linear average can't capture.

**Notebook**: [`notebooks/market_transformer_return_timing.ipynb`](notebooks/market_transformer_return_timing.ipynb)
implements the causal Transformer + RF/NN baselines, the rolling 5-year
training scheme, the post-ML recalibration step, and the market-timing
backtest (Sharpe ratio vs. buy-and-hold and vs. the average-return
benchmark), pulling daily Fama-French market excess returns via
`pandas_datareader`.

**Output contract**: `{as_of_date, raw_forecast, calibrated_forecast, timing_signal: "long_market"|"risk_free"}`
— feeds Ledger's overall exposure sizing, not an individual ticker's
report card.

**Caveats**:
- This forecasts the *aggregate market*, not any individual ticker — it's
  a systematic overlay on overall exposure, complementary to but distinct
  from Skill 6.1's per-stock forecast.
- Faithfully reproducing the paper means retraining once per rolling year
  across ~90 years of data — expensive. The notebook's default config
  uses a shorter recent window and coarser refit cadence to stay runnable;
  scale up for a fuller replication.
- Validate the paper's reported numbers independently (walk-forward, not
  reused test years) before trusting them for real capital decisions —
  same standing caveat as Skill 6.1.

---

## 7. Fundamental & Pattern Skills (Finviz-style)

- **Chart patterns**: wedge up/down, ascending/descending triangle,
  channel up/down, head & shoulders, double top/bottom — flagged with the
  price range they imply and the level that confirms/invalidates the pattern.
- **Key levels**: horizontal support/resistance from recent swing extremes.
- **Insider activity**: recent insider buy/sell filings — recent insider
  buying is a soft bullish tell, especially if clustered.
- **Valuation/positioning stats**: P/E ratio, short ratio (short-squeeze
  potential when short ratio is high + price breaking structure bullishly),
  relative volume.
- **Fundamentals over time**: revenue growth & margin trend over 3 years —
  flag acceleration or deterioration and note likely driver (pricing, volume,
  cost structure) rather than just the number.

---

## 8. Sentiment, News & Backtesting Skills

- **Stocktwits sentiment score** — crowd positioning; extreme bullish crowd
  sentiment + weakening structure is a contrarian caution flag, not a buy signal.
- **Yahoo Finance news** — recent headlines as qualitative catalyst context
  for any spike or structure break.
- **6-month backtest** — replay Yahoo Finance historical OHLCV to check
  which RSI thresholds/entry rules would have worked best for *this specific*
  ticker recently, rather than assuming fixed 30/70 levels apply universally.

---

## 9. Real-Time Scanner Skill

Screen a ticker list (or "all TradingView tickers") for early spike setups:
volume surge (last 1–5 days/hours), price acceleration vs. 50/200 MA, RSI
breakout from oversold, MACD crossover. Score continuation probability 1–10,
list key levels to watch, and flag likely catalyst (earnings, news, insider
activity, sector move) using Skills 5–8 above. Also flag tickers "coiling"
(low volatility contraction) as pre-breakout candidates.

---

## 10. Report Card Output Format

Every ticker Alpha reports on should produce:

```
Ticker: <SYMBOL>
Signal: Strong Buy | Buy | Neutral | Sell | Strong Sell
Entry: <price>            Stop-loss: <price>
Target(s): <price(s)>     Risk/Reward: <ratio>
Confidence: <rating + one-line why, esp. if analysts disagreed>
Key levels: <from Skills 3 & 4>
Model forecast: <Skill 6 next-day prediction + directional confidence, labeled as a learned signal>
Confluence notes: <Skill 5 readings + any pattern/sentiment flags>
```

Delivered via a local Flask web app listing scanned tickers with their
signal, entry/exit, and report card — read-only by default.

---

## 11. Daily Email Alerts

Alpha sends a daily digest to **panda.shobhika@gmail.com** in addition to the
Flask app — a push notification rather than something the user has to
remember to check.

- **Trigger**: scheduled once per trading day (e.g. after market close, or
  before open for a pre-market brief) — not on every scanner tick.
- **Content**: report cards (§10) for watchlist tickers with a signal change
  since the last email, plus any new liquidity sweep, CHoCH, or unmitigated
  order-block touch (§3–4) and any Wick scanner spike flags (§9). Skip
  tickers with no change to keep the email short — a digest, not a dump.
- **Delivery**: standard SMTP (e.g. Python `smtplib`) or a transactional
  email API, with credentials read from environment variables / a local
  `.env` file — never hardcoded in source or committed to the repo.
- **Safety boundary**: this skill only sends notifications. It must never
  trigger an order or any action requiring the human-confirmation step in
  §13 — email is one-way, informational output only.

---

## 12. Context & Resource Budgets

The agent's job each run is a single daily (or intraday) decision, not a
research paper — keep what gets fed into the model small and bounded.

- **Historical data window**: default OHLCV pull is **60 bars**, not 500/a
  full year. 60 daily bars is enough for the structure/liquidity/composite
  skills above (Skills 3–5) to read trend, recent swings, and current
  volatility, and matches the per-sample lookback the Skill 6 forecaster
  uses at inference time. Only pull a longer window when a skill explicitly
  needs it — e.g. the 6-month backtest in Skill 8, or training/retraining
  the Skill 6 model (which needs years of history) — those are separate,
  explicitly-invoked actions with their own larger budget, not the default
  per-decision fetch.
- **Journal**: don't feed the agent its last N raw journal entries for
  context. Maintain a rolling **journal summary file** (one doc, updated
  after each session) that captures the state that actually matters —
  open positions, recent signal changes, what worked/didn't — and pass
  *that* summary in, not the full entry history. Only pull a specific full
  entry if the agent needs to check an exact past decision.
- **Tool call budget**: cap tool calls per agent run (mirrors Claude Code's
  own per-session tool-call cap) so a scanner pass (Wick) or a multi-analyst
  Alpha run can't spiral into an unbounded research loop across tickers.
  Set the cap per task type (e.g. single-ticker report card vs. full-market
  scan) rather than one global number, since a scan over many tickers
  legitimately needs more calls than a single report card.

---

## 13. Safety Boundary — Execution

All skills in this document are analysis/notification only. Actual broker order placement or Polymarket
buy/sell (item 7 of the original `trading` note) requires an explicit human
confirmation step in the app before any order is sent — the agent should
never auto-execute a trade. This mirrors the standing rule that financial
transactions and personalized investment advice are not something the
assistant performs autonomously.
