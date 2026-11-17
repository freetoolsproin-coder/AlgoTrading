# NIFTY & SENSEX pre-breakout CE/PE scanner

One Python application that watches the **NIFTY 50** and **SENSEX** indices for pre-breakout setups, confirms
breakouts and breakdowns on completed 5-minute bars, picks a real **CE** or **PE** option contract for each
confirmed setup, and sends alerts. It paper-trades those setups and can mirror one model's trades to a broker.

Three scoring models run side by side on the same levels and breakout tests, each with its own paper book:

| Model | What it is | Trades / alerts |
|---|---|---|
| **v1** | the original 8-family score (location, level, compression, structure, trend, momentum, participation, context) | main model: pushes alerts |
| v2 | 5-minute context + 1-minute confirmation, 7 components | shadow, dashboard only |
| **v3** | 7 independent strategy modules + market regime + false-breakout / trap engine + derivatives intelligence | shadow, dashboard only (`[SCORE_V3] push`) |

**Paper trading by default.** Orders go through a simulated broker unless live trading is armed for the day (see
*Live trading* below). Scores are setup-strength scores, not probabilities, and no backtest so far has shown any of
the models to be profitable after costs.

## Supported instruments

| Index | Upstox index key | Options | Option segment |
|---|---|---|---|
| NIFTY 50 | `NSE_INDEX\|Nifty 50` | NIFTY CE, NIFTY PE | NSE_FO |
| SENSEX | `BSE_INDEX\|SENSEX` | SENSEX CE, SENSEX PE | BSE_FO |

Defined once, in `src/config/instruments.py`. Every other index, and all individual stock F&O, is rejected
(config, mock data, backtest, instrument master). Expiries, lot sizes and strike steps are read from the Upstox
instrument master and option chain every day; they are not hard-coded.

## Requirements

- Windows 10/11 (also runs on macOS and Linux)
- Python 3.10 or newer (3.12 recommended; your `.venv` is 3.12)
- An Upstox account with an API app, for live data. Mock mode needs nothing.

## Installation

From the project folder (`D:\trade_Prash`), in a VS Code terminal or any terminal:

```powershell
python -m venv .venv                      # skip if .venv already exists
.venv\Scripts\activate                    # macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `requests`, `websockets`, `protobuf`. Redis, the data hub and Streamlit are no
longer needed.

## Configuration

### Secrets: `.env`

Copy `.env.example` to `.env` and fill it in. `.env` is git-ignored. Environment variables with the same names
win over `.env`.

| Variable | Needed for |
|---|---|
| `UPSTOX_API_KEY`, `UPSTOX_API_SECRET`, `UPSTOX_REDIRECT_URI` | the daily login (`python -m src.broker.auth`) |
| `UPSTOX_ACCESS_TOKEN` | optional; used instead of `upstox_token.json` |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | optional; Telegram alerts |

Tokens and keys are never written to logs; the logger also masks anything that looks like a token.

### Daily Upstox token

Upstox tokens expire every day. Each morning run **Upstox login (daily token)** in VS Code (or
`python -m src.broker.auth`), open the printed URL, log in, and paste the redirected URL back. The token is saved
to `upstox_token.json`; a running scanner picks it up within a minute. Without a token the scanner keeps
running and says so on the dashboard and in the log.

### Scanner settings: `config/scanner.ini`

Everything tunable is in this one file: data-quality limits, timeframes, levels, scoring weights, thresholds,
confirmation rules, trade plan, option filters, risk limits, notification toggles, the dashboard port and
brokerage/charges. Strategy and risk values are reloaded within one cycle when you save the file (invalid values
are rejected and the previous settings kept). `[SERVICE] indices` and `[DASHBOARD]` changes need a restart.

Other files in `config/`:
- `nse_holidays.csv`: exchange holidays. Add next year's list when it is published.
- `events.csv`: RBI policy, budget and similar events. New entries are blocked from 30 minutes before to 30
  minutes after the event (or the whole day if no time is given).

## Running

One command starts everything (market data, WebSocket, scanner, notifications, dashboard) in one process:

```powershell
python -m src.main               # live
python -m src.main --mock        # synthetic data, no credentials, 60x clock (a session takes ~6 minutes)
python -m src.main --once        # one cycle, print a summary, exit
python -m src.main --help        # all options (--no-dashboard, --no-websocket, --speed, --debug, ...)
```

It does not open terminals, command prompts or browser tabs. Open the dashboard yourself:
**http://127.0.0.1:8502** (JSON at `/api/state`, health check at `/health`). The same page is also written to
`output/prebreakout_dashboard.html` every cycle.

Stop it with **Ctrl+C** (or the stop button in your IDE). Shutdown finishes the running cycle, closes the
WebSocket, stops the dashboard, sends queued notifications (5 s max), flushes the logs and releases the lock.
Starting a second copy while one is running is refused, so there is never a duplicate scanner, WebSocket
connection or dashboard.

### VS Code (F5)

1. Open the `D:\trade_Prash` folder in VS Code (File > Open Folder).
2. Once, in the VS Code terminal: `python -m src.tools.setup_vscode`. It copies the run configurations from
   `ide/vscode/` into `.vscode/` (the migration below does this too).
3. Install the **Python** extension (Microsoft) if VS Code asks. The interpreter is preset to `.venv` in
   `.vscode/settings.json`; if it is not picked up, use *Python: Select Interpreter* and choose `.venv`.
4. Open **Run and Debug** (Ctrl+Shift+D), pick a configuration and press **F5**:

| Configuration | What it runs |
|---|---|
| Backtest v3 vs v1 (NIFTY, research report) | `python -m src.backtest.v3_report` |
| Live trading: status / ARM for today / disarm | `python -m src.execution.live_gate --status / --arm / --disarm` |
| Scanner (live) | `python -m src.main` |
| Scanner (mock data, no credentials) | `python -m src.main --mock` |
| Scanner: one cycle and exit | `python -m src.main --once` |
| Upstox login (daily token) | `python -m src.broker.auth` |
| Backtest NIFTY / Backtest SENSEX | `python -m src.backtest --instrument ...` |
| Train ML filter | `python -m src.scanner.ml_filter --train` |
| Tests | `python -m unittest discover -s tests -t . -v` |

Terminal > Run Task offers *Install / update dependencies*, *Run tests* and *Migrate old layout (dry run)*.
The Testing panel (beaker icon) discovers the unit tests too.

### PyCharm or any other IDE

Create a Python run configuration with **Module name** `src.main` (not a script path), **working directory**
the project folder, and the `.venv` interpreter. Add `--mock` as a parameter for mock mode. The same works for
`src.backtest`, `src.broker.auth` and `unittest`.

## How it works

| Step | Module |
|---|---|
| Upstox REST client, token, instrument master | `src/broker/client.py` |
| Market data (candles, quotes, option chain, VIX, breadth) | `src/broker/market_data.py` |
| WebSocket last prices (indices, futures, VIX) | `src/broker/websocket_feed.py` |
| Synthetic market for `--mock` | `src/broker/mock_data.py` |
| Indicators: EMA, RSI, MACD, ADX, ATR, Bollinger, VWAP, swings | `src/scanner/indicators.py` |
| **Index direction**: levels, 0-100 CE / PE scores, breakout tests, underlying plan | `src/scanner/strategies.py` |
| **CE / PE signals**: WATCH, CONFIRMED, ENTRY SETUP, INVALIDATED, MISSED | `src/scanner/signals.py` |
| Score v2 (shadow comparison) and ML filter | `src/scanner/score_v2.py`, `src/scanner/ml_filter.py` |
| **Option contract selection** and premium plan | `src/options/chain.py`, `src/options/selection.py` |
| Risk limits, session gates, lot sizing | `src/risk/manager.py`, `src/risk/position_sizing.py` |
| Paper positions and charges | `src/execution/paper_trading.py`, `src/execution/costs.py` |
| One evaluation cycle; storage | `src/scanner/engine.py`, `src/scanner/store.py` |
| Telegram, Windows popup, CSV | `src/notifications/service.py` |
| Dashboard server and page | `src/dashboard/app.py`, `src/dashboard/view.py` |
| Lifecycle (start once, Ctrl+C, shutdown) | `src/app.py`, entry point `src/main.py` |

The strategy (levels, score families, stages, confirmation quality, option filters, R:R after costs, risk
rules) is unchanged from the previous pre-breakout service; a side-by-side run on identical data produced the
same scores and the same alerts. Index spot has no volume, so the near-month index futures supply volume,
VWAP and OI. Missing data is shown as missing and that score family is marked "no data"; nothing is estimated
except option greeks absent from the feed, which are Black-Scholes estimates labelled "estimated".

## The v3 multi-factor layer

| Step | Module |
|---|---|
| Extended causal features: Keltner squeeze, NR4 / NR7, inside bars, ATR / realised-vol percentiles, ROC, persistence, VWAP bands / acceptance / anchored VWAP, BOS / CHoCH, equal highs / lows, liquidity sweeps, initial balance, 5 / 15 / 30-min opening ranges, 15 / 30-min trend | `src/scanner/features_ext.py` |
| Market regime: strong bull / strong bear / sideways / compression / expansion / high-vol reversal / low-vol range / uncertain | `src/scanner/regime.py` |
| Strategy modules 1-7: volatility compression, market structure, momentum, VWAP & acceptance, opening range, liquidity sweep & reversal, derivatives - each with CE / PE confidence, evidence and an invalidation level | `src/scanner/modules.py` |
| Derivatives intelligence from the live chain: PCR, OI walls, OI added today, ATM IV and its 30-minute change, IV percentile (after 20 sessions of history) | `src/options/derivatives.py` |
| False-breakout / trap engine: no volume, weak momentum, immediate rejection, into the next level, sweep against, failed VWAP, IV spike, extended, poor R:R, illiquid options, divergence | `src/scanner/traps.py` |
| v3 score: spec weights 20 / 15 / 15 / 15 / 10 / 15 / 10, adapted to the regime, minus trap points; regime "uncertain", a strong opposite trend or too many traps veto the entry | `src/scanner/score_v3.py` |

v3 scores sit on a lower scale than v1, so `[SCORE_V3]` carries its own thresholds. They come from
`python -m src.backtest.v3_report`, which fits them on the first 70% of the history only (the same exceedance
rate as v1's thresholds) and judges every v3 variant on the remaining 30%. Any strategy setting can be overridden
for the v3 lane in `[SCORE_V3]` (for example `exit_profile = ladder`, `entry_mode = early`).

In the backtest there is no option chain and no futures volume, so the Derivatives and Participation families are
"no data" there (the score is rescaled to the families that have data) and VWAP is an index TWAP proxy. Live, all
seven families are scored.

## Trade management

- **Entry modes** (`[CONFIRMATION] entry_mode`): `close` (confirmed close beyond the level), `retest` (pull-back
  to the level), `early` (before the break: score >= `early_min_score`, within `early_max_distance_atr` of the
  level, a close beyond the previous bar's high / low; sized at `early_risk_mult` of the normal risk, invalidated
  by its stop).
- **Stop models** (`[TRADE_PLAN] stop_model`): `level` (default: the broken level - `sl_atr` x ATR), `swing`
  (beyond the last confirmed swing), `atr` (entry - `atr_stop_mult` x ATR). Every stop is capped at `max_sl_atr`.
  The premium stop is the option repriced with delta + gamma - theta at the underlying stop, floored at
  `max_premium_loss_pct`.
- **Exit profiles** (`exit_profile`): `classic` (default: half at T1 with breakeven, rest at T2) or `ladder`
  (book `ladder_book_pct` % of the original quantity at each `ladder_r` multiple of risk - default 25% at 1R,
  25% at 1.5R, the rest trails). After the first scale-out: `trail_mode` = `atr` / `swing` trailing on the
  underlying, `vwap_exit`, `momentum_exit`. Time stop and square-off always apply.

## Paper trading and risk

- **Paper trading mode** (`[SERVICE] paper_trading = true`, the default): every ENTRY SETUP opens a simulated
  position, bought at the ask plus slippage and sold at the bid minus slippage, net of brokerage, STT, exchange,
  SEBI, stamp duty and GST.
- **Alert-only mode** (`paper_trading = false`): the same alerts, no positions.
- Live order execution does not exist in this code, in either mode.
- All risk settings are in `[RISK]` and `[TRADE_PLAN]`: risk per trade (whole lots; a trade is skipped when one
  lot risks too much), maximum daily loss, trades per index and per session, maximum open positions, same-side
  exposure, losing-streak stop, premium stop cap, targets, minimum R:R, time stop, square-off, expiry-day
  cutoff, gap and VIX gates.
- Portfolio limits (`[RISK]`, 0 = off): `max_daily_loss_pct` (the tighter of it and `max_daily_loss_rupees`
  applies), `max_weekly_loss_pct` (Monday to now, realised + open), `max_open_premium_pct`, `max_lots_per_expiry`,
  and `correlated_risk_mult` - NIFTY and SENSEX positions on the same side are treated as one correlated bet.
  Sizing is `floor(risk budget / risk per lot) x lot size`, then reduced (never increased) to fit these limits.
  There is no martingale, no averaging down and no size increase after a loss.
- **Kill switch:** create an empty file named `KILL_SWITCH` in the project folder to block new entries and
  square off open paper positions at the bid (and every live position, when live trading is armed). Delete it
  to resume.

## Order execution and live trading

`[EXECUTION]` mirrors one lane's paper trades (`lane = v1 | v2 | v3`) to a broker through
`src/execution/order_manager.py`. The strategy never talks to a broker; brokers implement
`src/execution/broker_api.py` (`paper_broker.py`, `upstox_broker.py`), so another broker is one new class.

| `mode` | What happens |
|---|---|
| `off` | paper book only |
| `paper` (default) | every paper decision goes through the full order path to a simulated broker: validation, journal (`data/execution.db`), partial fills, repricing, resting stop, reconciliation |
| `live` | real Upstox orders - only while armed for today; otherwise it behaves exactly like `paper` and the dashboard says why |

Every live entry is checked first: data freshness, quote age, F&O contract key and expiry, market session and last
entry time, available margin, `max_live_lots`, stop validity, position limits, duplicate positions / orders, broker
connectivity, daily order count. Entries are LIMIT orders that are cancelled (never chased) if not filled in
`order_timeout_seconds`; exits are LIMIT orders re-priced to the bid, then MARKET with exchange protection. A
protective SL order rests at the broker after each fill and moves with the paper stop. Every order has an
idempotency key and broker tag, so a timeout never causes a duplicate order. Broker positions are reconciled
every `reconcile_seconds`; a mismatch blocks new live entries until it is resolved.

**Live-trading activation (deliberate, every day):**
1. Run on paper first and compare the lanes over several weeks; check `output/v3_backtest_report.html`.
2. `config/scanner.ini`: `[EXECUTION] mode = live`, choose `lane`, keep `max_live_lots = 1` to start.
3. `.env`: `LIVE_TRADING_CONFIRM=I ACCEPT REAL-MONEY RISK`.
4. Each morning after the Upstox login: VS Code *Live trading: ARM for today* (or
   `python -m src.execution.live_gate --arm`). It checks the token, funds and open positions, shows the limits and
   asks you to type `ARM <date>`. The arm expires at the end of the day.
5. `python -m src.execution.live_gate --status` shows what is missing; `--disarm` stops live orders at once.
   Mock mode never trades live.
- Mock-mode trades are stored separately and never mix with live paper results.

## Movement triggers (5 / 10 / 15 minutes) - Flask service

A separate web service watches NIFTY, SENSEX and India VIX for fast moves and fires alerts. It runs beside the
scanner (it does not need it) and never places orders.

    python -m src.movement            live Upstox data; page at http://127.0.0.1:8503
    python -m src.movement --mock     synthetic data, no credentials (add --notify to also send alerts)

VS Code: *Movement triggers (live, Flask :8503)* or *Movement triggers (mock data)*.

For every window (5, 10, 15 min by default) it shows the move in points and %, the range, **strength** (move /
(1-min ATR x sqrt(window)), about 1 is an ordinary move), futures **RVOL**, **consistency** (share of candles closing
in the move's direction) and the **futures OI change** read as long / short build-up, short covering or long
unwinding, plus RSI(1m), futures vs VWAP and the day's change.

A trigger fires when a move reaches `move_pct` (or `move_points_<INDEX>`) for its window. It is **CONFIRMED** when
every switched-on check (`min_strength`, `min_rvol`, `min_consistency`, `oi_confirm`) passes, **STRONG** when it is
also `strong_mult` x the threshold, otherwise **UNCONFIRMED**. An extra **ALL** trigger fires when every window moves
the same way; India VIX has its own `vix_move_pct`. Each index / window / direction then waits `cooldown_minutes`.
No window is measured across the overnight gap, and stale data never triggers.

| Technology | Used for |
|---|---|
| Flask + Waitress | web page and JSON API (`/api/state`, `/api/events`, `/api/config`, `/health`) |
| Server-Sent Events | `/api/stream` pushes every cycle and each trigger to the page as it happens |
| SQLite | every trigger in `data/movement.db` (also `output/movement_alerts.csv`) |
| Telegram / Windows popup | the existing notifier, from `push_min_quality` up |
| Webhook | `MOVEMENT_WEBHOOK_URL` in `.env`: each pushed trigger POSTed as JSON (n8n, Zapier, your own bot) |

### Paper trading on the triggers: momentum and scalp books

Each qualifying trigger opens a **simulated** trade: UP buys a CE, DOWN buys a PE. No orders are ever placed.

| Book | Opens on | Defaults |
|---|---|---|
| momentum `[PAPER_MOMENTUM]` | the 5 / 10 / 15-min and ALL triggers above | stop 20 %, target 40 %, breakeven at +15 %, trail 10 %, 30 min max |
| scalp `[PAPER_SCALP]` | its own 1 / 3-min triggers (`trigger_*` keys) | stop 8 %, target 12 %, breakeven at +5 %, trail 4 %, exit if the index gives back half the move, 8 min max |

The contract comes from the live option chain with the `[OPTIONS]` filters (delta 0.35-0.70, premium, spread,
volume, OI). Entry at the fresh ask + slippage, exits at the bid - slippage, net of `[COSTS]`. Open trades are
marked on a fresh option quote every cycle and are never exited on a missing quote. Each book has one position per
index, a daily trade cap, a daily loss limit (realised + open) that closes the book for the day, a wait after each
trade (longer after a loss), `last_entry` / `square_off` times, an opposite-trigger exit and the `KILL_SWITCH` file.
Trades are stored in `data/movement.db` and restored after a restart.

Every entry and exit shows a banner with a sound on the page (green CALL BUY, red PUT BUY), a Windows popup
(`trade_popup` in `[MOVEMENT]`), a browser notification after **Enable browser alerts**, and Telegram / the webhook
when configured. Trades can be closed from the page. All values are untested starting points.

Defaults are in `[MOVEMENT]` of `config/scanner.ini`. Rules changed on the page are saved in
`data/movement_rules.json` and apply on the next cycle; **Reset to ini** removes them. Rule changes are accepted from
this computer only, or from anywhere with `MOVEMENT_API_TOKEN` (sent as `X-API-Token`). The thresholds are
starting hypotheses, not tested values.

### Backtesting the two books

    python -m src.movement.backtest --instrument NIFTY                 both books, 2023-01 to today
    python -m src.movement.backtest --instrument SENSEX --book scalp
    python -m src.movement.backtest --instrument NIFTY --set PAPER_SCALP.stop_pct=10 --set MOVEMENT.min_strength=2

Every minute of the cached 1-minute history (`data/history/`, older NIFTY / VIX in `orb_data/`; `--download` fetches
what is missing with an Upstox token) goes through the live `MovementMonitor` and `PaperTrader` code, so cooldowns,
entry blocks and exits are exactly the service's. Between minutes open trades are also marked on the candle's
adverse and then favourable extreme (a candle touching both stop and target counts as the stop). `--set` changes
any ini value for that run only; nothing in `config/scanner.ini` is changed. Months run in parallel processes
(about 20 minutes per index for 2023-2026 on 8 cores).

The report (`output/movement_backtest_<INDEX>.txt`, trades in `..._trades.csv`) gives per book: trades, win rate,
average win / loss, profit factor, net P&L after costs, max drawdown, results by year, entry window, side, quality,
exit reason and hour, and the first 70 % / last 30 % of the trading days as an out-of-sample check.

**Approximations:** there is no historical option data, so premiums are **modelled** (Black-Scholes on the index,
IV = India VIX, nearest weekly expiry, strike by the `[OPTIONS]` delta / premium filters, `assumed_spread_pct`
spread, `slippage_pct` on both legs, `[COSTS]` charges). No IV smile, no IV jump on fast moves, no real spreads.
Index spot history has no futures volume, so the **rvol check is switched off** in the backtest (with it on, every
trigger would be UNCONFIRMED and nothing would trade; `--keep-rvol` shows that). Treat the results as a filter for
bad settings, not as an estimate of live P&L.

### Strategy library: 54 strategies, each its own paper book

`src/movement/strategies.py` (engine + the first 15) and `src/movement/library.py` (the rest), run by the same
service on COMPLETED bars. A signal is an **event** on the last finished bar (a break, cross or flip), fired once
per bar, never on a forming bar or stale data. UP buys a CE and DOWN buys a PE in that strategy's own paper book,
with the same contract choice, fills, charges and safety rules as above.

```powershell
python -m src.movement.strategies --list      # every strategy: category, CE rule, PE rule, data, timeframe, exits
```

| Category | Key | Strategy | CE when (PE = the mirror) | Data |
|---|---|---|---|---|
| Trend following | `ema_pullback` | EMA Pullback | EMA20 > EMA50, pull-back to EMA20, close above the previous high | spot index bars |
| Trend following | `adx_trend` | ADX Trend Following | +DI over -DI, ADX >= adx_min and rising, above EMA50 | spot index bars |
| Trend following | `supertrend` | Supertrend | Supertrend flips up | spot index bars |
| Trend following | `ma_crossover` | Moving Average Crossover | SMA fast crosses above SMA slow | spot index bars |
| Trend following | `ema_9_21` | EMA 9/21 | EMA9 crosses above EMA21, close above EMA9 | spot index bars |
| Trend following | `ema_20_50_200` | EMA 20/50/200 | EMA20 > EMA50 > EMA200 and close > EMA20 becomes true | spot index bars (205+ bars of the timeframe) |
| Trend following | `donchian` | Donchian Channel Breakout | first close above the previous N-bar high | spot index bars |
| Trend following | `ichimoku` | Ichimoku Cloud | Tenkan crosses above Kijun, close above the cloud | spot index bars |
| Trend following | `psar` | Parabolic SAR | SAR flips below price | spot index bars |
| Momentum | `breakout_momentum` | Breakout Momentum | close above the N-bar high on an impulse bar (body, RVOL or range/ATR, RSI) | spot bars (+ futures volume if any) |
| Momentum | `rsi_macd_divergence` | RSI + MACD Divergence | bullish RSI divergence + MACD histogram turning up | spot index bars |
| Momentum | `first_hour` | First-Hour Momentum | after 10:15, first close above the first-hour high of an up first hour | spot index bars |
| Momentum | `rsi_reversal` | RSI Reversal | RSI crosses back above oversold + close above the previous high | spot index bars |
| Momentum | `macd_crossover` | MACD Crossover | MACD crosses above signal (below zero if zero_filter = 1) | spot index bars |
| Momentum | `stochastic` | Stochastic Oscillator | %K crosses above %D from the oversold zone | spot index bars |
| Momentum | `roc` | Rate of Change | N-bar ROC rises through +threshold | spot index bars |
| Momentum | `cci` | CCI Momentum | CCI rises through +level | spot index bars |
| Momentum | `williams_r` | Williams %R | %R rises through the upper level, close above EMA20 | spot index bars |
| Price action | `vwap_reversal` | VWAP Reversal | futures stretched below VWAP, reversal bar | index futures bars with volume |
| Price action | `orb` | Opening Range Breakout | first close above the opening range | spot index bars |
| Price action | `sr_breakout` | Support / Resistance Breakout | close breaks a resistance tested min_touches times | spot index bars |
| Price action | `pdh_pdl_breakout` | Previous Day High/Low Breakout | first close above yesterday's high (+buffer) | spot index bars + previous day candle |
| Price action | `inside_bar` | Inside Bar Breakout | close above the mother bar after an inside bar | spot index bars |
| Price action | `engulfing` | Engulfing Candlestick | bullish engulfing after an N-bar decline | spot index bars |
| Price action | `pin_bar` | Pin Bar Reversal | long lower wick rejecting a new N-bar low | spot index bars |
| Price action | `vwap_breakout` | VWAP Breakout | futures close crosses above VWAP with rvol >= min_rvol | index futures bars with volume |
| Volatility | `atr_breakout` | ATR Volatility Breakout | first close above open + k x daily ATR | spot index bars + daily candles |
| Volatility | `bollinger_breakout` | Bollinger Bands Breakout | first close above the upper band, bands widening | spot index bars |
| Volatility | `bollinger_squeeze` | Bollinger Squeeze | bands leave the Keltner channel upward after a squeeze | spot index bars |
| Volatility | `keltner` | Keltner Channel | first close above the upper channel | spot index bars |
| Volatility | `volatility_expansion` | Volatility Expansion | wide bar closing above a narrowest-of-N bar | spot index bars |
| Volatility | `india_vix` | India VIX Strategy | VIX down >= min_change_pct in `minutes` and a new N-bar high | spot index bars + India VIX |
| Mean reversion | `bollinger_reversion` | Bollinger Band Mean Reversion | close back inside the lower band in a range | spot index bars |
| Mean reversion | `rsi_mean_reversion` | RSI Mean Reversion | RSI(2) leaves the oversold extreme in a range (ADX low) | spot index bars |
| Mean reversion | `vwap_mean_reversion` | VWAP Mean Reversion | futures back above -z sd from VWAP | index futures bars with volume |
| Mean reversion | `zscore_reversion` | Z-Score Reversion | z-score crosses back above -z_entry | spot index bars |
| Mean reversion | `ma_reversion` | Moving Average Reversion | stretched below SMA by stretch_atr ATR, then a close above the previous high | spot index bars |
| Combination | `volume_profile` | Volume Profile + S/R | rejection of VAL / PDL or acceptance above VAH | index futures bars with volume |
| Combination | `mtf_confirmation` | Multi-Timeframe Confirmation | 15-min up-trend + 5-min trigger + 1-min momentum | spot index bars |
| Combination | `supertrend_rsi` | Supertrend + RSI | Supertrend up and RSI >= rsi_min (newly) | spot index bars |
| Combination | `macd_adx` | MACD + ADX Confirmation | MACD cross up with ADX >= adx_min and +DI > -DI | spot index bars |
| Combination | `vwap_ema` | VWAP + EMA Confluence | futures above VWAP and EMA9 > EMA21 (newly both) | index futures bars with volume |
| Combination | `fibonacci` | Fibonacci Retracement | up swing, pull-back into the 50-61.8% zone, close above the previous high | spot index bars |
| Combination | `pivot_points` | Pivot Point Trading | breakout: first close above R1 / bounce: rejection of S1 | spot index bars + previous day candle |
| Combination | `cpr_breakout` | CPR Breakout | narrow CPR day, first close above the CPR top | spot index bars + previous day candle |
| Two indices | `relative_strength` | Relative Strength Rotation | this index outperforms the other by min_edge_pct and makes a new N-bar high | NIFTY and SENSEX bars |
| Two indices | `pair_trading` | Pair Trading | log-price spread vs the other index crosses back above -z (this index cheap) | NIFTY and SENSEX bars |
| Two indices | `stat_arb` | Statistical Arbitrage | rolling-OLS residual z crosses back above -z, half-life <= max_half_life | NIFTY and SENSEX bars |
| Options | `oi_buildup` | OI Buildup | long build-up (futures price and OI up) | index futures bars with OI |
| Options | `pcr` | PCR Strategy | PCR up by min_change to >= bull_level, price above EMA20 | live option chain OI |
| Options | `premium_momentum` | Options Premium Momentum | ATM CE premium up and PE down by set % in N min | live option chain premiums |
| Options | `gamma_iv` | Gamma and IV Options Strategy | net gamma (CE gamma x OI - PE gamma x OI) negative, ATM IV rising, new N-bar high | live option chain with gamma and OI |
| AI / ensemble | `ensemble` | Multi-Indicator Ensemble | net bullish vote of 7 indicators reaches min_votes | spot index bars |
| AI / ensemble | `ml_signal` | Machine Learning Signal Classification | model P(up move) crosses its threshold (0 = the model's own, set on the training period) | spot index bars + a trained model (python -m src.movement.ml_signal --train) |

- **Timeframes:** `[STRATEGIES] timeframe` (1, 3, 5, 10, 15, 30 or 60 min) is the default; any strategy can set its
  own `timeframe` in `[STRATEGY_<KEY>]`. Daily / weekly bars are not used for entries: every book is intraday and
  squared off the same day (daily candles feed pivots, CPR, previous-day levels and ATR).
- **Stop-loss, target, trailing stop, R:R, limits:** premium % per book - `stop_pct`, `target_pct`, `breakeven_pct`,
  `trail_pct`, `max_hold_minutes`, `max_trades`, `max_daily_loss`, `reentry_minutes`, `last_entry`, `square_off` -
  shared in `[STRATEGIES]`, overridable per strategy. R:R = target / stop (shown by `--list` and on `/strategies`).
- **Data:** strategies that need futures volume / OI (VWAP family, volume profile, OI build-up) say "no data" for
  SENSEX, whose futures report none. PCR, premium momentum, India VIX and gamma / IV build their own history after
  start-up. The two-index strategies need NIFTY and SENSEX both enabled; the two indices move almost together, so
  their spreads are small next to option costs.
- **No look-ahead:** `tests/test_library.py` runs every strategy at several times of day with and without the data
  after that time and requires identical answers.
- Every parameter is an untested starting value. Nothing here is known to be profitable.

#### Machine-learning signal (`ml_signal`) and the ML page

```powershell
python -m src.movement.ml_signal --train --instrument NIFTY      # 5-min bars, 6-bar horizon, last 30% held out
python -m src.movement.ml_signal --status                        # trained models and their test-period report
```

Two L2-regularised logistic regressions (numpy only) predict a move of `min_move_atr` x ATR up / down within
`horizon` bars of the same session, from 18 features of completed bars. The split is chronological: the latest
30% of days never touch training and give the reported AUC / precision; the firing threshold is set on the end of
the training period. The saved model (`data/ml_signal_<INDEX>_<tf>m.json`) remembers its last training day and the
strategy gives **no signal on any day the model trained on**, so its backtest is always out-of-sample.

Flask (same service, http://127.0.0.1:8503): `/strategies` (library with live results), `/ml` (train from the
browser, test-period report, live P(up) / P(down)), and the JSON endpoints `GET /api/strategies`,
`GET /api/ml/models`, `POST /api/ml/train` (this computer only, or `X-API-Token`), `GET /api/ml/jobs/<id>`,
`GET /api/ml/predict`.

#### Backtesting the strategies

```powershell
python -m src.movement.backtest --instrument NIFTY --strategies all
python -m src.movement.backtest --instrument NIFTY --strategies supertrend,ema_9_21 --set STRATEGY_SUPERTREND.mult=2
```

Every minute of history goes through the live monitor, strategy engine and paper trader (modelled option prices,
spread, slippage, charges). `output/strategy_backtest_<INDEX>.txt` has a summary ranked on the out-of-sample 30%,
then per strategy: in-sample / out-of-sample rows, gross profit / loss, avg win / avg loss, consecutive wins and
losses, Sharpe and Sortino of daily P&L, Monte Carlo drawdown, and splits by side, exit reason and year;
`..._trades.csv` has each trade with the running equity of its book. Strategies that cannot be tested on index
history (futures volume / OI, real chain OI, two indices) are listed with the reason.

## Trading console (web UI) - `python -m src.console`

A signed-in web console over everything above: **http://127.0.0.1:8510**. It runs beside the scanner and the movement
service (it does not need them running) and **never places, modifies or cancels orders**.

```powershell
python -m src.console --create-admin      # once: first admin (password typed in the terminal)
python -m src.console                     # start; VS Code: "Trading console (web, http://127.0.0.1:8510)"
python -m src.console --reset-password U  # forgotten password / unlock
python -m src.console --verify-audit      # check the audit log's hash chain
```

| Page | What it shows |
|---|---|
| Overview | live, paper and backtest results side by side, **never added together**; today's risk usage, open positions, alerts, orders |
| Performance / Trade blotter | net after charges, win rate, profit factor, expectancy, drawdown, losing run, charges and slippage, Sharpe / Sortino / VaR (only with 20+ trading days), bootstrap interval (30+ trades), equity and daily charts, by book / month / exit / side |
| Paper trading | every paper book (scanner v1 / v2 / v3, movement and strategy books) and how fills are simulated |
| Backtests | result sets from `output/*_trades.csv`, the HTML reports (sandboxed), and running a backtest (admin) |
| Risk & safety | every configured limit, today's usage per lane, charges, historical drawdown / VaR per book; **engage the kill switch** (any user), release it (admin, typed confirmation), **disarm** live trading |
| Broker monitor | Upstox profile, funds, positions, order book and fills, read with GET requests only; the order engine's journal |
| System health | scanner heartbeat, market data, WebSocket, token expiry, broker API, services on 8502 / 8503, databases, disk, recent log errors |
| Audit log / Reports | hash-chained append-only log of sign-ins, safety actions, credential / user changes, backtests and downloads; CSV / JSON exports and a printable performance report |
| Credentials (admin) | API keys in an encrypted vault (Fernet; key protected by Windows DPAPI or `CONSOLE_MASTER_KEY`). `get_secret` reads environment, then `.env`, then the vault |
| ML lab | the ML trade filter's walk-forward results: the four approval checks (pass / fail), hit rates and mean R of kept vs dropped setups, calibration chart, strongest features, live training samples; retrain it in the background (admin) |
| AI assistant | ask questions in plain language; Claude (`ANTHROPIC_API_KEY`, `[CONSOLE] ai_model`) answers only from read-only lookups into the console's own calculations, names the data source, never sums overlapping books, gives no trading advice; questions and token use are audited |
| Docs, FAQ, Risk disclosure | operating guide, data definitions, fees, risk controls, live safeguards, security; every user accepts the disclosure once |

**Data sources.** *Live* = fills at a real broker from `data/execution.db` (positions whose broker is not the paper
broker). *Paper* = `data/prebreakout.db` and `data/movement.db`. *Backtest* = `output/*_trades.csv`. Databases are
opened read-only; mock-mode records are excluded; missing data is shown as missing, never estimated.

**Security.** PBKDF2-SHA256 passwords, optional TOTP two-factor (`require_2fa_for_admin`), lockout after repeated
failures, idle timeout, CSRF tokens on every form, strict Content-Security-Policy, HttpOnly SameSite=Strict cookies.
Admins manage settings, credentials, users and backtests; viewers read everything and can use the two safe actions.
Arming live trading stays a terminal step (`python -m src.execution.live_gate --arm`); `LIVE_TRADING_CONFIRM` can
only be set in `.env`. It binds to 127.0.0.1; for phone / remote use put it behind a VPN or HTTPS reverse proxy and set
`secure_cookies = true` in `[CONSOLE]`.

## Outputs

| Path | Content |
|---|---|
| `data/prebreakout.db` | alerts, paper trades, bar scores, ML samples |
| `data/console/` | console users, audit log, encrypted vault and its key, session key (back it up with `data/`) |
| `output/prebreakout_alerts.csv` | every alert |
| `output/prebreakout_state.json`, `output/prebreakout_dashboard.html` | latest dashboard state |
| `output/prebreakout_backtest_report[_SENSEX].html` | backtest report |
| `logs/scanner.log` | one log for the whole application, rotated daily, 14 days kept |
| `data/cache/`, `data/history/` | instrument master / constituents cache, backtest history |

## Backtest and ML filter

```powershell
python -m src.backtest                      # NIFTY from data/history (downloads missing months; needs a token)
python -m src.backtest --instrument SENSEX
python -m src.scanner.ml_filter --train     # walk-forward; it may block entries only if it beats the rules
```

Options in the backtest are modelled (Black-Scholes with IV = India VIX); no historical option quotes exist.
The last NIFTY results (Oct 2026) were roughly break-even after costs: treat the alerts as a watchlist.

## Moving from the old setup (one time)

The old programs (Slope engine, data hub, tick streamer, market analysis, swing, ORB, regime, stock F&O
breakout, stock options, the phone control server and every `.bat` launcher including `start_all.bat`) are
not part of this application. To move them aside and carry your history over:

1. Close every window `start_all.bat` opened.
2. `python -m src.tools.migrate_layout` shows what it will do; add `--apply` to do it. It copies the Upstox
   and Telegram secrets from `config.ini` into `.env` (only missing keys, values never printed), moves
   `prebreakout.db` and the ML model into `data/`, copies NIFTY / SENSEX / VIX history into `data/history/`,
   installs the VS Code run configurations, and moves everything else into `legacy/`. Nothing is deleted.
3. `legacy/config.ini` still has the old secret and Telegram token in plain text: delete it once `.env`
   works, and regenerate both credentials.

## Testing

```powershell
python -m unittest discover -s tests -t . -v
```

No network or credentials are needed; Upstox, the WebSocket and Telegram are replaced by fakes. Tests write
only to a temporary folder (`SCANNER_RUNTIME_DIR`), so they can run while the scanner is running.

| File | Covers |
|---|---|
| `tests/test_instruments.py` | only NIFTY / SENSEX; other indices and stock F&O rejected (removed); CE and PE selectable for both; no leftover references |
| `tests/test_strategy.py` | indicators, no look-ahead, stages, options, risk, paper trading, alerts, mock sessions, restart without duplicate alerts, kill switch |
| `tests/test_broker.py` | REST client (401, 429, rate limit, bad keys, v3 endpoints), token handling, market data for both indices, WebSocket parse / subscribe / reconnect / stop |
| `tests/test_app.py` | single entry point, Ctrl+C and cancellation shutdown, no duplicate instance or service, dashboard routes, log redaction |
| `tests/test_backtest.py`, `tests/test_migration.py` | backtest end to end for both indices; the layout migration |
| `tests/test_v3.py` | v3 features (no look-ahead), regime, modules, traps, derivatives, v3 score and veto, stop models, early entries, exit ladder / trailing, portfolio risk, order manager (idempotency, validation, partial fills, repricing, stop, reconciliation, emergency close, journal), Upstox order requests, live gate, engine wiring |
| `tests/test_console.py` | console: password hashing, TOTP (RFC 6238), lockout, CSRF, roles, idle timeout, risk-disclosure gate, vault encryption and tamper detection, audit hash chain, metrics, live / paper / backtest separation, mock exclusion, read-only databases and GET-only broker access, kill switch / disarm, no order code, CSV formula escaping |
| `tests/test_movement_backtest.py` | movement-books backtest: replay through the live monitor / paper trader, modelled entry price, no look-ahead, rvol switch, statistics, 70 / 30 split |
| `tests/test_strategies.py`, `tests/test_library.py` | every strategy: metadata, no look-ahead (with / without future data), rules on built scenarios, per-strategy timeframe, once-per-bar evaluation; ML labels within a session, chronological split, refusal on training days; Flask `/api/strategies` and `/api/ml/*`; per-strategy backtest and its statistics |

## Known limitations

- The backtest models options with Black-Scholes and IV = India VIX; there are no historical option prices, bid/ask
  or chains, so the derivatives module and option liquidity filters are untested historically.
- The cached history has no futures volume / OI: participation and true VWAP are scored live only.
- Only NIFTY history is cached; SENSEX results need SENSEX history (`python -m src.backtest --instrument SENSEX`
  downloads it with a token).
- Regime thresholds, module weights and trap points are research starting values, not fitted ones.
- Event risk uses `config/events.csv` only; if it has no entries for the coming days, event filtering is simply
  not active (the scanner does not invent events).
- The Upstox order endpoints follow the Upstox developer documentation at the time of writing; verify them before
  arming live trading.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No Upstox access token` / `401` in the log | Run *Upstox login (daily token)*; the scanner picks the token up by itself. |
| `Missing in .env: UPSTOX_API_KEY ...` | Fill in `.env` (copy `.env.example`), or run the migration step above. |
| `Another scanner is already running` | A copy is already running (check other terminals). The lock clears automatically when it exits. |
| `Dashboard not started: ... in use` | Another program uses port 8502: change `[DASHBOARD] port`. The scanner keeps running. |
| `ModuleNotFoundError: No module named 'src'` | Run from the project folder with `python -m src.main` (not `python src/main.py`). |
| `ModuleNotFoundError: websockets` (or pandas ...) | `python -m pip install -r requirements.txt` with the `.venv` interpreter. |
| Index shows NO TRADE "data stale" | Check the internet connection and the token; it recovers by itself and sends a "fresh again" alert. |
| No Telegram messages | Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`; check `[NOTIFY]`. The log says why at start-up. |
| `config/scanner.ini ... invalid` | The message names the setting; the previous settings stay active until it is fixed. |

## NIFTY 50 stock options (CE / PE plans + research report)

A separate service for the 50 NIFTY stocks; the NIFTY / SENSEX index scanner above is unchanged.

    python -m src.stocks            live Upstox data (after python -m src.broker.auth), http://127.0.0.1:8504
    python -m src.stocks --mock     synthetic market, no token, any time of day
    python -m src.stocks --once     one scan printed in the terminal

* **Signals page** (`/`): every stock gets VALID / WATCH / NO TRADE. A plan has the option contract (strike with
  delta near 0.55, nearest monthly expiry with at least 3 days left), the premium to buy at, stop-loss, Target 1,
  Target 2, the matching spot levels, lot size, capital and rupee risk per lot. Click a row for every check.
* **Validity**: the stop sits behind the last 3 bars / Supertrend (0.6 to 1.8 ATR on 15-minute bars); targets are
  1.5 R and 2.5 R pulled in before the nearest resistance or support (previous-day high/low, pivots, OI walls,
  20-day and 52-week extremes); T1 must fit the range the stock can still move today; reward:risk must stay
  >= 1.2 on the premium after theta and charges; the strike must be liquid (OI, spread <= 5%); entries only
  09:30-14:30, square-off 15:15.
* **Paper book**: a VALID plan opens one paper trade per stock per day; after T1 the stop moves to cost.
  Saved in data/stocks/book_<date>.json.
* **Search bar**: type a symbol or company name; the report shows price and 52-week range, returns vs NIFTY,
  volatility and beta, daily trend and momentum, pivots and swing levels, PCR / max pain / OI walls / ATM IV,
  today's option plan and a bullish / bearish / neutral verdict with its reasons. Any NSE stock can be searched
  in live mode (options only for the NIFTY 50). Optional fundamentals: `python -m pip install yfinance`.
* Settings: `[STOCK_OPTIONS]` in config/scanner.ini. Tests: `python -m unittest tests.test_stocks`.
