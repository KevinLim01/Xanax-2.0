# stock_signal_app_weekly_opportunity

This version changes the model target from **Monday reference → Friday close** to **weekly trade opportunity**.

The model now asks:

```text
Will this stock give a meaningful UP or DOWN move at any point during the week?
```

It does **not** treat Friday close as the main truth anymore. Friday/held result is secondary.

## Main changes

- Trains on weekly high/low opportunity from the Monday reference price.
- Uses `OPPORTUNITY_THRESHOLD_PCT=1.0` by default.
- Adds ticker archetypes:
  - `MOMENTUM_HIGH_BETA`
  - `DEFENSIVE_STABLE`
  - `FINANCIAL`
  - `CYCLICAL`
  - `MEGA_CAP_TECH`
  - `NORMAL`
- Blocks aggressive high-conviction SELL calls on TSLA/NVDA/PLTR-style names unless bearish confirmation is strong enough.
- Caps weak mean-reversion short conviction.
- Adds setup types, including:
  - `MOMENTUM_CONTINUATION`
  - `MEAN_REVERSION_SHORT`
  - `BREAKDOWN_CONTINUATION`
  - `OVERSOLD_BOUNCE`
- Adds weekly-opportunity metadata to every output:
  - `target_type`
  - `success_threshold_pct`
  - `ticker_archetype`
  - `setup_type`
  - `momentum_continuation_score`
  - `bearish_confirmations`
  - `bearish_confirmation_details`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python main.py init-db
```

## Run one ticker

```bash
python main.py run --ticker TSLA
python main.py debug-run --ticker NVDA
```

## Screen a universe

```bash
python main.py screen --universe top50
python main.py screen --universe custom
python main.py screen --universe top50 --limit 15
python main.py screen --universe custom --all
```

## Interpretation

- `forecast_direction` = expected weekly opportunity direction.
- `final_action` = whether the setup is tradeable enough: BUY / SELL / WATCH.
- `expected_move_pct` = estimated best favorable weekly opportunity move, not Friday-close return.
- `conviction_score` = opportunity conviction, not a true probability.
- `momentum_continuation_score` = how much the setup looks like a continuation runner.
- `bearish_confirmations` = how many bearish checks support a SELL/DOWN call.

## Important scoring logic

For high-beta momentum names such as TSLA/NVDA/PLTR/AMD:

```text
Do not short only because the stock already went up.
SELL/DOWN needs confirmation.
If confirmation is weak, downgrade to WATCH or block high conviction.
```

For slower defensive names, mean-reversion logic still matters more.

## Weekly truth rule for your tracker

Use the app/dashboard truth logic like this:

```text
BUY/UP is true if the stock moves at least +1.0% above the Monday reference at any point during the week.
SELL/DOWN is true if the stock moves at least -1.0% below the Monday reference at any point during the week.
Friday close is only the held result, not the main verdict.
```

Recommended verdict labels:

```text
GOOD_CALL
TRADEABLE_BUT_MESSY
BARELY_TRUE
BAD_CALL
```

The helper module `app/weekly_opportunity.py` contains reusable verdict logic for the tracker/dashboard.

## Model 2.0 Phase 1: Alpaca Paper-Trading Add-On

This version keeps the existing weekly opportunity model and adds an execution layer around it.

The model still decides:

```txt
Ticker
BUY / SELL / WATCH
UP / DOWN / NEUTRAL
Conviction score
Expected move
Estimated edge
Setup type
Reason
```

The new Alpaca add-on decides:

```txt
Whether the signal is eligible to trade
Whether the trade is long or short
How many shares to paper trade
Whether Alpaca allows the asset
Whether a duplicate position already exists
Whether to block the trade for risk reasons
When to exit open positions
```

### Safety Defaults

By default, the add-on is conservative:

```env
ALPACA_TRADING_MODE=paper
ALLOW_LIVE_TRADING=false
AUTO_TRADE_ENABLED=false
```

`AUTO_TRADE_ENABLED=false` means the code can run and log decisions, but it will not send Alpaca orders unless you turn it on.

To enable paper orders after adding your Alpaca paper keys:

```env
AUTO_TRADE_ENABLED=true
ALPACA_TRADING_MODE=paper
ALLOW_LIVE_TRADING=false
```

Do not use live trading until the paper system has been tested for a long period.

### New Files

```txt
app/trading/alpaca_broker.py
app/trading/decision_engine.py
app/trading/risk_manager.py
app/trading/exit_engine.py
app/trading/position_monitor.py
app/trading/trade_logger.py
app/trading/autotrader.py
```

### New Commands

Run the current model for one ticker, then let the add-on decide whether to paper trade it:

```bash
python main.py trade-run --ticker TSLA --dry-run
```

Send the paper order if eligible:

```bash
python main.py trade-run --ticker TSLA
```

Run the existing screen, rank the model's best names, and let the add-on paper trade the eligible ones:

```bash
python main.py trade-screen --universe top50 --limit 25 --dry-run
```

Send eligible paper orders:

```bash
python main.py trade-screen --universe top50 --limit 25
```

Monitor open Alpaca positions and close positions when exit rules hit:

```bash
python main.py monitor-positions --dry-run
```

Actually close positions when exit rules hit:

```bash
python main.py monitor-positions
```

### Current Entry Rules

The first version is simple on purpose:

```txt
BUY + UP + conviction >= AUTO_TRADE_MIN_CONVICTION = long candidate
SELL + DOWN + conviction >= AUTO_TRADE_MIN_CONVICTION = short candidate
WATCH / NEUTRAL = no trade
WEAK edge = no trade when AUTO_TRADE_REQUIRE_MODERATE_EDGE=true
Already holding ticker = blocked
Market closed = blocked when AUTO_TRADE_REQUIRE_MARKET_OPEN=true
Not shortable = blocked for short trades
Too much exposure = blocked
```

### Current Exit Rules

```txt
Take-profit hit = close
Stop-loss hit = close
Friday forced exit time hit = close
Otherwise hold
```

Defaults:

```env
AUTO_TRADE_TAKE_PROFIT_PCT=3.0
AUTO_TRADE_STOP_LOSS_PCT=1.5
AUTO_TRADE_FORCE_EXIT_FRIDAY_HOUR=15
AUTO_TRADE_FORCE_EXIT_FRIDAY_MINUTE=45
```

### Database Additions

The existing SQLite database now also stores:

```txt
trade_decisions
position_checks
```

These tables log every blocked trade, dry-run trade, sent order, position check, and exit decision.

## Continuous Mode

The project now has a continuous paper-trading loop.

It does two jobs:

```txt
1. Monitor open Alpaca positions every few minutes.
2. Run the model screen on a schedule and open new eligible paper trades.
```

This is the command:

```bash
python main.py live-loop --universe top50 --limit 25
```

Safer first test:

```bash
python main.py live-loop --universe top50 --limit 5 --dry-run --cycles 2
```

Use `--cycles` only for testing. If you omit it, the loop keeps running until you stop it with `Ctrl+C`.

### Continuous Mode Settings

Add these to your local `.env`:

```env
LIVE_LOOP_ENABLED=true
LIVE_LOOP_MODEL_SCAN_INTERVAL_MINUTES=60
LIVE_LOOP_POSITION_MONITOR_INTERVAL_MINUTES=5
LIVE_LOOP_NO_NEW_TRADES_AFTER_HOUR=15
LIVE_LOOP_NO_NEW_TRADES_AFTER_MINUTE=0
LIVE_LOOP_MAX_NEW_TRADES_PER_DAY=5
LIVE_LOOP_REQUIRE_MARKET_OPEN=true
```

Recommended starting setup:

```env
ALPACA_TRADING_MODE=paper
ALLOW_LIVE_TRADING=false
AUTO_TRADE_ENABLED=true
LIVE_LOOP_ENABLED=true
AUTO_TRADE_MAX_POSITION_SIZE_USD=100
AUTO_TRADE_MAX_TOTAL_EXPOSURE_USD=500
LIVE_LOOP_MAX_NEW_TRADES_PER_DAY=5
```

### What Continuous Mode Does

```txt
Every 5 minutes:
- checks open Alpaca positions
- exits if take-profit hits
- exits if stop-loss hits
- exits near Friday close

Every 60 minutes:
- runs the current model screen
- ranks the candidates
- sends only eligible paper trades to Alpaca
- blocks duplicates, low-conviction trades, weak-edge trades, unshortable shorts, and over-exposure
```

It does not place new entries after the no-new-trades cutoff time. The default is 3:00 PM.

### Command Examples

Dry-run continuous test:

```bash
python main.py live-loop --universe top50 --limit 5 --dry-run --cycles 2
```

Paper-trading loop:

```bash
python main.py live-loop --universe top50 --limit 25
```

Custom universe loop:

```bash
python main.py live-loop --universe custom --limit 50
```

Faster testing intervals:

```bash
python main.py live-loop --universe top50 --limit 5 --dry-run --cycles 3 --scan-interval-minutes 1 --monitor-interval-minutes 1
```

### Best Practice

Use continuous mode for monitoring, but do not let it overtrade. A good first setup is:

```txt
Model scan: every 60 minutes
Position monitor: every 5 minutes
No new trades after: 3:00 PM
Max new trades per day: 5
Paper trading only
```

## Options Mode: Simple Long Calls/Puts

This version can also paper trade **simple long options** through Alpaca.

It only supports:

```txt
BUY / UP signal  -> buy 1 long call
SELL / DOWN signal -> buy 1 long put
Exit rule hit -> close the option position
```

It deliberately does **not** support short options, naked calls, spreads, or multi-leg strategies yet.

Alpaca's Python SDK supports options contract lookup and options orders, and Alpaca says paper accounts can access Level 3 options strategies, while live options trading requires approval. Keep this project in paper mode while testing. See Alpaca's options docs for account requirements and API behavior.

### Enable Options Paper Trading

In your local `.env`:

```env
ALPACA_TRADING_MODE=paper
ALLOW_LIVE_TRADING=false
AUTO_TRADE_ENABLED=true

AUTO_TRADE_INSTRUMENT=options
AUTO_TRADE_ALLOW_OPTIONS=true
AUTO_TRADE_OPTIONS_CONTRACTS_PER_TRADE=1
AUTO_TRADE_OPTIONS_MIN_DTE=7
AUTO_TRADE_OPTIONS_MAX_DTE=21
AUTO_TRADE_OPTIONS_CALL_STRIKE_OFFSET_PCT=0.0
AUTO_TRADE_OPTIONS_PUT_STRIKE_OFFSET_PCT=0.0
AUTO_TRADE_OPTIONS_CLOSE_EXPIRING_WITHIN_DAYS=2
```

Use stock + options together only after testing options by themselves:

```env
AUTO_TRADE_INSTRUMENT=both
```

### Test Options Without Orders

```bash
python main.py trade-run --ticker TSLA --dry-run
python main.py trade-screen --universe top50 --limit 10 --dry-run
python main.py live-loop --universe top50 --limit 5 --dry-run --cycles 2
```

### Send Paper Options Orders

```bash
python main.py trade-run --ticker TSLA
python main.py trade-screen --universe top50 --limit 10
python main.py live-loop --universe top50 --limit 25
```

### Options Guardrails

```txt
- paper mode only
- long calls and long puts only
- no short options
- no spreads
- no duplicate stock/option position on the same underlying
- exits use the same take-profit, stop-loss, and Friday rules
- extra expiration-risk exit closes options near expiration
```

## Running From GitHub Actions With Secrets

This version can run from GitHub without uploading `.env`.

Use this only for **paper trading** first. GitHub will store the code, while GitHub Secrets provide the keys at runtime.

### 1. Add GitHub Secrets

Go to:

```txt
Your private GitHub repo
→ Settings
→ Secrets and variables
→ Actions
→ New repository secret
```

Add:

```txt
ALPACA_API_KEY
ALPACA_SECRET_KEY
GEMINI_API_KEY   optional, only if you use Gemini text agents
```

Do not add these to `.env.example`, README, or any code file.

### 2. Add GitHub Variables

Go to:

```txt
Settings
→ Secrets and variables
→ Actions
→ Variables
```

Start with these safer values:

```txt
AUTO_TRADE_ENABLED=false
AUTO_TRADE_INSTRUMENT=stock
AUTO_TRADE_ALLOW_OPTIONS=false
AUTO_TRADE_MAX_TRADES_PER_RUN=5
AUTO_TRADE_MAX_POSITION_SIZE_USD=100
AUTO_TRADE_MAX_TOTAL_EXPOSURE_USD=500
AUTO_TRADE_MIN_CONVICTION=60
```

When dry-runs work and you want GitHub to place **paper** orders, change:

```txt
AUTO_TRADE_ENABLED=true
```

Keep this hardcoded in the workflow:

```txt
ALPACA_TRADING_MODE=paper
ALLOW_LIVE_TRADING=false
```

### 3. GitHub Workflows Included

```txt
.github/workflows/paper-trade-scan.yml
.github/workflows/paper-position-monitor.yml
```

The scan workflow runs the model and opens eligible paper trades.

The monitor workflow checks existing paper positions and exits by stop-loss, take-profit, Friday exit, or expiring-options rules.

### 4. Manual Test From GitHub

Open the repo on GitHub:

```txt
Actions
→ Paper trade scan
→ Run workflow
→ dry_run=true
```

Then test the monitor:

```txt
Actions
→ Paper position monitor
→ Run workflow
→ dry_run=true
```

Only after dry-runs look right, set the repo variable:

```txt
AUTO_TRADE_ENABLED=true
```

Then run without dry-run.

### 5. Local `.env` Still Works

Local runs still use your local `.env` file. GitHub Actions ignores `.env` and uses Secrets instead.

```txt
Local Mac run: reads .env
GitHub Actions run: reads GitHub Secrets and Variables
GitHub repo: stores no real keys
```

### 6. Continuous Bot Warning

GitHub Actions is good for scheduled scans and scheduled monitoring. It is not ideal for a nonstop 24/7 trading bot. For true nonstop running, use a VPS later. For now, the included schedule is the safer first step.
