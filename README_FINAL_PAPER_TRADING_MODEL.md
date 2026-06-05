# Final Paper Trading Model

This version is for real Alpaca paper trading, not simulation.

## Strategy

- Monday: main basket buying, up to 15 trades.
- Tuesday/Wednesday: strict second-chance buying, up to 7 trades.
- Thursday/Friday: monitor and sell only.
- Long/BUY/UP only.
- No shorts.
- No options.
- History-based exits stay on.
- 8% emergency stop-loss stays on.
- Reinvestment is controlled by the capital allocation agent.

## Reinvestment logic

The bot starts with a base capital amount, usually $5,000.

If Alpaca paper equity increases by $1,000 from the starting equity snapshot, the bot allows about $6,000 total exposure.

Example:

```env
AUTO_TRADE_BASE_CAPITAL_USD=5000
AUTO_TRADE_STARTING_ACCOUNT_EQUITY_USD=100000
```

If current Alpaca paper equity becomes $101,000, the bot uses:

```text
$5,000 base capital + $1,000 paper profit = $6,000 allowed exposure
```

Position size is usually 6% of allowed exposure:

```text
$5,000 × 0.06 = $300
$6,000 × 0.06 = $360
```

## Variables to set in GitHub Variables

Set this to your Alpaca paper equity when you start this final model:

```env
AUTO_TRADE_STARTING_ACCOUNT_EQUITY_USD=YOUR_CURRENT_ALPACA_PAPER_EQUITY
```

Recommended values:

```env
AUTO_TRADE_INSTRUMENT=stock
AUTO_TRADE_ALLOW_OPTIONS=false
AUTO_TRADE_ALLOW_SHORTS=false
AUTO_TRADE_REINVEST_ENABLED=true
AUTO_TRADE_BASE_CAPITAL_USD=5000
AUTO_TRADE_REINVEST_MAX_TOTAL_EXPOSURE_USD=25000
AUTO_TRADE_REINVEST_POSITION_FRACTION=0.06
AUTO_TRADE_REINVEST_MAX_POSITION_SIZE_USD=1200
AUTO_TRADE_STOP_LOSS_PCT=8
MONITOR_USE_MODEL_RERUN=true
MONITOR_USE_HISTORY_EXIT=true
MONITOR_EXIT_ON_SIGNAL_FLIP=true
MONITOR_EXIT_ON_LOW_CONVICTION=true
MONITOR_LOW_CONVICTION_EXIT_THRESHOLD=50
MONITOR_HISTORY_PROFIT_CAPTURE_RATIO=0.85
MONITOR_HISTORY_MIN_SAMPLE_SIZE=50
```

## Workflows

- `.github/workflows/paper-trade-scan.yml` runs the Monday main basket.
- `.github/workflows/paper-trade-second-chance.yml` runs Tuesday/Wednesday strict second-chance buys.
- `.github/workflows/paper-position-monitor.yml` runs Tuesday-Friday monitor/sell logic.

## Safety

If `AUTO_TRADE_STARTING_ACCOUNT_EQUITY_USD` is not set, reinvestment falls back to the fixed base exposure instead of accidentally using the whole Alpaca paper account.
