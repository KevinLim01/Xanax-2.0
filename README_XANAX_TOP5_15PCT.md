# Xanax top-5 / 15% paper-trading model

This version is for real Alpaca paper trading, not simulation.

Rules:

- Stock only.
- Long/BUY/UP only.
- No shorts.
- No options.
- Buy only the top 5 ranked/highest-certainty candidates per run.
- Each new position targets 15% of current allowed/reinvested capital.
- Weekly/account reinvestment uses Alpaca paper equity compared with `AUTO_TRADE_STARTING_ACCOUNT_EQUITY_USD`.
- History-based exits, model-rerun exits, low-conviction profitable exits, Friday exits, and 8% emergency stop remain enabled.
- GitHub workflow files are manual-only; no built-in cron schedules.

Important variables:

```env
AUTO_TRADE_INSTRUMENT=stock
AUTO_TRADE_ALLOW_OPTIONS=false
AUTO_TRADE_ALLOW_SHORTS=false
AUTO_TRADE_MAX_TRADES_PER_RUN=5
AUTO_TRADE_REINVEST_ENABLED=true
AUTO_TRADE_BASE_CAPITAL_USD=5000
AUTO_TRADE_STARTING_ACCOUNT_EQUITY_USD=<your current Alpaca paper equity>
AUTO_TRADE_REINVEST_POSITION_FRACTION=0.15
AUTO_TRADE_STOP_LOSS_PCT=8
```
