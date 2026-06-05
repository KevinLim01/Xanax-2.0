# Stock-only Long/Buy Weekly Reinvestment Simulation

This model adds a simulation-only strategy named:

```bash
stock_long_reinvest_weekly
```

It tests the exact idea:

- Buy only LONG / BUY / UP signals.
- No shorts.
- No options.
- Buy Monday, Tuesday, and Wednesday.
- Monday allows up to 15 trades.
- Tuesday allows up to 7 trades.
- Wednesday allows up to 7 trades.
- Thursday and Friday are sell-only.
- Uses history-based exits.
- Keeps the 8% emergency stop.
- Starts with $5,000 simulated capital.
- Reinvests profits weekly. If capital grows from $5,000 to $6,000, the next week simulates using $6,000 as the exposure base.
- Position size is 6% of capital, so it starts near $300 per position and scales up/down with account size.

Run:

```bash
python main.py simulate-weekly-strategy --universe custom --years 2 --strategy stock_long_reinvest_weekly
```

It creates:

```text
data/strategy_simulation_summary.csv
data/strategy_simulation_trades.csv
data/strategy_simulation_weekly_equity.csv
```

Compare it against stock-only non-reinvested:

```bash
python main.py simulate-weekly-strategy --universe custom --years 2 --strategy monday_tuesday_wednesday
```

Or run all strategies:

```bash
python main.py simulate-weekly-strategy --universe custom --years 2 --strategy all
```
