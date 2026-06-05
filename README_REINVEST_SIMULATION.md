# Weekly Reinvestment Simulation Update

This update adds a new strategy simulation mode:

```bash
python main.py simulate-weekly-strategy --universe custom --years 2 --strategy tuned_reinvest_weekly
```

It keeps the tuned weekly strategy rules:

- Monday buying enabled, up to 15 trades
- Tuesday buying enabled, up to 7 trades
- Wednesday buying enabled, up to 7 trades
- Thursday/Friday are sell-only in the simulated logic
- LONG/UP momentum setups are favored
- Shorts are allowed only in extreme cases
- Weak UP_OPPORTUNITY and BREAKDOWN_CONTINUATION setups are blocked unless top-tier
- History-based exits are used
- 8% emergency stop is kept
- Long-call proxy options are allowed for strong LONG/UP momentum setups

New compounding behavior:

- Starts with $5,000 simulated capital
- Uses about 6% of current capital per position, which equals $300 at the starting balance
- At the end of each simulated week, realized weekly profit/loss is added to next week's capital
- If a week makes $50, the next week simulates with $5,050 capital
- If a week loses $50, the next week simulates with $4,950 capital

The simulator writes three CSV files:

```text
data/strategy_simulation_summary.csv
data/strategy_simulation_trades.csv
data/strategy_simulation_weekly_equity.csv
```

For comparison across all strategies, run:

```bash
python main.py simulate-weekly-strategy --universe custom --years 2 --strategy all
```
