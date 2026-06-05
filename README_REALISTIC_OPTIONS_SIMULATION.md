# Realistic Options Simulation Upgrade

This version keeps the prior tuned weekly strategy and weekly reinvestment simulator, but replaces the simple long-call multiplier with a more realistic long-call proxy.

## What changed

The old options simulator estimated option gains mostly by multiplying the stock move. This version now subtracts estimated frictions:

- delta-based option exposure
- bid/ask spread cost
- theta decay by days held
- IV crush on winning/up moves
- liquidity/slippage penalty
- max option premium loss cap
- 8% underlying emergency stop

It still does **not** use real historical option-chain pricing. It is a conservative proxy meant to avoid fantasy option profits.

## Main command

```bash
python main.py simulate-weekly-strategy --universe custom --years 2 --strategy all
```

## Realistic options strategy name

```bash
python main.py simulate-weekly-strategy --universe custom --years 2 --strategy tuned_with_realistic_long_calls
```

The old strategy name still works:

```bash
python main.py simulate-weekly-strategy --universe custom --years 2 --strategy tuned_with_long_calls
```

but it now uses the more realistic option proxy.

## Reinvestment strategy

```bash
python main.py simulate-weekly-strategy --universe custom --years 2 --strategy tuned_reinvest_weekly
```

## Output files

```text
data/strategy_simulation_summary.csv
data/strategy_simulation_trades.csv
data/strategy_simulation_weekly_equity.csv
```

## Option proxy assumptions

Defaults are inside `backtests/weekly_strategy_simulator.py`:

```text
option_delta_proxy = 0.55
option_leverage_cap = 4.0
option_spread_cost_pct = 6.0
option_theta_decay_pct_per_day = 3.0
option_iv_crush_pct = 6.0
option_liquidity_slippage_pct = 2.0
option_max_loss_pct = 60.0
option_stop_loss_pct = 60.0
option_min_profit_target_pct = 12.0
```

These are deliberately conservative compared with the earlier simple proxy.
