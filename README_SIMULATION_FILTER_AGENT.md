# Simulation Filter Agent

This version keeps the no-weekend Xanax model and adds a simulation-backed guardrail.

## What it adds

- `app/agents/simulation_filter_agent.py`
- `backtests/simulation_live_filter_builder.py`
- `data/xanax_no_weekend_2y_simulation_trades.csv`
- `data/simulation_live_filter_summary.csv`
- `data/simulation_live_filter_config.json`
- `.github/workflows/build-simulation-live-filter.yml`

## What it does

The agent reads `data/simulation_live_filter_summary.csv` and checks the current ticker/setup/day against the two-year no-weekend simulation.

It can:

- boost conviction when a setup/day/ticker profile worked well in the simulation
- reduce conviction when the profile was weak
- block a trade if the simulation profile was bad enough

It is a guardrail, not the main model.

## Rebuild the CSV

```bash
python main.py build-simulation-live-filter \
  --input data/xanax_no_weekend_2y_simulation_trades.csv \
  --output data/simulation_live_filter_summary.csv \
  --config-output data/simulation_live_filter_config.json \
  --min-sample-size 20 \
  --max-adjustment 10
```

## Live settings

```txt
SIMULATION_FILTER_ENABLED=true
SIMULATION_FILTER_BLOCK_ENABLED=true
SIMULATION_FILTER_SUMMARY_PATH=data/simulation_live_filter_summary.csv
SIMULATION_FILTER_MIN_SAMPLE_SIZE=20
SIMULATION_FILTER_MAX_ADJUSTMENT=10
SIMULATION_FILTER_HIGH_CONVICTION_OVERRIDE=90
```

## Notes

This version does not include the Friday-to-Monday weekend agent.
