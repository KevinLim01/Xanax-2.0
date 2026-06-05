# Options History Dataset Builder

This adds a research dataset for deciding whether the model should ever use long-call options.

It does **not** use real historical option-chain data. It creates a conservative options proxy using the underlying stock move and subtracts realistic estimated costs:

- delta-based exposure
- bid/ask spread cost
- theta decay
- IV crush
- liquidity/slippage penalty
- option premium stop
- 8% underlying emergency stop

## Run

```bash
python main.py build-options-history-dataset --universe custom --years 2
```

For a longer test:

```bash
python main.py build-options-history-dataset --universe custom --years 5
```

## Outputs

```text
data/options_history_raw.csv
data/options_history_summary.csv
data/options_history_ticker_summary.csv
```

## Main file to inspect

Use:

```text
data/options_history_summary.csv
```

Look for rows where:

```text
recommended_option_allowed = True
option_type = LONG_CALL_PROXY
setup_type = MOMENTUM_CONTINUATION
direction = UP
sample_size >= 50
option_profitable_rate >= 55
average_net_option_return_pct >= 5
```

If there are few or no rows like that, options should stay disabled.

## What this is for

This creates the dataset needed for a future:

```text
app/agents/options_history_agent.py
```

That agent would act as a gatekeeper:

```text
Use stock by default.
Only allow long calls if the options history dataset says this exact setup historically worked after costs.
```
