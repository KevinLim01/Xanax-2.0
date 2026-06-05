# Tuned Long/UP + Options Model

This update keeps the Monday + Tuesday + Wednesday buying schedule, but changes the entry gates to match the simulation results:

- Strongly favors `BUY / UP` trades.
- Prioritizes `MOMENTUM_CONTINUATION`.
- Allows `UP_OPPORTUNITY` only when it is top-tier.
- Allows shorts only in extreme `BREAKDOWN_CONTINUATION` cases.
- Keeps history-based exits and the 8% emergency stop.
- Adds optional long-call support for high-quality `BUY / UP / MOMENTUM_CONTINUATION` trades.

## Live workflow behavior

- Monday: up to 15 trades.
- Tuesday: up to 7 strict second-chance trades.
- Wednesday: up to 7 strict second-chance trades.
- Thursday/Friday: monitor/sell only.

## Options

Options are blocked by default unless you turn them on in GitHub Variables or `.env`:

```env
AUTO_TRADE_INSTRUMENT=both
AUTO_TRADE_ALLOW_OPTIONS=true
OPTIONS_LONG_UP_ONLY=true
OPTIONS_LONG_UP_MIN_CONVICTION=78
OPTIONS_LONG_UP_MIN_HISTORY_RATE=72.0
OPTIONS_LONG_UP_MIN_HISTORY_SAMPLE_SIZE=100
OPTIONS_LONG_UP_ALLOWED_SETUP=MOMENTUM_CONTINUATION
AUTO_TRADE_OPTIONS_CONTRACTS_PER_TRADE=1
AUTO_TRADE_OPTIONS_MIN_DTE=7
AUTO_TRADE_OPTIONS_MAX_DTE=21
AUTO_TRADE_OPTIONS_MAX_CONTRACT_PRICE=5.00
```

In `both` mode, qualified long/up momentum signals prefer a long call instead of also buying stock. This avoids doubling the same thesis.

## Simulation

Run:

```bash
python main.py simulate-weekly-strategy --universe custom --years 2 --strategy all
```

The simulator now includes:

- `monday_only`
- `monday_tuesday`
- `monday_tuesday_wednesday`
- `tuned_with_long_calls`

The options simulation uses a simple long-call proxy. It is useful for comparing rough behavior, not for exact option pricing.

Outputs:

```text
data/strategy_simulation_summary.csv
data/strategy_simulation_trades.csv
```
