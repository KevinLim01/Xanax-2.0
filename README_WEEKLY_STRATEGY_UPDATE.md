# Weekly strategy update

This version uses the current history-based monitor and adds a staged weekly entry schedule.

## Live schedule

- Monday: main basket scan. Up to 15 trades.
- Tuesday/Wednesday: second-chance scan. Up to 7 trades, stricter filter only.
- Thursday/Friday: no buying. Monitor/sell only.

## Workflows

- `.github/workflows/paper-trade-scan.yml`
  - Monday main basket.
  - Runs 6 scan chunks.
  - Executes the final ranked list with `--max-trade-slots 15`.

- `.github/workflows/paper-trade-second-chance.yml`
  - Tuesday/Wednesday second-chance entries.
  - Runs 6 scan chunks with `--second-chance`.
  - Executes final ranked list with `--second-chance --max-trade-slots 7`.

- `.github/workflows/paper-position-monitor.yml`
  - Tue-Fri monitor/sell only.
  - Uses model rerun exits, history peak exits, low-conviction profitable exits, 8% emergency stop, and Friday force exit.

## New command-line option

`trade-screen` and `execute-ranked-trades` now support:

```bash
--second-chance
```

This forces stricter Tue/Wed buying rules.

## Second-chance defaults

```env
SECOND_CHANCE_MIN_CONVICTION=70
SECOND_CHANCE_MIN_HISTORY_RATE=70
SECOND_CHANCE_MIN_HISTORY_SAMPLE_SIZE=50
SECOND_CHANCE_MAX_SPREAD_PCT=0.25
SECOND_CHANCE_REQUIRE_INTRADAY_CONFIRMATION=true
SECOND_CHANCE_REQUIRE_MODERATE_EDGE=true
```

## Strategy simulator

New command:

```bash
python main.py simulate-weekly-strategy --universe custom --years 2 --strategy all
```

It compares:

- `monday_only`
- `monday_tuesday`
- `monday_tuesday_wednesday`

Outputs:

```text
data/strategy_simulation_trades.csv
data/strategy_simulation_summary.csv
```

This simulator is price-history based. It does not place orders and does not use Alpaca.
