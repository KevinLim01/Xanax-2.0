# Xanax updates applied

This package updates the existing Xanax model in-place. It does not add the final Friday-to-Monday/weekend agent phase.

## Entry and allocation updates

- Top-5 concentrated portfolio defaults.
- Fixed max active positions: `AUTO_TRADE_MAX_ACTIVE_POSITIONS=5`.
- Slot-only buying: if 5 positions are already open, new buys are blocked; if 4 are open, only 1 new buy can be approved; if 3 are open, only 2 can be approved.
- Position fraction changed from 15% to 20% by default: `AUTO_TRADE_REINVEST_POSITION_FRACTION=0.20`.
- Normal minimum conviction default raised from 40/60-style behavior to 55.
- Day-adjusted conviction floors added:
  - Monday: 55
  - Tuesday: 60
  - Wednesday: 65
  - Thursday: 70
  - Friday: 999, blocking normal Friday buys by default

## Entry quality filters

- Already-ran-too-much filter: blocks late entries when the recent move is already near the historical average best move.
- Ticker penalty filter: uses `data/history_ticker_summary.csv` to block weaker-conviction trades in tickers with poor success rates or large adverse moves.
- Gap risk filter: supports blocking risky overnight-gap names when gap metrics are present in a candidate row.
- Rank stability filter: added but disabled by default because the current scan output does not yet persist rank-stability counts.
- Soft earnings risk filter: blocks earnings-adjacent trades only when conviction is below the high-conviction threshold; it fails open if earnings dates cannot be fetched.

## Exit updates

- Conviction-based history exits:
  - lower conviction uses 85% of historical average best move
  - mid conviction uses 90%
  - high conviction uses 95%
- Profit protection state file added: `data/profit_protection_state.json`.
- Profit protection exits a winner if it reached the activation level and then falls back to the floor.

## Workflows updated

- `.github/workflows/paper-trade-scan.yml`
- `.github/workflows/paper-trade-second-chance.yml`
- `.github/workflows/paper-position-monitor.yml`

## Not included yet

- Weekend dataset builder.
- Weekend simulation.
- Friday-to-Monday agent.
- Friday workflow / Monday forced weekend exit.
