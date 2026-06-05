# Final Xanax No-Weekend + Simulation Filter Notes

Included updates:

- Top 5 fixed-slot portfolio
- 20% position sizing
- Slot-only buying
- Higher day-based conviction floors
- Already-ran-too-much filter
- Ticker penalty filter
- Gap risk filter
- Soft earnings risk filter
- Profit protection
- Conviction-based exits
- New simulation-backed filter agent
- No weekend / Friday-to-Monday agent

Recommended paper variables:

```txt
AUTO_TRADE_MAX_ACTIVE_POSITIONS=5
AUTO_TRADE_REINVEST_POSITION_FRACTION=0.20
AUTO_TRADE_MAX_TRADES_PER_RUN=5
AUTO_TRADE_MIN_CONVICTION=55
DAY_MIN_CONVICTION_MONDAY=55
DAY_MIN_CONVICTION_TUESDAY=60
DAY_MIN_CONVICTION_WEDNESDAY=65
DAY_MIN_CONVICTION_THURSDAY=70
DAY_MIN_CONVICTION_FRIDAY=999
SIMULATION_FILTER_ENABLED=true
SIMULATION_FILTER_BLOCK_ENABLED=true
SIMULATION_FILTER_MIN_SAMPLE_SIZE=20
SIMULATION_FILTER_MAX_ADJUSTMENT=10
SIMULATION_FILTER_HIGH_CONVICTION_OVERRIDE=90
```

Test first with dry run.
