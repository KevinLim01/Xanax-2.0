# XANAX Manual Paper Trading Dashboard

This is the website/dashboard wrapper around your existing Xanax model.

It keeps the model code intact and adds:

- clean net-worth graph, with no right-side numbers and no grid rows
- always-available **Run Monitor Check** button
- **Run Scan** button that runs the 15 custom ticker chunks locally/server-side
- **Execute Top Trades** button that reads `scan_results/` and sends orders through Alpaca
- latest scan results table
- open positions table
- recent command logs

## Run locally

```bash
cd xanax-dashboard
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill in .env with your real Alpaca + Gemini keys
streamlit run dashboard_app.py
```

## What the buttons do

### Run Scan
Runs this for each chunk:

```bash
python main.py trade-screen --universe custom --chunk-index 0 --chunk-count 15 --recommend-only --second-chance --output scan_results/chunk_0.json
```

Then repeats for chunk 1 through 14.

### Execute Top Trades
Runs:

```bash
python main.py execute-ranked-trades --input-dir scan_results --max-trade-slots 5 --second-chance
```

Turn on **Dry run** in the sidebar to test without buying.

### Run Monitor Check
Always available. Runs:

```bash
python main.py monitor-positions
```

Turn on **Dry run** to test without closing positions.

## Secrets

Do not commit `.env`.

For local runs, put real values in `.env`.

For Streamlit/Render/Railway, put the same keys in the host's environment variables/secrets page.

Required:

```env
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
ALPACA_BASE_URL=https://paper-api.alpaca.markets
GEMINI_API_KEY=
ALPACA_TRADING_MODE=paper
ALLOW_LIVE_TRADING=false
AUTO_TRADE_ENABLED=true
```

## GitHub Actions

This dashboard does not need GitHub Actions to run scans or monitor checks. GitHub can just store the code.
