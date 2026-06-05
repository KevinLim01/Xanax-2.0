from __future__ import annotations

import argparse
import csv
from io import StringIO
from pathlib import Path

import requests

URL = "https://datahub.io/core/s-and-p-500-companies/_r/-/data/constituents.csv"

# Yahoo Finance uses hyphen class tickers instead of dot class tickers.
YFINANCE_SYMBOL_FIXES = {
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
}

# Optional practical fixes. Leave this dict small and explicit.
# If a future source update changes these, the source value will still be printed in the CSV.
PRACTICAL_SYMBOL_FIXES = {
    # Uncomment only if your fetcher fails on current source symbols.
    # "FISV": "FI",
    # "MRSH": "MMC",
}


def load_sp500_symbols(yfinance_format: bool = True) -> list[str]:
    response = requests.get(URL, timeout=30)
    response.raise_for_status()

    reader = csv.DictReader(StringIO(response.text))
    symbols: list[str] = []

    for row in reader:
        symbol = (row.get("Symbol") or "").strip().upper()
        if not symbol:
            continue

        if yfinance_format:
            symbol = YFINANCE_SYMBOL_FIXES.get(symbol, symbol)
            symbol = PRACTICAL_SYMBOL_FIXES.get(symbol, symbol)

        symbols.append(symbol)

    seen: set[str] = set()
    deduped: list[str] = []
    for symbol in symbols:
        if symbol not in seen:
            deduped.append(symbol)
            seen.add(symbol)

    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(description="Update stock_signal_app universe with current S&P 500 tickers.")
    parser.add_argument(
        "--output",
        default="data/tickers_sp500.txt",
        help="Output path. Use data/tickers_custom.txt if you want to run with --universe custom.",
    )
    parser.add_argument(
        "--raw-symbols",
        action="store_true",
        help="Keep dot tickers like BRK.B instead of Yahoo-style BRK-B.",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    symbols = load_sp500_symbols(yfinance_format=not args.raw_symbols)
    out_path.write_text("\n".join(symbols) + "\n")

    print(f"Wrote {len(symbols)} symbols to {out_path}")
    print("First 10:", ", ".join(symbols[:10]))
    print("Last 10:", ", ".join(symbols[-10:]))


if __name__ == "__main__":
    main()
