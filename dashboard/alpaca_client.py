from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import pandas as pd
import requests


def _secret(name: str, default: str = "") -> str:
    # 1. Try normal environment variable first
    val = os.getenv(name)
    if val:
        return str(val).strip()

    # 2. Try Streamlit secrets directly
    try:
        import streamlit as st

        val = st.secrets.get(name, default)
        if val:
            return str(val).strip()
    except Exception:
        pass

    return default


def _base_url() -> str:
    return _secret("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")


def _headers() -> dict[str, str]:
    key = _secret("ALPACA_API_KEY")
    secret = _secret("ALPACA_SECRET_KEY")

    if not key or not secret:
        raise RuntimeError(
            "Missing ALPACA_API_KEY or ALPACA_SECRET_KEY. "
            "Check Streamlit Secrets, then reboot the app."
        )

    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }


def _get(path: str, params: dict[str, Any] | None = None) -> Any:
    url = f"{_base_url()}{path}"
    r = requests.get(url, headers=_headers(), params=params or {}, timeout=30)
    r.raise_for_status()
    return r.json()


def get_account() -> dict[str, Any]:
    return _get("/v2/account")


def get_positions() -> list[dict[str, Any]]:
    return _get("/v2/positions")


def get_portfolio_history(period: str = "1M", timeframe: str = "1D") -> pd.DataFrame:
    data = _get(
        "/v2/account/portfolio/history",
        {
            "period": period,
            "timeframe": timeframe,
            "intraday_reporting": "continuous",
        },
    )

    timestamps = data.get("timestamp") or []
    equity = data.get("equity") or []

    if not timestamps or not equity:
        return pd.DataFrame(columns=["time", "equity"])

    rows = []

    for ts, eq in zip(timestamps, equity):
        try:
            t = datetime.fromtimestamp(int(ts))
            v = float(eq) if eq is not None else None
        except Exception:
            continue

        if v is not None:
            rows.append({"time": t, "equity": v})

    return pd.DataFrame(rows)


def positions_dataframe() -> pd.DataFrame:
    try:
        positions = get_positions()
    except Exception:
        return pd.DataFrame(
            columns=["Ticker", "Qty", "Entry", "Current", "Market Value", "P/L", "P/L %"]
        )

    rows = []

    for p in positions:
        rows.append(
            {
                "Ticker": str(p.get("symbol", "")).upper(),
                "Qty": float(p.get("qty", 0) or 0),
                "Entry": float(p.get("avg_entry_price", 0) or 0),
                "Current": float(p.get("current_price", 0) or 0),
                "Market Value": float(p.get("market_value", 0) or 0),
                "P/L": float(p.get("unrealized_pl", 0) or 0),
                "P/L %": float(p.get("unrealized_plpc", 0) or 0) * 100,
            }
        )

    return pd.DataFrame(rows)
