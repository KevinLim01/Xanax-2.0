from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests


def _secret(name: str, default: str = "") -> str:
    val = os.getenv(name)
    if val:
        return str(val).strip()

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
    response = requests.get(url, headers=_headers(), params=params or {}, timeout=30)
    response.raise_for_status()
    return response.json()


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

    rows = []

    for ts, eq in zip(timestamps, equity):
        try:
            rows.append(
                {
                    "time": datetime.fromtimestamp(int(ts)),
                    "equity": float(eq),
                }
            )
        except Exception:
            continue

    return pd.DataFrame(rows)


def get_trade_activities(days_back: int = 400) -> pd.DataFrame:
    after = (datetime.utcnow() - timedelta(days=days_back)).isoformat() + "Z"

    try:
        data = _get(
            "/v2/account/activities/FILL",
            {
                "after": after,
                "direction": "asc",
                "page_size": 100,
            },
        )
    except Exception:
        return pd.DataFrame(columns=["time", "symbol", "side", "qty", "price"])

    rows = []

    for item in data:
        try:
            rows.append(
                {
                    "time": pd.to_datetime(item.get("transaction_time")).to_pydatetime(),
                    "symbol": str(item.get("symbol", "")).upper(),
                    "side": str(item.get("side", "")).lower(),
                    "qty": float(item.get("qty") or 0),
                    "price": float(item.get("price") or 0),
                }
            )
        except Exception:
            continue

    return pd.DataFrame(rows)


def filter_trades_for_chart(trades: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or hist.empty:
        return pd.DataFrame(columns=["time", "symbol", "side", "qty", "price", "equity"])

    start = hist["time"].min()
    end = hist["time"].max()

    trades = trades[(trades["time"] >= start) & (trades["time"] <= end)].copy()

    if trades.empty:
        return pd.DataFrame(columns=["time", "symbol", "side", "qty", "price", "equity"])

    hist_sorted = hist.sort_values("time")[["time", "equity"]].copy()
    trades_sorted = trades.sort_values("time").copy()

    merged = pd.merge_asof(
        trades_sorted,
        hist_sorted,
        on="time",
        direction="nearest",
    )

    return merged


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
