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
            "pnl_reset": "no_reset",
        },
    )

    timestamps = data.get("timestamp") or []
    equity_values = data.get("equity") or []
    profit_loss_values = data.get("profit_loss") or []

    rows = []

    for ts, eq in zip(timestamps, equity_values):
        try:
            if eq is None:
                continue

            rows.append(
                {
                    "time": datetime.fromtimestamp(int(ts)),
                    "equity": float(eq),
                    "source": "equity",
                }
            )
        except Exception:
            continue

    df = pd.DataFrame(rows)

    if not df.empty and df["equity"].nunique() > 1:
        return df

    if timestamps and profit_loss_values:
        try:
            account = get_account()
            current_equity = float(
                account.get("equity")
                or account.get("portfolio_value")
                or 0
            )

            pnl_clean = []

            for pnl in profit_loss_values:
                try:
                    pnl_clean.append(float(pnl or 0))
                except Exception:
                    pnl_clean.append(0.0)

            if pnl_clean and current_equity:
                latest_pnl = pnl_clean[-1]
                rebuilt_rows = []

                for ts, pnl in zip(timestamps, pnl_clean):
                    rebuilt_rows.append(
                        {
                            "time": datetime.fromtimestamp(int(ts)),
                            "equity": current_equity - latest_pnl + pnl,
                            "source": "profit_loss",
                        }
                    )

                rebuilt_df = pd.DataFrame(rebuilt_rows)

                if not rebuilt_df.empty and rebuilt_df["equity"].nunique() > 1:
                    return rebuilt_df
        except Exception:
            pass

    return pd.DataFrame(columns=["time", "equity", "source"])


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
