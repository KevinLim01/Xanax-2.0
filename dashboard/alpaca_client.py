from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
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


def _data_url() -> str:
    return _secret("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets").rstrip("/")


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


def _data_get(path: str, params: dict[str, Any] | None = None) -> Any:
    url = f"{_data_url()}{path}"
    response = requests.get(url, headers=_headers(), params=params or {}, timeout=30)
    response.raise_for_status()
    return response.json()


def _to_utc_iso(value: datetime | str) -> str:
    if isinstance(value, str):
        dt = pd.to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        else:
            dt = dt.tz_convert("UTC")
        return dt.isoformat().replace("+00:00", "Z")

    dt = value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    return dt.isoformat().replace("+00:00", "Z")


def _safe_datetime(value: Any) -> datetime | None:
    try:
        dt = pd.to_datetime(value)
        if pd.isna(dt):
            return None
        return dt.to_pydatetime()
    except Exception:
        return None


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
            dt = _safe_datetime(item.get("transaction_time"))
            if dt is None:
                continue

            rows.append(
                {
                    "time": dt,
                    "symbol": str(item.get("symbol", "")).upper(),
                    "side": str(item.get("side", "")).lower(),
                    "qty": float(item.get("qty") or 0),
                    "price": float(item.get("price") or 0),
                }
            )
        except Exception:
            continue

    return pd.DataFrame(rows)


def get_ticker_price_path(
    symbol: str,
    start_time: datetime | str,
    end_time: datetime | str | None = None,
    timeframe: str = "5Min",
    feed: str = "iex",
) -> pd.DataFrame:
    """
    Gets actual stock price movement while a trade was open.

    This is what the graph should use as the main line/candles.
    Alpaca trade activities should only be used as BUY/SELL markers.
    """

    symbol = str(symbol or "").upper().strip()

    if not symbol:
        return pd.DataFrame(
            columns=["time", "symbol", "open", "high", "low", "close", "volume"]
        )

    if end_time is None:
        end_time = datetime.now(timezone.utc)

    params = {
        "timeframe": timeframe,
        "start": _to_utc_iso(start_time),
        "end": _to_utc_iso(end_time),
        "adjustment": "raw",
        "feed": feed,
        "limit": 10000,
    }

    try:
        data = _data_get(f"/v2/stocks/{symbol}/bars", params)
    except Exception:
        return pd.DataFrame(
            columns=["time", "symbol", "open", "high", "low", "close", "volume"]
        )

    bars = data.get("bars") or []

    rows = []

    for bar in bars:
        try:
            dt = _safe_datetime(bar.get("t"))
            if dt is None:
                continue

            rows.append(
                {
                    "time": dt,
                    "symbol": symbol,
                    "open": float(bar.get("o") or 0),
                    "high": float(bar.get("h") or 0),
                    "low": float(bar.get("l") or 0),
                    "close": float(bar.get("c") or 0),
                    "volume": float(bar.get("v") or 0),
                }
            )
        except Exception:
            continue

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(
            columns=["time", "symbol", "open", "high", "low", "close", "volume"]
        )

    return df.sort_values("time").reset_index(drop=True)


def infer_trade_windows(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Turns raw BUY/SELL fills into open trade windows.

    Example:
    BUY BKNG Monday 10:02
    SELL BKNG Wednesday 11:14

    Result:
    BKNG open_time=Monday 10:02, close_time=Wednesday 11:14
    """

    if trades.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "open_time",
                "close_time",
                "entry_price",
                "exit_price",
                "entry_qty",
                "exit_qty",
                "is_open",
            ]
        )

    required = {"time", "symbol", "side", "qty", "price"}
    if not required.issubset(set(trades.columns)):
        return pd.DataFrame(
            columns=[
                "symbol",
                "open_time",
                "close_time",
                "entry_price",
                "exit_price",
                "entry_qty",
                "exit_qty",
                "is_open",
            ]
        )

    rows = []
    open_lots: dict[str, dict[str, Any]] = {}

    df = trades.copy()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["side"] = df["side"].astype(str).str.lower()
    df = df.sort_values("time")

    for _, trade in df.iterrows():
        symbol = str(trade["symbol"]).upper()
        side = str(trade["side"]).lower()
        qty = float(trade.get("qty") or 0)
        price = float(trade.get("price") or 0)
        time = trade.get("time")

        if not symbol or qty <= 0 or price <= 0:
            continue

        if side == "buy":
            if symbol not in open_lots:
                open_lots[symbol] = {
                    "symbol": symbol,
                    "open_time": time,
                    "entry_price": price,
                    "entry_qty": qty,
                    "remaining_qty": qty,
                }
            else:
                lot = open_lots[symbol]
                old_qty = float(lot["remaining_qty"])
                new_qty = old_qty + qty

                lot["entry_price"] = (
                    float(lot["entry_price"]) * old_qty + price * qty
                ) / new_qty
                lot["entry_qty"] = float(lot["entry_qty"]) + qty
                lot["remaining_qty"] = new_qty

        elif side == "sell":
            if symbol not in open_lots:
                continue

            lot = open_lots[symbol]
            remaining = float(lot["remaining_qty"])
            sold_qty = min(qty, remaining)
            new_remaining = remaining - sold_qty

            if new_remaining <= 0.000001:
                rows.append(
                    {
                        "symbol": symbol,
                        "open_time": lot["open_time"],
                        "close_time": time,
                        "entry_price": float(lot["entry_price"]),
                        "exit_price": price,
                        "entry_qty": float(lot["entry_qty"]),
                        "exit_qty": qty,
                        "is_open": False,
                    }
                )
                del open_lots[symbol]
            else:
                lot["remaining_qty"] = new_remaining

    for symbol, lot in open_lots.items():
        rows.append(
            {
                "symbol": symbol,
                "open_time": lot["open_time"],
                "close_time": None,
                "entry_price": float(lot["entry_price"]),
                "exit_price": None,
                "entry_qty": float(lot["entry_qty"]),
                "exit_qty": None,
                "is_open": True,
            }
        )

    return pd.DataFrame(rows)


def get_trade_price_path(
    symbol: str,
    open_time: datetime | str,
    close_time: datetime | str | None = None,
    timeframe: str = "5Min",
) -> pd.DataFrame:
    """
    Convenience wrapper for dashboard charting.

    Use this when the dashboard already knows the trade window.
    """

    if close_time is None:
        close_time = datetime.now(timezone.utc)

    return get_ticker_price_path(
        symbol=symbol,
        start_time=open_time,
        end_time=close_time,
        timeframe=timeframe,
    )


def build_trade_chart_data(
    symbol: str,
    trades: pd.DataFrame,
    open_time: datetime | str,
    close_time: datetime | str | None = None,
    timeframe: str = "5Min",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
    1. price_path_df = actual stock movement
    2. marker_df = BUY/SELL markers

    The dashboard should graph price_path_df as the main chart,
    then overlay marker_df.
    """

    symbol = str(symbol or "").upper().strip()

    price_path = get_trade_price_path(
        symbol=symbol,
        open_time=open_time,
        close_time=close_time,
        timeframe=timeframe,
    )

    if trades.empty:
        markers = pd.DataFrame(columns=["time", "symbol", "side", "qty", "price"])
    else:
        markers = trades.copy()
        markers["symbol"] = markers["symbol"].astype(str).str.upper()
        markers = markers[markers["symbol"] == symbol].copy()

        start_dt = pd.to_datetime(open_time)
        end_dt = pd.to_datetime(close_time or datetime.now(timezone.utc))

        markers["time"] = pd.to_datetime(markers["time"])
        markers = markers[
            (markers["time"] >= start_dt)
            & (markers["time"] <= end_dt)
        ].copy()

        markers = markers.sort_values("time").reset_index(drop=True)

    return price_path, markers


def filter_trades_for_chart(trades: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """
    Old portfolio-equity marker helper.

    Keep this for the existing dashboard.
    New ticker-specific charts should use:
    - get_ticker_price_path()
    - build_trade_chart_data()
    """

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
