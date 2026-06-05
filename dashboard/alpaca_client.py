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
    dt = pd.to_datetime(value, utc=True)
    return dt.isoformat().replace("+00:00", "Z")


def _safe_datetime(value: Any) -> datetime | None:
    try:
        dt = pd.to_datetime(value, utc=True)
        if pd.isna(dt):
            return None
        return dt.to_pydatetime()
    except Exception:
        return None


def _empty_equity_curve() -> pd.DataFrame:
    return pd.DataFrame(columns=["time", "equity", "source"])


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(columns=["time", "symbol", "side", "qty", "price"])


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
                    "time": pd.to_datetime(int(ts), unit="s", utc=True),
                    "equity": float(eq),
                    "source": "alpaca_equity",
                }
            )
        except Exception:
            continue

    df = pd.DataFrame(rows)

    if not df.empty and df["equity"].nunique() > 1:
        return df.sort_values("time").reset_index(drop=True)

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
                            "time": pd.to_datetime(int(ts), unit="s", utc=True),
                            "equity": current_equity - latest_pnl + pnl,
                            "source": "alpaca_profit_loss",
                        }
                    )

                rebuilt_df = pd.DataFrame(rebuilt_rows)

                if not rebuilt_df.empty and rebuilt_df["equity"].nunique() > 1:
                    return rebuilt_df.sort_values("time").reset_index(drop=True)
        except Exception:
            pass

    return _empty_equity_curve()


def get_trade_activities(days_back: int = 400) -> pd.DataFrame:
    after = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat().replace(
        "+00:00",
        "Z",
    )

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
        return _empty_trades()

    rows = []

    for item in data:
        try:
            dt = _safe_datetime(item.get("transaction_time"))
            if dt is None:
                continue

            rows.append(
                {
                    "time": dt,
                    "symbol": str(item.get("symbol", "")).upper().strip(),
                    "side": str(item.get("side", "")).lower().strip(),
                    "qty": float(item.get("qty") or 0),
                    "price": float(item.get("price") or 0),
                }
            )
        except Exception:
            continue

    if not rows:
        return _empty_trades()

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"])

    return df.sort_values("time").reset_index(drop=True)


def get_ticker_price_path(
    symbol: str,
    start_time: datetime | str,
    end_time: datetime | str | None = None,
    timeframe: str = "5Min",
    feed: str = "iex",
) -> pd.DataFrame:
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

    if not rows:
        return pd.DataFrame(
            columns=["time", "symbol", "open", "high", "low", "close", "volume"]
        )

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"])

    return df.sort_values("time").reset_index(drop=True)


def build_mark_to_market_equity_curve(
    trades: pd.DataFrame,
    current_equity: float,
    days_back: int = 10,
    timeframe: str = "5Min",
    feed: str = "iex",
) -> pd.DataFrame:
    """
    Builds the main dashboard graph from each stock's actual price movement.

    This is the correct graph:
    cash + sum(qty held per ticker * ticker close price at each timestamp)

    This means every held stock is calculated individually, then combined into
    one final portfolio curve.
    """

    if trades is None or trades.empty:
        return _empty_equity_curve()

    now = datetime.now(timezone.utc)
    start_time = now - timedelta(days=int(days_back))

    fills = trades.copy()

    required = {"time", "symbol", "side", "qty", "price"}
    if not required.issubset(set(fills.columns)):
        return _empty_equity_curve()

    fills["time"] = pd.to_datetime(fills["time"], utc=True, errors="coerce")
    fills = fills.dropna(subset=["time"])

    fills["symbol"] = fills["symbol"].astype(str).str.upper().str.strip()
    fills["side"] = fills["side"].astype(str).str.lower().str.strip()
    fills["qty"] = pd.to_numeric(fills["qty"], errors="coerce").fillna(0.0)
    fills["price"] = pd.to_numeric(fills["price"], errors="coerce").fillna(0.0)

    fills = fills[
        (fills["symbol"] != "")
        & (fills["qty"] > 0)
        & (fills["price"] > 0)
        & (fills["side"].isin(["buy", "sell"]))
    ].copy()

    if fills.empty:
        return _empty_equity_curve()

    symbols = sorted(fills["symbol"].unique().tolist())
    bar_frames = []

    for symbol in symbols:
        bars = get_ticker_price_path(
            symbol=symbol,
            start_time=start_time,
            end_time=now,
            timeframe=timeframe,
            feed=feed,
        )

        if bars.empty:
            continue

        bars = bars.copy()
        bars["time"] = pd.to_datetime(bars["time"], utc=True, errors="coerce")
        bars = bars.dropna(subset=["time"])
        bars["close"] = pd.to_numeric(bars["close"], errors="coerce")
        bars = bars.dropna(subset=["close"])
        bars["symbol"] = symbol

        if not bars.empty:
            bar_frames.append(bars[["time", "symbol", "close"]])

    if not bar_frames:
        return _empty_equity_curve()

    bars_all = pd.concat(bar_frames, ignore_index=True)
    bars_all = bars_all.sort_values(["symbol", "time"]).reset_index(drop=True)

    chart_times = sorted(set(bars_all["time"].tolist()))

    if not chart_times:
        return _empty_equity_curve()

    rows = []

    for t in chart_times:
        fills_until_t = fills[fills["time"] <= t]

        cash_flow = 0.0
        holdings_value = 0.0

        if not fills_until_t.empty:
            for _, fill in fills_until_t.iterrows():
                qty = float(fill["qty"])
                price = float(fill["price"])
                side = str(fill["side"])

                if side == "buy":
                    cash_flow -= qty * price
                elif side == "sell":
                    cash_flow += qty * price

            for symbol in symbols:
                symbol_fills = fills_until_t[fills_until_t["symbol"] == symbol]

                if symbol_fills.empty:
                    continue

                bought = symbol_fills[symbol_fills["side"] == "buy"]["qty"].sum()
                sold = symbol_fills[symbol_fills["side"] == "sell"]["qty"].sum()
                qty_held = float(bought - sold)

                if qty_held <= 0:
                    continue

                symbol_prices = bars_all[
                    (bars_all["symbol"] == symbol)
                    & (bars_all["time"] <= t)
                ]

                if symbol_prices.empty:
                    continue

                latest_price = float(symbol_prices["close"].iloc[-1])
                holdings_value += qty_held * latest_price

        rows.append(
            {
                "time": t,
                "cash_flow": cash_flow,
                "holdings_value": holdings_value,
            }
        )

    curve = pd.DataFrame(rows)

    if curve.empty:
        return _empty_equity_curve()

    latest_holdings = float(curve["holdings_value"].iloc[-1])
    latest_cash_flow = float(curve["cash_flow"].iloc[-1])

    cash_offset = float(current_equity or 0) - latest_holdings - latest_cash_flow

    curve["equity"] = cash_offset + curve["cash_flow"] + curve["holdings_value"]
    curve["source"] = "mark_to_market"

    curve = curve[["time", "equity", "source"]].copy()
    curve["time"] = pd.to_datetime(curve["time"], utc=True, errors="coerce")
    curve = curve.dropna(subset=["time"])

    return curve.sort_values("time").reset_index(drop=True)


def filter_trades_for_chart(trades: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    if trades is None or hist is None or trades.empty or hist.empty:
        return pd.DataFrame(columns=["time", "symbol", "side", "qty", "price", "equity"])

    if "time" not in trades.columns or "time" not in hist.columns:
        return pd.DataFrame(columns=["time", "symbol", "side", "qty", "price", "equity"])

    trades = trades.copy()
    hist = hist.copy()

    trades["time"] = pd.to_datetime(trades["time"], utc=True, errors="coerce")
    hist["time"] = pd.to_datetime(hist["time"], utc=True, errors="coerce")

    trades = trades.dropna(subset=["time"])
    hist = hist.dropna(subset=["time"])

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
