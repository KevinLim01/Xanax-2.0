from __future__ import annotations

import time
from datetime import timedelta

import pandas as pd
import yfinance as yf

from app.utils import utc_now


def _yahoo_ticker(ticker: str) -> str:
    """
    Yahoo uses '-' for share classes:
    BRK.B -> BRK-B
    BF.B  -> BF-B
    """
    return str(ticker).strip().upper().replace(".", "-")


def _clean_download(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError(f"No market data returned for {ticker}")

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(columns=str.lower)

    if "adj close" in df.columns and "close" not in df.columns:
        df["close"] = df["adj close"]

    df.index = pd.to_datetime(df.index).tz_localize(None)

    needed = {"open", "high", "low", "close"}
    missing = needed - set(df.columns)

    if missing:
        raise ValueError(f"Market data for {ticker} missing columns: {sorted(missing)}")

    return df


def _download_with_retries(
    ticker: str,
    *,
    attempts: int = 3,
    sleep_seconds: float = 1.5,
    **kwargs,
) -> pd.DataFrame:
    """
    yfinance sometimes returns empty data even for valid tickers.
    Retry before treating it as a real failure.
    """
    yahoo_symbol = _yahoo_ticker(ticker)
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            df = yf.download(
                yahoo_symbol,
                progress=False,
                threads=False,
                **kwargs,
            )

            if df is not None and not df.empty:
                return df

            last_error = ValueError(f"Empty Yahoo response for {ticker}")

        except Exception as exc:
            last_error = exc

        time.sleep(sleep_seconds * attempt)

    raise ValueError(f"No usable Yahoo market data for {ticker} after {attempts} attempts: {last_error}")


def _history_with_retries(
    ticker: str,
    *,
    attempts: int = 3,
    sleep_seconds: float = 1.5,
    **kwargs,
) -> pd.DataFrame:
    yahoo_symbol = _yahoo_ticker(ticker)
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            df = yf.Ticker(yahoo_symbol).history(**kwargs)

            if df is not None and not df.empty:
                return df

            last_error = ValueError(f"Empty Yahoo history response for {ticker}")

        except Exception as exc:
            last_error = exc

        time.sleep(sleep_seconds * attempt)

    raise ValueError(f"No usable Yahoo history data for {ticker} after {attempts} attempts: {last_error}")


def fetch_daily_history(ticker: str, years: int = 5) -> pd.DataFrame:
    ticker = _yahoo_ticker(ticker)
    start = (utc_now() - timedelta(days=365 * years + 30)).date().isoformat()
    period = f"{max(1, years)}y"

    # First try: start-date fetch
    try:
        df = _download_with_retries(
            ticker,
            start=start,
            auto_adjust=True,
        )
        return _clean_download(df, ticker)
    except Exception:
        pass

    # Second try: period fetch
    try:
        df = _download_with_retries(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
        )
        return _clean_download(df, ticker)
    except Exception:
        pass

    # Third try: Ticker().history()
    df = _history_with_retries(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
    )
    return _clean_download(df, ticker)


def fetch_vix_history(years: int = 5) -> pd.DataFrame:
    period = f"{max(1, years)}y"

    attempts = [
        lambda: _download_with_retries(
            "^VIX",
            period=period,
            interval="1d",
            auto_adjust=True,
        ),
        lambda: _history_with_retries(
            "^VIX",
            period=period,
            interval="1d",
            auto_adjust=True,
        ),
        lambda: _download_with_retries(
            "^VIX",
            period="3y",
            interval="1d",
            auto_adjust=True,
        ),
        lambda: _history_with_retries(
            "^VIX",
            period="3y",
            interval="1d",
            auto_adjust=True,
        ),
        lambda: _download_with_retries(
            "VIXY",
            period=period,
            interval="1d",
            auto_adjust=True,
        ),
    ]

    last_error: Exception | None = None

    for fetcher in attempts:
        try:
            df = fetcher()
            return _clean_download(df, "^VIX")
        except Exception as exc:
            last_error = exc

    raise ValueError(f"No usable VIX data after fallbacks: {last_error}")


def fetch_recent_intraday_history(
    ticker: str,
    interval: str = "30m",
    period: str = "60d",
) -> pd.DataFrame:
    ticker = _yahoo_ticker(ticker)

    # Yahoo can be flaky with 60d intraday. Try shorter fallbacks before failing.
    periods_to_try = [period, "30d", "10d", "5d"]

    last_error: Exception | None = None

    for p in periods_to_try:
        try:
            df = _download_with_retries(
                ticker,
                period=p,
                interval=interval,
                auto_adjust=True,
                prepost=False,
            )
            return _clean_download(df, ticker)
        except Exception as exc:
            last_error = exc

    # Last fallback: Ticker().history()
    for p in periods_to_try:
        try:
            df = _history_with_retries(
                ticker,
                period=p,
                interval=interval,
                auto_adjust=True,
                prepost=False,
            )
            return _clean_download(df, ticker)
        except Exception as exc:
            last_error = exc

    raise ValueError(f"No usable intraday data for {ticker}: {last_error}")


def monday_noon_window_average_from_intraday(
    intraday_df: pd.DataFrame,
    monday_date: pd.Timestamp,
    window_start: str = "11:30",
    window_end: str = "12:30",
) -> float | None:
    if intraday_df is None or intraday_df.empty:
        return None

    idx = pd.to_datetime(intraday_df.index).tz_localize(None)
    df = intraday_df.copy()
    df.index = idx

    day_df = df[df.index.normalize() == pd.Timestamp(monday_date).normalize()]
    if day_df.empty:
        return None

    start_ts = pd.Timestamp(f"{pd.Timestamp(monday_date).date()} {window_start}")
    end_ts = pd.Timestamp(f"{pd.Timestamp(monday_date).date()} {window_end}")

    window_df = day_df[(day_df.index >= start_ts) & (day_df.index <= end_ts)]

    if window_df.empty:
        return None

    if "close" in window_df.columns:
        return float(window_df["close"].mean())

    numeric_cols = [c for c in ["open", "high", "low", "close"] if c in window_df.columns]

    if not numeric_cols:
        return None

    return float(window_df[numeric_cols].mean(axis=1).mean())