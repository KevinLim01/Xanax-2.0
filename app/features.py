from __future__ import annotations

import numpy as np
import pandas as pd

from app.fetchers.market_data import (
    fetch_recent_intraday_history,
    monday_noon_window_average_from_intraday,
)
from app.models import WeeklyFeatureRow


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def prepare_feature_frame(price_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    close = price_df["close"].copy()
    df = pd.DataFrame(index=close.index)
    df["close"] = close
    df["prev_1d_return"] = close.pct_change(1)
    df["prev_5d_return"] = close.pct_change(5)
    df["prev_20d_return"] = close.pct_change(20)
    df["rsi_14"] = compute_rsi(close, 14)

    rolling_mean = close.rolling(20).mean()
    rolling_std = close.rolling(20).std()
    df["zscore_20"] = ((close - rolling_mean) / rolling_std).replace([np.inf, -np.inf], np.nan)

    df["ma_gap_5_20"] = close.rolling(5).mean() / close.rolling(20).mean() - 1
    df["ma_gap_10_50"] = close.rolling(10).mean() / close.rolling(50).mean() - 1
    df["realized_vol_20"] = close.pct_change().rolling(20).std() * np.sqrt(252)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd - signal

    vix = vix_df["close"].reindex(df.index).ffill()
    df["vix_close"] = vix
    df["vix_zscore_20"] = ((vix - vix.rolling(20).mean()) / vix.rolling(20).std()).replace([np.inf, -np.inf], np.nan)

    return df.dropna().copy()


def monday_noon_proxy_from_daily(day_row: pd.Series) -> float:
    cols = {c.lower(): c for c in day_row.index}
    needed = ["open", "high", "low", "close"]
    if all(k in cols for k in needed):
        o = float(day_row[cols["open"]])
        h = float(day_row[cols["high"]])
        l = float(day_row[cols["low"]])
        c = float(day_row[cols["close"]])
        return (o + h + l + c) / 4.0

    if "close" in cols:
        return float(day_row[cols["close"]])
    raise ValueError("Could not compute Monday reference from daily row.")


def _weekly_opportunity_target(week_df: pd.DataFrame, reference_price: float, neutral_band: float) -> tuple[float, int, float, float]:
    """Return signed best opportunity target, class, max_up, max_down_abs.

    This trains the model on what the user actually wants: did the stock give a real tradable
    move at any point during the week, not where it closed Friday.
    """
    cols = {c.lower(): c for c in week_df.columns}
    high_col = cols.get("high", cols.get("close"))
    low_col = cols.get("low", cols.get("close"))
    close_col = cols.get("close")
    if high_col is None or low_col is None or close_col is None:
        raise ValueError("price_df must contain close and preferably high/low")

    max_up = float(week_df[high_col].max() / reference_price - 1.0)
    max_down_abs = float(1.0 - week_df[low_col].min() / reference_price)
    friday_return = float(week_df[close_col].iloc[-1] / reference_price - 1.0)

    if max_up < neutral_band and max_down_abs < neutral_band:
        return friday_return, 0, max_up, max_down_abs

    # If both sides triggered, pick the larger opportunity. If basically tied, use Friday direction.
    if max_up >= neutral_band and max_down_abs >= neutral_band:
        if max_up >= max_down_abs * 1.10:
            return max_up, 1, max_up, max_down_abs
        if max_down_abs >= max_up * 1.10:
            return -max_down_abs, -1, max_up, max_down_abs
        if friday_return > 0:
            return max_up, 1, max_up, max_down_abs
        if friday_return < 0:
            return -max_down_abs, -1, max_up, max_down_abs
        return 0.0, 0, max_up, max_down_abs

    if max_up >= neutral_band:
        return max_up, 1, max_up, max_down_abs
    return -max_down_abs, -1, max_up, max_down_abs


def build_training_rows(price_df: pd.DataFrame, feature_df: pd.DataFrame, neutral_band: float) -> pd.DataFrame:
    rows = []
    by_week = price_df.copy()
    by_week["weekday"] = by_week.index.weekday
    by_week["week"] = by_week.index.to_period("W-FRI")

    for _, week_df in by_week.groupby("week"):
        mondays = week_df[week_df["weekday"] == 0]
        if mondays.empty:
            continue

        monday_idx = mondays.index[0]
        if monday_idx not in feature_df.index or monday_idx not in price_df.index:
            continue

        monday_entry = monday_noon_proxy_from_daily(price_df.loc[monday_idx])
        signed_best_move, target_class, max_up, max_down_abs = _weekly_opportunity_target(
            week_df, monday_entry, neutral_band
        )

        feat = feature_df.loc[monday_idx]
        rows.append(
            {
                "asof_date": monday_idx.date().isoformat(),
                "monday_close": monday_entry,
                "target_return": signed_best_move,
                "target_class": target_class,
                "max_up_opportunity": max_up,
                "max_down_opportunity": max_down_abs,
                **feat.to_dict(),
            }
        )

    return pd.DataFrame(rows)


def current_week_feature_row(price_df: pd.DataFrame, feature_df: pd.DataFrame, ticker: str) -> WeeklyFeatureRow:
    recent = price_df.copy()
    recent["weekday"] = recent.index.weekday
    recent["week"] = recent.index.to_period("W-FRI")
    last_week_df = list(recent.groupby("week"))[-1][1]
    monday_rows = last_week_df[last_week_df["weekday"] == 0]

    if not monday_rows.empty:
        idx = monday_rows.index[0]
        monday_entry = None
        try:
            intraday_df = fetch_recent_intraday_history(ticker, interval="30m", period="60d")
            monday_entry = monday_noon_window_average_from_intraday(intraday_df, idx)
        except Exception:
            monday_entry = None

        if monday_entry is None:
            monday_entry = monday_noon_proxy_from_daily(price_df.loc[idx])

        window = "THIS_WEEK_MONDAY_REFERENCE_TO_FRIDAY_CLOSE"
    else:
        idx = feature_df.index[-1]
        monday_entry = monday_noon_proxy_from_daily(price_df.loc[idx])
        window = "NEXT_5_TRADING_DAYS_OPPORTUNITY_PROXY"

    feat = feature_df.loc[idx]
    return WeeklyFeatureRow(
        asof_date=idx.date().isoformat(),
        monday_close=float(monday_entry),
        expected_window=window,
        prev_1d_return=float(feat["prev_1d_return"]),
        prev_5d_return=float(feat["prev_5d_return"]),
        prev_20d_return=float(feat["prev_20d_return"]),
        rsi_14=float(feat["rsi_14"]),
        zscore_20=float(feat["zscore_20"]),
        ma_gap_5_20=float(feat["ma_gap_5_20"]),
        ma_gap_10_50=float(feat["ma_gap_10_50"]),
        realized_vol_20=float(feat["realized_vol_20"]),
        vix_close=float(feat["vix_close"]),
        vix_zscore_20=float(feat["vix_zscore_20"]),
        macd_hist=float(feat["macd_hist"]),
    )
