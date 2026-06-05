from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class OptionsHistoryConfig:
    years: int = 2
    output_dir: str | Path = "data"
    history_summary_path: str | Path = "data/history_setup_summary.csv"

    # Entry filters. This dataset is intentionally broad enough to research,
    # but still focused on the type of option trade you are considering.
    min_history_sample: int = 50
    min_underlying_price: float = 5.0
    min_avg_volume: float = 500_000
    entry_weekdays: tuple[int, ...] = (0, 1, 2)  # Monday, Tuesday, Wednesday

    # Option test grid. These are proxy choices, not real historical chains.
    dte_choices: tuple[int, ...] = (7, 14, 21)
    delta_choices: tuple[float, ...] = (0.35, 0.55, 0.70)

    # Conservative friction assumptions.
    leverage_cap: float = 4.0
    base_spread_cost_pct: float = 6.0
    theta_decay_pct_per_day_7d: float = 4.0
    theta_decay_pct_per_day_14d: float = 3.0
    theta_decay_pct_per_day_21d: float = 2.2
    iv_crush_pct: float = 6.0
    liquidity_slippage_pct: float = 2.0
    max_loss_pct: float = 60.0
    option_stop_loss_pct: float = 60.0
    underlying_emergency_stop_pct: float = 8.0
    history_capture_ratio: float = 0.85
    option_success_threshold_pct: float = 5.0


@dataclass(frozen=True)
class SignalRow:
    ticker: str
    date: pd.Timestamp
    action: str
    direction: str
    setup_type: str
    conviction: int
    history_rate: float
    history_avg_best_pct: float
    history_sample: int
    rank_score: float
    ret_5d: float
    ret_20d: float
    rsi_14: float
    vol_ratio: float
    avg_volume_20d: float


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def _load_history_summary(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()

    df = pd.read_csv(p)
    for col in ["setup_type", "forecast_direction", "primary_regime", "ticker_archetype"]:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN").astype(str).str.upper().str.strip()
    return df


def _history_for_setup(history_df: pd.DataFrame, setup_type: str, direction: str) -> tuple[float, float, int]:
    if history_df.empty:
        return 55.0, 3.0, 0

    mask = (history_df["setup_type"] == setup_type) & (history_df["forecast_direction"] == direction)
    hit = history_df.loc[mask].copy()
    if hit.empty:
        return 55.0, 3.0, 0

    hit = hit.sort_values("sample_size", ascending=False)
    row = hit.iloc[0]
    return (
        _safe_float(row.get("true_during_week_rate"), 55.0),
        max(0.5, _safe_float(row.get("average_best_correct_return_pct"), 3.0)),
        _safe_int(row.get("sample_size"), 0),
    )


def _download_prices(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    prices: dict[str, pd.DataFrame] = {}

    for i, ticker in enumerate(tickers, start=1):
        print(f"[{i}/{len(tickers)}] downloading {ticker}...")
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.rename(columns={"Adj Close": "Adj_Close"})
            required = {"Open", "High", "Low", "Close", "Volume"}
            if not required.issubset(df.columns):
                continue

            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            df = df.sort_index()
            df["ret_5d"] = df["Close"].pct_change(5) * 100
            df["ret_20d"] = df["Close"].pct_change(20) * 100
            df["avg_volume_20d"] = df["Volume"].rolling(20).mean()
            df["vol_ratio"] = df["Volume"] / df["avg_volume_20d"]
            df["rsi_14"] = _rsi(df["Close"], 14)
            prices[ticker] = df.dropna(subset=["ret_5d", "ret_20d", "rsi_14", "avg_volume_20d"])
        except Exception as exc:
            print(f"  failed {ticker}: {exc}")

    return prices


def _make_signal(ticker: str, as_of: pd.Timestamp, df: pd.DataFrame, history_df: pd.DataFrame) -> SignalRow | None:
    if as_of not in df.index:
        return None

    row = df.loc[as_of]
    price = _safe_float(row.get("Open"))
    if price <= 0:
        return None

    ret5 = _safe_float(row.get("ret_5d"))
    ret20 = _safe_float(row.get("ret_20d"))
    rsi = _safe_float(row.get("rsi_14"), 50.0)
    vol_ratio = _safe_float(row.get("vol_ratio"), 1.0)
    avg_volume = _safe_float(row.get("avg_volume_20d"), 0.0)

    score = (0.55 * ret5) + (0.25 * ret20) + (2.0 * (vol_ratio - 1.0))

    if score >= 0:
        direction = "UP"
        action = "BUY"
        setup_type = "MOMENTUM_CONTINUATION" if ret5 >= 0 else "UP_OPPORTUNITY"
        chase_penalty = max(0.0, rsi - 78.0) * 0.25
        raw_strength = max(0.0, score - chase_penalty)
    else:
        # Keep downside rows in the raw dataset for comparison, but the options
        # gate will only approve LONG_CALL candidates unless you later add puts.
        direction = "DOWN"
        action = "SELL"
        setup_type = "BREAKDOWN_CONTINUATION" if ret5 <= 0 else "DOWN_OPPORTUNITY"
        chase_penalty = max(0.0, 22.0 - rsi) * 0.25
        raw_strength = max(0.0, abs(score) - chase_penalty)

    history_rate, avg_best, sample = _history_for_setup(history_df, setup_type, direction)
    conviction = int(max(0, min(100, 45 + raw_strength * 4 + (history_rate - 55) * 0.55)))

    rank_score = conviction + history_rate * 0.5 + abs(score)
    if action == "BUY" and direction == "UP" and setup_type == "MOMENTUM_CONTINUATION":
        rank_score += 18
    elif action == "BUY" and direction == "UP":
        rank_score += 4
    elif action == "SELL" and direction == "DOWN":
        rank_score -= 20

    return SignalRow(
        ticker=ticker,
        date=as_of,
        action=action,
        direction=direction,
        setup_type=setup_type,
        conviction=conviction,
        history_rate=history_rate,
        history_avg_best_pct=avg_best,
        history_sample=sample,
        rank_score=rank_score,
        ret_5d=ret5,
        ret_20d=ret20,
        rsi_14=rsi,
        vol_ratio=vol_ratio,
        avg_volume_20d=avg_volume,
    )


def _theta_per_day(dte: int, config: OptionsHistoryConfig) -> float:
    if dte <= 7:
        return config.theta_decay_pct_per_day_7d
    if dte <= 14:
        return config.theta_decay_pct_per_day_14d
    return config.theta_decay_pct_per_day_21d


def _spread_cost(delta: float, avg_volume: float, config: OptionsHistoryConfig) -> float:
    # Illiquid underlying volume usually means worse option fills. This is only a
    # proxy because free data does not provide historical option-chain bid/ask.
    volume_penalty = 0.0
    if avg_volume < 1_000_000:
        volume_penalty = 4.0
    elif avg_volume < 2_000_000:
        volume_penalty = 2.0

    # Lower-delta options usually have wider percentage spreads.
    delta_penalty = 2.0 if delta < 0.45 else 0.0
    return config.base_spread_cost_pct + volume_penalty + delta_penalty


def _option_return_pct(
    underlying_pct: float,
    *,
    delta: float,
    dte: int,
    days_held: int,
    avg_volume: float,
    config: OptionsHistoryConfig,
) -> float:
    gross = underlying_pct * delta * config.leverage_cap
    spread = _spread_cost(delta, avg_volume, config)
    theta = max(0, days_held) * _theta_per_day(dte, config)
    iv = config.iv_crush_pct if underlying_pct > 0 else 0.0
    slippage = config.liquidity_slippage_pct
    net = gross - spread - theta - iv - slippage
    return max(-config.max_loss_pct, net)


def _simulate_option_trade(
    sig: SignalRow,
    df: pd.DataFrame,
    *,
    dte: int,
    delta: float,
    config: OptionsHistoryConfig,
) -> dict[str, Any] | None:
    if sig.date not in df.index:
        return None

    entry_row = df.loc[sig.date]
    entry_price = _safe_float(entry_row.get("Open"))
    if entry_price < config.min_underlying_price:
        return None
    if sig.avg_volume_20d < config.min_avg_volume:
        return None

    target_underlying_pct = max(0.25, sig.history_avg_best_pct * config.history_capture_ratio)
    stop_underlying_pct = -config.underlying_emergency_stop_pct

    rows_after = df.loc[df.index >= sig.date].head(6)  # entry day through Friday-ish window
    if rows_after.empty:
        return None

    best_underlying_pct = 0.0
    worst_underlying_pct = 0.0
    exit_day = None
    exit_reason = "NO_EXIT"
    exit_underlying_pct = 0.0
    option_exit_pct = 0.0
    days_held = 0

    for day, bar in rows_after.iterrows():
        high = _safe_float(bar.get("High"))
        low = _safe_float(bar.get("Low"))
        close = _safe_float(bar.get("Close"))
        if min(high, low, close) <= 0:
            continue

        days_held = max(0, int((day - sig.date).days))
        high_pct = ((high - entry_price) / entry_price) * 100.0
        low_pct = ((low - entry_price) / entry_price) * 100.0
        close_pct = ((close - entry_price) / entry_price) * 100.0
        best_underlying_pct = max(best_underlying_pct, high_pct)
        worst_underlying_pct = min(worst_underlying_pct, low_pct)

        high_option_pct = _option_return_pct(
            high_pct,
            delta=delta,
            dte=dte,
            days_held=days_held,
            avg_volume=sig.avg_volume_20d,
            config=config,
        )
        low_option_pct = _option_return_pct(
            low_pct,
            delta=delta,
            dte=dte,
            days_held=days_held,
            avg_volume=sig.avg_volume_20d,
            config=config,
        )
        close_option_pct = _option_return_pct(
            close_pct,
            delta=delta,
            dte=dte,
            days_held=days_held,
            avg_volume=sig.avg_volume_20d,
            config=config,
        )

        if low_pct <= stop_underlying_pct:
            exit_day = day
            exit_reason = "UNDERLYING_8PCT_EMERGENCY_STOP"
            exit_underlying_pct = stop_underlying_pct
            option_exit_pct = _option_return_pct(
                stop_underlying_pct,
                delta=delta,
                dte=dte,
                days_held=days_held,
                avg_volume=sig.avg_volume_20d,
                config=config,
            )
            break

        if low_option_pct <= -config.option_stop_loss_pct:
            exit_day = day
            exit_reason = "OPTION_PREMIUM_STOP"
            exit_underlying_pct = low_pct
            option_exit_pct = -config.option_stop_loss_pct
            break

        if high_pct >= target_underlying_pct:
            exit_day = day
            exit_reason = "HISTORY_TARGET_OPTION_PROXY"
            exit_underlying_pct = target_underlying_pct
            option_exit_pct = _option_return_pct(
                target_underlying_pct,
                delta=delta,
                dte=dte,
                days_held=days_held,
                avg_volume=sig.avg_volume_20d,
                config=config,
            )
            break

        if day.weekday() == 4:
            exit_day = day
            exit_reason = "FRIDAY_EXIT"
            exit_underlying_pct = close_pct
            option_exit_pct = close_option_pct
            break

    if exit_day is None:
        # Fallback to last available close in the week window.
        last_day = rows_after.index[-1]
        last_close = _safe_float(rows_after.iloc[-1].get("Close"))
        if last_close <= 0:
            return None
        days_held = max(0, int((last_day - sig.date).days))
        exit_day = last_day
        exit_reason = "LAST_AVAILABLE_EXIT"
        exit_underlying_pct = ((last_close - entry_price) / entry_price) * 100.0
        option_exit_pct = _option_return_pct(
            exit_underlying_pct,
            delta=delta,
            dte=dte,
            days_held=days_held,
            avg_volume=sig.avg_volume_20d,
            config=config,
        )

    gross_option_pct = exit_underlying_pct * delta * config.leverage_cap
    spread_cost = _spread_cost(delta, sig.avg_volume_20d, config)
    theta_cost = days_held * _theta_per_day(dte, config)
    iv_cost = config.iv_crush_pct if exit_underlying_pct > 0 else 0.0
    total_cost = spread_cost + theta_cost + iv_cost + config.liquidity_slippage_pct

    return {
        "ticker": sig.ticker,
        "entry_date": sig.date.date().isoformat(),
        "entry_weekday": sig.date.day_name(),
        "exit_date": exit_day.date().isoformat(),
        "days_held": days_held,
        "action": sig.action,
        "direction": sig.direction,
        "setup_type": sig.setup_type,
        "conviction": sig.conviction,
        "history_rate": round(sig.history_rate, 3),
        "history_avg_best_pct": round(sig.history_avg_best_pct, 3),
        "history_sample": sig.history_sample,
        "rank_score": round(sig.rank_score, 3),
        "ret_5d": round(sig.ret_5d, 3),
        "ret_20d": round(sig.ret_20d, 3),
        "rsi_14": round(sig.rsi_14, 3),
        "vol_ratio": round(sig.vol_ratio, 3),
        "avg_volume_20d": round(sig.avg_volume_20d, 0),
        "underlying_entry_price": round(entry_price, 4),
        "underlying_exit_pct": round(exit_underlying_pct, 3),
        "underlying_best_pct": round(best_underlying_pct, 3),
        "underlying_worst_pct": round(worst_underlying_pct, 3),
        "option_type": "LONG_CALL_PROXY" if sig.direction == "UP" and sig.action == "BUY" else "NOT_ALLOWED_FOR_LIVE",
        "dte": dte,
        "delta_proxy": delta,
        "leverage_cap": config.leverage_cap,
        "gross_option_return_pct": round(gross_option_pct, 3),
        "spread_cost_pct": round(spread_cost, 3),
        "theta_cost_pct": round(theta_cost, 3),
        "iv_crush_cost_pct": round(iv_cost, 3),
        "liquidity_slippage_pct": round(config.liquidity_slippage_pct, 3),
        "total_option_cost_pct": round(total_cost, 3),
        "net_option_return_pct": round(option_exit_pct, 3),
        "option_success": bool(option_exit_pct >= config.option_success_threshold_pct),
        "option_profitable": bool(option_exit_pct > 0),
        "exit_reason": exit_reason,
    }


def _bin_conviction(value: int) -> str:
    if value >= 90:
        return "90+"
    if value >= 80:
        return "80-89"
    if value >= 70:
        return "70-79"
    if value >= 60:
        return "60-69"
    return "UNDER_60"


def _bin_history_rate(value: float) -> str:
    if value >= 80:
        return "80+"
    if value >= 75:
        return "75-79"
    if value >= 70:
        return "70-74"
    if value >= 65:
        return "65-69"
    return "UNDER_65"


def _make_summary(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = raw.copy()
    df["conviction_bin"] = df["conviction"].apply(lambda x: _bin_conviction(int(x)))
    df["history_rate_bin"] = df["history_rate"].apply(lambda x: _bin_history_rate(float(x)))

    group_cols = [
        "option_type",
        "setup_type",
        "direction",
        "entry_weekday",
        "dte",
        "delta_proxy",
        "conviction_bin",
        "history_rate_bin",
    ]

    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            sample_size=("ticker", "count"),
            option_success_rate=("option_success", "mean"),
            option_profitable_rate=("option_profitable", "mean"),
            average_net_option_return_pct=("net_option_return_pct", "mean"),
            median_net_option_return_pct=("net_option_return_pct", "median"),
            average_underlying_exit_pct=("underlying_exit_pct", "mean"),
            average_days_held=("days_held", "mean"),
            average_total_option_cost_pct=("total_option_cost_pct", "mean"),
            average_history_rate=("history_rate", "mean"),
            average_conviction=("conviction", "mean"),
        )
        .reset_index()
    )

    for col in ["option_success_rate", "option_profitable_rate"]:
        summary[col] = summary[col] * 100.0

    summary["recommended_option_allowed"] = (
        (summary["sample_size"] >= 50)
        & (summary["option_profitable_rate"] >= 55.0)
        & (summary["average_net_option_return_pct"] >= 5.0)
        & (summary["setup_type"] == "MOMENTUM_CONTINUATION")
        & (summary["direction"] == "UP")
        & (summary["option_type"] == "LONG_CALL_PROXY")
    )

    summary = summary.sort_values(
        ["recommended_option_allowed", "average_net_option_return_pct", "option_profitable_rate", "sample_size"],
        ascending=[False, False, False, False],
    )

    ticker_summary = (
        df.groupby(["ticker", "option_type", "setup_type", "dte", "delta_proxy"], dropna=False)
        .agg(
            sample_size=("ticker", "count"),
            option_success_rate=("option_success", "mean"),
            option_profitable_rate=("option_profitable", "mean"),
            average_net_option_return_pct=("net_option_return_pct", "mean"),
            median_net_option_return_pct=("net_option_return_pct", "median"),
            average_total_option_cost_pct=("total_option_cost_pct", "mean"),
        )
        .reset_index()
    )
    ticker_summary["option_success_rate"] = ticker_summary["option_success_rate"] * 100.0
    ticker_summary["option_profitable_rate"] = ticker_summary["option_profitable_rate"] * 100.0
    ticker_summary = ticker_summary.sort_values(
        ["average_net_option_return_pct", "option_profitable_rate", "sample_size"],
        ascending=[False, False, False],
    )

    return summary, ticker_summary


def build_options_history_dataset(
    tickers: list[str],
    *,
    years: int = 2,
    output_dir: str | Path = "data",
) -> dict[str, Any]:
    config = OptionsHistoryConfig(years=years, output_dir=output_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    end_dt = pd.Timestamp.utcnow().tz_localize(None).normalize() + pd.Timedelta(days=1)
    start_dt = end_dt - pd.Timedelta(days=365 * years + 80)

    history_df = _load_history_summary(config.history_summary_path)
    prices = _download_prices(tickers, start_dt.date().isoformat(), end_dt.date().isoformat())
    if not prices:
        raise RuntimeError("No price data downloaded for options history dataset.")

    rows: list[dict[str, Any]] = []
    all_days = sorted(set().union(*[set(df.index) for df in prices.values()]))
    all_days = [d for d in all_days if d >= start_dt + pd.Timedelta(days=70)]

    for day in all_days:
        if day.weekday() not in config.entry_weekdays:
            continue

        for ticker, df in prices.items():
            if day not in df.index:
                continue

            sig = _make_signal(ticker, day, df, history_df)
            if sig is None:
                continue
            if sig.history_sample < config.min_history_sample:
                continue

            # Only long calls are possible in this version. We still keep non-long rows
            # as NOT_ALLOWED_FOR_LIVE so the summary can prove why they are blocked.
            for dte in config.dte_choices:
                for delta in config.delta_choices:
                    row = _simulate_option_trade(sig, df, dte=dte, delta=delta, config=config)
                    if row is not None:
                        rows.append(row)

    raw = pd.DataFrame(rows)
    summary, ticker_summary = _make_summary(raw)

    raw_path = output / "options_history_raw.csv"
    summary_path = output / "options_history_summary.csv"
    ticker_path = output / "options_history_ticker_summary.csv"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    ticker_summary.to_csv(ticker_path, index=False)

    return {
        "years": years,
        "tickers_requested": len(tickers),
        "tickers_downloaded": len(prices),
        "raw_rows": int(len(raw)),
        "summary_rows": int(len(summary)),
        "ticker_summary_rows": int(len(ticker_summary)),
        "raw_path": str(raw_path),
        "summary_path": str(summary_path),
        "ticker_summary_path": str(ticker_path),
    }
