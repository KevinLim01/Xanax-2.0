from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    monday_trades: int
    tuesday_trades: int
    wednesday_trades: int
    max_total_exposure_usd: float = 5000.0
    max_position_size_usd: float = 300.0
    stop_loss_pct: float = 8.0
    history_capture_ratio: float = 0.85

    # Tuned entry rules from the first simulation.
    monday_min_long_conviction: int = 60
    second_chance_min_long_conviction: int = 70
    min_long_history_rate: float = 60.0
    second_chance_min_long_history_rate: float = 70.0

    # UP_OPPORTUNITY and BREAKDOWN_CONTINUATION are allowed only when top-tier.
    allow_shorts: bool = True
    top_tier_up_opp_conviction: int = 82
    top_tier_up_opp_history_rate: float = 75.0
    extreme_short_conviction: int = 88
    extreme_short_history_rate: float = 78.0

    # Realistic option proxy simulation for LONG/UP only.
    # This is still a proxy, not real historical option-chain pricing, but it is much
    # more conservative than multiplying stock returns. It subtracts estimated
    # bid/ask spread, theta decay, IV crush, and liquidity slippage.
    allow_long_call_options: bool = True
    option_long_min_conviction: int = 78
    option_long_min_history_rate: float = 72.0
    option_max_contract_price: float = 5.0
    option_contracts: int = 1
    option_delta_proxy: float = 0.55
    option_leverage_cap: float = 4.0
    option_spread_cost_pct: float = 6.0
    option_theta_decay_pct_per_day: float = 3.0
    option_iv_crush_pct: float = 6.0
    option_liquidity_slippage_pct: float = 2.0
    option_max_loss_pct: float = 60.0
    option_stop_loss_pct: float = 60.0
    option_min_profit_target_pct: float = 12.0

    # Weekly compounding/reinvestment simulation.
    # When enabled, the next week's exposure cap and position size grow/shrink
    # based on profits/losses realized in prior weeks.
    reinvest_weekly: bool = False
    starting_capital_usd: float = 5000.0
    position_size_fraction: float = 0.06  # $300 on a $5,000 starting account
    min_position_size_usd: float = 50.0


@dataclass
class SimPosition:
    ticker: str
    side: str  # LONG or SHORT
    instrument: str  # STOCK or LONG_CALL_PROXY
    entry_date: pd.Timestamp
    entry_price: float
    capital_at_risk: float
    shares: float
    setup_type: str
    forecast_direction: str
    conviction: int
    history_rate: float
    history_avg_best_pct: float
    max_favorable_pct: float = 0.0
    max_adverse_pct: float = 0.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def _load_history_summary(path: str | Path = "data/history_setup_summary.csv") -> pd.DataFrame:
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
        int(_safe_float(row.get("sample_size"), 0)),
    )


def _download_prices(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
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
            df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
            df["rsi_14"] = _rsi(df["Close"], 14)
            out[ticker] = df.dropna(subset=["ret_5d", "ret_20d", "rsi_14"])
        except Exception as exc:
            print(f"  failed {ticker}: {exc}")
    return out


def _make_signal(ticker: str, as_of: pd.Timestamp, df: pd.DataFrame, history_df: pd.DataFrame) -> dict[str, Any] | None:
    if as_of not in df.index:
        return None
    row = df.loc[as_of]
    ret5 = _safe_float(row.get("ret_5d"))
    ret20 = _safe_float(row.get("ret_20d"))
    rsi = _safe_float(row.get("rsi_14"), 50.0)
    vol_ratio = _safe_float(row.get("vol_ratio"), 1.0)

    score = (0.55 * ret5) + (0.25 * ret20) + (2.0 * (vol_ratio - 1.0))

    if score >= 0:
        direction = "UP"
        action = "BUY"
        setup_type = "MOMENTUM_CONTINUATION" if ret5 >= 0 else "UP_OPPORTUNITY"
        chase_penalty = max(0.0, rsi - 78.0) * 0.25
        raw_strength = max(0.0, score - chase_penalty)
    else:
        direction = "DOWN"
        action = "SELL"
        setup_type = "BREAKDOWN_CONTINUATION" if ret5 <= 0 else "DOWN_OPPORTUNITY"
        chase_penalty = max(0.0, 22.0 - rsi) * 0.25
        raw_strength = max(0.0, abs(score) - chase_penalty)

    history_rate, avg_best, sample = _history_for_setup(history_df, setup_type, direction)
    conviction = int(max(0, min(100, 45 + raw_strength * 4 + (history_rate - 55) * 0.55)))

    # Strongly favor long/up momentum in ranking.
    rank_score = conviction + history_rate * 0.5 + abs(score)
    if action == "BUY" and direction == "UP" and setup_type == "MOMENTUM_CONTINUATION":
        rank_score += 18
    elif action == "BUY" and direction == "UP":
        rank_score += 4
    elif action == "SELL" and direction == "DOWN":
        rank_score -= 20

    return {
        "ticker": ticker,
        "date": as_of,
        "action": action,
        "direction": direction,
        "setup_type": setup_type,
        "conviction": conviction,
        "history_rate": history_rate,
        "history_avg_best_pct": avg_best,
        "history_sample": sample,
        "rank_score": rank_score,
    }


def _passes_tuned_entry(sig: dict[str, Any], config: StrategyConfig, *, second_chance: bool) -> tuple[bool, str]:
    action = str(sig.get("action", "")).upper()
    direction = str(sig.get("direction", "")).upper()
    setup = str(sig.get("setup_type", "")).upper()
    conviction = int(_safe_float(sig.get("conviction"), 0))
    history_rate = _safe_float(sig.get("history_rate"), 0.0)
    sample = int(_safe_float(sig.get("history_sample"), 0))

    if sample < 50:
        return False, "history sample too small"

    if action == "BUY" and direction == "UP":
        min_conv = config.second_chance_min_long_conviction if second_chance else config.monday_min_long_conviction
        min_hist = config.second_chance_min_long_history_rate if second_chance else config.min_long_history_rate

        if setup == "MOMENTUM_CONTINUATION":
            if conviction < min_conv:
                return False, "long momentum conviction too low"
            if history_rate < min_hist:
                return False, "long momentum history rate too low"
            return True, "long/up momentum passed"

        if setup == "UP_OPPORTUNITY":
            if conviction >= config.top_tier_up_opp_conviction and history_rate >= config.top_tier_up_opp_history_rate:
                return True, "top-tier up opportunity passed"
            return False, "up opportunity blocked unless top-tier"

        return False, "unsupported long setup"

    if action == "SELL" and direction == "DOWN":
        if not config.allow_shorts:
            return False, "shorts disabled for this strategy"

        # Shorts are allowed only in extreme cases.
        if setup != "BREAKDOWN_CONTINUATION":
            return False, "short blocked unless breakdown continuation"
        if conviction < config.extreme_short_conviction:
            return False, "short conviction not extreme enough"
        if history_rate < config.extreme_short_history_rate:
            return False, "short history rate not extreme enough"
        return True, "extreme short passed"

    return False, "not a directional trade"


def _choose_instrument(sig: dict[str, Any], config: StrategyConfig) -> str:
    if not config.allow_long_call_options:
        return "STOCK"
    action = str(sig.get("action", "")).upper()
    direction = str(sig.get("direction", "")).upper()
    setup = str(sig.get("setup_type", "")).upper()
    conviction = int(_safe_float(sig.get("conviction"), 0))
    history_rate = _safe_float(sig.get("history_rate"), 0.0)

    if (
        action == "BUY"
        and direction == "UP"
        and setup == "MOMENTUM_CONTINUATION"
        and conviction >= config.option_long_min_conviction
        and history_rate >= config.option_long_min_history_rate
    ):
        return "LONG_CALL_PROXY"
    return "STOCK"


def _option_proxy_pnl_pct(underlying_pct: float, config: StrategyConfig, *, days_held: int) -> float:
    """Conservative long-call return proxy.

    This does not use a real historical option chain. It estimates the option
    return from the underlying stock move, then subtracts realistic frictions:
      - bid/ask spread cost
      - theta decay by days held
      - IV crush on winning/large up moves
      - liquidity/slippage penalty

    The output is percent return on option premium/capital at risk.
    """
    gross = underlying_pct * config.option_delta_proxy * config.option_leverage_cap

    theta_cost = max(0, int(days_held)) * config.option_theta_decay_pct_per_day
    iv_cost = config.option_iv_crush_pct if underlying_pct > 0 else 0.0
    total_cost = (
        config.option_spread_cost_pct
        + theta_cost
        + iv_cost
        + config.option_liquidity_slippage_pct
    )

    net = gross - total_cost
    return max(-config.option_max_loss_pct, net)


def _exit_position(pos: SimPosition, day: pd.Timestamp, bar: pd.Series, config: StrategyConfig) -> dict[str, Any] | None:
    open_p = _safe_float(bar.get("Open"))
    high = _safe_float(bar.get("High"))
    low = _safe_float(bar.get("Low"))
    close = _safe_float(bar.get("Close"))
    if min(open_p, high, low, close) <= 0:
        return None

    target_pct = max(0.25, pos.history_avg_best_pct * config.history_capture_ratio)
    stop_pct = config.stop_loss_pct

    if pos.side == "LONG":
        high_underlying_pct = ((high - pos.entry_price) / pos.entry_price) * 100
        low_underlying_pct = ((low - pos.entry_price) / pos.entry_price) * 100
        close_underlying_pct = ((close - pos.entry_price) / pos.entry_price) * 100
        pos.max_favorable_pct = max(pos.max_favorable_pct, high_underlying_pct)
        pos.max_adverse_pct = min(pos.max_adverse_pct, low_underlying_pct)

        if pos.instrument == "LONG_CALL_PROXY":
            days_held = max(0, int((day - pos.entry_date).days))
            high_pnl_pct = _option_proxy_pnl_pct(high_underlying_pct, config, days_held=days_held)
            low_pnl_pct = _option_proxy_pnl_pct(low_underlying_pct, config, days_held=days_held)
            close_pnl_pct = _option_proxy_pnl_pct(close_underlying_pct, config, days_held=days_held)

            # The option still uses the stock model's history-based target, but the
            # option return itself includes spread/theta/IV/liquidity costs. The 8%
            # emergency cutoff remains based on the underlying move.
            if low_underlying_pct <= -stop_pct:
                pnl_pct = max(-config.option_max_loss_pct, _option_proxy_pnl_pct(-stop_pct, config, days_held=days_held))
                reason = "UNDERLYING_8PCT_EMERGENCY_STOP"
            elif low_pnl_pct <= -config.option_stop_loss_pct:
                pnl_pct = -config.option_stop_loss_pct
                reason = "OPTION_PREMIUM_STOP"
            elif high_underlying_pct >= target_pct:
                pnl_pct = _option_proxy_pnl_pct(target_pct, config, days_held=days_held)
                pnl_pct = max(config.option_min_profit_target_pct, pnl_pct) if pnl_pct > 0 else pnl_pct
                reason = "HISTORY_TARGET_OPTION_PROXY"
            elif day.weekday() == 4:
                pnl_pct = close_pnl_pct
                reason = "FRIDAY_EXIT"
            else:
                return None
            pnl = pos.capital_at_risk * (pnl_pct / 100.0)
            exit_price = pos.entry_price * (1 + close_underlying_pct / 100.0)
        else:
            if low <= pos.entry_price * (1 - stop_pct / 100):
                exit_price = pos.entry_price * (1 - stop_pct / 100)
                reason = "STOP_LOSS"
            elif high >= pos.entry_price * (1 + target_pct / 100):
                exit_price = pos.entry_price * (1 + target_pct / 100)
                reason = "HISTORY_TARGET"
            elif day.weekday() == 4:
                exit_price = close
                reason = "FRIDAY_EXIT"
            else:
                return None
            pnl = (exit_price - pos.entry_price) * pos.shares
            pnl_pct = ((exit_price - pos.entry_price) / pos.entry_price) * 100
    else:
        favorable = ((pos.entry_price - low) / pos.entry_price) * 100
        adverse = ((pos.entry_price - high) / pos.entry_price) * 100
        pos.max_favorable_pct = max(pos.max_favorable_pct, favorable)
        pos.max_adverse_pct = min(pos.max_adverse_pct, adverse)
        if high >= pos.entry_price * (1 + stop_pct / 100):
            exit_price = pos.entry_price * (1 + stop_pct / 100)
            reason = "STOP_LOSS"
        elif low <= pos.entry_price * (1 - target_pct / 100):
            exit_price = pos.entry_price * (1 - target_pct / 100)
            reason = "HISTORY_TARGET"
        elif day.weekday() == 4:
            exit_price = close
            reason = "FRIDAY_EXIT"
        else:
            return None
        pnl = (pos.entry_price - exit_price) * pos.shares
        pnl_pct = ((pos.entry_price - exit_price) / pos.entry_price) * 100

    return {
        "strategy": config.name,
        "ticker": pos.ticker,
        "side": pos.side,
        "instrument": pos.instrument,
        "entry_date": pos.entry_date.date().isoformat(),
        "exit_date": day.date().isoformat(),
        "entry_price": round(pos.entry_price, 4),
        "exit_price": round(exit_price, 4),
        "capital_at_risk": round(pos.capital_at_risk, 2),
        "shares": round(pos.shares, 6),
        "pnl_usd": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 3),
        "exit_reason": reason,
        "setup_type": pos.setup_type,
        "direction": pos.forecast_direction,
        "conviction": pos.conviction,
        "history_rate": round(pos.history_rate, 2),
        "history_avg_best_pct": round(pos.history_avg_best_pct, 3),
        "option_delta_proxy": round(config.option_delta_proxy, 3) if pos.instrument == "LONG_CALL_PROXY" else None,
        "option_spread_cost_pct": round(config.option_spread_cost_pct, 3) if pos.instrument == "LONG_CALL_PROXY" else None,
        "option_theta_decay_pct_per_day": round(config.option_theta_decay_pct_per_day, 3) if pos.instrument == "LONG_CALL_PROXY" else None,
        "option_iv_crush_pct": round(config.option_iv_crush_pct, 3) if pos.instrument == "LONG_CALL_PROXY" else None,
        "option_liquidity_slippage_pct": round(config.option_liquidity_slippage_pct, 3) if pos.instrument == "LONG_CALL_PROXY" else None,
        "max_favorable_pct": round(pos.max_favorable_pct, 3),
        "max_adverse_pct": round(pos.max_adverse_pct, 3),
    }


def _week_key(day: pd.Timestamp) -> tuple[int, int]:
    iso = day.isocalendar()
    return int(iso.year), int(iso.week)


def _effective_capital(config: StrategyConfig, current_capital: float) -> float:
    return current_capital if config.reinvest_weekly else config.max_total_exposure_usd


def _effective_position_size(config: StrategyConfig, current_capital: float, exposure: float) -> float:
    cap = _effective_capital(config, current_capital)
    remaining = max(0.0, cap - exposure)
    if config.reinvest_weekly:
        size = max(config.min_position_size_usd, current_capital * config.position_size_fraction)
        return min(size, remaining)
    return min(config.max_position_size_usd, remaining)


def simulate_strategy(
    tickers: list[str],
    *,
    years: int = 2,
    strategies: list[str] | None = None,
    output_dir: str | Path = "data",
) -> dict[str, Any]:
    end_dt = pd.Timestamp.utcnow().tz_localize(None).normalize() + pd.Timedelta(days=1)
    start_dt = end_dt - pd.Timedelta(days=365 * years + 80)

    history_df = _load_history_summary()
    prices = _download_prices(tickers, start_dt.date().isoformat(), end_dt.date().isoformat())
    if not prices:
        raise RuntimeError("No price data downloaded for simulation.")

    all_days = sorted(set().union(*[set(df.index) for df in prices.values()]))
    all_days = [d for d in all_days if d >= start_dt + pd.Timedelta(days=70)]

    configs = {
        "monday_only": StrategyConfig(
            "monday_only",
            monday_trades=15,
            tuesday_trades=0,
            wednesday_trades=0,
            allow_long_call_options=False,
        ),
        "monday_tuesday": StrategyConfig(
            "monday_tuesday",
            monday_trades=15,
            tuesday_trades=7,
            wednesday_trades=0,
            allow_long_call_options=False,
        ),
        "monday_tuesday_wednesday": StrategyConfig(
            "monday_tuesday_wednesday",
            monday_trades=15,
            tuesday_trades=7,
            wednesday_trades=7,
            allow_long_call_options=False,
        ),
        "tuned_with_realistic_long_calls": StrategyConfig(
            "tuned_with_realistic_long_calls",
            monday_trades=15,
            tuesday_trades=7,
            wednesday_trades=7,
            allow_long_call_options=True,
        ),
        # Kept as an alias so old commands/results still work, but now it uses the
        # same realistic option proxy rather than the old simple multiplier proxy.
        "tuned_with_long_calls": StrategyConfig(
            "tuned_with_long_calls",
            monday_trades=15,
            tuesday_trades=7,
            wednesday_trades=7,
            allow_long_call_options=True,
        ),
        "tuned_reinvest_weekly": StrategyConfig(
            "tuned_reinvest_weekly",
            monday_trades=15,
            tuesday_trades=7,
            wednesday_trades=7,
            allow_long_call_options=True,
            reinvest_weekly=True,
            starting_capital_usd=5000.0,
            max_total_exposure_usd=5000.0,
            max_position_size_usd=300.0,
            position_size_fraction=0.06,
        ),
        "stock_long_reinvest_weekly": StrategyConfig(
            "stock_long_reinvest_weekly",
            monday_trades=15,
            tuesday_trades=7,
            wednesday_trades=7,
            allow_long_call_options=False,
            allow_shorts=False,
            reinvest_weekly=True,
            starting_capital_usd=5000.0,
            max_total_exposure_usd=5000.0,
            max_position_size_usd=300.0,
            position_size_fraction=0.06,
            monday_min_long_conviction=60,
            second_chance_min_long_conviction=70,
            min_long_history_rate=60.0,
            second_chance_min_long_history_rate=70.0,
        ),
    }
    selected = [configs[name] for name in (strategies or list(configs)) if name in configs]

    trades: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    final_capital_by_strategy: dict[str, float] = {}

    for config in selected:
        print(f"Simulating {config.name}...")
        open_positions: dict[str, SimPosition] = {}
        exposure = 0.0
        current_capital = float(config.starting_capital_usd if config.reinvest_weekly else config.max_total_exposure_usd)
        current_week_key: tuple[int, int] | None = None
        current_week_pnl = 0.0
        current_week_trades = 0
        current_week_start_capital = current_capital

        def close_week_if_needed(next_day: pd.Timestamp | None) -> None:
            nonlocal current_week_key, current_week_pnl, current_week_trades, current_week_start_capital, current_capital
            if current_week_key is None:
                if next_day is not None:
                    current_week_key = _week_key(next_day)
                    current_week_start_capital = current_capital
                return
            if next_day is not None and _week_key(next_day) == current_week_key:
                return

            year, week = current_week_key
            ending_capital = current_capital + current_week_pnl if config.reinvest_weekly else current_capital
            weekly_rows.append(
                {
                    "strategy": config.name,
                    "iso_year": year,
                    "iso_week": week,
                    "week_start_capital": round(current_week_start_capital, 2),
                    "week_pnl_usd": round(current_week_pnl, 2),
                    "week_trades": current_week_trades,
                    "week_return_pct": round((current_week_pnl / current_week_start_capital) * 100, 3)
                    if current_week_start_capital > 0
                    else 0.0,
                    "week_end_capital": round(ending_capital, 2),
                }
            )

            if config.reinvest_weekly:
                current_capital = max(100.0, ending_capital)

            current_week_pnl = 0.0
            current_week_trades = 0
            if next_day is not None:
                current_week_key = _week_key(next_day)
                current_week_start_capital = current_capital
            else:
                current_week_key = None

        for day in all_days:
            close_week_if_needed(day)

            for ticker, pos in list(open_positions.items()):
                df = prices.get(ticker)
                if df is None or day not in df.index:
                    continue
                exit_row = _exit_position(pos, day, df.loc[day], config)
                if exit_row is not None:
                    exit_row["week_key"] = f"{_week_key(day)[0]}-{_week_key(day)[1]:02d}"
                    exit_row["capital_at_week_start"] = round(current_week_start_capital, 2)
                    exit_row["capital_after_prior_weeks"] = round(current_capital, 2)
                    trades.append(exit_row)
                    exposure = max(0.0, exposure - pos.capital_at_risk)
                    current_week_pnl += float(exit_row["pnl_usd"])
                    current_week_trades += 1
                    del open_positions[ticker]

            if day.weekday() not in {0, 1, 2}:
                continue

            allowed = 0
            second_chance = False
            if day.weekday() == 0:
                allowed = config.monday_trades
            elif day.weekday() == 1:
                allowed = config.tuesday_trades
                second_chance = True
            elif day.weekday() == 2:
                allowed = config.wednesday_trades
                second_chance = True

            if allowed <= 0:
                continue

            cap = _effective_capital(config, current_capital)
            if exposure >= cap:
                continue

            slots_left = allowed
            candidates: list[dict[str, Any]] = []
            for ticker, df in prices.items():
                if ticker in open_positions or day not in df.index:
                    continue
                sig = _make_signal(ticker, day, df, history_df)
                if not sig:
                    continue
                ok, _reason = _passes_tuned_entry(sig, config, second_chance=second_chance)
                if ok:
                    candidates.append(sig)

            candidates.sort(key=lambda x: x["rank_score"], reverse=True)
            for sig in candidates:
                if slots_left <= 0:
                    break
                cap = _effective_capital(config, current_capital)
                if exposure >= cap:
                    break
                df = prices[sig["ticker"]]
                row = df.loc[day]
                entry_price = _safe_float(row.get("Open"))
                if entry_price <= 0:
                    continue

                instrument = _choose_instrument(sig, config)
                if instrument == "LONG_CALL_PROXY":
                    if config.reinvest_weekly:
                        base_size = _effective_position_size(config, current_capital, exposure)
                        position_size = min(base_size, cap - exposure)
                    else:
                        position_size = min(
                            config.option_max_contract_price * 100 * config.option_contracts,
                            config.max_position_size_usd,
                            cap - exposure,
                        )
                else:
                    position_size = _effective_position_size(config, current_capital, exposure)

                if position_size <= 25:
                    continue

                shares = position_size / entry_price
                side = "LONG" if sig["action"] == "BUY" else "SHORT"
                open_positions[sig["ticker"]] = SimPosition(
                    ticker=sig["ticker"],
                    side=side,
                    instrument=instrument,
                    entry_date=day,
                    entry_price=entry_price,
                    capital_at_risk=position_size,
                    shares=shares,
                    setup_type=sig["setup_type"],
                    forecast_direction=sig["direction"],
                    conviction=sig["conviction"],
                    history_rate=sig["history_rate"],
                    history_avg_best_pct=sig["history_avg_best_pct"],
                )
                exposure += position_size
                slots_left -= 1

        # Force-close leftover open positions on last available date.
        if all_days:
            last_day = all_days[-1]
            for ticker, pos in list(open_positions.items()):
                df = prices.get(ticker)
                if df is not None and last_day in df.index:
                    close_bar = df.loc[last_day].copy()
                    close_bar["High"] = close_bar["Close"]
                    close_bar["Low"] = close_bar["Close"]
                    exit_row = _exit_position(pos, last_day, close_bar, config)
                    if exit_row is not None:
                        exit_row["week_key"] = f"{_week_key(last_day)[0]}-{_week_key(last_day)[1]:02d}"
                        exit_row["capital_at_week_start"] = round(current_week_start_capital, 2)
                        exit_row["capital_after_prior_weeks"] = round(current_capital, 2)
                        trades.append(exit_row)
                        current_week_pnl += float(exit_row["pnl_usd"])
                        current_week_trades += 1
                del open_positions[ticker]

        close_week_if_needed(None)
        final_capital_by_strategy[config.name] = current_capital

    trades_df = pd.DataFrame(trades)
    weekly_df = pd.DataFrame(weekly_rows)
    if trades_df.empty:
        summary_df = pd.DataFrame()
    else:
        summary_df = (
            trades_df.groupby("strategy")
            .agg(
                trades=("ticker", "count"),
                total_pnl_usd=("pnl_usd", "sum"),
                avg_pnl_usd=("pnl_usd", "mean"),
                avg_pnl_pct=("pnl_pct", "mean"),
                win_rate_pct=("pnl_usd", lambda s: float((s > 0).mean() * 100)),
                avg_max_favorable_pct=("max_favorable_pct", "mean"),
                avg_max_adverse_pct=("max_adverse_pct", "mean"),
            )
            .reset_index()
        )
        summary_df["starting_capital_usd"] = summary_df["strategy"].map(
            lambda name: configs[str(name)].starting_capital_usd if configs[str(name)].reinvest_weekly else configs[str(name)].max_total_exposure_usd
        )
        summary_df["ending_capital_usd"] = summary_df.apply(
            lambda row: final_capital_by_strategy.get(str(row["strategy"]), row["starting_capital_usd"] + row["total_pnl_usd"])
            if configs[str(row["strategy"])].reinvest_weekly
            else row["starting_capital_usd"] + row["total_pnl_usd"],
            axis=1,
        )
        summary_df["total_return_pct"] = (
            (summary_df["ending_capital_usd"] - summary_df["starting_capital_usd"])
            / summary_df["starting_capital_usd"]
            * 100
        )
        summary_df["is_reinvested"] = summary_df["strategy"].map(lambda name: bool(configs[str(name)].reinvest_weekly))
        summary_df["total_pnl_usd"] = summary_df["total_pnl_usd"].round(2)
        summary_df["avg_pnl_usd"] = summary_df["avg_pnl_usd"].round(2)
        summary_df["avg_pnl_pct"] = summary_df["avg_pnl_pct"].round(3)
        summary_df["win_rate_pct"] = summary_df["win_rate_pct"].round(2)
        summary_df["starting_capital_usd"] = summary_df["starting_capital_usd"].round(2)
        summary_df["ending_capital_usd"] = summary_df["ending_capital_usd"].round(2)
        summary_df["total_return_pct"] = summary_df["total_return_pct"].round(2)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trades_path = out / "strategy_simulation_trades.csv"
    summary_path = out / "strategy_simulation_summary.csv"
    weekly_path = out / "strategy_simulation_weekly_equity.csv"
    trades_df.to_csv(trades_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    weekly_df.to_csv(weekly_path, index=False)

    print(f"Saved trades: {trades_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved weekly equity: {weekly_path}")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))

    return {
        "trades_path": str(trades_path),
        "summary_path": str(summary_path),
        "weekly_equity_path": str(weekly_path),
        "trades": int(len(trades_df)),
        "summary_rows": int(len(summary_df)),
        "weekly_rows": int(len(weekly_df)),
    }
