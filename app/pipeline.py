from __future__ import annotations

import math
from statistics import mean
from typing import Any

import pandas as pd

try:
    from app.agents.agent_common import AgentSignal
    from app.agents.final_score_blender import FinalScoreBlender
    from app.agents.intraday_confirmation_agent import IntradayConfirmationAgent
    from app.agents.liquidity_agent import LiquidityAgent
    from app.agents.relative_strength_agent import RelativeStrengthAgent
except Exception:  # Keeps the old pipeline runnable if agent files are not installed yet.
    AgentSignal = None  # type: ignore[assignment]
    FinalScoreBlender = None  # type: ignore[assignment]
    IntradayConfirmationAgent = None  # type: ignore[assignment]
    LiquidityAgent = None  # type: ignore[assignment]
    RelativeStrengthAgent = None  # type: ignore[assignment]

from app.config import settings
from app.archetypes import (
    bearish_confirmation_count,
    bullish_continuation_score,
    is_dangerous_short,
    is_momentum_name,
    setup_type_for_signal,
    ticker_archetype,
)
from app.database import save_run
from app.domain_maps import (
    any_exposure,
    is_financial_name,
    is_high_beta_name,
    is_megacap_tech_name,
    is_strict_sector_name,
)
from app.fetchers.news_data import fetch_news_modifier
from app.fetchers.polymarket_data import fetch_polymarket_modifier
from app.fetchers.sector_readthrough_data import fetch_sector_readthrough_modifier
from app.fetchers.truthsocial_data import fetch_truthsocial_modifier
from app.models import ModifierSignal, PredictionResult
from app.training import FEATURE_COLUMNS, train_for_ticker
from app.utils import logger, utc_now

ALIASES = {
    "GOOG": "GOOGL",
    "BRK.B": "BRK-B",
}


def normalize_ticker(ticker: str) -> str:
    return ALIASES.get(ticker.strip().upper(), ticker.strip().upper())


def _settings_bool(name: str, default: bool) -> bool:
    value = getattr(settings, name, default)
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _settings_float(name: str, default: float) -> float:
    try:
        return float(getattr(settings, name, default))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _sector_key_for_agent(ticker: str) -> str:
    if _is_semi_or_ai_name(ticker):
        return "SEMIS"
    if is_financial_name(ticker):
        return "FINANCIALS"
    if _is_readthrough_friendly(ticker):
        return "TECH"
    if is_megacap_tech_name(ticker):
        return "TECH"
    if is_high_beta_name(ticker):
        return "CONSUMER_DISCRETIONARY"
    return "DEFAULT"


def _benchmark_for_agent(ticker: str) -> str:
    sector_key = _sector_key_for_agent(ticker)
    return {
        "SEMIS": "SMH",
        "FINANCIALS": "XLF",
        "TECH": "QQQ",
        "CONSUMER_DISCRETIONARY": "XLY",
        "DEFAULT": "SPY",
    }.get(sector_key, "SPY")


def _empty_agent_signal(ticker: str, agent_name: str, reason: str):
    if AgentSignal is None:
        return None
    return AgentSignal(
        agent_name=agent_name,
        ticker=ticker,
        score=0.0,
        direction="NEUTRAL",
        confidence=0.0,
        reason=reason,
        metrics={},
    )


def _download_recent_bars(symbols: list[str], period: str = "5d", interval: str = "5m") -> dict[str, pd.DataFrame]:
    """Best-effort intraday bars for the new agents.

    The main model still works if yfinance is unavailable or rate limited. In that case the
    new agents return neutral signals rather than breaking the weekly pipeline.
    """
    try:
        import yfinance as yf
    except Exception as exc:
        logger.warning("yfinance unavailable for new agents: %s", exc)
        return {}

    out: dict[str, pd.DataFrame] = {}
    for symbol in sorted(set(symbols)):
        try:
            hist = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
            if hist is None or hist.empty:
                continue
            clean = hist.rename(columns={c: str(c).lower() for c in hist.columns})
            if "close" in clean.columns:
                out[symbol] = clean
        except Exception as exc:
            logger.warning("Could not fetch intraday bars for %s: %s", symbol, exc)
    return out


def _fetch_liquidity_inputs(ticker: str, fallback_price: float | None = None) -> dict[str, Any]:
    """Best-effort quote/liquidity snapshot.

    Missing bid/ask should not kill the pipeline. It only reduces the usefulness of the
    liquidity agent for that run.
    """
    try:
        import yfinance as yf
        yt = yf.Ticker(ticker)
        fast = getattr(yt, "fast_info", {}) or {}
        info = {}
        try:
            info = yt.info or {}
        except Exception:
            info = {}

        def pick(*names: str) -> float | None:
            for name in names:
                try:
                    value = fast.get(name) if hasattr(fast, "get") else getattr(fast, name, None)
                except Exception:
                    value = None
                if value is None:
                    value = info.get(name)
                parsed = _safe_float(value, None)
                if parsed is not None:
                    return parsed
            return None

        last_price = pick("last_price", "lastPrice", "regularMarketPrice", "currentPrice") or fallback_price
        avg_volume = pick("ten_day_average_volume", "tenDayAverageVolume", "averageVolume", "averageDailyVolume10Day")
        bid = pick("bid")
        ask = pick("ask")

        return {
            "bid": bid,
            "ask": ask,
            "last_price": last_price,
            "avg_daily_volume": avg_volume,
            "dollar_volume": (last_price * avg_volume) if last_price and avg_volume else None,
        }
    except Exception as exc:
        logger.warning("Could not fetch liquidity inputs for %s: %s", ticker, exc)
        return {
            "bid": None,
            "ask": None,
            "last_price": fallback_price,
            "avg_daily_volume": None,
            "dollar_volume": None,
        }


def _run_market_quality_agents(
    ticker: str,
    direction_hint: str,
    fallback_price: float | None,
) -> tuple[list[Any], dict[str, dict[str, Any]]]:
    """Run the new agents without making them mandatory dependencies."""
    signals: list[Any] = []
    metrics: dict[str, dict[str, Any]] = {}

    if AgentSignal is None:
        return signals, metrics

    use_rs = _settings_bool("use_relative_strength_agent", True)
    use_intraday = _settings_bool("use_intraday_confirmation_agent", True)
    use_liquidity = _settings_bool("use_liquidity_agent", True)

    benchmark = _benchmark_for_agent(ticker)
    sector_key = _sector_key_for_agent(ticker)
    bars = _download_recent_bars([ticker, benchmark]) if (use_rs or use_intraday) else {}

    if use_rs and RelativeStrengthAgent is not None:
        try:
            sig = RelativeStrengthAgent().evaluate(ticker, bars, sector_key=sector_key, benchmark=benchmark)
        except Exception as exc:
            sig = _empty_agent_signal(ticker, "relative_strength_agent", f"Relative strength agent failed: {exc}")
        if sig is not None:
            signals.append(sig)
            metrics[sig.agent_name] = sig.metrics

    if use_intraday and IntradayConfirmationAgent is not None:
        try:
            sig = IntradayConfirmationAgent().evaluate(ticker, bars.get(ticker, pd.DataFrame()), intended_bias=direction_hint)
        except Exception as exc:
            sig = _empty_agent_signal(ticker, "intraday_confirmation_agent", f"Intraday confirmation agent failed: {exc}")
        if sig is not None:
            signals.append(sig)
            metrics[sig.agent_name] = sig.metrics

    if use_liquidity and LiquidityAgent is not None:
        try:
            liq_inputs = _fetch_liquidity_inputs(ticker, fallback_price=fallback_price)
            sig = LiquidityAgent().evaluate(ticker, **liq_inputs)
        except Exception as exc:
            sig = _empty_agent_signal(ticker, "liquidity_agent", f"Liquidity agent failed: {exc}")
        if sig is not None:
            signals.append(sig)
            metrics[sig.agent_name] = sig.metrics

    return signals, metrics


def _blend_new_agent_score(ticker: str, raw_score: float, signals: list[Any]) -> tuple[float, int, list[str], dict[str, dict[str, Any]]]:
    if not signals or FinalScoreBlender is None:
        return raw_score, 0, [], {}

    try:
        blended = FinalScoreBlender().blend(ticker, raw_score, signals)
        return (
            float(blended.adjusted_score),
            int(blended.conviction_delta),
            list(blended.reasons),
            dict(blended.agent_metrics),
        )
    except Exception as exc:
        logger.warning("FinalScoreBlender failed for %s: %s", ticker, exc)
        return raw_score, 0, [f"new_agents: scoring blend failed: {exc}"], {}


def _apply_liquidity_gate(
    action: str,
    edge: str,
    size: str,
    conviction: int,
    agent_metrics: dict[str, dict[str, Any]],
    risk_flags: list[str],
) -> tuple[str, str, str, int]:
    liq = agent_metrics.get("liquidity_agent", {}) if agent_metrics else {}
    spread_pct = _safe_float(liq.get("spread_pct"), None)
    dollar_volume = _safe_float(liq.get("dollar_volume"), None)

    max_spread = _settings_float("max_spread_pct", 0.50)
    warn_spread = _settings_float("liquidity_warning_spread_pct", 0.25)
    min_dollar_volume = _settings_float("min_dollar_volume", 10_000_000)

    if spread_pct is not None and spread_pct > max_spread:
        risk_flags.append(f"liquidity: blocked trade because spread is {spread_pct:.3f}%")
        return "WATCH", "WEAK", "0%", min(conviction, 45)

    if dollar_volume is not None and dollar_volume < min_dollar_volume:
        risk_flags.append(f"liquidity: blocked trade because dollar volume is only ${dollar_volume:,.0f}")
        return "WATCH", "WEAK", "0%", min(conviction, 45)

    if spread_pct is not None and spread_pct > warn_spread:
        risk_flags.append(f"liquidity: spread is elevated at {spread_pct:.3f}%")
        conviction = max(5, conviction - 8)
        if conviction < 45:
            return "WATCH", "WEAK", "0%", conviction

    return action, edge, size, conviction


def _is_readthrough_friendly(ticker: str) -> bool:
    return any_exposure(
        ticker,
        {
            "ai_compute", "server_cpu", "cloud_capex", "semi", "semi_equipment",
            "foundry", "memory", "data_center", "enterprise_software",
        },
    )


def _is_semi_or_ai_name(ticker: str) -> bool:
    return any_exposure(
        ticker,
        {"ai_compute", "semi", "server_cpu", "semi_equipment", "foundry", "memory", "data_center"},
    )


def _is_ev_or_story_stock(ticker: str) -> bool:
    return any_exposure(ticker, {"ev", "autonomy"})


def _positive_trend(row: dict[str, Any]) -> bool:
    return (
        float(row["ma_gap_10_50"]) > 0
        and float(row["macd_hist"]) > 0
        and float(row["prev_5d_return"]) > 0
    )


def _negative_trend(row: dict[str, Any]) -> bool:
    return (
        float(row["ma_gap_10_50"]) < 0
        and float(row["macd_hist"]) < 0
        and float(row["prev_5d_return"]) < 0
    )


def _overextension_penalty(ticker: str, row: dict[str, Any]) -> dict[str, float]:
    rsi = float(row["rsi_14"])
    prev_5d = float(row["prev_5d_return"])
    prev_20d = float(row["prev_20d_return"])
    z20 = float(row["zscore_20"])

    penalty = 0.0

    if rsi >= 92:
        penalty += 0.22
    elif rsi >= 88:
        penalty += 0.16
    elif rsi >= 82:
        penalty += 0.08

    if prev_5d >= 0.25:
        penalty += 0.22
    elif prev_5d >= 0.18:
        penalty += 0.14
    elif prev_5d >= 0.12:
        penalty += 0.07

    if prev_20d >= 0.45:
        penalty += 0.12
    elif prev_20d >= 0.30:
        penalty += 0.07

    if z20 >= 2.4:
        penalty += 0.12
    elif z20 >= 1.9:
        penalty += 0.07

    if _is_semi_or_ai_name(ticker):
        penalty *= 0.84

    if is_momentum_name(ticker):
        penalty *= 0.58
    elif _is_ev_or_story_stock(ticker):
        penalty *= 0.90

    penalty = max(0.0, min(0.38, penalty))

    return {
        "score_penalty": penalty,
        "move_multiplier": max(0.62, 1.0 - 0.45 * penalty / 0.38) if penalty > 0 else 1.0,
        "conviction_penalty": round(18 * penalty / 0.38) if penalty > 0 else 0,
        "is_overextended": penalty >= 0.12,
    }


def _fresh_setup_bonus(ticker: str, row: dict[str, Any]) -> dict[str, float]:
    rsi = float(row["rsi_14"])
    prev_5d = float(row["prev_5d_return"])
    prev_20d = float(row["prev_20d_return"])
    z20 = float(row["zscore_20"])

    bullish_bonus = 0.0
    bearish_bonus = 0.0

    if _positive_trend(row):
        if 52 <= rsi <= 74 and 0.00 <= prev_5d <= 0.10 and z20 <= 1.4:
            bullish_bonus += 0.10
        elif 48 <= rsi <= 78 and prev_5d <= 0.12 and z20 <= 1.7:
            bullish_bonus += 0.05

    if _negative_trend(row):
        if 26 <= rsi <= 48 and -0.12 <= prev_5d <= 0.0 and z20 >= -1.4:
            bearish_bonus += 0.10
        elif 22 <= rsi <= 52 and prev_5d >= -0.15 and z20 >= -1.8:
            bearish_bonus += 0.05

    if abs(prev_20d) > 0.35:
        bullish_bonus *= 0.8
        bearish_bonus *= 0.8

    if _is_semi_or_ai_name(ticker):
        bullish_bonus *= 1.10

    return {
        "bullish_bonus": bullish_bonus,
        "bearish_bonus": bearish_bonus,
        "is_fresh": max(bullish_bonus, bearish_bonus) >= 0.08,
    }


def _high_beta_negative_override(
    ticker: str,
    row: dict[str, Any],
    news: ModifierSignal,
    probs: dict[str, float],
    expected_move: float,
) -> dict[str, Any] | None:
    if not _is_ev_or_story_stock(ticker):
        return None

    negative_reaction_signal = float(news.metadata.get("negative_reaction_signal", 0.0)) if getattr(news, "metadata", None) else 0.0
    catalyst_quality = float(news.metadata.get("catalyst_quality", 0.0)) if getattr(news, "metadata", None) else 0.0
    mixed_news = bool(news.metadata.get("mixed_news", False)) if getattr(news, "metadata", None) else False

    bad_trend = float(row["ma_gap_10_50"]) < 0 and float(row["macd_hist"]) < 0
    weak_setup = catalyst_quality < 0.66
    market_not_convinced = probs["UP"] <= 0.70 and probs["DOWN"] >= 0.12

    if (
        negative_reaction_signal <= -0.60
        and bad_trend
        and weak_setup
        and (mixed_news or market_not_convinced or expected_move < 0.03)
    ):
        conviction = 74 if negative_reaction_signal <= -1.0 else 62
        edge = "STRONG" if conviction >= 70 else "MODERATE"
        return {
            "forecast_direction": "DOWN",
            "final_action": "SELL",
            "conviction_score": conviction,
            "estimated_edge": edge,
            "suggested_position_size": "1%" if conviction >= 70 else "0.5%",
            "override_reason": "EV/story-stock negative-reaction override triggered.",
        }

    return None


def _regime_from_row(row: dict[str, Any], news: ModifierSignal, sector_rt: ModifierSignal, ticker: str) -> str:
    news_quality = float(news.metadata.get("catalyst_quality", 0.0)) if getattr(news, "metadata", None) else 0.0
    sector_quality = float(sector_rt.metadata.get("readthrough_quality", 0.0)) if getattr(sector_rt, "metadata", None) else 0.0

    if float(row["vix_close"]) >= 28 or float(row["vix_zscore_20"]) >= 1.25:
        return "HIGH_VOL"

    if _is_readthrough_friendly(ticker):
        if news_quality >= 0.56 or sector_quality >= 0.58:
            return "EVENTFUL"
    else:
        if news_quality >= 0.64 or sector_quality >= 0.68:
            return "EVENTFUL"

    if abs(float(row["prev_5d_return"])) >= 0.09 or abs(float(row["zscore_20"])) >= 2.25:
        return "EVENTFUL"

    return "NORMAL"


def _predict_base_probs(bundle, row_df: pd.DataFrame) -> dict[str, float]:
    labels = ["DOWN", "NEUTRAL", "UP"]

    def convert_probs(model_probs, classes):
        out = {k: 0.0 for k in labels}
        for cls, prob in zip(classes, model_probs):
            mapped = "DOWN" if cls == -1 else "UP" if cls == 1 else "NEUTRAL"
            out[mapped] = float(prob)
        return out

    p1 = convert_probs(bundle.logistic.predict_proba(row_df)[0], bundle.logistic.classes_)
    p2 = convert_probs(bundle.gbt_classifier.predict_proba(row_df)[0], bundle.gbt_classifier.classes_)
    p3 = convert_probs(bundle.rf_classifier.predict_proba(row_df)[0], bundle.rf_classifier.classes_)

    base = {k: (0.33 * p1[k] + 0.42 * p2[k] + 0.25 * p3[k]) for k in labels}
    total = sum(base.values())
    return {k: base[k] / total for k in labels}


def _softmax_shift(base_probs: dict[str, float], shifts: dict[str, float]) -> dict[str, float]:
    labels = ["DOWN", "NEUTRAL", "UP"]
    logits = {k: math.log(max(base_probs.get(k, 1e-6), 1e-6)) for k in labels}
    for k, v in shifts.items():
        logits[k] += v
    exps = {k: math.exp(v) for k, v in logits.items()}
    total = sum(exps.values())
    return {k: exps[k] / total for k in labels}


def _modifier_weights_for_regime(ticker: str, news: ModifierSignal) -> dict[str, float]:
    news_q = float(news.metadata.get("catalyst_quality", 0.0)) if getattr(news, "metadata", None) else 0.0
    guidance = float(news.metadata.get("guidance_signal", 0.0)) if getattr(news, "metadata", None) else 0.0
    relief = float(news.metadata.get("relief_rally_signal", 0.0)) if getattr(news, "metadata", None) else 0.0

    weights = {"news": 0.22, "sector": 0.14, "poly": 0.07, "truth": 0.02}

    if news_q >= 0.68 or abs(guidance) >= 0.18:
        weights["news"] += 0.06
    if relief >= 0.12:
        weights["news"] += 0.03

    if is_strict_sector_name(ticker):
        weights["news"] *= 0.90
        weights["sector"] *= 0.78

    if _is_readthrough_friendly(ticker):
        weights["news"] *= 1.06
        weights["sector"] *= 1.18

    if is_high_beta_name(ticker) and not _is_ev_or_story_stock(ticker):
        weights["news"] *= 0.99
        weights["sector"] *= 1.02

    if _is_ev_or_story_stock(ticker):
        weights["news"] *= 0.95
        weights["sector"] *= 0.92

    if is_financial_name(ticker):
        weights["news"] *= 0.88
        weights["sector"] *= 0.72

    return weights


def _apply_modifiers(
    ticker: str,
    base_probs: dict[str, float],
    row: dict[str, Any],
    news: ModifierSignal,
    poly: ModifierSignal,
    truth: ModifierSignal,
    sector_rt: ModifierSignal,
) -> tuple[dict[str, float], float, list[str], list[str], float]:
    shifts = {"UP": 0.0, "NEUTRAL": 0.0, "DOWN": 0.0}
    expected_move_delta = 0.0
    drivers: list[str] = []
    risks: list[str] = []
    validity_parts: list[float] = []

    weights = _modifier_weights_for_regime(ticker, news)

    signal_map = [
        (news, weights["news"]),
        (sector_rt, weights["sector"]),
        (poly, weights["poly"]),
        (truth, weights["truth"]),
    ]

    mixed_news = bool(news.metadata.get("mixed_news", False)) if getattr(news, "metadata", None) else False
    catalyst_quality = float(news.metadata.get("catalyst_quality", 0.0)) if getattr(news, "metadata", None) else 0.0
    guidance_signal = float(news.metadata.get("guidance_signal", 0.0)) if getattr(news, "metadata", None) else 0.0
    relief_signal = float(news.metadata.get("relief_rally_signal", 0.0)) if getattr(news, "metadata", None) else 0.0
    negative_reaction_signal = float(news.metadata.get("negative_reaction_signal", 0.0)) if getattr(news, "metadata", None) else 0.0
    sector_quality = float(sector_rt.metadata.get("readthrough_quality", 0.0)) if getattr(sector_rt, "metadata", None) else 0.0

    for signal, weight in signal_map:
        if signal.validity <= 0:
            continue

        effective = signal.bias * signal.validity * weight
        validity_parts.append(signal.validity)

        if signal.name == "news":
            effective *= (0.55 + 0.70 * catalyst_quality)
            effective += 0.10 * guidance_signal + 0.05 * relief_signal + 1.20 * negative_reaction_signal
            if mixed_news:
                if _is_readthrough_friendly(ticker) and sector_quality >= 0.70:
                    effective *= 0.94
                else:
                    effective *= 0.82

        if signal.name == "sector_readthrough":
            effective *= (0.45 + 0.80 * sector_quality)

            if is_strict_sector_name(ticker) and sector_quality < 0.72:
                effective *= 0.60

            if _is_readthrough_friendly(ticker):
                if sector_quality >= 0.70:
                    effective *= 1.24
                elif sector_quality >= 0.58:
                    effective *= 1.12

            if is_financial_name(ticker) and sector_quality < 0.78:
                effective *= 0.55

        if signal.name == "polymarket":
            effective *= 0.90

        if signal.name == "truthsocial":
            effective *= 0.55
            if _is_semi_or_ai_name(ticker):
                effective *= 0.20

        if effective > 0:
            shifts["UP"] += effective
            shifts["DOWN"] -= effective * 0.34
        elif effective < 0:
            shifts["DOWN"] += abs(effective)
            shifts["UP"] -= abs(effective) * 0.34

        expected_move_delta += effective * 0.013

        if _is_semi_or_ai_name(ticker) and signal.name == "sector_readthrough" and signal.bias > 0:
            if sector_quality >= 0.45 and _positive_trend(row):
                expected_move_delta += 0.022
            elif sector_quality >= 0.35 and _positive_trend(row):
                expected_move_delta += 0.012

        if _is_semi_or_ai_name(ticker) and signal.name == "news" and catalyst_quality >= 0.25 and _positive_trend(row):
            expected_move_delta += 0.006

        if signal.details:
            drivers.append(f"{signal.name}: {signal.details[0]}")
        if signal.validity < 0.40:
            risks.append(f"{signal.name}: weak validity")

    adjusted = _softmax_shift(base_probs, shifts)
    validity = mean(validity_parts) if validity_parts else 0.0
    return adjusted, expected_move_delta, drivers[:4], risks[:4], validity


def _signal_score(
    ticker: str,
    probs: dict[str, float],
    expected_move: float,
    row: dict[str, Any],
    model_validity: float,
    news: ModifierSignal,
    sector_rt: ModifierSignal,
) -> float:
    prob_edge = probs["UP"] - probs["DOWN"]
    move_edge = math.tanh(expected_move * 16.0)

    reversal = 0.0
    if float(row["rsi_14"]) < 24 and float(row["zscore_20"]) < -1.8:
        reversal += 0.07
    if float(row["rsi_14"]) > 76 and float(row["zscore_20"]) > 1.8:
        reversal -= 0.025 if is_momentum_name(ticker) else 0.07

    trend_penalty = -0.04 if (float(row["ma_gap_10_50"]) < 0 and float(row["macd_hist"]) < 0) else 0.0

    news_quality = float(news.metadata.get("catalyst_quality", 0.0)) if getattr(news, "metadata", None) else 0.0
    mixed_news = bool(news.metadata.get("mixed_news", False)) if getattr(news, "metadata", None) else False
    guidance_signal = float(news.metadata.get("guidance_signal", 0.0)) if getattr(news, "metadata", None) else 0.0
    relief_signal = float(news.metadata.get("relief_rally_signal", 0.0)) if getattr(news, "metadata", None) else 0.0
    negative_reaction_signal = float(news.metadata.get("negative_reaction_signal", 0.0)) if getattr(news, "metadata", None) else 0.0
    sector_quality = float(sector_rt.metadata.get("readthrough_quality", 0.0)) if getattr(sector_rt, "metadata", None) else 0.0

    score = 0.58 * prob_edge + 0.30 * move_edge + reversal + trend_penalty
    score += 0.10 * guidance_signal + 0.05 * relief_signal + 1.50 * negative_reaction_signal

    if mixed_news:
        if _is_readthrough_friendly(ticker) and sector_quality >= 0.70:
            score *= 0.94
        else:
            score *= 0.84

    if is_strict_sector_name(ticker):
        if news_quality < 0.68:
            score *= 0.72
        if mixed_news:
            score *= 0.88

    if is_megacap_tech_name(ticker):
        if score < 0 and expected_move > -0.020 and probs["DOWN"] < 0.55:
            score *= 0.18

    if _is_readthrough_friendly(ticker):
        if sector_quality >= 0.72:
            score += 0.10
        elif sector_quality >= 0.60:
            score += 0.05

        if expected_move >= 0.03 and probs["DOWN"] <= 0.12:
            score += 0.08
        elif expected_move >= 0.02 and probs["DOWN"] <= 0.15:
            score += 0.04

    if _is_semi_or_ai_name(ticker) and _positive_trend(row):
        if probs["UP"] - probs["DOWN"] >= 0.35:
            score += 0.06
        elif probs["UP"] - probs["DOWN"] >= 0.20:
            score += 0.03

    fresh = _fresh_setup_bonus(ticker, row)
    if score > 0:
        score += fresh["bullish_bonus"]
    elif score < 0:
        score -= fresh["bearish_bonus"]

    if is_financial_name(ticker):
        if news_quality < 0.72:
            score *= 0.70
        if sector_quality < 0.78:
            score *= 0.78
        if expected_move > 0 and probs["UP"] < 0.62:
            score *= 0.78

    if _is_ev_or_story_stock(ticker) and float(row["ma_gap_10_50"]) < 0 and float(row["macd_hist"]) < 0 and news_quality < 0.60:
        score -= 0.07 if is_momentum_name(ticker) else 0.12

    if _is_ev_or_story_stock(ticker) and negative_reaction_signal < 0:
        score += (1.25 if is_momentum_name(ticker) else 2.40) * negative_reaction_signal

    if is_momentum_name(ticker):
        continuation = bullish_continuation_score(ticker, row, probs, expected_move)
        if continuation >= 0.72 and score < 0:
            score *= 0.35
        elif continuation >= 0.60 and score < 0:
            score *= 0.55
        if continuation >= 0.72 and expected_move > -0.010 and probs["UP"] >= probs["DOWN"]:
            score += 0.055

    overext = _overextension_penalty(ticker, row)
    if overext["is_overextended"] and score > 0:
        score -= 0.45 * overext["score_penalty"]

    score *= (0.68 + 0.55 * model_validity)
    return float(max(-1.0, min(1.0, score)))


def _direction_from_score(ticker: str, score: float, expected_move: float, probs: dict[str, float]) -> str:
    if score >= settings.neutral_score_band:
        return "UP"
    if score <= -settings.neutral_score_band:
        if is_megacap_tech_name(ticker) and expected_move > -0.020 and probs["DOWN"] < 0.55:
            return "NEUTRAL"
        return "DOWN"
    return "NEUTRAL"


def _conviction_from_score(
    ticker: str,
    score: float,
    validity: float,
    news: ModifierSignal,
    sector_rt: ModifierSignal,
    probs: dict[str, float],
    expected_move: float,
    row: dict[str, Any],
) -> int:
    catalyst_quality = float(news.metadata.get("catalyst_quality", 0.0)) if getattr(news, "metadata", None) else 0.0
    mixed_news = bool(news.metadata.get("mixed_news", False)) if getattr(news, "metadata", None) else False
    negative_reaction_signal = float(news.metadata.get("negative_reaction_signal", 0.0)) if getattr(news, "metadata", None) else 0.0
    sector_quality = float(sector_rt.metadata.get("readthrough_quality", 0.0)) if getattr(sector_rt, "metadata", None) else 0.0

    raw = abs(score) * 100 * (0.72 + 0.55 * validity)

    if catalyst_quality >= 0.72:
        raw += 8
    if sector_quality >= 0.75:
        raw += 5
    if mixed_news:
        if _is_readthrough_friendly(ticker) and sector_quality >= 0.70:
            raw -= 3
        else:
            raw -= 8
    if is_strict_sector_name(ticker) and catalyst_quality < 0.68:
        raw -= 8

    if _is_readthrough_friendly(ticker):
        if expected_move >= 0.05 and probs["DOWN"] <= 0.10:
            raw += 16
        elif expected_move >= 0.03 and probs["DOWN"] <= 0.12:
            raw += 10
        elif expected_move >= 0.02 and probs["DOWN"] <= 0.15:
            raw += 6

        if sector_quality >= 0.70:
            raw += 6

    if _is_semi_or_ai_name(ticker) and _positive_trend(row):
        if probs["UP"] >= 0.62 and expected_move >= 0.03:
            raw += 6
        elif probs["UP"] >= 0.58 and expected_move >= 0.02:
            raw += 3

    fresh = _fresh_setup_bonus(ticker, row)
    if score > 0:
        raw += 10 * fresh["bullish_bonus"]
    elif score < 0:
        raw += 10 * fresh["bearish_bonus"]

    if is_financial_name(ticker):
        if catalyst_quality < 0.72:
            raw -= 8
        if sector_quality < 0.78:
            raw -= 6

    if _is_ev_or_story_stock(ticker) and float(row["ma_gap_10_50"]) < 0 and float(row["macd_hist"]) < 0 and catalyst_quality < 0.60:
        raw -= 14

    if _is_ev_or_story_stock(ticker) and negative_reaction_signal < 0:
        raw += (24 if is_momentum_name(ticker) else 45) * negative_reaction_signal

    overext = _overextension_penalty(ticker, row)
    raw -= overext["conviction_penalty"]

    return int(max(5, min(95, round(raw))))


def _map_action(
    ticker: str,
    row: dict[str, Any],
    direction: str,
    score: float,
    expected_move: float,
    conviction: int,
    news: ModifierSignal,
    sector_rt: ModifierSignal,
    probs: dict[str, float],
) -> tuple[str, str, str]:
    abs_score = abs(score)

    catalyst_quality = (
        float(news.metadata.get("catalyst_quality", 0.0))
        if getattr(news, "metadata", None)
        else 0.0
    )

    mixed_news = (
        bool(news.metadata.get("mixed_news", False))
        if getattr(news, "metadata", None)
        else False
    )

    negative_reaction_signal = (
        float(news.metadata.get("negative_reaction_signal", 0.0))
        if getattr(news, "metadata", None)
        else 0.0
    )

    sector_quality = (
        float(sector_rt.metadata.get("readthrough_quality", 0.0))
        if getattr(sector_rt, "metadata", None)
        else 0.0
    )

    overext = _overextension_penalty(ticker, row)
    fresh = _fresh_setup_bonus(ticker, row)

    if direction == "UP":
        can_buy = (
            abs_score >= settings.buy_score_min
            and expected_move >= settings.buy_expected_move_min
            and (
                catalyst_quality >= 0.58
                or sector_quality >= 0.68
                or conviction >= 58
                or fresh["bullish_bonus"] >= 0.04
            )
        )

        if is_strict_sector_name(ticker) and catalyst_quality < 0.72:
            can_buy = False

        if mixed_news and not _is_readthrough_friendly(ticker):
            can_buy = False

        if _is_ev_or_story_stock(ticker) and negative_reaction_signal <= -0.20:
            can_buy = False

        if is_financial_name(ticker) and (
            catalyst_quality < 0.76 or sector_quality < 0.80
        ):
            can_buy = False

        # Do not let stretched continuation names become BUY unless the setup is genuinely strong.
        if (
            overext["is_overextended"]
            and conviction < 65
            and fresh["bullish_bonus"] < 0.10
        ):
            can_buy = False

        if _is_readthrough_friendly(ticker):
            if (
                conviction >= 45
                and expected_move >= 0.015
                and probs["DOWN"] <= 0.18
            ):
                can_buy = True

            if (
                sector_quality >= 0.62
                and expected_move >= 0.015
                and probs["DOWN"] <= 0.18
            ):
                can_buy = True

            if mixed_news and sector_quality >= 0.72 and catalyst_quality >= 0.50:
                can_buy = True

        # Re-check after readthrough override so overextended names do not sneak through.
        if (
            overext["is_overextended"]
            and conviction < 65
            and fresh["bullish_bonus"] < 0.10
        ):
            can_buy = False

        if can_buy:
            if conviction >= 70:
                return "BUY", "STRONG", "1%"

            if conviction >= 45:
                return "BUY", "MODERATE", "0.5%"

        if conviction >= 70:
            return "WATCH", "MODERATE", "0%"

        return "WATCH", "WEAK", "0%"

    if direction == "DOWN":
        can_sell = (
            abs_score >= settings.sell_score_min
            and (
                expected_move <= -settings.sell_expected_move_min
                or negative_reaction_signal <= -0.25
            )
            and (
                catalyst_quality >= 0.64
                or sector_quality >= 0.74
                or conviction >= 45
                or negative_reaction_signal <= -0.35
            )
        )

        if is_strict_sector_name(ticker) and mixed_news and catalyst_quality < 0.70:
            can_sell = False

        if (
            _is_ev_or_story_stock(ticker)
            and (
                expected_move <= -0.005
                or negative_reaction_signal <= -0.25
            )
            and probs["UP"] <= 0.40
            and probs["DOWN"] >= 0.12
        ):
            can_sell = True

        if can_sell:
            if conviction >= 70:
                return "SELL", "STRONG", "1%"

            if conviction >= 45:
                return "SELL", "MODERATE", "0.5%"

    if conviction >= 70:
        return "WATCH", "MODERATE", "0%"

    return "WATCH", "WEAK", "0%"


def run_pipeline(ticker: str | None = None) -> dict[str, Any]:
    ticker = normalize_ticker(ticker or settings.default_ticker)
    logger.info("Running trained weekly pipeline for %s", ticker)

    bundle = train_for_ticker(ticker)
    row = bundle.current_row
    row_df = pd.DataFrame([{col: row[col] for col in FEATURE_COLUMNS}], columns=FEATURE_COLUMNS)

    base_probs = _predict_base_probs(bundle, row_df)
    expected_move = float(bundle.regressor.predict(row_df)[0])

    news = fetch_news_modifier(ticker) if settings.news_enabled else ModifierSignal(name="news")
    poly = fetch_polymarket_modifier(ticker) if settings.polymarket_enabled else ModifierSignal(name="polymarket")
    truth = fetch_truthsocial_modifier(ticker) if settings.truthsocial_enabled else ModifierSignal(name="truthsocial")
    sector_rt = fetch_sector_readthrough_modifier(ticker) if settings.sector_readthrough_enabled else ModifierSignal(name="sector_readthrough")

    regime = _regime_from_row(row, news, sector_rt, ticker)

    adjusted_probs, move_delta, modifier_drivers, risk_flags, modifier_validity = _apply_modifiers(
        ticker, base_probs, row, news, poly, truth, sector_rt
    )
    expected_move += move_delta

    overext = _overextension_penalty(ticker, row)
    if expected_move > 0 and overext["is_overextended"]:
        expected_move *= overext["move_multiplier"]

    fresh = _fresh_setup_bonus(ticker, row)
    if expected_move > 0 and fresh["bullish_bonus"] > 0:
        expected_move *= 1.0 + 0.35 * fresh["bullish_bonus"]
    if expected_move < 0 and fresh["bearish_bonus"] > 0:
        expected_move *= 1.0 + 0.35 * fresh["bearish_bonus"]

    override = _high_beta_negative_override(ticker, row, news, adjusted_probs, expected_move)

    validation = bundle.validation
    base_validity = mean(
        [
            validation["logistic_accuracy"],
            validation["gbt_accuracy"],
            validation["rf_accuracy"],
            validation["ensemble_accuracy"],
        ]
    )
    model_validity = round(0.70 * base_validity + 0.30 * modifier_validity, 4)

    raw_score = _signal_score(ticker, adjusted_probs, expected_move, row, model_validity, news, sector_rt)

    pre_agent_direction = _direction_from_score(ticker, raw_score, expected_move, adjusted_probs)
    new_agent_signals, new_agent_metrics = _run_market_quality_agents(
        ticker=ticker,
        direction_hint=pre_agent_direction,
        fallback_price=_safe_float(row.get("current_price"), _safe_float(row.get("close"), None)),
    )
    raw_score, new_agent_conviction_delta, new_agent_reasons, blended_agent_metrics = _blend_new_agent_score(
        ticker, raw_score, new_agent_signals
    )
    if blended_agent_metrics:
        new_agent_metrics.update(blended_agent_metrics)

    archetype = ticker_archetype(ticker)
    momentum_score = bullish_continuation_score(ticker, row, adjusted_probs, expected_move)
    bearish_count, bearish_details = bearish_confirmation_count(
        ticker,
        row,
        adjusted_probs,
        expected_move,
        news.metadata if getattr(news, "metadata", None) else {},
        sector_rt.metadata if getattr(sector_rt, "metadata", None) else {},
    )

    if settings.enable_momentum_override and is_momentum_name(ticker) and raw_score < 0:
        if bearish_count < settings.momentum_sell_confirmation_min:
            risk_flags.append(
                "momentum: blocked aggressive SELL because high-beta name lacks bearish confirmation"
            )
            raw_score *= 0.25
            if momentum_score >= 0.70 and adjusted_probs["UP"] >= adjusted_probs["DOWN"] and expected_move > -0.010:
                raw_score = max(raw_score, settings.neutral_score_band + 0.025)
                expected_move = max(expected_move, 0.010)

    direction = _direction_from_score(ticker, raw_score, expected_move, adjusted_probs)
    setup_type = setup_type_for_signal(ticker, direction, row, expected_move)
    conviction = _conviction_from_score(ticker, raw_score, model_validity, news, sector_rt, adjusted_probs, expected_move, row)
    conviction = int(max(5, min(95, conviction + new_agent_conviction_delta)))

    if direction == "DOWN" and is_dangerous_short(ticker) and bearish_count < settings.momentum_sell_confirmation_min:
        conviction = max(5, conviction - settings.dangerous_short_conviction_penalty)

    if direction == "DOWN" and setup_type == "MEAN_REVERSION_SHORT" and bearish_count < settings.momentum_sell_confirmation_min:
        conviction = min(conviction, settings.reversal_conviction_cap)

    action, edge, size = _map_action(ticker, row, direction, raw_score, expected_move, conviction, news, sector_rt, adjusted_probs)
    action, edge, size, conviction = _apply_liquidity_gate(
        action, edge, size, conviction, new_agent_metrics, risk_flags
    )

    if override:
        if not (is_momentum_name(ticker) and bearish_count < settings.momentum_sell_confirmation_min):
            direction = override["forecast_direction"]
            action = override["final_action"]
            conviction = override["conviction_score"]
            edge = override["estimated_edge"]
            size = override["suggested_position_size"]
            setup_type = setup_type_for_signal(ticker, direction, row, expected_move)
        else:
            risk_flags.append("momentum: negative override ignored because confirmation count is too low")
            override = None

    # If an override changed the action, the liquidity gate still gets the last word.
    action, edge, size, conviction = _apply_liquidity_gate(
        action, edge, size, conviction, new_agent_metrics, risk_flags
    )

    drivers = []
    if float(row["rsi_14"]) < 24:
        drivers.append("technical: deeply oversold setup")
    elif float(row["rsi_14"]) > 76:
        drivers.append("technical: overbought setup")

    if overext["is_overextended"]:
        drivers.append("risk: stock is extended after a large recent run")
    if fresh["is_fresh"]:
        drivers.append("setup: fresher trend setup than last week’s crowded winners")

    if abs(expected_move) >= 0.012:
        drivers.append(f"model: expected move {expected_move:.2%}")

    drivers.extend(modifier_drivers)
    drivers.extend(new_agent_reasons)
    if override:
        drivers.insert(0, override["override_reason"])
    drivers = drivers[:6] or ["model: no strong validated edge"]

    conflicts = []
    if float(row["ma_gap_10_50"]) < 0:
        conflicts.append("trend: short/long moving-average gap is negative")
    if float(row["macd_hist"]) < 0:
        conflicts.append("trend: MACD histogram is negative")
    if bool(news.metadata.get("mixed_news", False)):
        conflicts.append("news: catalyst flow is mixed")
    if overext["is_overextended"]:
        conflicts.append("momentum: prior move may already be overextended")

    intraday_metrics = new_agent_metrics.get("intraday_confirmation_agent", {}) if new_agent_metrics else {}
    if intraday_metrics.get("confirms_bias") is False:
        conflicts.append("intraday: live tape conflicts with intended direction")

    conflicts = conflicts[:4]

    result = PredictionResult(
        ticker=ticker,
        run_timestamp=utc_now().isoformat(timespec="seconds"),
        forecast_window=row["expected_window"],
        model_validity=model_validity,
        primary_regime=regime,
        forecast_direction=direction,
        final_action=action,
        conviction_score=conviction,
        estimated_edge=edge,
        suggested_position_size=size,
        expected_move_pct=round(expected_move * 100, 2),
        probability_up=round(adjusted_probs["UP"], 4),
        probability_neutral=round(adjusted_probs["NEUTRAL"], 4),
        probability_down=round(adjusted_probs["DOWN"], 4),
        raw_signal_score=round(raw_score, 4),
        drivers=drivers,
        conflicts=conflicts,
        risk_flags=risk_flags[:4],
        reason=(
            f"Ensemble weekly opportunity model trained on {bundle.training_rows} Monday-reference rows; "
            f"ensemble validation opportunity accuracy={validation['ensemble_accuracy']:.2%}. "
            f"Target is a meaningful intraweek tradeable move, not only Friday close. "
            f"Momentum-stock SELL calls require confirmation before conviction can rise. "
            f"New market-quality agents check relative strength, intraday confirmation, and liquidity."
        ),
        monday_close_reference=row["monday_close"],
        target_type="WEEKLY_TRADE_OPPORTUNITY",
        success_threshold_pct=settings.opportunity_threshold_pct,
        ticker_archetype=archetype,
        setup_type=setup_type,
        opportunity_conviction_score=conviction,
        friday_hold_conviction_score=None,
        bearish_confirmations=bearish_count,
        bearish_confirmation_details=bearish_details[:4],
        momentum_continuation_score=round(momentum_score, 3),
    )

    debug_payload = {
        "ticker": ticker,
        "validation": validation,
        "training_rows": bundle.training_rows,
        "current_row": row,
        "base_probs": base_probs,
        "adjusted_probs": adjusted_probs,
        "expected_move": expected_move,
        "raw_signal_score": raw_score,
        "overextension": overext,
        "fresh_setup": fresh,
        "archetype": archetype,
        "setup_type": setup_type,
        "momentum_continuation_score": momentum_score,
        "bearish_confirmations": bearish_count,
        "bearish_confirmation_details": bearish_details,
        "override": override,
        "modifiers": {
            "news": news.to_dict(),
            "polymarket": poly.to_dict(),
            "truthsocial": truth.to_dict(),
            "sector_readthrough": sector_rt.to_dict(),
        },
        "new_agents": {
            signal.agent_name: {
                "score": signal.score,
                "direction": signal.direction,
                "confidence": signal.confidence,
                "risk_level": signal.risk_level,
                "reason": signal.reason,
                "metrics": signal.metrics,
            }
            for signal in new_agent_signals
        },
        "new_agent_metrics": new_agent_metrics,
        "new_agent_conviction_delta": new_agent_conviction_delta,
        "final_output": result.to_dict(),
    }

    run_id = save_run(ticker, result.run_timestamp, result.forecast_window, result.to_dict(), debug_payload)
    logger.info("Saved run %s", run_id)
    return {"run_id": run_id, "ticker": ticker, "final_output": result.to_dict(), "debug": debug_payload}
