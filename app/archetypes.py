from __future__ import annotations

from typing import Any

MOMENTUM_HIGH_BETA = {
    "TSLA", "NVDA", "AMD", "PLTR", "SMCI", "APP", "COIN", "MSTR", "META", "AMZN", "NFLX",
}

DANGEROUS_SHORT_NAMES = {
    "TSLA", "NVDA", "PLTR", "SMCI", "COIN", "MSTR", "AMD", "APP",
}

DEFENSIVE_STABLE = {
    "KO", "PG", "JNJ", "PEP", "WMT", "COST", "MCD", "PM", "MRK", "ABBV", "ABT",
}

FINANCIALS = {"JPM", "GS", "BAC", "WFC", "MS", "C"}

CYCLICALS = {"CAT", "DE", "NUE", "X", "F", "GM", "GE", "LIN", "XOM", "CVX"}

MEGA_CAP_TECH = {"AAPL", "MSFT", "GOOGL", "GOOG", "META", "AMZN", "NVDA"}


def ticker_archetype(ticker: str) -> str:
    t = ticker.upper().strip()
    if t in MOMENTUM_HIGH_BETA:
        return "MOMENTUM_HIGH_BETA"
    if t in DEFENSIVE_STABLE:
        return "DEFENSIVE_STABLE"
    if t in FINANCIALS:
        return "FINANCIAL"
    if t in CYCLICALS:
        return "CYCLICAL"
    if t in MEGA_CAP_TECH:
        return "MEGA_CAP_TECH"
    return "NORMAL"


def is_momentum_name(ticker: str) -> bool:
    return ticker_archetype(ticker) == "MOMENTUM_HIGH_BETA"


def is_dangerous_short(ticker: str) -> bool:
    return ticker.upper().strip() in DANGEROUS_SHORT_NAMES


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def bullish_continuation_score(ticker: str, row: dict[str, Any], probs: dict[str, float], expected_move: float) -> float:
    """0-1 score for weekly continuation setups.

    This is intentionally simple and old-school: price trend first, then probability and expected move.
    It stops TSLA/NVDA-style names from being punished just because they already moved.
    """
    score = 0.0
    prev_5d = _safe_float(row.get("prev_5d_return"))
    prev_20d = _safe_float(row.get("prev_20d_return"))
    ma_gap = _safe_float(row.get("ma_gap_10_50"))
    macd = _safe_float(row.get("macd_hist"))
    rsi = _safe_float(row.get("rsi_14"), 50.0)
    z20 = _safe_float(row.get("zscore_20"))

    if ma_gap > 0:
        score += 0.18
    if macd > 0:
        score += 0.18
    if prev_5d > 0:
        score += 0.16
    if prev_20d > 0:
        score += 0.10
    if 50 <= rsi <= 84:
        score += 0.10
    elif 84 < rsi <= 92 and is_momentum_name(ticker):
        score += 0.05
    if z20 <= 2.4:
        score += 0.06
    if probs.get("UP", 0.0) >= probs.get("DOWN", 0.0):
        score += 0.12
    if expected_move > 0:
        score += 0.10
    if is_momentum_name(ticker):
        score += 0.10

    return max(0.0, min(1.0, score))


def bearish_confirmation_count(
    ticker: str,
    row: dict[str, Any],
    probs: dict[str, float],
    expected_move: float,
    news_meta: dict[str, Any] | None = None,
    sector_meta: dict[str, Any] | None = None,
) -> tuple[int, list[str]]:
    news_meta = news_meta or {}
    sector_meta = sector_meta or {}
    confirmations: list[str] = []

    if _safe_float(row.get("ma_gap_10_50")) < 0:
        confirmations.append("below 10/50 trend")
    if _safe_float(row.get("macd_hist")) < 0:
        confirmations.append("negative MACD")
    if _safe_float(row.get("prev_5d_return")) < -0.015:
        confirmations.append("recent 5-day weakness")
    if probs.get("DOWN", 0.0) >= 0.55 and probs.get("DOWN", 0.0) > probs.get("UP", 0.0):
        confirmations.append("model probability favors downside")
    if expected_move <= -0.012:
        confirmations.append("expected opportunity move is negative")
    if _safe_float(news_meta.get("negative_reaction_signal")) <= -0.25:
        confirmations.append("negative news reaction")
    if _safe_float(news_meta.get("catalyst_quality")) >= 0.70 and _safe_float(news_meta.get("guidance_signal")) < -0.10:
        confirmations.append("bad catalyst/guidance")
    if _safe_float(sector_meta.get("readthrough_quality")) >= 0.74 and _safe_float(sector_meta.get("sector_bias")) < -0.05:
        confirmations.append("negative sector read-through")

    return len(confirmations), confirmations


def setup_type_for_signal(ticker: str, direction: str, row: dict[str, Any], expected_move: float) -> str:
    prev_5d = _safe_float(row.get("prev_5d_return"))
    rsi = _safe_float(row.get("rsi_14"), 50.0)
    z20 = _safe_float(row.get("zscore_20"))
    ma_gap = _safe_float(row.get("ma_gap_10_50"))
    macd = _safe_float(row.get("macd_hist"))

    if direction == "UP":
        if ma_gap > 0 and macd > 0 and prev_5d > 0:
            return "MOMENTUM_CONTINUATION"
        if rsi < 32 and z20 < -1.2:
            return "OVERSOLD_BOUNCE"
        if expected_move > 0:
            return "UP_OPPORTUNITY"
    if direction == "DOWN":
        if ma_gap < 0 and macd < 0 and prev_5d < 0:
            return "BREAKDOWN_CONTINUATION"
        if rsi > 78 and z20 > 1.7:
            return "MEAN_REVERSION_SHORT"
        if expected_move < 0:
            return "DOWN_OPPORTUNITY"
    return "NO_CLEAN_SETUP"
