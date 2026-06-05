from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass
class WeeklyOpportunityVerdict:
    ticker: str
    direction: str
    reference_price: float
    threshold_pct: float
    weekly_opportunity_result: str
    first_correct_time: str | None
    best_correct_time: str | None
    best_correct_price: float | None
    max_favorable_move_pct: float
    max_adverse_move_pct: float
    best_exit_pl_per_share: float
    held_to_end_pl_per_share: float
    missed_profit_gap_per_share: float
    error_type: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_direction(direction: str) -> str:
    d = str(direction or "").upper().strip()

    if d in {"BUY", "BULLISH", "LONG"}:
        return "UP"

    if d in {"SELL", "BEARISH", "SHORT"}:
        return "DOWN"

    return d


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _time_to_string(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return str(value)


def _is_monday_timestamp(value: Any) -> bool:
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return False
        return ts.weekday() == 0
    except Exception:
        text = str(value).lower()
        return "mon" in text or "monday" in text


def _hit_is_only_monday_touch(hits: pd.Series) -> bool:
    if hits.empty:
        return False

    if len(hits) != 1:
        return False

    return _is_monday_timestamp(hits.index[0])


def classify_opportunity(
    direction: str,
    max_favorable_move_pct: float,
    first_correct_time: str | None,
    threshold_pct: float = 1.0,
    hit_count: int = 0,
    only_monday_touch: bool = False,
) -> tuple[str, str]:
    """Return verdict and error type for a weekly opportunity call.

    Rule:
    - A single Monday-only touch does not count as a real weekly success.
    - A real success needs either a strong move or more than a tiny one-time touch.
    """

    direction = _normalize_direction(direction)

    if direction not in {"UP", "DOWN"}:
        return "NO_CALL", "NO_DIRECTION"

    if only_monday_touch:
        return "BAD_CALL", "ONLY_TRUE_ON_MONDAY"

    if hit_count <= 0:
        if 0.25 <= max_favorable_move_pct < threshold_pct:
            return "BARELY_TRUE", "WEAK_TRUE_ONLY"
        return "BAD_CALL", "NEVER_GAVE_REAL_MOVE"

    if max_favorable_move_pct >= 2.5:
        return "GOOD_CALL", "NONE"

    if max_favorable_move_pct >= threshold_pct:
        return "TRADEABLE_BUT_MESSY", "NONE"

    if 0.25 <= max_favorable_move_pct < threshold_pct:
        return "BARELY_TRUE", "WEAK_TRUE_ONLY"

    return "BAD_CALL", "NEVER_GAVE_REAL_MOVE"


def calculate_weekly_opportunity_verdict(
    ticker: str,
    price_df: pd.DataFrame,
    direction: str,
    reference_price: float,
    threshold_pct: float = 1.0,
) -> WeeklyOpportunityVerdict:
    """Evaluate an already-made weekly call using daily or intraday OHLC rows.

    Required:
    - price_df must contain close.
    - high/low are preferred. If missing, close is used.

    This is for auditing model calls, not for making predictions.
    """

    if price_df is None or price_df.empty:
        raise ValueError("price_df is empty")

    df = price_df.copy().sort_index()

    cols = {str(c).lower(): c for c in df.columns}

    close_col = cols.get("close")
    if close_col is None:
        raise ValueError("price_df must have a close column")

    high_col = cols.get("high", close_col)
    low_col = cols.get("low", close_col)

    for col in {close_col, high_col, low_col}:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[close_col])

    if df.empty:
        raise ValueError("price_df has no usable close values")

    direction = _normalize_direction(direction)
    ref = _safe_float(reference_price)

    if ref <= 0:
        raise ValueError("reference_price must be greater than 0")

    final_price = _safe_float(df.iloc[-1][close_col])

    if direction == "UP":
        favorable_series = df[high_col].astype(float) / ref - 1.0
        adverse_series = df[low_col].astype(float) / ref - 1.0

        best_idx = favorable_series.idxmax()
        best_price = _safe_float(df.loc[best_idx, high_col])

        best_pl = best_price - ref
        held_pl = final_price - ref

        max_fav = float(favorable_series.max() * 100.0)
        max_adv = float(adverse_series.min() * 100.0)

    elif direction == "DOWN":
        favorable_series = 1.0 - df[low_col].astype(float) / ref
        adverse_series = 1.0 - df[high_col].astype(float) / ref

        best_idx = favorable_series.idxmax()
        best_price = _safe_float(df.loc[best_idx, low_col])

        best_pl = ref - best_price
        held_pl = ref - final_price

        max_fav = float(favorable_series.max() * 100.0)
        max_adv = float(adverse_series.min() * 100.0)

    else:
        return WeeklyOpportunityVerdict(
            ticker=str(ticker).upper().strip(),
            direction=direction,
            reference_price=round(ref, 4),
            threshold_pct=threshold_pct,
            weekly_opportunity_result="NO_CALL",
            first_correct_time=None,
            best_correct_time=None,
            best_correct_price=None,
            max_favorable_move_pct=0.0,
            max_adverse_move_pct=0.0,
            best_exit_pl_per_share=0.0,
            held_to_end_pl_per_share=0.0,
            missed_profit_gap_per_share=0.0,
            error_type="NO_DIRECTION",
        )

    threshold_decimal = threshold_pct / 100.0
    hits = favorable_series[favorable_series >= threshold_decimal]

    hit_count = int(len(hits))
    only_monday_touch = _hit_is_only_monday_touch(hits)

    first_time = _time_to_string(hits.index[0]) if not hits.empty else None
    best_time = _time_to_string(best_idx)

    verdict, error_type = classify_opportunity(
        direction=direction,
        max_favorable_move_pct=max_fav,
        first_correct_time=first_time,
        threshold_pct=threshold_pct,
        hit_count=hit_count,
        only_monday_touch=only_monday_touch,
    )

    # Missed profit gap should show how much better the best possible exit was
    # compared with holding to the end. Never let it become negative.
    missed_gap = max(0.0, best_pl - held_pl)

    return WeeklyOpportunityVerdict(
        ticker=str(ticker).upper().strip(),
        direction=direction,
        reference_price=round(ref, 4),
        threshold_pct=threshold_pct,
        weekly_opportunity_result=verdict,
        first_correct_time=first_time,
        best_correct_time=best_time,
        best_correct_price=round(best_price, 4),
        max_favorable_move_pct=round(max_fav, 2),
        max_adverse_move_pct=round(max_adv, 2),
        best_exit_pl_per_share=round(best_pl, 4),
        held_to_end_pl_per_share=round(held_pl, 4),
        missed_profit_gap_per_share=round(missed_gap, 4),
        error_type=error_type,
    )
