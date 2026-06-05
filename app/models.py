from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class WeeklyFeatureRow:
    asof_date: str
    monday_close: float
    expected_window: str
    prev_1d_return: float
    prev_5d_return: float
    prev_20d_return: float
    rsi_14: float
    zscore_20: float
    ma_gap_5_20: float
    ma_gap_10_50: float
    realized_vol_20: float
    vix_close: float
    vix_zscore_20: float
    macd_hist: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModifierSignal:
    name: str
    bias: float = 0.0
    validity: float = 0.0
    details: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PredictionResult:
    ticker: str
    run_timestamp: str
    forecast_window: str
    model_validity: float
    primary_regime: str
    forecast_direction: str
    final_action: str
    conviction_score: int
    estimated_edge: str
    suggested_position_size: str
    expected_move_pct: float
    probability_up: float
    probability_neutral: float
    probability_down: float
    raw_signal_score: float
    drivers: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    reason: str = ""
    monday_close_reference: float | None = None

    # New weekly-opportunity fields. Kept optional/defaulted so old DB JSON stays readable.
    target_type: str = "WEEKLY_TRADE_OPPORTUNITY"
    success_threshold_pct: float = 1.0
    ticker_archetype: str = "NORMAL"
    setup_type: str = "NO_CLEAN_SETUP"
    opportunity_conviction_score: int | None = None
    friday_hold_conviction_score: int | None = None
    bearish_confirmations: int = 0
    bearish_confirmation_details: list[str] = field(default_factory=list)
    momentum_continuation_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
