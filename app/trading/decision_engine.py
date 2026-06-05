from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class TradeSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"


class TradeInstrument(str, Enum):
    STOCK = "stock"
    OPTION = "option"


@dataclass(frozen=True)
class TradeDecision:
    ticker: str
    side: TradeSide
    model_action: str
    model_direction: str
    conviction_score: int
    estimated_edge: str
    expected_move_pct: float
    setup_type: str
    reason: str
    take_profit_pct: float
    stop_loss_pct: float
    trailing_stop_pct: float
    instrument: TradeInstrument = TradeInstrument.STOCK
    underlying_ticker: str | None = None
    option_symbol: str | None = None
    option_type: str | None = None

    @property
    def should_trade(self) -> bool:
        return self.side == TradeSide.LONG

    @property
    def is_option(self) -> bool:
        return False


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


class TradeDecisionEngine:
    """
    Xanax stock-only decision engine.

    Rules:
      - stock only
      - long only
      - BUY / UP only
      - no shorts
      - no options
    """

    def __init__(
        self,
        min_conviction: int,
        allow_shorts: bool,
        default_take_profit_pct: float,
        default_stop_loss_pct: float,
        default_trailing_stop_pct: float,
        require_moderate_edge: bool = True,
        instrument_mode: str = "stock",
        allow_options: bool = False,
    ) -> None:
        self.min_conviction = min_conviction
        self.allow_shorts = False
        self.default_take_profit_pct = default_take_profit_pct
        self.default_stop_loss_pct = default_stop_loss_pct
        self.default_trailing_stop_pct = default_trailing_stop_pct
        self.require_moderate_edge = require_moderate_edge
        self.instrument_mode = "stock"
        self.allow_options = False

    def from_model_output(self, final_output: dict[str, Any]) -> TradeDecision:
        return self.from_model_output_all(final_output)[0]

    def from_model_output_all(self, final_output: dict[str, Any]) -> list[TradeDecision]:
        ticker = str(final_output.get("ticker", "")).strip().upper()
        action = str(final_output.get("final_action", "WATCH")).strip().upper()
        direction = str(final_output.get("forecast_direction", "NEUTRAL")).strip().upper()
        edge = str(final_output.get("estimated_edge", "WEAK")).strip().upper()
        setup_type = str(final_output.get("setup_type", "NO_CLEAN_SETUP")).strip().upper()
        conviction = _safe_int(final_output.get("conviction_score"), 0)
        expected_move_pct = _safe_float(final_output.get("expected_move_pct"), 0.0)
        reason = str(final_output.get("reason", ""))

        base = dict(
            ticker=ticker,
            underlying_ticker=ticker,
            model_action=action,
            model_direction=direction,
            conviction_score=conviction,
            estimated_edge=edge,
            expected_move_pct=expected_move_pct,
            setup_type=setup_type,
            take_profit_pct=self._target_from_expected_move(expected_move_pct),
            stop_loss_pct=self.default_stop_loss_pct,
            trailing_stop_pct=self.default_trailing_stop_pct,
        )

        if not ticker:
            return [self._hold(base, "Missing ticker.")]

        if conviction < self.min_conviction:
            return [self._hold(base, f"No trade: conviction {conviction} is below minimum {self.min_conviction}.")]

        if self.require_moderate_edge and edge not in {"MODERATE", "STRONG"}:
            return [self._hold(base, f"No trade: estimated edge is {edge}, not MODERATE/STRONG.")]

        if action == "BUY" and direction == "UP":
            return [
                TradeDecision(
                    side=TradeSide.LONG,
                    instrument=TradeInstrument.STOCK,
                    option_type=None,
                    reason=f"Stock trade allowed by xanax: BUY/UP with conviction {conviction}. {reason}",
                    **base,
                )
            ]

        if action == "SELL" and direction == "DOWN":
            return [self._hold(base, "No trade: xanax is long-only, so SELL/DOWN is blocked.")]

        return [self._hold(base, f"No trade: model output is {action}/{direction}.")]

    def _target_from_expected_move(self, expected_move_pct: float) -> float:
        move = abs(expected_move_pct)
        if move <= 0:
            return self.default_take_profit_pct
        return max(1.0, min(move, 6.0))

    @staticmethod
    def _hold(base: dict[str, Any], reason: str) -> TradeDecision:
        return TradeDecision(
            side=TradeSide.HOLD,
            instrument=TradeInstrument.STOCK,
            option_type=None,
            reason=reason,
            **base,
        )
