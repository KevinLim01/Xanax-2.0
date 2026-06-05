from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

from app.config import settings
from app.trading.alpaca_broker import PositionSnapshot


MARKET_TZ = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class ExitDecision:
    should_exit: bool
    reason: str
    pnl_pct: float

    # Main classification used by PositionMonitor.
    # Values:
    #   HOLD
    #   TAKE_PROFIT
    #   STOP_LOSS
    #   FRIDAY_FORCE_EXIT
    #   OPTION_EXPIRATION
    exit_type: str = "HOLD"

    # Extra diagnostics saved by TradeLogger when the columns exist.
    take_profit_pct: float | None = None
    stop_loss_pct: float | None = None
    adaptive_take_profit_pct: float | None = None
    expected_move_pct: float | None = None


class ExitEngine:
    def __init__(
        self,
        take_profit_pct: float,
        stop_loss_pct: float,
        force_exit_friday_hour: int,
        force_exit_friday_minute: int,
    ) -> None:
        self.take_profit_pct = float(take_profit_pct)
        self.stop_loss_pct = float(stop_loss_pct)
        self.force_exit_friday_hour = int(force_exit_friday_hour)
        self.force_exit_friday_minute = int(force_exit_friday_minute)

        self.no_same_day_profit_exit = settings.no_same_day_profit_exit
        self.allow_same_day_stop_loss = settings.allow_same_day_stop_loss
        self.allow_same_day_friday_exit = settings.allow_same_day_friday_exit
        self.allow_same_day_option_expiration_exit = settings.allow_same_day_option_expiration_exit

        # Kept for compatibility, but the new monitor treats TAKE_PROFIT as advisory.
        self.adaptive_take_profit_enabled = settings.adaptive_take_profit_enabled
        self.adaptive_take_profit_floor_pct = settings.adaptive_take_profit_floor_pct
        self.adaptive_take_profit_cap_pct = settings.adaptive_take_profit_cap_pct
        self.adaptive_take_profit_move_fraction = settings.adaptive_take_profit_move_fraction

    def check(
        self,
        position: PositionSnapshot,
        now: datetime | None = None,
        opened_at: datetime | None = None,
    ) -> ExitDecision:
        now = now or datetime.now(MARKET_TZ)

        if now.tzinfo is None:
            now = now.replace(tzinfo=MARKET_TZ)

        now_et = now.astimezone(MARKET_TZ)

        pnl_pct = self._pnl_pct(position)
        same_day_position = self._is_same_market_day(opened_at, now_et)

        expected_move_pct = self._expected_move_pct(position)
        effective_take_profit_pct = self._effective_take_profit_pct(expected_move_pct)

        if self._is_option_expiring_soon(position):
            if same_day_position and not self.allow_same_day_option_expiration_exit:
                return self._decision(
                    False,
                    f"Hold. Same-day option expiration exit blocked. P/L is {pnl_pct:.2f}%.",
                    pnl_pct,
                    effective_take_profit_pct,
                    expected_move_pct,
                    exit_type="HOLD",
                )

            return self._decision(
                True,
                f"Option expiration risk exit. Contract expires {position.expiration_date}.",
                pnl_pct,
                effective_take_profit_pct,
                expected_move_pct,
                exit_type="OPTION_EXPIRATION",
            )

        if pnl_pct <= -self.stop_loss_pct:
            if same_day_position and not self.allow_same_day_stop_loss:
                return self._decision(
                    False,
                    f"Hold. Same-day stop-loss blocked. P/L is {pnl_pct:.2f}%.",
                    pnl_pct,
                    effective_take_profit_pct,
                    expected_move_pct,
                    exit_type="HOLD",
                )

            return self._decision(
                True,
                f"Stop-loss hit at {pnl_pct:.2f}%.",
                pnl_pct,
                effective_take_profit_pct,
                expected_move_pct,
                exit_type="STOP_LOSS",
            )

        if self._is_friday_exit(now_et):
            if same_day_position and not self.allow_same_day_friday_exit:
                return self._decision(
                    False,
                    f"Hold. Same-day Friday forced exit blocked. P/L is {pnl_pct:.2f}%.",
                    pnl_pct,
                    effective_take_profit_pct,
                    expected_move_pct,
                    exit_type="HOLD",
                )

            return self._decision(
                True,
                "Friday forced exit time reached.",
                pnl_pct,
                effective_take_profit_pct,
                expected_move_pct,
                exit_type="FRIDAY_FORCE_EXIT",
            )

        if pnl_pct >= effective_take_profit_pct:
            if same_day_position and self.no_same_day_profit_exit:
                return self._decision(
                    False,
                    (
                        f"Hold. Take-profit hit at {pnl_pct:.2f}% "
                        f"(target {effective_take_profit_pct:.2f}%), "
                        "but same-day profit exits are blocked until next trading day."
                    ),
                    pnl_pct,
                    effective_take_profit_pct,
                    expected_move_pct,
                    exit_type="HOLD",
                )

            return self._decision(
                True,
                f"Take-profit hit at {pnl_pct:.2f}% (target {effective_take_profit_pct:.2f}%).",
                pnl_pct,
                effective_take_profit_pct,
                expected_move_pct,
                exit_type="TAKE_PROFIT",
            )

        return self._decision(
            False,
            f"Hold. P/L is {pnl_pct:.2f}%. Advisory old take-profit target is {effective_take_profit_pct:.2f}%.",
            pnl_pct,
            effective_take_profit_pct,
            expected_move_pct,
            exit_type="HOLD",
        )

    def _decision(
        self,
        should_exit: bool,
        reason: str,
        pnl_pct: float,
        effective_take_profit_pct: float,
        expected_move_pct: float | None,
        exit_type: str = "HOLD",
    ) -> ExitDecision:
        return ExitDecision(
            should_exit=should_exit,
            reason=reason,
            pnl_pct=round(float(pnl_pct), 4),
            exit_type=exit_type,
            take_profit_pct=round(float(effective_take_profit_pct), 4),
            stop_loss_pct=round(float(self.stop_loss_pct), 4),
            adaptive_take_profit_pct=round(float(effective_take_profit_pct), 4),
            expected_move_pct=round(float(expected_move_pct), 4) if expected_move_pct is not None else None,
        )

    def _effective_take_profit_pct(self, expected_move_pct: float | None) -> float:
        if not self.adaptive_take_profit_enabled:
            return self.take_profit_pct

        if expected_move_pct is None or expected_move_pct <= 0:
            return self.take_profit_pct

        adaptive = abs(expected_move_pct) * self.adaptive_take_profit_move_fraction

        adaptive = max(self.adaptive_take_profit_floor_pct, adaptive)
        adaptive = min(self.adaptive_take_profit_cap_pct, adaptive)

        return min(self.take_profit_pct, adaptive)

    def _expected_move_pct(self, position: PositionSnapshot) -> float | None:
        for attr in ("expected_move_pct", "model_expected_move_pct", "target_expected_move_pct"):
            value = getattr(position, attr, None)
            parsed = self._safe_float_or_none(value)
            if parsed is not None:
                return abs(parsed)

        model_output = getattr(position, "model_output", None)
        if isinstance(model_output, dict):
            for key in ("expected_move_pct", "model_expected_move_pct", "target_expected_move_pct"):
                parsed = self._safe_float_or_none(model_output.get(key))
                if parsed is not None:
                    return abs(parsed)

        return None

    def _pnl_pct(self, position: PositionSnapshot) -> float:
        entry = self._safe_float(getattr(position, "avg_entry_price", 0.0))
        current = self._safe_float(getattr(position, "current_price", 0.0))

        if entry <= 0 or current <= 0:
            return 0.0

        qty = self._position_qty(position)

        if qty > 0:
            return ((current - entry) / entry) * 100.0

        if qty < 0:
            return ((entry - current) / entry) * 100.0

        side = str(getattr(position, "side", "") or "").strip().lower()

        if side == "long":
            return ((current - entry) / entry) * 100.0

        if side == "short":
            return ((entry - current) / entry) * 100.0

        return 0.0

    def _position_qty(self, position: PositionSnapshot) -> float:
        for attr in ("qty", "quantity", "shares"):
            value = getattr(position, attr, None)
            parsed = self._safe_float_or_none(value)
            if parsed is not None:
                return parsed

        return 0.0

    def _is_same_market_day(self, opened_at: datetime | None, now_et: datetime) -> bool:
        if opened_at is None:
            return False

        if opened_at.tzinfo is None:
            opened_at = opened_at.replace(tzinfo=MARKET_TZ)

        opened_et = opened_at.astimezone(MARKET_TZ)
        return opened_et.date() == now_et.date()

    def _is_friday_exit(self, now: datetime) -> bool:
        if now.weekday() != 4:
            return False

        if now.hour > self.force_exit_friday_hour:
            return True

        return now.hour == self.force_exit_friday_hour and now.minute >= self.force_exit_friday_minute

    def _is_option_expiring_soon(self, position: PositionSnapshot) -> bool:
        expiration_date = getattr(position, "expiration_date", None)

        if not expiration_date:
            return False

        try:
            exp = date.fromisoformat(str(expiration_date))
        except ValueError:
            return False

        days_left = (exp - date.today()).days
        return days_left <= settings.auto_trade_options_close_expiring_within_days

    def _safe_float(self, value, default: float = 0.0) -> float:
        parsed = self._safe_float_or_none(value)
        return default if parsed is None else parsed

    def _safe_float_or_none(self, value) -> float | None:
        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None
