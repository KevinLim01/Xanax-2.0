from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import settings


@dataclass(frozen=True)
class CapitalAllocation:
    reinvest_enabled: bool
    base_capital_usd: float
    starting_account_equity_usd: float | None
    current_account_equity_usd: float | None
    estimated_profit_usd: float
    allowed_total_exposure_usd: float
    target_position_size_usd: float
    reason: str


class CapitalAllocationAgent:
    """Controls how much the paper bot is allowed to invest.

    This is the live/paper equivalent of the reinvestment simulation:
      - start with AUTO_TRADE_BASE_CAPITAL_USD, usually 5000
      - compare current Alpaca equity to AUTO_TRADE_STARTING_ACCOUNT_EQUITY_USD
      - if the paper account is up $1000, allowed exposure becomes about $6000
      - position size becomes a fraction of allowed exposure, now 15% for the top-5 concentrated model

    It does not choose stocks. It only sets the weekly money cap.
    """

    name = "capital_allocation_agent"

    def __init__(self, broker: Any) -> None:
        self.broker = broker

    def evaluate(self) -> CapitalAllocation:
        base_capital = float(settings.auto_trade_base_capital_usd)

        if not settings.auto_trade_reinvest_enabled:
            return self._fixed(base_capital, "Reinvestment disabled; using fixed exposure settings.")

        starting_equity = float(settings.auto_trade_starting_account_equity_usd)
        if starting_equity <= 0:
            return self._fixed(
                base_capital,
                "Reinvestment enabled, but AUTO_TRADE_STARTING_ACCOUNT_EQUITY_USD is not set; using fixed base capital.",
            )

        current_equity = self._account_equity()
        if current_equity is None:
            return self._fixed(base_capital, "Could not read Alpaca account equity; using fixed base capital.")

        profit = current_equity - starting_equity
        allowed = base_capital + profit
        allowed = max(float(settings.auto_trade_reinvest_min_total_exposure_usd), allowed)
        allowed = min(float(settings.auto_trade_reinvest_max_total_exposure_usd), allowed)

        target_position = allowed * float(settings.auto_trade_reinvest_position_fraction)
        target_position = max(float(settings.auto_trade_reinvest_min_position_size_usd), target_position)
        target_position = min(float(settings.auto_trade_reinvest_max_position_size_usd), target_position)

        return CapitalAllocation(
            reinvest_enabled=True,
            base_capital_usd=round(base_capital, 2),
            starting_account_equity_usd=round(starting_equity, 2),
            current_account_equity_usd=round(current_equity, 2),
            estimated_profit_usd=round(profit, 2),
            allowed_total_exposure_usd=round(allowed, 2),
            target_position_size_usd=round(target_position, 2),
            reason=(
                f"Reinvestment cap active: base=${base_capital:,.2f}, "
                f"estimated paper P/L=${profit:,.2f}, allowed exposure=${allowed:,.2f}, "
                f"target position=${target_position:,.2f} ({float(settings.auto_trade_reinvest_position_fraction) * 100:.1f}% per stock)."
            ),
        )

    def _fixed(self, base_capital: float, reason: str) -> CapitalAllocation:
        allowed = float(settings.auto_trade_max_total_exposure_usd)
        target_position = float(settings.auto_trade_max_position_size_usd)
        return CapitalAllocation(
            reinvest_enabled=False,
            base_capital_usd=round(base_capital, 2),
            starting_account_equity_usd=None,
            current_account_equity_usd=None,
            estimated_profit_usd=0.0,
            allowed_total_exposure_usd=round(allowed, 2),
            target_position_size_usd=round(target_position, 2),
            reason=reason,
        )

    def _account_equity(self) -> float | None:
        try:
            account = self.broker.get_account()
        except Exception:
            return None

        for attr in ("equity", "portfolio_value", "cash"):
            value = getattr(account, attr, None)
            parsed = self._safe_float(value)
            if parsed is not None and parsed > 0:
                return parsed

        return None

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None
