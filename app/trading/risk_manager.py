from __future__ import annotations

from dataclasses import dataclass

from app.config import settings
from app.agents.capital_allocation_agent import CapitalAllocationAgent, CapitalAllocation
from app.trading.alpaca_broker import AlpacaBroker
from app.trading.decision_engine import TradeDecision, TradeSide, TradeInstrument


@dataclass(frozen=True)
class RiskResult:
    approved: bool
    reason: str
    position_size_usd: float = 0.0
    estimated_price: float = 0.0
    qty: float = 0.0
    option_symbol: str | None = None
    option_type: str | None = None
    option_expiration: str | None = None
    option_strike: float | None = None


class RiskManager:
    def __init__(
        self,
        broker: AlpacaBroker,
        max_position_size_usd: float,
        max_total_exposure_usd: float,
        require_market_open: bool = True,
    ) -> None:
        self.broker = broker
        self.max_position_size_usd = max_position_size_usd
        self.max_total_exposure_usd = max_total_exposure_usd
        self.require_market_open = require_market_open
        self.capital_agent = CapitalAllocationAgent(broker)
        self._reserved_new_positions = 0

    def check_entry(self, decision: TradeDecision) -> RiskResult:
        if not decision.should_trade:
            return RiskResult(False, decision.reason)

        if self.require_market_open and not self.broker.is_market_open():
            return RiskResult(False, "Market is closed.")

        if decision.instrument == TradeInstrument.OPTION:
            return self._check_option_entry(decision)

        return self._check_stock_entry(decision)

    def _check_stock_entry(self, decision: TradeDecision) -> RiskResult:
        # Final live model is stock-only, long-only by default.
        if decision.side == TradeSide.SHORT and not settings.auto_trade_allow_shorts:
            return RiskResult(False, "Shorts are disabled by the final stock-long model.")

        if self.broker.has_position_for_underlying(decision.ticker):
            return RiskResult(False, "Already holding this ticker or an option on this ticker; duplicate trade blocked.")

        active_positions = self._active_position_count()
        max_active = max(1, int(settings.auto_trade_max_active_positions))
        available_slots = max_active - active_positions - self._reserved_new_positions
        if available_slots <= 0:
            return RiskResult(
                False,
                f"No portfolio slot left. Active positions={active_positions}, reserved this run={self._reserved_new_positions}, max active={max_active}.",
            )

        try:
            if not self.broker.is_tradable(decision.ticker):
                return RiskResult(False, "Ticker is not tradable on Alpaca.")
        except Exception as exc:
            return RiskResult(False, f"Could not verify Alpaca tradability: {exc}")

        if decision.side == TradeSide.SHORT:
            try:
                if not self.broker.is_shortable(decision.ticker):
                    return RiskResult(False, "Ticker is not shortable on Alpaca.")
            except Exception as exc:
                return RiskResult(False, f"Could not verify Alpaca shortability: {exc}")

        allocation = self.capital_agent.evaluate()
        current_exposure = self._current_exposure()
        remaining_exposure = max(0.0, allocation.allowed_total_exposure_usd - current_exposure)

        if remaining_exposure <= 0:
            return RiskResult(False, f"Max exposure reached. {allocation.reason}")

        position_size = min(allocation.target_position_size_usd, remaining_exposure)
        if position_size <= 0:
            return RiskResult(False, f"No remaining exposure available. {allocation.reason}")

        account = self.broker.get_account()
        buying_power = float(account.buying_power)
        if buying_power < position_size:
            return RiskResult(False, f"Not enough Alpaca buying power for ${position_size:.2f}.")

        price = self.broker.get_latest_price(decision.ticker)
        if price <= 0:
            return RiskResult(False, "Invalid latest price.")

        qty = position_size / price
        if decision.side == TradeSide.SHORT:
            qty = int(qty)
            if qty < 1:
                return RiskResult(False, "Position size is too small to short at least 1 share.")

        self._reserved_new_positions += 1

        return RiskResult(
            approved=True,
            reason=(
                f"Approved for slot {active_positions + self._reserved_new_positions}/{max_active}. "
                f"{allocation.reason}"
            ),
            position_size_usd=position_size,
            estimated_price=price,
            qty=qty,
        )

    def _check_option_entry(self, decision: TradeDecision) -> RiskResult:
        # Options are deliberately blocked in the final live paper model.
        if not settings.auto_trade_allow_options:
            return RiskResult(False, "Options are disabled by the final stock-only model.")

        if settings.alpaca_trading_mode.lower() != "paper":
            return RiskResult(False, "Options automation is blocked outside paper mode in this project.")

        if decision.side != TradeSide.LONG:
            return RiskResult(False, "Only long calls/puts are allowed. Short options are blocked.")

        underlying = decision.underlying_ticker or decision.ticker
        if self.broker.has_position_for_underlying(underlying):
            return RiskResult(False, "Already holding this ticker or an option on this ticker; duplicate trade blocked.")

        option_type = decision.option_type
        if option_type not in {"call", "put"}:
            return RiskResult(False, "Missing option type. Expected call or put.")

        try:
            choice = self.broker.choose_simple_option_contract(
                underlying=underlying,
                option_type=option_type,
                min_dte=settings.auto_trade_options_min_dte,
                max_dte=settings.auto_trade_options_max_dte,
                strike_offset_pct=(
                    settings.auto_trade_options_call_strike_offset_pct
                    if option_type == "call"
                    else settings.auto_trade_options_put_strike_offset_pct
                ),
                max_contract_price=settings.auto_trade_options_max_contract_price,
            )
        except Exception as exc:
            return RiskResult(False, f"Could not choose option contract: {exc}")

        contracts = max(1, int(settings.auto_trade_options_contracts_per_trade))
        account = self.broker.get_account()
        options_buying_power = getattr(account, "options_buying_power", None)
        if options_buying_power is not None:
            try:
                if float(options_buying_power) <= 0:
                    return RiskResult(False, "No options buying power available.")
            except Exception:
                pass

        allocation = self.capital_agent.evaluate()
        rough_exposure = min(
            allocation.target_position_size_usd,
            settings.auto_trade_options_max_contract_price * 100 * contracts,
        )
        if not self._exposure_allowed(rough_exposure, allocation.allowed_total_exposure_usd):
            return RiskResult(False, f"Max total exposure would be exceeded. {allocation.reason}")

        return RiskResult(
            approved=True,
            reason=(
                f"Approved long {choice.option_type}: {choice.option_symbol}, "
                f"strike={choice.strike_price}, exp={choice.expiration_date}, dte={choice.days_to_expiration}. "
                f"{allocation.reason}"
            ),
            position_size_usd=rough_exposure,
            estimated_price=choice.underlying_price,
            qty=float(contracts),
            option_symbol=choice.option_symbol,
            option_type=choice.option_type,
            option_expiration=choice.expiration_date,
            option_strike=choice.strike_price,
        )

    def _current_exposure(self) -> float:
        positions = self.broker.list_positions()
        return sum(abs(p.market_value) for p in positions)

    def _active_position_count(self) -> int:
        return len(self.broker.list_positions())

    def available_slots(self) -> int:
        max_active = max(1, int(settings.auto_trade_max_active_positions))
        return max(0, max_active - self._active_position_count() - self._reserved_new_positions)

    def _exposure_allowed(self, additional_exposure: float, allowed_total_exposure_usd: float | None = None) -> bool:
        allowed = float(allowed_total_exposure_usd if allowed_total_exposure_usd is not None else self.max_total_exposure_usd)
        return self._current_exposure() + additional_exposure <= allowed
