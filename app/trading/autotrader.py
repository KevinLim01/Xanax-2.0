from __future__ import annotations

import inspect
from datetime import datetime
from typing import Any

from app.config import settings
from app.trading.alpaca_broker import AlpacaBroker
from app.trading.decision_engine import TradeDecisionEngine, TradeSide, TradeInstrument
from app.trading.exit_engine import ExitEngine
from app.trading.position_monitor import PositionMonitor
from app.trading.risk_manager import RiskManager
from app.trading.trade_logger import TradeLogger


def build_broker() -> AlpacaBroker:
    paper = settings.alpaca_trading_mode.lower() == "paper"
    return AlpacaBroker(settings.alpaca_api_key, settings.alpaca_secret_key, paper=paper)


def _day_adjusted_min_conviction() -> int:
    weekday = datetime.now().weekday()  # Monday=0
    by_day = {
        0: settings.day_min_conviction_monday,
        1: settings.day_min_conviction_tuesday,
        2: settings.day_min_conviction_wednesday,
        3: settings.day_min_conviction_thursday,
        4: settings.day_min_conviction_friday,
    }
    return max(int(settings.auto_trade_min_conviction), int(by_day.get(weekday, settings.auto_trade_min_conviction)))


def build_decision_engine() -> TradeDecisionEngine:
    return TradeDecisionEngine(
        min_conviction=_day_adjusted_min_conviction(),
        allow_shorts=settings.auto_trade_allow_shorts,
        default_take_profit_pct=settings.auto_trade_take_profit_pct,
        default_stop_loss_pct=settings.auto_trade_stop_loss_pct,
        default_trailing_stop_pct=settings.auto_trade_trailing_stop_pct,
        require_moderate_edge=settings.auto_trade_require_moderate_edge,
        instrument_mode=settings.auto_trade_instrument,
        allow_options=settings.auto_trade_allow_options,
    )


def build_exit_engine() -> ExitEngine:
    return ExitEngine(
        take_profit_pct=settings.auto_trade_take_profit_pct,
        stop_loss_pct=settings.auto_trade_stop_loss_pct,
        force_exit_friday_hour=settings.auto_trade_force_exit_friday_hour,
        force_exit_friday_minute=settings.auto_trade_force_exit_friday_minute,
    )


def _build_monitor_kwargs() -> dict[str, Any]:
    """
    Build optional PositionMonitor dependencies without breaking older monitor code.

    The new exit design needs the monitor to rerun the model for held positions and use
    the saved history summary. Older PositionMonitor versions do not accept these args,
    so this function checks the constructor before passing them.
    """
    kwargs: dict[str, Any] = {}

    try:
        params = inspect.signature(PositionMonitor).parameters
    except Exception:
        return kwargs

    if "model_runner" in params:
        from app.pipeline import run_pipeline

        kwargs["model_runner"] = run_pipeline

    if "history_agent" in params:
        try:
            from app.agents.history_lookup_agent import HistoryLookupAgent

            kwargs["history_agent"] = HistoryLookupAgent()
        except Exception as exc:
            print(f"History lookup agent unavailable for monitor: {exc}")

    return kwargs


class AutoTrader:
    def __init__(self) -> None:
        if settings.alpaca_trading_mode.lower() == "live" and not settings.allow_live_trading:
            raise RuntimeError("Live trading is blocked. Set ALPACA_TRADING_MODE=paper.")

        self.broker = build_broker()
        self.decision_engine = build_decision_engine()
        self.risk_manager = RiskManager(
            broker=self.broker,
            max_position_size_usd=settings.auto_trade_max_position_size_usd,
            max_total_exposure_usd=settings.auto_trade_max_total_exposure_usd,
            require_market_open=settings.auto_trade_require_market_open,
        )
        self.logger = TradeLogger()

    def process_model_output(self, final_output: dict[str, Any], dry_run: bool = False) -> bool:
        decisions = self.decision_engine.from_model_output_all(final_output)
        any_approved_or_sent = False

        for decision in decisions:
            if hasattr(self.risk_manager, "available_slots") and self.risk_manager.available_slots() <= 0:
                print("No portfolio slots left for this run. Skipping remaining candidates.")
                break
            if not dry_run and not settings.auto_trade_enabled:
                from app.trading.risk_manager import RiskResult

                risk = RiskResult(
                    False,
                    "AUTO_TRADE_ENABLED=false. Set it to true only when ready for Alpaca paper orders.",
                )
                print(
                    f"{decision.ticker}: {decision.instrument.value.upper()} {decision.side.value} | "
                    f"conviction={decision.conviction_score} | risk=False | {risk.reason}"
                )
                self.logger.log_trade_decision(decision, risk, "AUTO_TRADE_DISABLED", final_output)
                continue

            risk = self.risk_manager.check_entry(decision)

            label = decision.ticker
            if decision.instrument == TradeInstrument.OPTION and risk.option_symbol:
                label = f"{decision.ticker} -> {risk.option_symbol}"

            print(
                f"{label}: {decision.instrument.value.upper()} {decision.side.value} | "
                f"conviction={decision.conviction_score} | "
                f"risk={risk.approved} | {risk.reason}"
            )

            if not risk.approved:
                self.logger.log_trade_decision(decision, risk, "BLOCKED", final_output)
                continue

            if dry_run:
                if decision.instrument == TradeInstrument.OPTION:
                    print(
                        f"[DRY RUN] Would buy {int(risk.qty)} {risk.option_type} option contract(s): "
                        f"{risk.option_symbol}."
                    )
                else:
                    print(
                        f"[DRY RUN] Would send {decision.side.value} stock order for "
                        f"{decision.ticker}, qty={risk.qty:.4f}, about ${risk.position_size_usd:.2f}."
                    )
                self.logger.log_trade_decision(decision, risk, "DRY_RUN_APPROVED", final_output)
                any_approved_or_sent = True
                continue

            if decision.instrument == TradeInstrument.OPTION:
                if not risk.option_symbol:
                    self.logger.log_trade_decision(decision, risk, "BLOCKED", final_output)
                    continue

                order = self.broker.submit_option_market_order(
                    risk.option_symbol,
                    "buy",
                    int(risk.qty),
                )

            elif decision.side == TradeSide.LONG:
                order = self.broker.submit_market_order_by_qty(
                    decision.ticker,
                    "buy",
                    risk.qty,
                )

            elif decision.side == TradeSide.SHORT:
                order = self.broker.submit_market_order_by_qty(
                    decision.ticker,
                    "sell",
                    risk.qty,
                )

            else:
                self.logger.log_trade_decision(decision, risk, "NO_TRADE", final_output)
                continue

            order_id = str(getattr(order, "id", "unknown"))
            print(
                f"Order sent: {label} {decision.instrument.value.upper()} {decision.side.value}. "
                f"Alpaca order id: {order_id}"
            )
            self.logger.log_trade_decision(decision, risk, "ORDER_SENT", final_output, order_id)
            any_approved_or_sent = True

        return any_approved_or_sent


def monitor_positions_once(dry_run: bool = False) -> None:
    broker = build_broker()
    monitor = PositionMonitor(
        broker,
        build_exit_engine(),
        TradeLogger(),
        **_build_monitor_kwargs(),
    )
    monitor.run_once(dry_run=dry_run)
