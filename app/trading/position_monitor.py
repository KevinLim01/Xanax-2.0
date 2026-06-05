from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
import json

from app.config import settings
from app.trading.alpaca_broker import AlpacaBroker
from app.trading.exit_engine import ExitEngine
from app.trading.trade_logger import TradeLogger


@dataclass
class MonitorExitDecision:
    should_exit: bool
    reason: str
    exit_type: str = "MODEL_HISTORY_MONITOR"
    current_profit_pct: float | None = None

    def __getattr__(self, name: str) -> Any:
        return None


class PositionMonitor:
    def __init__(
        self,
        broker: AlpacaBroker,
        exit_engine: ExitEngine,
        logger: TradeLogger,
        model_runner: Callable[[str], dict[str, Any]] | None = None,
        history_agent: Any | None = None,
    ) -> None:
        self.broker = broker
        self.exit_engine = exit_engine
        self.logger = logger
        self.model_runner = model_runner
        self.history_agent = history_agent

    def run_once(self, dry_run: bool = False) -> None:
        positions = self.broker.list_positions()
        if not positions:
            print("No open Alpaca positions.")
            return

        for position in positions:
            opened_at = self.broker.get_latest_filled_order_time(position.ticker)
            base_decision = self.exit_engine.check(position, opened_at=opened_at)

            current_profit_pct = self._current_profit_pct(position)
            same_day = self._is_same_day_position(opened_at)

            model_output = self._rerun_model_output(position)

            final_decision = self._combine_exit_decisions(
                position=position,
                opened_at=opened_at,
                base_decision=base_decision,
                current_profit_pct=current_profit_pct,
                same_day=same_day,
                model_output=model_output,
            )

            opened_text = opened_at.isoformat() if opened_at is not None else "unknown"
            profit_text = "N/A" if current_profit_pct is None else f"{current_profit_pct:+.2f}%"

            print(
                f"{position.ticker}: opened/latest fill={opened_text} | "
                f"profit={profit_text} | {final_decision.reason}"
            )

            if final_decision.should_exit:
                if dry_run:
                    print(f"[DRY RUN] Would close {position.ticker}.")
                    self.logger.log_position_check(
                        position,
                        final_decision,
                        "DRY_RUN_EXIT",
                        model_output=model_output,
                    )
                else:
                    order = self.broker.close_position(position.ticker)
                    order_id = str(getattr(order, "id", "unknown"))
                    print(f"Closed {position.ticker}. Alpaca order id: {order_id}")
                    self.logger.log_position_check(
                        position,
                        final_decision,
                        "EXIT_SENT",
                        alpaca_order_id=order_id,
                        model_output=model_output,
                    )
            else:
                self.logger.log_position_check(
                    position,
                    final_decision,
                    "HOLDING",
                    model_output=model_output,
                )

    def _rerun_model_output(self, position: Any) -> dict[str, Any]:
        if not self._setting_bool("monitor_use_model_rerun", True):
            return {}

        if self.model_runner is None:
            return {}

        ticker = str(getattr(position, "ticker", "")).upper().strip()
        if not ticker:
            return {}

        try:
            result = self.model_runner(ticker)
            final = dict(result.get("final_output", {}))
            return final
        except Exception as exc:
            return {
                "ticker": ticker,
                "monitor_model_error": str(exc),
                "final_action": "WATCH",
                "forecast_direction": "NEUTRAL",
                "conviction_score": 0,
                "setup_type": "MODEL_RERUN_FAILED",
            }

    def _combine_exit_decisions(
        self,
        *,
        position: Any,
        opened_at: datetime | None,
        base_decision: Any,
        current_profit_pct: float | None,
        same_day: bool,
        model_output: dict[str, Any],
    ) -> Any:
        """
        New monitor logic:

        - No same-day selling except emergency/stop-loss style exits.
        - Fixed 3% / 5% take-profit style exits are ignored.
        - Reruns model for held tickers.
        - Exits if the model flips against the position.
        - Exits if conviction collapses while already profitable.
        - Exits if current profit reaches the historical average best move.
        - Keeps Friday/emergency stop protection.
        """

        if getattr(base_decision, "should_exit", False):
            if self._should_honor_base_exit(base_decision, same_day=same_day):
                return base_decision

        if same_day:
            return MonitorExitDecision(
                should_exit=False,
                reason="Holding because same-day selling is blocked unless an emergency stop/Friday rule triggers.",
                current_profit_pct=current_profit_pct,
            )

        model_decision = self._model_based_exit(position, current_profit_pct, model_output)
        if model_decision.should_exit:
            return model_decision

        profit_protection_decision = self._profit_protection_exit(position, current_profit_pct)
        if profit_protection_decision.should_exit:
            return profit_protection_decision

        history_decision = self._history_peak_exit(position, current_profit_pct, model_output)
        if history_decision.should_exit:
            return history_decision

        if getattr(base_decision, "should_exit", False):
            return MonitorExitDecision(
                should_exit=False,
                reason=f"Holding because old fixed take-profit exit was ignored: {getattr(base_decision, 'reason', '')}",
                current_profit_pct=current_profit_pct,
            )

        return MonitorExitDecision(
            should_exit=False,
            reason=f"Holding. Base exit check: {getattr(base_decision, 'reason', 'No exit reason returned.')}",
            current_profit_pct=current_profit_pct,
        )

    def _should_honor_base_exit(self, decision: Any, *, same_day: bool) -> bool:
        exit_type = str(getattr(decision, "exit_type", "")).upper()
        reason = str(getattr(decision, "reason", "")).lower()

        if exit_type in {"STOP_LOSS", "FRIDAY_FORCE_EXIT", "OPTION_EXPIRATION"}:
            return True

        if exit_type == "TAKE_PROFIT":
            return False

        emergency_words = [
            "stop",
            "stop-loss",
            "loss",
            "emergency",
            "risk",
            "friday",
            "force",
            "forced",
            "market close",
            "expiration",
        ]

        take_profit_words = [
            "take profit",
            "take-profit",
            "profit target",
            "target profit",
            "trailing",
            "trail",
            "3%",
            "5%",
        ]

        if any(word in reason for word in emergency_words):
            return True

        if any(word in reason for word in take_profit_words):
            return False

        if same_day:
            return False

        return False

    def _model_based_exit(
        self,
        position: Any,
        current_profit_pct: float | None,
        model_output: dict[str, Any],
    ) -> MonitorExitDecision:
        if not self._setting_bool("monitor_use_model_rerun", True):
            return self._hold("Model rerun exit disabled.", current_profit_pct)

        if self.model_runner is None:
            return self._hold("Model rerun unavailable.", current_profit_pct)

        ticker = str(getattr(position, "ticker", "")).upper().strip()
        if not ticker:
            return self._hold("Model rerun skipped because ticker is missing.", current_profit_pct)

        if model_output.get("monitor_model_error"):
            return self._hold(
                f"Model rerun failed for {ticker}: {model_output.get('monitor_model_error')}",
                current_profit_pct,
            )

        if not model_output:
            return self._hold("Model rerun returned no output.", current_profit_pct)

        final_action = str(model_output.get("final_action", "WATCH")).upper()
        forecast_direction = str(model_output.get("forecast_direction", "NEUTRAL")).upper()
        conviction = self._safe_int(model_output.get("conviction_score"), 0)
        setup_type = str(model_output.get("setup_type", "NO_CLEAN_SETUP")).upper()

        position_bias = self._position_bias(position)

        opposite_direction = (
            (position_bias == "UP" and forecast_direction == "DOWN")
            or (position_bias == "DOWN" and forecast_direction == "UP")
        )

        opposite_action = (
            (position_bias == "UP" and final_action == "SELL")
            or (position_bias == "DOWN" and final_action == "BUY")
        )

        low_conviction_threshold = self._setting_int("monitor_low_conviction_exit_threshold", 50)
        exit_on_flip = self._setting_bool("monitor_exit_on_signal_flip", True)
        exit_on_low_conviction = self._setting_bool("monitor_exit_on_low_conviction", True)

        if exit_on_flip and (opposite_direction or opposite_action):
            return MonitorExitDecision(
                should_exit=True,
                reason=(
                    f"Exit: rerun model flipped against position. "
                    f"Current model={final_action}/{forecast_direction}, conviction={conviction}, setup={setup_type}."
                ),
                exit_type="MODEL_FLIP_EXIT",
                current_profit_pct=current_profit_pct,
            )

        if (
            exit_on_low_conviction
            and conviction < low_conviction_threshold
            and current_profit_pct is not None
            and current_profit_pct >= 0
        ):
            return MonitorExitDecision(
                should_exit=True,
                reason=(
                    f"Exit: rerun model conviction fell below {low_conviction_threshold} "
                    f"while position is profitable. Current conviction={conviction}, "
                    f"model={final_action}/{forecast_direction}, setup={setup_type}."
                ),
                exit_type="LOW_CONVICTION_PROFIT_EXIT",
                current_profit_pct=current_profit_pct,
            )

        return self._hold(
            f"Model rerun still supports hold: {final_action}/{forecast_direction}, conviction={conviction}, setup={setup_type}.",
            current_profit_pct,
        )

    def _history_peak_exit(
        self,
        position: Any,
        current_profit_pct: float | None,
        model_output: dict[str, Any],
    ) -> MonitorExitDecision:
        if not self._setting_bool("monitor_use_history_exit", True):
            return self._hold("History exit disabled.", current_profit_pct)

        if self.history_agent is None:
            return self._hold("History agent unavailable.", current_profit_pct)

        if current_profit_pct is None or current_profit_pct <= 0:
            return self._hold("History peak exit skipped because position is not profitable.", current_profit_pct)

        ticker = str(getattr(position, "ticker", "")).upper().strip()
        if not ticker:
            return self._hold("History peak exit skipped because ticker is missing.", current_profit_pct)

        if model_output.get("monitor_model_error"):
            return self._hold(
                f"History peak exit skipped because model rerun failed for {ticker}: {model_output.get('monitor_model_error')}",
                current_profit_pct,
            )

        if not model_output:
            return self._hold("History peak exit skipped because model rerun output is missing.", current_profit_pct)

        setup_type = model_output.get("setup_type")
        forecast_direction = model_output.get("forecast_direction")
        primary_regime = model_output.get("primary_regime", "NORMAL")
        ticker_archetype = model_output.get("ticker_archetype", "NORMAL")

        try:
            signal = self.history_agent.evaluate(
                ticker=ticker,
                setup_type=setup_type,
                forecast_direction=forecast_direction,
                primary_regime=primary_regime,
                ticker_archetype=ticker_archetype,
            )
        except Exception as exc:
            return self._hold(f"History lookup failed for {ticker}: {exc}", current_profit_pct)

        metrics = signal.metrics if isinstance(signal.metrics, dict) else {}
        matched = bool(metrics.get("matched", False))
        avg_best = self._safe_float(metrics.get("average_best_correct_return_pct"), None)
        sample_size = self._safe_int(metrics.get("sample_size"), 0)
        success_rate = self._safe_float(metrics.get("true_during_week_rate"), None)

        model_output["history_average_best_correct_return_pct"] = avg_best
        model_output["history_average_adverse_move_pct"] = metrics.get("average_adverse_move_pct")
        model_output["history_sample_size"] = sample_size
        model_output["history_true_during_week_rate"] = success_rate
        model_output["history_score_adjustment"] = metrics.get("recommended_score_adjustment")
        model_output["history_match_level"] = metrics.get("match_level")
        model_output["history_lookup_reason"] = signal.reason

        if not matched or avg_best is None or avg_best <= 0:
            return self._hold("History peak exit skipped because no reliable average best move was found.", current_profit_pct)

        min_sample = self._setting_int("monitor_history_min_sample_size", 50)
        if sample_size < min_sample:
            return self._hold(
                f"History peak exit skipped because sample size is too small: {sample_size} < {min_sample}.",
                current_profit_pct,
            )

        conviction = self._safe_int(model_output.get("conviction_score"), 0)
        capture_ratio = self._history_capture_ratio_for_conviction(conviction)
        trigger = avg_best * capture_ratio

        if current_profit_pct >= trigger:
            rate_text = "N/A" if success_rate is None else f"{success_rate:.2f}%"
            return MonitorExitDecision(
                should_exit=True,
                reason=(
                    f"Exit: profit {current_profit_pct:.2f}% reached history-based peak trigger "
                    f"{trigger:.2f}% ({capture_ratio:.0%} of historical avg best move {avg_best:.2f}%). "
                    f"History sample={sample_size:,}, true-during-week rate={rate_text}."
                ),
                exit_type="HISTORY_PEAK_EXIT",
                current_profit_pct=current_profit_pct,
            )

        return self._hold(
            f"History hold: profit {current_profit_pct:.2f}% has not reached trigger {trigger:.2f}% "
            f"from historical avg best move {avg_best:.2f}%.",
            current_profit_pct,
        )

    def _history_capture_ratio_for_conviction(self, conviction: int) -> float:
        if conviction >= self._setting_int("monitor_history_high_conviction", 75):
            return self._setting_float("monitor_history_profit_capture_ratio_high", 0.95)
        if conviction >= self._setting_int("monitor_history_mid_conviction", 65):
            return self._setting_float("monitor_history_profit_capture_ratio_mid", 0.90)
        return self._setting_float("monitor_history_profit_capture_ratio", 0.85)

    def _profit_protection_exit(self, position: Any, current_profit_pct: float | None) -> MonitorExitDecision:
        if not self._setting_bool("monitor_profit_protection_enabled", True):
            return self._hold("Profit protection disabled.", current_profit_pct)

        ticker = str(getattr(position, "ticker", "")).upper().strip()
        if not ticker or current_profit_pct is None:
            return self._hold("Profit protection skipped because ticker/profit is missing.", current_profit_pct)

        state = self._load_profit_state()
        row = state.get(ticker, {}) if isinstance(state.get(ticker), dict) else {}
        prior_max = self._safe_float(row.get("max_profit_pct"), current_profit_pct)
        max_profit = max(float(prior_max or 0.0), float(current_profit_pct))

        state[ticker] = {
            "max_profit_pct": round(max_profit, 4),
            "last_profit_pct": round(float(current_profit_pct), 4),
            "updated_at": datetime.now().isoformat(),
        }
        self._save_profit_state(state)

        activation = self._setting_float("monitor_profit_protection_activation_pct", 2.0)
        floor = self._setting_float("monitor_profit_protection_floor_pct", 0.5)

        if max_profit >= activation and current_profit_pct <= floor:
            return MonitorExitDecision(
                should_exit=True,
                reason=(
                    f"Exit: profit protection triggered. Position reached max profit {max_profit:.2f}% "
                    f"but fell back to {current_profit_pct:.2f}% <= floor {floor:.2f}%."
                ),
                exit_type="PROFIT_PROTECTION_EXIT",
                current_profit_pct=current_profit_pct,
            )

        return self._hold(
            f"Profit protection hold: max profit {max_profit:.2f}%, current {current_profit_pct:.2f}%.",
            current_profit_pct,
        )

    def _load_profit_state(self) -> dict[str, Any]:
        path = self._profit_state_path()
        try:
            if path.exists():
                data = json.loads(path.read_text())
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return {}

    def _save_profit_state(self, state: dict[str, Any]) -> None:
        path = self._profit_state_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(state, indent=2, sort_keys=True))
        except Exception:
            pass

    def _profit_state_path(self) -> Path:
        path = Path(str(getattr(settings, "monitor_profit_protection_state_path", "data/profit_protection_state.json")))
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parents[2] / path

    def _current_profit_pct(self, position: Any) -> float | None:
        direct = self._safe_float_any(
            self._first_attr(
                position,
                [
                    "unrealized_plpc",
                    "unrealized_intraday_plpc",
                    "profit_pct",
                    "current_profit_pct",
                ],
            )
        )

        if direct is not None:
            if abs(direct) <= 2:
                return direct * 100
            return direct

        entry = self._safe_float_any(
            self._first_attr(
                position,
                [
                    "avg_entry_price",
                    "average_entry_price",
                    "entry_price",
                    "cost_basis_price",
                ],
            )
        )
        current = self._safe_float_any(
            self._first_attr(
                position,
                [
                    "current_price",
                    "market_price",
                    "last_price",
                    "price",
                ],
            )
        )

        if entry is None or current is None or entry <= 0:
            return None

        raw = ((current - entry) / entry) * 100.0
        return raw * self._position_sign(position)

    def _position_bias(self, position: Any) -> str:
        return "UP" if self._position_sign(position) >= 0 else "DOWN"

    def _position_sign(self, position: Any) -> int:
        side = str(self._first_attr(position, ["side", "position_side", "asset_side"]) or "").lower()
        if side in {"short", "sell", "sold"}:
            return -1
        if side in {"long", "buy", "bought"}:
            return 1

        qty = self._safe_float_any(self._first_attr(position, ["qty", "quantity", "shares"]))
        if qty is not None and qty < 0:
            return -1

        return 1

    def _is_same_day_position(self, opened_at: datetime | None) -> bool:
        if opened_at is None:
            return False

        try:
            now = datetime.now(tz=opened_at.tzinfo) if opened_at.tzinfo else datetime.now()
            return opened_at.date() == now.date()
        except Exception:
            return False

    def _hold(self, reason: str, current_profit_pct: float | None = None) -> MonitorExitDecision:
        return MonitorExitDecision(
            should_exit=False,
            reason=reason,
            current_profit_pct=current_profit_pct,
        )

    @staticmethod
    def _first_attr(obj: Any, names: list[str]) -> Any:
        for name in names:
            if isinstance(obj, dict) and name in obj:
                return obj.get(name)
            if hasattr(obj, name):
                return getattr(obj, name)
        return None

    @staticmethod
    def _safe_float_any(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            if value is None:
                return default
            return int(float(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _setting_bool(name: str, default: bool) -> bool:
        value = getattr(settings, name, default)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _setting_int(name: str, default: int) -> int:
        try:
            return int(getattr(settings, name, default))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _setting_float(name: str, default: float) -> float:
        try:
            return float(getattr(settings, name, default))
        except (TypeError, ValueError):
            return default
