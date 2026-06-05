from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any


class TradeLogger:
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or os.getenv("DATABASE_PATH", "data/stock_signals.db")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _table_columns(self, conn, table_name: str) -> set[str]:
        try:
            return {
                row["name"]
                for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            }
        except Exception:
            return set()

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    status TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS position_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    ticker TEXT,
                    status TEXT
                )
                """
            )

            trade_extra_cols = {
                "instrument": "TEXT",
                "trade_action": "TEXT",
                "alpaca_side": "TEXT",
                "underlying_ticker": "TEXT",
                "option_symbol": "TEXT",
                "option_type": "TEXT",
                "conviction_score": "INTEGER",
                "source_action": "TEXT",
                "source_direction": "TEXT",
                "expected_move_pct": "REAL",
                "take_profit_pct": "REAL",
                "stop_loss_pct": "REAL",
                "decision_reason": "TEXT",
                "risk_approved": "INTEGER",
                "risk_reason": "TEXT",
                "position_size_usd": "REAL",
                "estimated_price": "REAL",
                "qty": "REAL",
                "contracts": "INTEGER",
                "alpaca_order_id": "TEXT",
                "model_action": "TEXT",
                "model_direction": "TEXT",
                "model_edge": "TEXT",
                "model_setup_type": "TEXT",
                "model_reason": "TEXT",
                "model_output_json": "TEXT",
                "raw_model_output_json": "TEXT",

                # New agent diagnostics.
                "relative_strength_score": "REAL",
                "relative_strength_reason": "TEXT",
                "relative_strength_vs": "TEXT",
                "intraday_confirmation_score": "REAL",
                "intraday_confirmation_reason": "TEXT",
                "vwap_status": "TEXT",
                "liquidity_score": "REAL",
                "spread_pct": "REAL",
                "dollar_volume": "REAL",
                "tradeable": "INTEGER",
                "liquidity_reason": "TEXT",

                # History agent diagnostics.
                "history_score_adjustment": "INTEGER",
                "history_true_during_week_rate": "REAL",
                "history_sample_size": "INTEGER",
                "history_match_level": "TEXT",
                "history_lookup_reason": "TEXT",
            }

            position_extra_cols = {
                "asset_class": "TEXT",
                "qty": "REAL",
                "side": "TEXT",
                "market_value": "REAL",
                "avg_entry_price": "REAL",
                "current_price": "REAL",
                "unrealized_pl": "REAL",
                "unrealized_pl_pct": "REAL",
                "pnl_pct": "REAL",
                "exit_reason": "TEXT",
                "exit_type": "TEXT",
                "alpaca_order_id": "TEXT",

                # Old exit diagnostics.
                "take_profit_pct": "REAL",
                "stop_loss_pct": "REAL",
                "adaptive_take_profit_pct": "REAL",
                "expected_move_pct": "REAL",

                # New monitor/path diagnostics.
                "current_profit_pct": "REAL",
                "model_action": "TEXT",
                "model_direction": "TEXT",
                "model_conviction": "INTEGER",
                "model_setup_type": "TEXT",
                "history_score_adjustment": "INTEGER",
                "history_true_during_week_rate": "REAL",
                "history_sample_size": "INTEGER",
                "history_average_best_correct_return_pct": "REAL",
                "history_average_adverse_move_pct": "REAL",
                "history_match_level": "TEXT",
                "history_lookup_reason": "TEXT",
                "model_output_json": "TEXT",
            }

            existing_trade_cols = self._table_columns(conn, "trade_decisions")
            for col, col_type in trade_extra_cols.items():
                if col not in existing_trade_cols:
                    conn.execute(f"ALTER TABLE trade_decisions ADD COLUMN {col} {col_type}")

            existing_position_cols = self._table_columns(conn, "position_checks")
            for col, col_type in position_extra_cols.items():
                if col not in existing_position_cols:
                    conn.execute(f"ALTER TABLE position_checks ADD COLUMN {col} {col_type}")

            conn.commit()

    def _tradeable_to_int(self, value: Any) -> int | None:
        if value is None:
            return None

        if isinstance(value, bool):
            return 1 if value else 0

        text = str(value).strip().lower()

        if text in {"1", "true", "yes", "y", "pass", "tradeable"}:
            return 1

        if text in {"0", "false", "no", "n", "fail", "blocked"}:
            return 0

        return None

    def log_trade_decision(
        self,
        decision,
        risk,
        status: str,
        model_output: dict[str, Any] | None = None,
        alpaca_order_id: str | None = None,
    ) -> None:
        self._insert_trade_decision(
            decision=decision,
            risk=risk,
            status=status,
            alpaca_order_id=alpaca_order_id,
            model_output=model_output,
        )

    def log_order_sent(
        self,
        decision,
        risk,
        status: str,
        alpaca_order_id: str | None,
        model_output: dict[str, Any] | None = None,
    ) -> None:
        self._insert_trade_decision(
            decision=decision,
            risk=risk,
            status=status,
            alpaca_order_id=alpaca_order_id,
            model_output=model_output,
        )

    def _insert_trade_decision(
        self,
        decision,
        risk,
        status: str,
        alpaca_order_id: str | None,
        model_output: dict[str, Any] | None = None,
    ) -> None:
        model_output = model_output or {}

        alpaca_side = getattr(decision, "alpaca_side", None)
        trade_action = str(getattr(decision, "trade_action", ""))
        side = alpaca_side or trade_action or "HOLD"

        row = {
            "created_at": "datetime('now')",
            "ticker": getattr(decision, "ticker", None),
            "side": side,
            "instrument": str(getattr(decision, "instrument", "stock") or "stock"),
            "trade_action": trade_action,
            "alpaca_side": alpaca_side,
            "underlying_ticker": getattr(decision, "underlying_ticker", None),
            "option_symbol": getattr(decision, "option_symbol", None),
            "option_type": getattr(decision, "option_type", None),
            "conviction_score": getattr(decision, "conviction_score", None),
            "source_action": getattr(decision, "source_action", None),
            "source_direction": getattr(decision, "source_direction", None),
            "expected_move_pct": getattr(decision, "expected_move_pct", None),
            "take_profit_pct": getattr(decision, "take_profit_pct", None),
            "stop_loss_pct": getattr(decision, "stop_loss_pct", None),
            "decision_reason": getattr(decision, "reason", None),
            "risk_approved": 1 if getattr(risk, "approved", False) else 0,
            "risk_reason": getattr(risk, "reason", None),
            "position_size_usd": getattr(risk, "position_size_usd", None),
            "estimated_price": getattr(risk, "estimated_price", None),
            "qty": getattr(risk, "qty", None),
            "contracts": getattr(risk, "contracts", None),
            "status": status,
            "alpaca_order_id": alpaca_order_id,
            "model_action": model_output.get("final_action") or model_output.get("action"),
            "model_direction": model_output.get("forecast_direction") or model_output.get("direction"),
            "model_edge": model_output.get("estimated_edge") or model_output.get("edge"),
            "model_setup_type": model_output.get("setup_type"),
            "model_reason": model_output.get("reason"),
            "model_output_json": json.dumps(model_output, default=str),
            "raw_model_output_json": json.dumps(model_output, default=str),

            # New agent diagnostics.
            "relative_strength_score": model_output.get("relative_strength_score"),
            "relative_strength_reason": model_output.get("relative_strength_reason"),
            "relative_strength_vs": model_output.get("relative_strength_vs")
            or model_output.get("relative_strength_benchmark"),
            "intraday_confirmation_score": model_output.get("intraday_confirmation_score"),
            "intraday_confirmation_reason": model_output.get("intraday_confirmation_reason"),
            "vwap_status": model_output.get("vwap_status"),
            "liquidity_score": model_output.get("liquidity_score"),
            "spread_pct": model_output.get("spread_pct"),
            "dollar_volume": model_output.get("dollar_volume"),
            "tradeable": self._tradeable_to_int(model_output.get("tradeable")),
            "liquidity_reason": model_output.get("liquidity_reason"),

            # History diagnostics.
            "history_score_adjustment": model_output.get("history_score_adjustment"),
            "history_true_during_week_rate": model_output.get("history_true_during_week_rate"),
            "history_sample_size": model_output.get("history_sample_size"),
            "history_match_level": model_output.get("history_match_level"),
            "history_lookup_reason": model_output.get("history_lookup_reason"),
        }

        with self._connect() as conn:
            existing_cols = self._table_columns(conn, "trade_decisions")
            insert_cols = [col for col in row.keys() if col in existing_cols]

            sql_cols = ", ".join(insert_cols)
            placeholders = []
            values = []

            for col in insert_cols:
                if col == "created_at":
                    placeholders.append("datetime('now')")
                else:
                    placeholders.append("?")
                    values.append(row[col])

            sql = f"""
                INSERT INTO trade_decisions ({sql_cols})
                VALUES ({", ".join(placeholders)})
            """

            conn.execute(sql, values)
            conn.commit()

    def log_position_check(
        self,
        position,
        exit_decision,
        status: str,
        alpaca_order_id: str | None = None,
        model_output: dict[str, Any] | None = None,
    ) -> None:
        model_output = model_output or {}

        pnl_pct = getattr(exit_decision, "pnl_pct", None)
        if pnl_pct is None:
            pnl_pct = getattr(exit_decision, "current_profit_pct", None)
        if pnl_pct is None:
            pnl_pct = getattr(position, "unrealized_pl_pct", None)

        row = {
            "created_at": "datetime('now')",
            "ticker": getattr(position, "ticker", None),
            "asset_class": getattr(position, "asset_class", None),
            "qty": getattr(position, "qty", None),
            "side": getattr(position, "side", None),
            "market_value": getattr(position, "market_value", None),
            "avg_entry_price": getattr(position, "avg_entry_price", None),
            "current_price": getattr(position, "current_price", None),
            "unrealized_pl": getattr(position, "unrealized_pl", None),
            "unrealized_pl_pct": getattr(position, "unrealized_pl_pct", None),
            "pnl_pct": pnl_pct,
            "exit_reason": getattr(exit_decision, "reason", None),
            "exit_type": getattr(exit_decision, "exit_type", None),
            "status": status,
            "alpaca_order_id": alpaca_order_id,
            "take_profit_pct": getattr(exit_decision, "take_profit_pct", None),
            "stop_loss_pct": getattr(exit_decision, "stop_loss_pct", None),
            "adaptive_take_profit_pct": getattr(exit_decision, "adaptive_take_profit_pct", None),
            "expected_move_pct": getattr(exit_decision, "expected_move_pct", None),
            "current_profit_pct": getattr(exit_decision, "current_profit_pct", None),

            # Model/history monitor fields. These are filled when position_monitor passes model_output.
            "model_action": model_output.get("final_action") or model_output.get("action"),
            "model_direction": model_output.get("forecast_direction") or model_output.get("direction"),
            "model_conviction": model_output.get("conviction_score"),
            "model_setup_type": model_output.get("setup_type"),
            "history_score_adjustment": model_output.get("history_score_adjustment"),
            "history_true_during_week_rate": model_output.get("history_true_during_week_rate"),
            "history_sample_size": model_output.get("history_sample_size"),
            "history_average_best_correct_return_pct": model_output.get("history_average_best_correct_return_pct"),
            "history_average_adverse_move_pct": model_output.get("history_average_adverse_move_pct"),
            "history_match_level": model_output.get("history_match_level"),
            "history_lookup_reason": model_output.get("history_lookup_reason"),
            "model_output_json": json.dumps(model_output, default=str) if model_output else None,
        }

        with self._connect() as conn:
            existing_cols = self._table_columns(conn, "position_checks")
            insert_cols = [col for col in row.keys() if col in existing_cols]

            sql_cols = ", ".join(insert_cols)
            placeholders = []
            values = []

            for col in insert_cols:
                if col == "created_at":
                    placeholders.append("datetime('now')")
                else:
                    placeholders.append("?")
                    values.append(row[col])

            sql = f"""
                INSERT INTO position_checks ({sql_cols})
                VALUES ({", ".join(placeholders)})
            """

            conn.execute(sql, values)
            conn.commit()
