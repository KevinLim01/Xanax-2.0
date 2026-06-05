from __future__ import annotations

import sqlite3
from typing import Any

from app.config import settings
from app.utils import ensure_parent, from_json, to_json

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    run_timestamp TEXT NOT NULL,
    forecast_window TEXT NOT NULL,
    prediction_json TEXT NOT NULL,
    debug_json TEXT,
    monday_close REAL,
    friday_close REAL,
    actual_return REAL,
    actual_direction TEXT,
    is_correct_direction INTEGER,

    relative_strength_score REAL,
    relative_strength_reason TEXT,
    relative_strength_vs TEXT,
    intraday_confirmation_score REAL,
    intraday_confirmation_reason TEXT,
    vwap_status TEXT,
    liquidity_score REAL,
    spread_pct REAL,
    dollar_volume REAL,
    tradeable INTEGER,
    liquidity_reason TEXT
);

CREATE TABLE IF NOT EXISTS trade_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    instrument TEXT DEFAULT 'stock',
    underlying_ticker TEXT,
    option_symbol TEXT,
    option_type TEXT,
    model_action TEXT,
    model_direction TEXT,
    conviction_score INTEGER,
    estimated_edge TEXT,
    expected_move_pct REAL,
    setup_type TEXT,
    take_profit_pct REAL,
    stop_loss_pct REAL,
    decision_reason TEXT,
    risk_approved INTEGER NOT NULL,
    risk_reason TEXT,
    position_size_usd REAL,
    estimated_price REAL,
    qty REAL,
    status TEXT NOT NULL,
    alpaca_order_id TEXT,
    model_output_json TEXT,

    relative_strength_score REAL,
    relative_strength_reason TEXT,
    relative_strength_vs TEXT,
    intraday_confirmation_score REAL,
    intraday_confirmation_reason TEXT,
    vwap_status TEXT,
    liquidity_score REAL,
    spread_pct REAL,
    dollar_volume REAL,
    tradeable INTEGER,
    liquidity_reason TEXT
);

CREATE TABLE IF NOT EXISTS position_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    ticker TEXT NOT NULL,
    qty REAL,
    side TEXT,
    market_value REAL,
    avg_entry_price REAL,
    current_price REAL,
    unrealized_pl REAL,
    pnl_pct REAL,
    exit_reason TEXT,
    status TEXT NOT NULL,
    alpaca_order_id TEXT,

    take_profit_pct REAL,
    stop_loss_pct REAL,
    adaptive_take_profit_pct REAL,
    expected_move_pct REAL,
    model_output_json TEXT
);
"""


RUNS_MIGRATIONS = {
    "relative_strength_score": "ALTER TABLE runs ADD COLUMN relative_strength_score REAL",
    "relative_strength_reason": "ALTER TABLE runs ADD COLUMN relative_strength_reason TEXT",
    "relative_strength_vs": "ALTER TABLE runs ADD COLUMN relative_strength_vs TEXT",
    "intraday_confirmation_score": "ALTER TABLE runs ADD COLUMN intraday_confirmation_score REAL",
    "intraday_confirmation_reason": "ALTER TABLE runs ADD COLUMN intraday_confirmation_reason TEXT",
    "vwap_status": "ALTER TABLE runs ADD COLUMN vwap_status TEXT",
    "liquidity_score": "ALTER TABLE runs ADD COLUMN liquidity_score REAL",
    "spread_pct": "ALTER TABLE runs ADD COLUMN spread_pct REAL",
    "dollar_volume": "ALTER TABLE runs ADD COLUMN dollar_volume REAL",
    "tradeable": "ALTER TABLE runs ADD COLUMN tradeable INTEGER",
    "liquidity_reason": "ALTER TABLE runs ADD COLUMN liquidity_reason TEXT",
}

TRADE_DECISION_MIGRATIONS = {
    "instrument": "ALTER TABLE trade_decisions ADD COLUMN instrument TEXT DEFAULT 'stock'",
    "underlying_ticker": "ALTER TABLE trade_decisions ADD COLUMN underlying_ticker TEXT",
    "option_symbol": "ALTER TABLE trade_decisions ADD COLUMN option_symbol TEXT",
    "option_type": "ALTER TABLE trade_decisions ADD COLUMN option_type TEXT",
    "relative_strength_score": "ALTER TABLE trade_decisions ADD COLUMN relative_strength_score REAL",
    "relative_strength_reason": "ALTER TABLE trade_decisions ADD COLUMN relative_strength_reason TEXT",
    "relative_strength_vs": "ALTER TABLE trade_decisions ADD COLUMN relative_strength_vs TEXT",
    "intraday_confirmation_score": "ALTER TABLE trade_decisions ADD COLUMN intraday_confirmation_score REAL",
    "intraday_confirmation_reason": "ALTER TABLE trade_decisions ADD COLUMN intraday_confirmation_reason TEXT",
    "vwap_status": "ALTER TABLE trade_decisions ADD COLUMN vwap_status TEXT",
    "liquidity_score": "ALTER TABLE trade_decisions ADD COLUMN liquidity_score REAL",
    "spread_pct": "ALTER TABLE trade_decisions ADD COLUMN spread_pct REAL",
    "dollar_volume": "ALTER TABLE trade_decisions ADD COLUMN dollar_volume REAL",
    "tradeable": "ALTER TABLE trade_decisions ADD COLUMN tradeable INTEGER",
    "liquidity_reason": "ALTER TABLE trade_decisions ADD COLUMN liquidity_reason TEXT",
}

POSITION_CHECK_MIGRATIONS = {
    "take_profit_pct": "ALTER TABLE position_checks ADD COLUMN take_profit_pct REAL",
    "stop_loss_pct": "ALTER TABLE position_checks ADD COLUMN stop_loss_pct REAL",
    "adaptive_take_profit_pct": "ALTER TABLE position_checks ADD COLUMN adaptive_take_profit_pct REAL",
    "expected_move_pct": "ALTER TABLE position_checks ADD COLUMN expected_move_pct REAL",
    "model_output_json": "ALTER TABLE position_checks ADD COLUMN model_output_json TEXT",
}


def connect() -> sqlite3.Connection:
    ensure_parent(settings.db_abspath)
    conn = sqlite3.connect(settings.db_abspath)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)
        _migrate_table(conn, "runs", RUNS_MIGRATIONS)
        _migrate_table(conn, "trade_decisions", TRADE_DECISION_MIGRATIONS)
        _migrate_table(conn, "position_checks", POSITION_CHECK_MIGRATIONS)
        conn.commit()


def _migrate_table(conn: sqlite3.Connection, table_name: str, migrations: dict[str, str]) -> None:
    existing = {
        row[1]
        for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }

    for column, sql in migrations.items():
        if column not in existing:
            conn.execute(sql)


def _get_nested(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = payload

    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)

    return default if current is None else current


def _agent_value(prediction: dict[str, Any], debug_payload: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in prediction:
        return prediction.get(key, default)

    final_output = debug_payload.get("final_output", {}) if isinstance(debug_payload, dict) else {}
    if isinstance(final_output, dict) and key in final_output:
        return final_output.get(key, default)

    return default


def _tradeable_to_int(value: Any) -> int | None:
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


def save_run(
    ticker: str,
    run_timestamp: str,
    forecast_window: str,
    prediction: dict[str, Any],
    debug_payload: dict[str, Any],
) -> int:
    init_db()

    relative_strength_score = _agent_value(prediction, debug_payload, "relative_strength_score")
    relative_strength_reason = _agent_value(prediction, debug_payload, "relative_strength_reason")
    relative_strength_vs = _agent_value(prediction, debug_payload, "relative_strength_vs")

    intraday_confirmation_score = _agent_value(prediction, debug_payload, "intraday_confirmation_score")
    intraday_confirmation_reason = _agent_value(prediction, debug_payload, "intraday_confirmation_reason")
    vwap_status = _agent_value(prediction, debug_payload, "vwap_status")

    liquidity_score = _agent_value(prediction, debug_payload, "liquidity_score")
    spread_pct = _agent_value(prediction, debug_payload, "spread_pct")
    dollar_volume = _agent_value(prediction, debug_payload, "dollar_volume")
    tradeable = _tradeable_to_int(_agent_value(prediction, debug_payload, "tradeable"))
    liquidity_reason = _agent_value(prediction, debug_payload, "liquidity_reason")

    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO runs (
                ticker,
                run_timestamp,
                forecast_window,
                prediction_json,
                debug_json,
                relative_strength_score,
                relative_strength_reason,
                relative_strength_vs,
                intraday_confirmation_score,
                intraday_confirmation_reason,
                vwap_status,
                liquidity_score,
                spread_pct,
                dollar_volume,
                tradeable,
                liquidity_reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                run_timestamp,
                forecast_window,
                to_json(prediction),
                to_json(debug_payload),
                relative_strength_score,
                relative_strength_reason,
                relative_strength_vs,
                intraday_confirmation_score,
                intraday_confirmation_reason,
                vwap_status,
                liquidity_score,
                spread_pct,
                dollar_volume,
                tradeable,
                liquidity_reason,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def load_recent_runs(limit: int = 20) -> list[dict[str, Any]]:
    init_db()

    with connect() as conn:
        rows = conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()

    out = []

    for row in rows:
        item = dict(row)
        item["prediction_json"] = from_json(item["prediction_json"], {})
        item["debug_json"] = from_json(item["debug_json"], {})
        out.append(item)

    return out
