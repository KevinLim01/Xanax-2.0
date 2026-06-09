from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from app.database import connect, init_db
from app.pipeline import normalize_ticker, run_pipeline
from app.trading.autotrader import AutoTrader, build_broker, monitor_positions_once
from app.utils import logger
from app.agents.history_lookup_agent import HistoryLookupAgent
from app.agents.simulation_filter_agent import SimulationFilterAgent

LINE = "─" * 98
ALIAS_GROUPS = [frozenset({"GOOG", "GOOGL"})]

_HISTORY_AGENT: HistoryLookupAgent | None = None
_SIMULATION_FILTER_AGENT: SimulationFilterAgent | None = None


def _get_history_agent() -> HistoryLookupAgent:
    global _HISTORY_AGENT
    if _HISTORY_AGENT is None:
        _HISTORY_AGENT = HistoryLookupAgent()
    return _HISTORY_AGENT


def _get_simulation_filter_agent() -> SimulationFilterAgent:
    global _SIMULATION_FILTER_AGENT
    if _SIMULATION_FILTER_AGENT is None:
        _SIMULATION_FILTER_AGENT = SimulationFilterAgent()
    return _SIMULATION_FILTER_AGENT


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _deep_find_value(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = _deep_find_value(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _deep_find_value(value, key)
            if found is not None:
                return found
    return None


def _agent_block(row: dict, agent_name: str) -> dict:
    agents = row.get("new_agents")
    if isinstance(agents, dict):
        block = agents.get(agent_name)
        if isinstance(block, dict):
            return block

    debug = row.get("debug")
    if isinstance(debug, dict):
        agents = debug.get("new_agents")
        if isinstance(agents, dict):
            block = agents.get(agent_name)
            if isinstance(block, dict):
                return block

    return {}


def _agent_metrics(row: dict, agent_name: str) -> dict:
    block = _agent_block(row, agent_name)
    metrics = block.get("metrics")
    if isinstance(metrics, dict):
        return metrics

    metrics_by_agent = row.get("new_agent_metrics")
    if isinstance(metrics_by_agent, dict):
        metrics = metrics_by_agent.get(agent_name)
        if isinstance(metrics, dict):
            return metrics

    return {}


def _safe_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "pass", "passed"}


def _tradeable_from_row(row: dict) -> bool:
    explicit = row.get("tradeable")
    if explicit is not None:
        return _safe_bool(explicit, True)

    liq = _agent_metrics(row, "liquidity_agent")
    spread_pct = _safe_float(liq.get("spread_pct"), 0.0)
    dollar_volume = _safe_float(liq.get("dollar_volume"), 0.0)

    if spread_pct and spread_pct > 0.50:
        return False
    if dollar_volume and dollar_volume < 10_000_000:
        return False

    return True


def _new_agent_summary(row: dict) -> str:
    rs_score = _safe_float(row.get("relative_strength_score"), 0.0)
    intra_score = _safe_float(row.get("intraday_confirmation_score"), 0.0)
    hist_adj = _safe_int(row.get("history_score_adjustment"), 0)
    hist_rate = row.get("history_true_during_week_rate")

    spread = row.get("spread_pct")
    spread_text = "N/A" if spread is None else f"{_safe_float(spread):.3f}%"
    tradeable = "PASS" if _tradeable_from_row(row) else "BLOCK"

    hist_rate_text = "N/A" if hist_rate is None else f"{_safe_float(hist_rate):.1f}%"
    hist_adj_text = f"{hist_adj:+d}"

    return (
        f"RS={rs_score:+.2f} | "
        f"Intra={intra_score:+.2f} | "
        f"Hist={hist_adj_text}/{hist_rate_text} | "
        f"Spread={spread_text} | "
        f"Liq={tradeable}"
    )


def _apply_history_lookup(row: dict) -> dict:
    """
    Apply the saved 5-year history summary to a current scan row.

    This does not rerun any historical backtest. It only reads:
      data/history_setup_summary.csv

    The result is intentionally stored on the row so screen, trade-screen,
    recommend-only JSON, and ranking can all use the same fields.
    """
    if row.get("history_adjusted") is True:
        return row

    try:
        agent = _get_history_agent()
        signal = agent.evaluate(
            ticker=str(row.get("ticker", "")),
            setup_type=row.get("setup_type"),
            forecast_direction=row.get("forecast_direction"),
            primary_regime=row.get("primary_regime", "NORMAL"),
            ticker_archetype=row.get("ticker_archetype", "NORMAL"),
        )

        metrics = signal.metrics if isinstance(signal.metrics, dict) else {}
        adjustment = _safe_int(metrics.get("recommended_score_adjustment"), 0)

        base_conviction = _safe_int(row.get("conviction_score"), 0)
        adjusted_conviction = max(0, min(100, base_conviction + adjustment))

        row["base_conviction_score"] = base_conviction
        row["conviction_score"] = adjusted_conviction
        row["history_adjusted"] = True

        row["history_lookup_score"] = _safe_float(signal.score, 0.0)
        row["history_lookup_direction"] = signal.direction
        row["history_lookup_confidence"] = _safe_float(signal.confidence, 0.0)
        row["history_lookup_reason"] = signal.reason

        row["history_match_level"] = metrics.get("match_level", "none")
        row["history_sample_size"] = _safe_int(metrics.get("sample_size"), 0)
        row["history_true_during_week_rate"] = metrics.get("true_during_week_rate")
        row["history_average_best_correct_return_pct"] = metrics.get("average_best_correct_return_pct")
        row["history_average_adverse_move_pct"] = metrics.get("average_adverse_move_pct")
        row["history_score_adjustment"] = adjustment

        return row

    except Exception as exc:
        row["history_adjusted"] = False
        row["history_lookup_score"] = 0.0
        row["history_lookup_direction"] = "NEUTRAL"
        row["history_lookup_confidence"] = 0.0
        row["history_lookup_reason"] = f"History lookup failed: {exc}"
        row["history_score_adjustment"] = 0
        return row

def _current_entry_day_name() -> str:
    return datetime.now().strftime("%A").upper()


def _apply_simulation_filter(row: dict) -> dict:
    """Apply the two-year no-weekend simulation guardrail to a scan row."""
    from app.config import settings

    if not settings.simulation_filter_enabled:
        row["simulation_filter_enabled"] = False
        return row

    if row.get("simulation_filter_adjusted") is True:
        return row

    try:
        agent = _get_simulation_filter_agent()
        signal = agent.evaluate(
            ticker=str(row.get("ticker", "")),
            setup_type=row.get("setup_type"),
            forecast_direction=row.get("forecast_direction"),
            entry_day=row.get("entry_day") or _current_entry_day_name(),
            conviction_score=row.get("conviction_score"),
        )

        metrics = signal.metrics if isinstance(signal.metrics, dict) else {}
        adjustment = _safe_int(metrics.get("recommended_score_adjustment"), 0)
        base_conviction = _safe_int(row.get("conviction_score"), 0)
        adjusted_conviction = max(0, min(100, base_conviction + adjustment))

        row["simulation_filter_enabled"] = True
        row["simulation_filter_adjusted"] = True
        row["simulation_base_conviction_score"] = base_conviction
        row["conviction_score"] = adjusted_conviction
        row["simulation_filter_score"] = _safe_float(signal.score, 0.0)
        row["simulation_filter_direction"] = signal.direction
        row["simulation_filter_confidence"] = _safe_float(signal.confidence, 0.0)
        row["simulation_filter_reason"] = signal.reason
        row["simulation_filter_match_level"] = metrics.get("match_level", "none")
        row["simulation_filter_sample_size"] = _safe_int(metrics.get("sample_size"), 0)
        row["simulation_filter_win_rate_pct"] = metrics.get("win_rate_pct")
        row["simulation_filter_avg_pnl_pct"] = metrics.get("avg_pnl_pct")
        row["simulation_filter_profit_factor"] = metrics.get("profit_factor")
        row["simulation_filter_score_adjustment"] = adjustment
        row["simulation_filter_block_trade"] = bool(metrics.get("block_trade", False))

        if adjustment:
            reasons = row.get("risk_flags")
            if not isinstance(reasons, list):
                reasons = []
            reasons.append(f"Simulation filter adjustment {adjustment:+d}: {signal.reason}")
            row["risk_flags"] = reasons

        return row

    except Exception as exc:
        row["simulation_filter_enabled"] = True
        row["simulation_filter_adjusted"] = False
        row["simulation_filter_score"] = 0.0
        row["simulation_filter_direction"] = "NEUTRAL"
        row["simulation_filter_confidence"] = 0.0
        row["simulation_filter_reason"] = f"Simulation filter failed: {exc}"
        row["simulation_filter_score_adjustment"] = 0
        row["simulation_filter_block_trade"] = False
        return row


def _enrich_screen_row(final: dict, full_result: dict) -> dict:
    """
    Pull ranking-only diagnostics and new-agent diagnostics out of the full pipeline result.
    This keeps the pipeline output compatible while letting screen/trade-screen rank better.
    """
    diagnostic_keys = [
        "fresh_setup_score",
        "prev_5d_return",
        "prev_20d_return",
        "rsi_14",
        "zscore_20",
        "overextension",
        "new_agent_metrics",
        "new_agent_conviction_delta",
    ]

    for key in diagnostic_keys:
        if key not in final:
            value = _deep_find_value(full_result, key)
            if value is not None:
                final[key] = value

    debug = full_result.get("debug", {}) if isinstance(full_result, dict) else {}
    new_agents = debug.get("new_agents", {}) if isinstance(debug, dict) else {}
    new_agent_metrics = debug.get("new_agent_metrics", {}) if isinstance(debug, dict) else {}

    if isinstance(new_agents, dict):
        final["new_agents"] = new_agents
    if isinstance(new_agent_metrics, dict):
        final["new_agent_metrics"] = new_agent_metrics

    rs = new_agents.get("relative_strength_agent", {}) if isinstance(new_agents, dict) else {}
    intra = new_agents.get("intraday_confirmation_agent", {}) if isinstance(new_agents, dict) else {}
    liq = new_agents.get("liquidity_agent", {}) if isinstance(new_agents, dict) else {}

    rs_metrics = rs.get("metrics", {}) if isinstance(rs, dict) else {}
    intra_metrics = intra.get("metrics", {}) if isinstance(intra, dict) else {}
    liq_metrics = liq.get("metrics", {}) if isinstance(liq, dict) else {}

    final["relative_strength_score"] = _safe_float(rs.get("score"), 0.0) if isinstance(rs, dict) else 0.0
    final["relative_strength_direction"] = str(rs.get("direction", "NEUTRAL")) if isinstance(rs, dict) else "NEUTRAL"
    final["relative_strength_reason"] = str(rs.get("reason", "")) if isinstance(rs, dict) else ""
    final["relative_strength_pct"] = _safe_float(rs_metrics.get("relative_strength_pct"), 0.0)
    final["relative_strength_benchmark"] = rs_metrics.get("benchmark", "N/A")

    final["intraday_confirmation_score"] = _safe_float(intra.get("score"), 0.0) if isinstance(intra, dict) else 0.0
    final["intraday_confirmation_direction"] = str(intra.get("direction", "NEUTRAL")) if isinstance(intra, dict) else "NEUTRAL"
    final["intraday_confirmation_reason"] = str(intra.get("reason", "")) if isinstance(intra, dict) else ""
    final["intraday_confirms_bias"] = intra_metrics.get("confirms_bias")
    final["vwap_distance_pct"] = _safe_float(intra_metrics.get("vwap_distance_pct"), 0.0)
    final["intraday_momentum_pct"] = _safe_float(intra_metrics.get("momentum_pct"), 0.0)

    final["liquidity_score"] = _safe_float(liq.get("score"), 0.0) if isinstance(liq, dict) else 0.0
    final["liquidity_direction"] = str(liq.get("direction", "NEUTRAL")) if isinstance(liq, dict) else "NEUTRAL"
    final["liquidity_reason"] = str(liq.get("reason", "")) if isinstance(liq, dict) else ""
    final["spread_pct"] = liq_metrics.get("spread_pct")
    final["dollar_volume"] = liq_metrics.get("dollar_volume")
    final["tradeable"] = _tradeable_from_row(final)

    fresh = final.get("fresh_setup_score")
    if fresh is None:
        overext = final.get("overextension")
        if isinstance(overext, dict):
            penalty = _safe_float(overext.get("score_penalty"), 0.0)
            final["fresh_setup_score"] = -penalty
        else:
            final["fresh_setup_score"] = 0.0

    final = _apply_history_lookup(final)
    final = _apply_simulation_filter(final)

    return final
def _print_result(result: dict) -> None:
    r = result["final_output"]
    enriched = _enrich_screen_row(dict(r), result)

    print()
    print(LINE)
    print(f" {r['ticker']}   {r['final_action']}   {r['forecast_direction']}")
    print(LINE)
    print(f"{'Window':26} {r.get('forecast_window', 'N/A')}")
    print(f"{'Target type':26} {r.get('target_type', 'WEEKLY_TRADE_OPPORTUNITY')}")
    print(f"{'Success threshold':26} {r.get('success_threshold_pct', 1.0)}%")
    print(f"{'Run timestamp':26} {r.get('run_timestamp', 'N/A')}")
    print(f"{'Primary regime':26} {r.get('primary_regime', 'N/A')}")
    print(f"{'Ticker archetype':26} {r.get('ticker_archetype', 'NORMAL')}")
    print(f"{'Setup type':26} {r.get('setup_type', 'NO_CLEAN_SETUP')}")
    print(f"{'Model validity':26} {r.get('model_validity', 'N/A')}")
    print(f"{'Conviction score':26} {r.get('conviction_score', 0)}")
    print(f"{'Expected move %':26} {r.get('expected_move_pct', 0)}")
    print(f"{'Raw signal score':26} {r.get('raw_signal_score', 0)}")
    print(f"{'Momentum score':26} {r.get('momentum_continuation_score', 0)}")
    print(f"{'Bearish confirmations':26} {r.get('bearish_confirmations', 0)}")
    print(
        f"{'Prob UP / NEUTRAL / DOWN':26} "
        f"{_safe_float(r.get('probability_up')):.2f} / "
        f"{_safe_float(r.get('probability_neutral')):.2f} / "
        f"{_safe_float(r.get('probability_down')):.2f}"
    )

    print()
    print("New agents")
    print(
        f"  • Relative strength: {_safe_float(enriched.get('relative_strength_score')):+.2f} "
        f"vs {enriched.get('relative_strength_benchmark', 'N/A')} "
        f"({_safe_float(enriched.get('relative_strength_pct')):+.2f}%)"
    )
    print(
        f"  • Intraday: {_safe_float(enriched.get('intraday_confirmation_score')):+.2f} | "
        f"momentum={_safe_float(enriched.get('intraday_momentum_pct')):+.2f}% | "
        f"VWAP distance={_safe_float(enriched.get('vwap_distance_pct')):+.2f}% | "
        f"confirms={enriched.get('intraday_confirms_bias', 'N/A')}"
    )
    spread = enriched.get("spread_pct")
    spread_text = "N/A" if spread is None else f"{_safe_float(spread):.3f}%"
    dollar_volume = enriched.get("dollar_volume")
    dv_text = "N/A" if dollar_volume is None else f"${_safe_float(dollar_volume):,.0f}"
    print(
        f"  • Liquidity: {_safe_float(enriched.get('liquidity_score')):+.2f} | "
        f"spread={spread_text} | dollar volume={dv_text} | "
        f"tradeable={'YES' if _tradeable_from_row(enriched) else 'NO'}"
    )
    hist_rate = enriched.get("history_true_during_week_rate")
    hist_rate_text = "N/A" if hist_rate is None else f"{_safe_float(hist_rate):.2f}%"
    print(
        f"  • History: adj={_safe_int(enriched.get('history_score_adjustment')):+d} | "
        f"rate={hist_rate_text} | "
        f"sample={_safe_int(enriched.get('history_sample_size'))} | "
        f"match={enriched.get('history_match_level', 'none')}"
    )
    sim_win = enriched.get("simulation_filter_win_rate_pct")
    sim_win_text = "N/A" if sim_win is None else f"{_safe_float(sim_win):.2f}%"
    print(
        f"  • Simulation filter: adj={_safe_int(enriched.get('simulation_filter_score_adjustment')):+d} | "
        f"win={sim_win_text} | "
        f"sample={_safe_int(enriched.get('simulation_filter_sample_size'))} | "
        f"match={enriched.get('simulation_filter_match_level', 'none')} | "
        f"block={enriched.get('simulation_filter_block_trade', False)}"
    )

    print()
    print("Drivers")
    for item in r.get("drivers", []) or ["None"]:
        print(f"  • {item}")

    print()
    print("Conflicts")
    for item in r.get("conflicts", []) or ["None"]:
        print(f"  • {item}")

    print()
    print("Risk flags")
    for item in r.get("risk_flags", []) or ["None"]:
        print(f"  • {item}")

    print()
    print("Reason")
    print(f"  {r.get('reason', 'No reason returned.')}")
    print(LINE)
    print()
def _load_universe(name: str) -> list[str]:
    root = Path(__file__).resolve().parent / "data"

    mapping = {
        "top50": root / "tickers_top50.txt",
        "custom": root / "tickers_custom.txt",
    }

    path = mapping[name]

    if not path.exists():
        raise FileNotFoundError(f"Missing universe file: {path}")

    return [
        normalize_ticker(line.strip())
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]



def _apply_chunk(
    tickers: list[str],
    chunk_index: int | None = None,
    chunk_count: int | None = None,
) -> list[str]:
    """Return one deterministic slice of the ticker universe for parallel scans.

    Example:
      --chunk-count 6 --chunk-index 0
      --chunk-count 6 --chunk-index 1
      ...
      --chunk-count 6 --chunk-index 5

    Uses modulo slicing so every chunk receives a balanced mix from the universe.
    """

    if chunk_index is None and chunk_count is None:
        return tickers

    if chunk_index is None or chunk_count is None:
        raise ValueError("Both --chunk-index and --chunk-count must be provided together.")

    if chunk_count <= 0:
        raise ValueError("--chunk-count must be greater than 0.")

    if chunk_index < 0 or chunk_index >= chunk_count:
        raise ValueError("--chunk-index must be between 0 and chunk_count - 1.")

    return [
        ticker
        for i, ticker in enumerate(tickers)
        if i % chunk_count == chunk_index
    ]


def _dedupe_alias_results(results: list[dict]) -> list[dict]:
    chosen: list[dict] = []
    used_groups: set[tuple[str, ...]] = set()
    used_tickers: set[str] = set()

    for result in results:
        ticker = normalize_ticker(result["ticker"])
        result["ticker"] = ticker

        group = next((grp for grp in ALIAS_GROUPS if ticker in grp), None)

        if group:
            group_key = tuple(sorted(group))

            if group_key in used_groups:
                continue

            group_items = [
                item
                for item in results
                if normalize_ticker(item["ticker"]) in group
            ]

            best = max(
                group_items,
                key=lambda item: (
                    _safe_int(item.get("conviction_score")),
                    abs(_safe_float(item.get("raw_signal_score"))),
                    abs(_safe_float(item.get("expected_move_pct"))),
                ),
            )

            best["ticker"] = normalize_ticker(best["ticker"])
            chosen.append(best)
            used_groups.add(group_key)
            used_tickers.add(best["ticker"])

        elif ticker not in used_tickers:
            chosen.append(result)
            used_tickers.add(ticker)

    return chosen


def _passes_display_filter(row: dict, show_all: bool = False) -> bool:
    if show_all:
        return True

    action = str(row.get("final_action", "WATCH")).upper()
    direction = str(row.get("forecast_direction", "NEUTRAL")).upper()
    edge = str(row.get("estimated_edge", "WEAK")).upper()
    conviction = _safe_int(row.get("conviction_score"))
    score = abs(_safe_float(row.get("raw_signal_score")))
    expected_move = abs(_safe_float(row.get("expected_move_pct")))
    fresh = _safe_float(row.get("fresh_setup_score"), 0.0)

    # Always show real calls.
    if action in {"BUY", "SELL"}:
        return True

    # Hide crowded continuation WATCH names.
    # This is what removes NVDA/AMD-style "already ran" names from the main screen.
    if action == "WATCH" and fresh < -0.12:
        return False

    # Show clean fresh setups first.
    if fresh >= 0.04 and direction in {"UP", "DOWN"}:
        return True

    # Keep meaningful non-BUY/SELL ideas.
    if edge in {"MODERATE", "STRONG"}:
        return True

    # Directional WATCH names need more than weak momentum.
    if direction in {"UP", "DOWN"} and conviction >= 28 and expected_move >= 1.0 and fresh >= -0.05:
        return True

    if direction in {"UP", "DOWN"} and score >= 0.30 and expected_move >= 1.5 and fresh >= 0.0:
        return True

    return False

def _setup_rank_score(row: dict) -> float:
    action = str(row.get("final_action", "WATCH")).upper()
    direction = str(row.get("forecast_direction", "NEUTRAL")).upper()

    conviction = _safe_int(row.get("conviction_score"))
    fresh = _safe_float(row.get("fresh_setup_score"), 0.0)
    raw = abs(_safe_float(row.get("raw_signal_score"), 0.0))
    expected_move_pct = abs(_safe_float(row.get("expected_move_pct"), 0.0))

    prev_5d = abs(_safe_float(row.get("prev_5d_return"), 0.0))
    prev_20d = abs(_safe_float(row.get("prev_20d_return"), 0.0))
    rsi = _safe_float(row.get("rsi_14"), 50.0)
    z20 = abs(_safe_float(row.get("zscore_20"), 0.0))

    rs_score = _safe_float(row.get("relative_strength_score"), 0.0)
    intra_score = _safe_float(row.get("intraday_confirmation_score"), 0.0)
    liq_score = _safe_float(row.get("liquidity_score"), 0.0)
    history_adjustment = _safe_int(row.get("history_score_adjustment"), 0)
    history_confidence = _safe_float(row.get("history_lookup_confidence"), 0.0)
    sim_adjustment = _safe_int(row.get("simulation_filter_score_adjustment"), 0)
    sim_confidence = _safe_float(row.get("simulation_filter_confidence"), 0.0)
    sim_block = bool(row.get("simulation_filter_block_trade", False))
    spread_pct = _safe_float(row.get("spread_pct"), 0.0)
    tradeable = _tradeable_from_row(row)

    score = 0.0

    if action in {"BUY", "SELL"}:
        score += 22
    elif action == "WATCH":
        score += 8

    if direction in {"UP", "DOWN"}:
        score += 5

    score += conviction * 0.50
    score += raw * 16
    score += expected_move_pct * 1.2
    score += fresh * 38

    if direction == "UP":
        score += rs_score * 8
        score += intra_score * 6
    elif direction == "DOWN":
        score -= rs_score * 8
        score -= intra_score * 6

    score += liq_score * 4
    score += history_adjustment * max(0.35, history_confidence)
    score += sim_adjustment * max(0.35, sim_confidence)
    if sim_block:
        score -= 25

    if not tradeable:
        score -= 35
    elif spread_pct >= 0.25:
        score -= 8

    if prev_5d >= 0.10:
        score -= 12
    elif prev_5d >= 0.07:
        score -= 6

    if prev_20d >= 0.25:
        score -= 16
    elif prev_20d >= 0.16:
        score -= 8

    if rsi >= 90:
        score -= 18
    elif rsi >= 84:
        score -= 9

    if z20 >= 2.4:
        score -= 12
    elif z20 >= 1.8:
        score -= 6

    setup_type = str(row.get("setup_type", "NO_CLEAN_SETUP")).upper()
    if action == "BUY" and direction == "UP" and setup_type == "MOMENTUM_CONTINUATION":
        score += 18
    elif action == "BUY" and direction == "UP" and setup_type == "UP_OPPORTUNITY":
        score -= 12
    elif action == "SELL" and direction == "DOWN":
        score -= 20

    return score
def _sort_results(results: list[dict]) -> list[dict]:
    return sorted(
        results,
        key=lambda row: (
            _setup_rank_score(row),
            _safe_float(row.get("fresh_setup_score"), 0.0),
            _safe_int(row.get("conviction_score")),
            abs(_safe_float(row.get("raw_signal_score"))),
        ),
        reverse=True,
    )


def _print_screen_table(results: list[dict]) -> None:
    print()
    print(
        f"{'Rank':<4}"
        f"{'Ticker':<8}"
        f"{'Action':<8}"
        f"{'Dir':<8}"
        f"{'Conv':<7}"
        f"{'Edge':<9}"
        f"{'SetupType':<24}"
        f"{'Score':<8}"
        f"{'Exp%':<8}"
        f"{'RS':<8}"
        f"{'Intra':<8}"
        f"{'Spread':<10}"
        f"{'Liq':<7}"
        f"{'Mom':<7}"
        f"{'Bear':<6}"
        f"{'Hist':<13}"
        f"{'Sim':<13}"
        f"{'Timestamp':<24}"
    )
    print("-" * 190)

    for i, r in enumerate(results, start=1):
        spread = r.get("spread_pct")
        spread_text = "N/A" if spread is None else f"{_safe_float(spread):.3f}%"
        liq_text = "PASS" if _tradeable_from_row(r) else "BLOCK"
        hist_rate = r.get("history_true_during_week_rate")
        hist_rate_text = "N/A" if hist_rate is None else f"{_safe_float(hist_rate):.1f}%"
        hist_text = f"{_safe_int(r.get('history_score_adjustment')):+d}/{hist_rate_text}"
        sim_win = r.get("simulation_filter_win_rate_pct")
        sim_win_text = "N/A" if sim_win is None else f"{_safe_float(sim_win):.1f}%"
        sim_text = f"{_safe_int(r.get('simulation_filter_score_adjustment')):+d}/{sim_win_text}"

        print(
            f"{i:<4}"
            f"{r.get('ticker', 'N/A'):<8}"
            f"{r.get('final_action', 'WATCH'):<8}"
            f"{r.get('forecast_direction', 'NEUTRAL'):<8}"
            f"{_safe_int(r.get('conviction_score')):<7}"
            f"{r.get('estimated_edge', 'WEAK'):<9}"
            f"{str(r.get('setup_type', 'NO_CLEAN_SETUP')):<24}"
            f"{_safe_float(r.get('raw_signal_score')):<8.3f}"
            f"{_safe_float(r.get('expected_move_pct')):<8.2f}"
            f"{_safe_float(r.get('relative_strength_score')):<8.2f}"
            f"{_safe_float(r.get('intraday_confirmation_score')):<8.2f}"
            f"{spread_text:<10}"
            f"{liq_text:<7}"
            f"{_safe_float(r.get('momentum_continuation_score')):<7.3f}"
            f"{_safe_int(r.get('bearish_confirmations')):<6}"
            f"{hist_text:<13}"
            f"{sim_text:<13}"
            f"{str(r.get('run_timestamp', 'N/A')):<24}"
        )

    print()

def _write_recommendation_output(
    output_path: str | Path,
    *,
    universe: str,
    chunk_index: int | None,
    chunk_count: int | None,
    candidates: list[dict],
    failures: list[tuple[str, str]],
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": datetime.now().isoformat(),
        "universe": universe,
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "failures": [
            {"ticker": ticker, "error": error}
            for ticker, error in failures
        ],
    }

    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Saved recommendation output: {path} ({len(candidates)} candidate(s))")


def _load_recommendation_files(input_dir: str | Path) -> list[dict]:
    root = Path(input_dir)

    if not root.exists():
        raise FileNotFoundError(f"Recommendation input directory does not exist: {root}")

    files = sorted(root.rglob("*.json"))

    if not files:
        raise FileNotFoundError(f"No JSON recommendation files found under: {root}")

    candidates: list[dict] = []

    for path in files:
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            print(f"Skipping unreadable recommendation file {path}: {exc}")
            continue

        rows = payload.get("candidates")
        if not isinstance(rows, list):
            print(f"Skipping {path}: missing candidates list.")
            continue

        for row in rows:
            if isinstance(row, dict):
                row = dict(row)
                row["_source_file"] = str(path)
                candidates.append(row)

    if not candidates:
        raise RuntimeError(f"Recommendation files were found, but no candidates were loaded from: {root}")

    return candidates


def _execute_ranked_trades(
    input_dir: str | Path,
    dry_run: bool = False,
    max_trade_slots: int | None = None,
    show_all: bool = False,
    limit: int | None = None,
    second_chance: bool = False,
) -> int:
    init_db()

    print()
    print(f"Loading scan recommendations from: {input_dir}")

    results = _load_recommendation_files(input_dir)
    results = _dedupe_alias_results(results)
    results = _sort_results(results)

    filtered = [
        row
        for row in results
        if _passes_display_filter(row, show_all=show_all)
    ]

    if second_chance:
        strict_rows: list[dict] = []
        for row in filtered:
            ok, reason = _passes_second_chance_filter(row)
            if ok:
                strict_rows.append(row)
            else:
                print(f"Second-chance skip {row.get('ticker', 'N/A')}: {reason}.")
        filtered = strict_rows

    if limit is not None:
        filtered = filtered[:limit]

    if not filtered:
        label = "second-chance filter" if second_chance else "display filter"
        print(f"No loaded recommendations passed the {label}. Nothing sent to Alpaca add-on.")
        return 0

    print()
    print(f"Loaded {len(results)} total recommendation(s).")
    print(f"{len(filtered)} recommendation(s) passed the final filter.")
    if second_chance:
        print("Second-chance mode is enabled: only strongest Tuesday/Wednesday signals are eligible.")
    print("Final executor will rank all chunks together, check real open slots, then send only selected trades.")
    print()

    _print_screen_table(filtered[:25])

    from app.config import settings

    trade_limit = settings.auto_trade_max_trades_per_run
    if max_trade_slots is not None:
        trade_limit = min(trade_limit, max_trade_slots)

    trader = AutoTrader()

    def _setting_int(*names: str, default: int | None = None) -> int | None:
        for name in names:
            if hasattr(settings, name):
                value = getattr(settings, name)
                if value is not None:
                    return _safe_int(value, default or 0)
        return default

    def _position_symbol(pos: Any) -> str:
        if isinstance(pos, dict):
            return normalize_ticker(str(pos.get("symbol", "")))
        return normalize_ticker(str(getattr(pos, "symbol", "")))

    def _active_position_symbols() -> set[str]:
        symbols: set[str] = set()

        broker = getattr(trader, "broker", None)
        if broker is None:
            broker = getattr(trader, "client", None)

        if broker is None:
            return symbols

        methods = [
            "get_all_positions",
            "list_positions",
            "get_positions",
            "positions",
        ]

        for method_name in methods:
            try:
                method = getattr(broker, method_name, None)
                if method is None:
                    continue

                positions = method() if callable(method) else method
                if positions is None:
                    continue

                for pos in positions:
                    symbol = _position_symbol(pos)
                    if symbol:
                        symbols.add(symbol)

                if symbols:
                    return symbols

            except Exception:
                continue

        return symbols

    max_active_positions = _setting_int(
        "auto_trade_max_active_positions",
        "auto_trade_max_positions",
        "auto_trade_max_open_positions",
        default=None,
    )

    active_symbols = _active_position_symbols()

    if max_active_positions is not None and active_symbols:
        open_slots = max(0, max_active_positions - len(active_symbols))

        print(
            f"Portfolio slots before execution: "
            f"active={len(active_symbols)}, "
            f"max_active={max_active_positions}, "
            f"open={open_slots}"
        )

        trade_limit = min(trade_limit, open_slots)

    if trade_limit <= 0:
        print("No open portfolio slots before execution. Nothing sent to Alpaca.")
        return 0

    eligible_rows: list[dict] = []

    for row in filtered:
        ticker = normalize_ticker(str(row.get("ticker", "")))

        if ticker in active_symbols:
            print(f"Skipping {ticker}: already has an active position.")
            continue

        ok, reason = _passes_trade_safety_gate(row)
        if not ok:
            print(f"Skipping {ticker or 'N/A'}: {reason}.")
            continue

        eligible_rows.append(row)

        if len(eligible_rows) >= trade_limit:
            break

    if not eligible_rows:
        print("No ranked recommendations passed the final trade safety gate.")
        return 0

    print()
    print(f"Selected {len(eligible_rows)} trade(s) for the actual open slot(s):")
    for i, row in enumerate(eligible_rows, start=1):
        print(
            f"  {i}. {row.get('ticker', 'N/A')} | "
            f"{row.get('final_action', 'WATCH')} | "
            f"{row.get('forecast_direction', 'NEUTRAL')} | "
            f"conviction={row.get('conviction_score', 0)} | "
            f"setup={row.get('setup_type', 'NO_CLEAN_SETUP')}"
        )
    print()

    trades_attempted = 0

    for row in eligible_rows:
        did_attempt = trader.process_model_output(row, dry_run=dry_run)
        if did_attempt:
            trades_attempted += 1

    print(f"Final executor trade attempts: {trades_attempted}")
    return trades_attempted

def _screen(
    universe: str,
    limit: int | None = None,
    show_all: bool = False,
    chunk_index: int | None = None,
    chunk_count: int | None = None,
) -> None:
    init_db()

    tickers = _load_universe(universe)
    tickers = _apply_chunk(tickers, chunk_index=chunk_index, chunk_count=chunk_count)

    if limit:
        tickers = tickers[:limit]

    total = len(tickers)

    print()
    if chunk_index is not None and chunk_count is not None:
        print(
            f"Running fresh model on chunk {chunk_index + 1}/{chunk_count} "
            f"with {total} ticker(s) from universe='{universe}'..."
        )
    else:
        print(f"Running fresh model on {total} tickers from universe='{universe}'...")
    print("The screen prints the actual pipeline final_output. It does not overwrite actions or directions.")
    print()

    results: list[dict] = []
    failures: list[tuple[str, str]] = []

    for idx, ticker in enumerate(tickers, start=1):
        ticker = normalize_ticker(ticker)
        print(f"[{idx}/{total}] Working on {ticker}...")

        try:
            out = run_pipeline(ticker)
            final = dict(out["final_output"])
            final = _enrich_screen_row(final, out)

            final["ticker"] = normalize_ticker(final.get("ticker", ticker))

            results.append(final)

            print(
                f"         Done: {final['ticker']} | "
                f"{final.get('final_action', 'WATCH')} | "
                f"{final.get('forecast_direction', 'NEUTRAL')} | "
                f"conviction={final.get('conviction_score', 0)} | "
                f"score={_safe_float(final.get('raw_signal_score')):.3f} | "
                f"setup_type={final.get('setup_type', 'NO_CLEAN_SETUP')} | "
                f"mom={_safe_float(final.get('momentum_continuation_score')):.3f} | "
                f"{_new_agent_summary(final)} | "
                f"timestamp={final.get('run_timestamp', 'N/A')}"
            )

        except Exception as exc:
            logger.warning("Failed on %s: %s", ticker, exc)
            failures.append((ticker, str(exc)))
            print(f"         Failed: {ticker} | {exc}")

    if failures and not results:
        raise RuntimeError(f"All tickers failed. First few errors: {failures[:5]}")

    results = _dedupe_alias_results(results)
    results = _sort_results(results)

    filtered = [
        row
        for row in results
        if _passes_display_filter(row, show_all=show_all)
    ]

    if not filtered:
        print()
        print("No names passed the display filter.")
        print("Run again with --all to print every ticker.")
        if failures:
            print()
            print("Some tickers failed:")
            for ticker, err in failures[:10]:
                print(f"  • {ticker}: {err}")
        print()
        return 0

    _print_screen_table(filtered[:25])

    if failures:
        print("Some tickers failed:")
        for ticker, err in failures[:10]:
            print(f"  • {ticker}: {err}")
        print()



def _run_and_trade_ticker(ticker: str, dry_run: bool = False) -> None:
    init_db()
    ticker = normalize_ticker(ticker)
    print(f"Running model and Alpaca paper-trade decision for {ticker}...")
    result = run_pipeline(ticker)
    final = dict(result["final_output"])
    _print_result(result)
    trader = AutoTrader()
    trader.process_model_output(final, dry_run=dry_run)




_TICKER_PENALTY_CACHE: dict[str, dict[str, Any]] | None = None


def _load_ticker_penalty_cache() -> dict[str, dict[str, Any]]:
    global _TICKER_PENALTY_CACHE
    if _TICKER_PENALTY_CACHE is not None:
        return _TICKER_PENALTY_CACHE

    from app.config import settings
    import csv

    path = Path(settings.ticker_penalty_summary_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path

    out: dict[str, dict[str, Any]] = {}
    try:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                ticker = str(row.get("ticker", "")).upper().strip()
                if ticker:
                    out[ticker] = row
    except Exception as exc:
        print(f"Ticker penalty file unavailable: {path} ({exc}). Continuing without ticker penalty filter.")

    _TICKER_PENALTY_CACHE = out
    return out


def _passes_already_ran_filter(row: dict) -> tuple[bool, str]:
    from app.config import settings

    if not settings.already_ran_filter_enabled:
        return True, "already-ran filter disabled"

    direction = str(row.get("forecast_direction", "NEUTRAL")).upper()
    if direction != "UP":
        return True, "already-ran filter applies only to UP trades"

    hist_sample = _safe_int(row.get("history_sample_size"), 0)
    if hist_sample < settings.already_ran_min_history_sample_size:
        return True, "not enough history for already-ran filter"

    avg_best = row.get("history_average_best_correct_return_pct")
    if avg_best is None:
        avg_best = row.get("average_best_correct_return_pct")
    avg_best_f = _safe_float(avg_best, 0.0)
    if avg_best_f <= 0:
        return True, "missing historical best move"

    current_move = row.get("current_week_return_pct")
    if current_move is None:
        current_move = row.get("week_to_date_return_pct")
    if current_move is None:
        current_move = row.get("prev_5d_return")
        current_move_f = _safe_float(current_move, 0.0) * 100.0
    else:
        current_move_f = _safe_float(current_move, 0.0)
        if abs(current_move_f) <= 1.0:
            current_move_f *= 100.0

    trigger = avg_best_f * float(settings.already_ran_capture_ratio)
    if current_move_f >= trigger:
        return False, (
            f"already-ran filter: recent move {current_move_f:.2f}% is already >= "
            f"{settings.already_ran_capture_ratio:.0%} of historical avg best move {avg_best_f:.2f}%"
        )

    return True, "passed already-ran filter"


def _passes_ticker_penalty_filter(row: dict) -> tuple[bool, str]:
    from app.config import settings

    if not settings.ticker_penalty_enabled:
        return True, "ticker penalty disabled"

    ticker = str(row.get("ticker", "")).upper().strip()
    if not ticker:
        return True, "missing ticker; ticker penalty skipped"

    metrics = _load_ticker_penalty_cache().get(ticker)
    if not metrics:
        return True, "no ticker penalty metrics"

    sample = _safe_int(metrics.get("sample_size"), 0)
    if sample < settings.ticker_penalty_min_sample_size:
        return True, "ticker sample too small for penalty"

    conviction = _safe_int(row.get("conviction_score"), 0)
    success = _safe_float(metrics.get("true_during_week_rate"), 0.0)
    adverse = abs(_safe_float(metrics.get("average_adverse_move_pct"), 0.0))

    if conviction >= settings.ticker_penalty_block_conviction_below:
        return True, "high conviction overrides ticker penalty"

    if success < settings.ticker_penalty_min_success_rate:
        return False, (
            f"ticker penalty: {ticker} historical success rate {success:.2f}% "
            f"below {settings.ticker_penalty_min_success_rate:.2f}%"
        )

    if adverse > settings.ticker_penalty_max_adverse_move_pct:
        return False, (
            f"ticker penalty: {ticker} average adverse move {adverse:.2f}% "
            f"above {settings.ticker_penalty_max_adverse_move_pct:.2f}%"
        )

    return True, "passed ticker penalty filter"


def _passes_gap_risk_filter(row: dict) -> tuple[bool, str]:
    from app.config import settings

    if not settings.gap_risk_filter_enabled:
        return True, "gap risk filter disabled"

    conviction = _safe_int(row.get("conviction_score"), 0)
    gap = row.get("average_overnight_gap_pct")
    if gap is None:
        gap = row.get("overnight_gap_risk_pct")
    if gap is None:
        gap = row.get("gap_risk_pct")
    if gap is None:
        return True, "gap risk metric unavailable"

    gap_f = abs(_safe_float(gap, 0.0))
    if gap_f <= 1.0:
        gap_f *= 100.0

    if gap_f > settings.gap_risk_max_overnight_gap_pct and conviction < settings.gap_risk_block_conviction_below:
        return False, (
            f"gap risk filter: average overnight gap {gap_f:.2f}% exceeds "
            f"{settings.gap_risk_max_overnight_gap_pct:.2f}% for sub-{settings.gap_risk_block_conviction_below} conviction"
        )

    return True, "passed gap risk filter"


def _passes_rank_stability_filter(row: dict) -> tuple[bool, str]:
    from app.config import settings

    if not settings.rank_stability_filter_enabled:
        return True, "rank stability disabled"

    seen = row.get("rank_stability_count")
    if seen is None:
        seen = row.get("recent_top10_seen_count")
    if seen is None:
        return True, "rank stability metric unavailable"

    seen_i = _safe_int(seen, 0)
    if seen_i < settings.rank_stability_min_seen_count:
        return False, f"rank stability filter: seen {seen_i} time(s), needs {settings.rank_stability_min_seen_count}"

    return True, "passed rank stability filter"


def _passes_simulation_filter(row: dict) -> tuple[bool, str]:
    from app.config import settings

    if not settings.simulation_filter_enabled or not settings.simulation_filter_block_enabled:
        return True, "simulation filter disabled"

    if not row.get("simulation_filter_adjusted"):
        row = _apply_simulation_filter(row)

    if bool(row.get("simulation_filter_block_trade", False)):
        conviction = _safe_int(row.get("conviction_score"), 0)
        if conviction >= settings.simulation_filter_high_conviction_override:
            return True, "high conviction overrides simulation block"
        return False, f"simulation filter block: {row.get('simulation_filter_reason', 'bad historical sim profile')}"

    return True, "passed simulation filter"


def _passes_soft_earnings_filter(row: dict) -> tuple[bool, str]:
    from app.config import settings

    if not settings.earnings_risk_filter_enabled:
        return True, "earnings risk filter disabled"

    conviction = _safe_int(row.get("conviction_score"), 0)
    ticker = str(row.get("ticker", "")).upper().strip()

    # Prefer model-provided dates if a future data agent supplies them.
    raw_date = row.get("earnings_date") or row.get("next_earnings_date")
    days_until: int | None = None
    if raw_date:
        try:
            earnings_date = datetime.fromisoformat(str(raw_date)[:10]).date()
            days_until = (earnings_date - date.today()).days
        except Exception:
            days_until = None

    if days_until is None and ticker:
        # Best-effort lookup. If Yahoo/calendar is unavailable, fail open.
        try:
            import yfinance as yf

            cal = yf.Ticker(ticker).calendar
            if hasattr(cal, "loc"):
                for key in ("Earnings Date", "Earnings Average", "Earnings Low", "Earnings High"):
                    if key in cal.index:
                        value = cal.loc[key]
                        if hasattr(value, "iloc"):
                            value = value.iloc[0]
                        earnings_date = datetime.fromisoformat(str(value)[:10]).date()
                        days_until = (earnings_date - date.today()).days
                        break
            elif isinstance(cal, dict):
                value = cal.get("Earnings Date") or cal.get("earningsDate")
                if isinstance(value, (list, tuple)) and value:
                    value = value[0]
                if value:
                    earnings_date = datetime.fromisoformat(str(value)[:10]).date()
                    days_until = (earnings_date - date.today()).days
        except Exception:
            return True, "earnings date unavailable; soft filter skipped"

    if days_until is None:
        return True, "earnings date unavailable"

    if 0 <= days_until <= settings.earnings_risk_days and conviction < settings.earnings_risk_min_conviction:
        return False, (
            f"soft earnings filter: earnings in {days_until} day(s), conviction {conviction} "
            f"below {settings.earnings_risk_min_conviction}"
        )

    return True, "passed soft earnings filter"

def _passes_trade_safety_gate(row: dict) -> tuple[bool, str]:
    from app.config import settings

    action = str(row.get("final_action", "WATCH")).upper()
    direction = str(row.get("forecast_direction", "NEUTRAL")).upper()
    setup_type = str(row.get("setup_type", "NO_CLEAN_SETUP")).upper()
    edge = str(row.get("estimated_edge", "WEAK")).upper()
    conviction = _safe_int(row.get("conviction_score"), 0)

    # Final xanax concentrated model: stock-only, LONG/BUY/UP only.
    if action != "BUY" or direction != "UP":
        return False, "only BUY/UP long stock trades are allowed in the top-5 xanax model"

    for check in (
        _passes_already_ran_filter,
        _passes_ticker_penalty_filter,
        _passes_gap_risk_filter,
        _passes_rank_stability_filter,
        _passes_simulation_filter,
        _passes_soft_earnings_filter,
    ):
        ok, reason = check(row)
        if not ok:
            return False, reason

    if not _tradeable_from_row(row):
        return False, "liquidity gate failed"

    spread_pct = row.get("spread_pct")
    spread_f = _safe_float(spread_pct) if spread_pct is not None else 0.0
    if spread_pct is not None and spread_f > 0.50:
        return False, f"spread too wide at {spread_f:.3f}%"

    intra_confirms = row.get("intraday_confirms_bias")
    if intra_confirms is False and conviction < 70:
        return False, "intraday agent conflicts with a sub-70 conviction trade"

    hist_rate = row.get("history_true_during_week_rate")
    hist_rate_f = _safe_float(hist_rate, 0.0) if hist_rate is not None else 0.0
    hist_sample = _safe_int(row.get("history_sample_size"), 0)

    # Tuned strategy from simulation: favor LONG/UP momentum.
    if action == "BUY" and direction == "UP":
        if setup_type == "MOMENTUM_CONTINUATION":
            if hist_rate is not None and hist_rate_f < settings.tuned_long_momentum_min_history_rate:
                return False, f"long momentum history rate {hist_rate_f:.2f}% below {settings.tuned_long_momentum_min_history_rate:.2f}%"
            return True, "passed tuned long/up momentum gate"

        if setup_type == "UP_OPPORTUNITY":
            if conviction < settings.tuned_up_opportunity_min_conviction:
                return False, f"UP_OPPORTUNITY conviction {conviction} below top-tier minimum {settings.tuned_up_opportunity_min_conviction}"
            if hist_rate is None or hist_rate_f < settings.tuned_up_opportunity_min_history_rate:
                return False, f"UP_OPPORTUNITY history rate {hist_rate_f:.2f}% below top-tier minimum {settings.tuned_up_opportunity_min_history_rate:.2f}%"
            if edge != "STRONG":
                return False, "UP_OPPORTUNITY requires STRONG edge"
            return True, "passed top-tier up opportunity gate"

        return False, f"unsupported long setup: {setup_type}"

    if action == "SELL" and direction == "DOWN":
        if settings.tuned_block_weak_shorts:
            if setup_type != "BREAKDOWN_CONTINUATION":
                return False, "short blocked unless setup is BREAKDOWN_CONTINUATION"
            if conviction < settings.tuned_short_min_conviction:
                return False, f"short conviction {conviction} below extreme minimum {settings.tuned_short_min_conviction}"
            if hist_rate is None or hist_rate_f < settings.tuned_short_min_history_rate:
                return False, f"short history rate {hist_rate_f:.2f}% below extreme minimum {settings.tuned_short_min_history_rate:.2f}%"
            if hist_sample < settings.tuned_short_min_history_sample_size:
                return False, f"short history sample {hist_sample} below {settings.tuned_short_min_history_sample_size}"
            if spread_pct is not None and spread_f > settings.tuned_short_max_spread_pct:
                return False, f"short spread {spread_f:.3f}% above max {settings.tuned_short_max_spread_pct:.3f}%"
            if settings.tuned_short_require_strong_edge and edge != "STRONG":
                return False, f"short requires STRONG edge, got {edge}"

        return True, "passed extreme short gate"

    return False, f"model action/direction is not eligible: {action}/{direction}"



def _passes_second_chance_filter(row: dict) -> tuple[bool, str]:
    """Strict Tuesday/Wednesday entry filter.

    This is intentionally stricter than the normal Monday scan. It is meant to use
    only leftover buying power after the Monday basket and only accept the clearest
    signals.
    """
    from app.config import settings

    action = str(row.get("final_action", "WATCH")).upper()
    direction = str(row.get("forecast_direction", "NEUTRAL")).upper()

    if action not in {"BUY", "SELL"}:
        return False, "second-chance requires BUY/SELL action"
    if direction not in {"UP", "DOWN"}:
        return False, "second-chance requires directional UP/DOWN forecast"

    conviction = _safe_int(row.get("conviction_score"), 0)
    if conviction < settings.second_chance_min_conviction:
        return False, f"second-chance conviction {conviction} below {settings.second_chance_min_conviction}"

    hist_rate = row.get("history_true_during_week_rate")
    if hist_rate is None:
        return False, "second-chance requires a history match"
    hist_rate_f = _safe_float(hist_rate, 0.0)
    if hist_rate_f < settings.second_chance_min_history_rate:
        return False, f"second-chance history rate {hist_rate_f:.2f}% below {settings.second_chance_min_history_rate:.2f}%"

    hist_sample = _safe_int(row.get("history_sample_size"), 0)
    if hist_sample < settings.second_chance_min_history_sample_size:
        return False, f"second-chance history sample {hist_sample} below {settings.second_chance_min_history_sample_size}"

    if not _tradeable_from_row(row):
        return False, "second-chance liquidity gate failed"

    spread_pct = row.get("spread_pct")
    if spread_pct is not None and _safe_float(spread_pct) > settings.second_chance_max_spread_pct:
        return False, f"second-chance spread too wide at {_safe_float(spread_pct):.3f}%"

    intra_confirms = row.get("intraday_confirms_bias")
    if settings.second_chance_require_intraday_confirmation and intra_confirms is False:
        return False, "second-chance intraday confirmation conflicts"

    edge = str(row.get("estimated_edge", "WEAK")).upper()
    if settings.second_chance_require_moderate_edge and edge not in {"MODERATE", "STRONG"}:
        return False, f"second-chance requires MODERATE/STRONG edge, got {edge}"

    tuned_ok, tuned_reason = _passes_trade_safety_gate(row)
    if not tuned_ok:
        return False, f"second-chance tuned strategy gate failed: {tuned_reason}"

    return True, "passed second-chance filter"

def _trade_screen(
    universe: str,
    limit: int | None = None,
    show_all: bool = False,
    dry_run: bool = False,
    max_trade_slots: int | None = None,
    chunk_index: int | None = None,
    chunk_count: int | None = None,
    recommend_only: bool = False,
    output: str | None = None,
    second_chance: bool = False,
) -> int:
    init_db()
    tickers = _load_universe(universe)
    tickers = _apply_chunk(tickers, chunk_index=chunk_index, chunk_count=chunk_count)

    if limit:
        tickers = tickers[:limit]

    total = len(tickers)
    print()
    if chunk_index is not None and chunk_count is not None:
        print(
            f"Running model screen and Alpaca paper-trade decisions on chunk {chunk_index + 1}/{chunk_count} "
            f"with {total} ticker(s) from universe='{universe}'..."
        )
    else:
        print(f"Running model screen and Alpaca paper-trade decisions on {total} tickers from universe='{universe}'...")
    print("The current model chooses candidates. The Alpaca add-on only trades candidates that pass risk checks.")
    if second_chance:
        print("Second-chance mode is enabled: Tue/Wed entries require stronger conviction/history/liquidity.")
    print()

    results: list[dict] = []
    failures: list[tuple[str, str]] = []

    for idx, ticker in enumerate(tickers, start=1):
        ticker = normalize_ticker(ticker)
        print(f"[{idx}/{total}] Working on {ticker}...")
        try:
            out = run_pipeline(ticker)
            final = dict(out["final_output"])
            final = _enrich_screen_row(final, out)
            final["ticker"] = normalize_ticker(final.get("ticker", ticker))
            results.append(final)
            print(
                f"         Done: {final['ticker']} | "
                f"{final.get('final_action', 'WATCH')} | "
                f"{final.get('forecast_direction', 'NEUTRAL')} | "
                f"conviction={final.get('conviction_score', 0)} | "
                f"edge={final.get('estimated_edge', 'WEAK')} | "
                f"setup_type={final.get('setup_type', 'NO_CLEAN_SETUP')} | "
                f"{_new_agent_summary(final)}"
            )
        except Exception as exc:
            logger.warning("Failed on %s: %s", ticker, exc)
            failures.append((ticker, str(exc)))
            print(f"         Failed: {ticker} | {exc}")

    if failures and not results:
        raise RuntimeError(f"All tickers failed. First few errors: {failures[:5]}")

    results = _dedupe_alias_results(results)
    results = _sort_results(results)
    filtered = [row for row in results if _passes_display_filter(row, show_all=show_all)]

    if second_chance:
        strict_rows: list[dict] = []
        for row in filtered:
            ok, reason = _passes_second_chance_filter(row)
            if ok:
                strict_rows.append(row)
            else:
                print(f"Second-chance skip {row.get('ticker', 'N/A')}: {reason}.")
        filtered = strict_rows

    if not filtered:
        label = "second-chance filter" if second_chance else "model display filter"
        print(f"No names passed the {label}. Nothing sent to Alpaca add-on.")
        if recommend_only and output:
            _write_recommendation_output(
                output,
                universe=universe,
                chunk_index=chunk_index,
                chunk_count=chunk_count,
                candidates=[],
                failures=failures,
            )
        return 0

    _print_screen_table(filtered[:25])

    if recommend_only:
        if not output:
            raise ValueError("--output must be provided when --recommend-only is used.")

        _write_recommendation_output(
            output,
            universe=universe,
            chunk_index=chunk_index,
            chunk_count=chunk_count,
            candidates=filtered,
            failures=failures,
        )
        print("Recommend-only mode is enabled. No Alpaca orders were sent from this chunk.")
        return len(filtered)

    trader = AutoTrader()
    trades_attempted = 0

    from app.config import settings
    trade_limit = settings.auto_trade_max_trades_per_run
    if max_trade_slots is not None:
        trade_limit = min(trade_limit, max_trade_slots)

    for row in filtered:
        if trades_attempted >= trade_limit:
            print(f"Auto-trade limit reached: {trade_limit}")
            break

        ok, reason = _passes_trade_safety_gate(row)
        if not ok:
            print(f"Skipping {row.get('ticker', 'N/A')}: {reason}.")
            continue

        did_attempt = trader.process_model_output(row, dry_run=dry_run)
        if did_attempt:
            trades_attempted += 1

    if failures:
        print()
        print("Some tickers failed:")
        for ticker, err in failures[:10]:
            print(f"  • {ticker}: {err}")
        print()

    return trades_attempted


def _today_order_count() -> int:
    init_db()
    today = date.today().isoformat()
    with connect() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS n
            FROM trade_decisions
            WHERE date(created_at) = ?
              AND status IN ('ORDER_SENT', 'DRY_RUN_APPROVED')
            """,
            (today,),
        ).fetchone()
    return int(row["n"] if row else 0)


def _past_no_new_trades_time(now: datetime) -> bool:
    from app.config import settings

    cutoff = now.replace(
        hour=settings.live_loop_no_new_trades_after_hour,
        minute=settings.live_loop_no_new_trades_after_minute,
        second=0,
        microsecond=0,
    )
    return now >= cutoff


def _market_is_open_for_loop(dry_run: bool) -> bool:
    from app.config import settings

    if not settings.live_loop_require_market_open:
        return True

    try:
        broker = build_broker()
        return broker.is_market_open()
    except Exception as exc:
        if dry_run:
            print(f"[DRY RUN] Could not verify market clock: {exc}")
        else:
            print(f"Could not verify market clock. Skipping this loop cycle. Reason: {exc}")
        return False


def _live_loop(
    universe: str,
    limit: int | None,
    show_all: bool,
    dry_run: bool,
    cycles: int | None,
    scan_interval_minutes: int | None,
    monitor_interval_minutes: int | None,
    chunk_index: int | None = None,
    chunk_count: int | None = None,
) -> None:
    from app.config import settings

    init_db()

    if not dry_run and not settings.live_loop_enabled:
        raise RuntimeError(
            "LIVE_LOOP_ENABLED=false. Set LIVE_LOOP_ENABLED=true in .env before running continuous paper trading."
        )

    scan_interval = max(1, scan_interval_minutes or settings.live_loop_model_scan_interval_minutes)
    monitor_interval = max(1, monitor_interval_minutes or settings.live_loop_position_monitor_interval_minutes)

    print()
    print("Starting continuous Alpaca paper-trading loop.")
    print(f"Universe: {universe}")
    print(f"Limit: {limit if limit else 'all'}")
    if chunk_index is not None and chunk_count is not None:
        print(f"Chunk: {chunk_index + 1}/{chunk_count}")
    print(f"Model scan interval: {scan_interval} minute(s)")
    print(f"Position monitor interval: {monitor_interval} minute(s)")
    print(f"No new trades after: {settings.live_loop_no_new_trades_after_hour:02d}:{settings.live_loop_no_new_trades_after_minute:02d}")
    print(f"Max new trades per day: {settings.live_loop_max_new_trades_per_day}")
    print(f"Dry run: {dry_run}")
    print("Press Ctrl+C to stop.")
    print()

    next_scan_at = datetime.now()
    next_monitor_at = datetime.now()
    completed_cycles = 0

    try:
        while True:
            now = datetime.now()

            if cycles is not None and completed_cycles >= cycles:
                print("Live loop stopped because requested cycle count was reached.")
                return

            market_open = _market_is_open_for_loop(dry_run=dry_run)

            if not market_open:
                print(f"{now:%Y-%m-%d %H:%M:%S} Market is closed or unavailable. Waiting...")
                time.sleep(min(60, monitor_interval * 60))
                completed_cycles += 1
                continue

            if now >= next_monitor_at:
                print()
                print(f"{now:%Y-%m-%d %H:%M:%S} Monitoring open positions...")
                monitor_positions_once(dry_run=dry_run)
                next_monitor_at = now + timedelta(minutes=monitor_interval)

            if now >= next_scan_at:
                today_count = _today_order_count()

                if _past_no_new_trades_time(now):
                    print(f"{now:%Y-%m-%d %H:%M:%S} New entries blocked by no-new-trades-after rule.")
                elif today_count >= settings.live_loop_max_new_trades_per_day:
                    print(
                        f"{now:%Y-%m-%d %H:%M:%S} New entries blocked: "
                        f"daily limit reached ({today_count}/{settings.live_loop_max_new_trades_per_day})."
                    )
                else:
                    remaining = settings.live_loop_max_new_trades_per_day - today_count
                    print()
                    print(f"{now:%Y-%m-%d %H:%M:%S} Running scheduled model scan. Remaining daily trade slots: {remaining}.")
                    _trade_screen(
                        universe,
                        limit,
                        show_all=show_all,
                        dry_run=dry_run,
                        max_trade_slots=remaining,
                        chunk_index=chunk_index,
                        chunk_count=chunk_count,
                    )

                next_scan_at = now + timedelta(minutes=scan_interval)

            completed_cycles += 1

            sleep_until = min(next_monitor_at, next_scan_at)
            sleep_seconds = max(1, min(60, int((sleep_until - datetime.now()).total_seconds())))
            time.sleep(sleep_seconds)

    except KeyboardInterrupt:
        print("Live loop stopped by user.")


def _check_github_env() -> None:
    """Print a safe environment check for local runs or GitHub Actions.

    This never prints secret values. It only reports whether required keys exist.
    """
    from app.config import settings

    checks = {
        "ALPACA_API_KEY": bool(settings.alpaca_api_key),
        "ALPACA_SECRET_KEY": bool(settings.alpaca_secret_key),
        "ALPACA_TRADING_MODE=paper": settings.alpaca_trading_mode.lower() == "paper",
        "ALLOW_LIVE_TRADING=false": settings.allow_live_trading is False,
        "AUTO_TRADE_ENABLED": settings.auto_trade_enabled,
        "LIVE_LOOP_ENABLED": settings.live_loop_enabled,
        "GEMINI_API_KEY optional/present": bool(settings.gemini_api_key),
    }

    print("Safe environment check. Secret values are not printed.")
    failed_required = False
    for name, ok in checks.items():
        label = "OK" if ok else "MISSING/OFF"
        print(f"{label:12} {name}")

    if not settings.alpaca_api_key or not settings.alpaca_secret_key:
        failed_required = True
    if settings.alpaca_trading_mode.lower() != "paper":
        failed_required = True
    if settings.allow_live_trading:
        failed_required = True

    if failed_required:
        raise SystemExit(2)

    print("Environment is ready for paper-mode commands.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Weekly multi-agent stock signal app")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db")
    sub.add_parser("check-env", help="Check local/GitHub Actions environment without printing secrets.")

    run_p = sub.add_parser("run")
    run_p.add_argument("--ticker", type=str, default=None)

    dbg_p = sub.add_parser("debug-run")
    dbg_p.add_argument("--ticker", type=str, default=None)

    screen_p = sub.add_parser("screen")
    screen_p.add_argument("--universe", choices=["top50", "custom"], default="top50")
    screen_p.add_argument("--limit", type=int, default=None)
    screen_p.add_argument(
        "--all",
        action="store_true",
        help="Print every ticker instead of only names that pass the display filter.",
    )
    screen_p.add_argument("--chunk-index", type=int, default=None, help="Zero-based chunk number for parallel scans.")
    screen_p.add_argument("--chunk-count", type=int, default=None, help="Total number of parallel chunks.")


    trade_p = sub.add_parser("trade-run", help="Run the current model for one ticker and send an eligible Alpaca paper trade.")
    trade_p.add_argument("--ticker", type=str, default=None)
    trade_p.add_argument("--dry-run", action="store_true", help="Do not send Alpaca orders; log the decision only.")

    trade_screen_p = sub.add_parser("trade-screen", help="Run the current screen and send eligible Alpaca paper trades.")
    trade_screen_p.add_argument("--universe", choices=["top50", "custom"], default="top50")
    trade_screen_p.add_argument("--limit", type=int, default=None)
    trade_screen_p.add_argument("--all", action="store_true", help="Consider every ticker instead of only display-filtered names.")
    trade_screen_p.add_argument("--dry-run", action="store_true", help="Do not send Alpaca orders; log decisions only.")
    trade_screen_p.add_argument("--chunk-index", type=int, default=None, help="Zero-based chunk number for parallel scans.")
    trade_screen_p.add_argument("--chunk-count", type=int, default=None, help="Total number of parallel chunks.")
    trade_screen_p.add_argument("--recommend-only", action="store_true", help="Save ranked recommendations and do not send Alpaca orders.")
    trade_screen_p.add_argument("--output", type=str, default=None, help="JSON output path for --recommend-only mode.")
    trade_screen_p.add_argument("--second-chance", action="store_true", help="Use stricter Tue/Wed second-chance entry filters.")

    execute_p = sub.add_parser("execute-ranked-trades", help="Load saved scan recommendations, rank all chunks together, and send eligible Alpaca paper trades.")
    execute_p.add_argument("--input-dir", type=str, required=True, help="Directory containing recommendation JSON files.")
    execute_p.add_argument("--dry-run", action="store_true", help="Do not send Alpaca orders; log decisions only.")
    execute_p.add_argument("--max-trade-slots", type=int, default=None, help="Optional cap on trades for this execution. Xanax top-5 uses 5.")
    execute_p.add_argument("--all", action="store_true", help="Consider every loaded recommendation instead of only display-filtered names.")
    execute_p.add_argument("--limit", type=int, default=None, help="Optional cap on final ranked recommendations considered.")
    execute_p.add_argument("--second-chance", action="store_true", help="Use stricter Tue/Wed second-chance entry filters.")

    monitor_p = sub.add_parser("monitor-positions", help="Monitor Alpaca positions and exit by take-profit, stop-loss, or Friday rule.")
    monitor_p.add_argument("--dry-run", action="store_true", help="Do not close positions; log exit decisions only.")

    live_p = sub.add_parser("live-loop", help="Continuously monitor positions and run scheduled model scans.")
    live_p.add_argument("--universe", choices=["top50", "custom"], default="top50")
    live_p.add_argument("--limit", type=int, default=None, help="Max tickers to scan each scheduled model pass.")
    live_p.add_argument("--all", action="store_true", help="Consider every ticker instead of only display-filtered names.")
    live_p.add_argument("--dry-run", action="store_true", help="Do not send or close Alpaca orders; log decisions only.")
    live_p.add_argument("--cycles", type=int, default=None, help="Optional test limit. Omit for continuous running.")
    live_p.add_argument("--scan-interval-minutes", type=int, default=None, help="Override LIVE_LOOP_MODEL_SCAN_INTERVAL_MINUTES.")
    live_p.add_argument("--monitor-interval-minutes", type=int, default=None, help="Override LIVE_LOOP_POSITION_MONITOR_INTERVAL_MINUTES.")
    live_p.add_argument("--chunk-index", type=int, default=None, help="Zero-based chunk number for parallel scans.")
    live_p.add_argument("--chunk-count", type=int, default=None, help="Total number of parallel chunks.")

    sim_p = sub.add_parser("simulate-weekly-strategy", help="Simulate Monday-only vs Tue/Wed second-chance strategy using price history.")
    sim_p.add_argument("--universe", choices=["top50", "custom"], default="custom")
    sim_p.add_argument("--years", type=int, default=2)
    sim_p.add_argument("--strategy", choices=["monday_only", "monday_tuesday", "monday_tuesday_wednesday", "tuned_with_realistic_long_calls", "tuned_with_long_calls", "tuned_reinvest_weekly", "stock_long_reinvest_weekly", "all"], default="all")
    sim_p.add_argument("--output-dir", type=str, default="data")

    opt_hist_p = sub.add_parser("build-options-history-dataset", help="Build options-history proxy dataset and summary CSVs for option eligibility research.")
    opt_hist_p.add_argument("--universe", choices=["top50", "custom"], default="custom")
    opt_hist_p.add_argument("--years", type=int, default=2)
    opt_hist_p.add_argument("--output-dir", type=str, default="data")

    sim_filter_p = sub.add_parser("build-simulation-live-filter", help="Build simulation_live_filter_summary.csv from Xanax no-weekend simulation trades.")
    sim_filter_p.add_argument("--input", type=str, required=True, help="Simulation trades CSV, usually xanax_no_weekend_2y_simulation_trades.csv")
    sim_filter_p.add_argument("--output", type=str, default="data/simulation_live_filter_summary.csv")
    sim_filter_p.add_argument("--config-output", type=str, default="data/simulation_live_filter_config.json")
    sim_filter_p.add_argument("--min-sample-size", type=int, default=20)
    sim_filter_p.add_argument("--max-adjustment", type=int, default=10)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "init-db":
        init_db()
        print("Database initialized.")
        return

    if args.command == "check-env":
        _check_github_env()
        return

    if args.command == "run":
        init_db()
        ticker = normalize_ticker(args.ticker or "AAPL")
        print(f"Running fresh model for {ticker}...")
        result = run_pipeline(ticker)
        print(f"Finished {ticker}.")
        _print_result(result)
        return

    if args.command == "debug-run":
        init_db()
        ticker = normalize_ticker(args.ticker or "AAPL")
        print(f"Running fresh debug model for {ticker}...")
        result = run_pipeline(ticker)
        print(f"Finished {ticker}.")
        print(json.dumps(result, indent=2, default=str))
        return


    if args.command == "trade-run":
        ticker = normalize_ticker(args.ticker or "AAPL")
        _run_and_trade_ticker(ticker, dry_run=args.dry_run)
        return

    if args.command == "trade-screen":
        _trade_screen(
            args.universe,
            args.limit,
            show_all=args.all,
            dry_run=args.dry_run,
            chunk_index=args.chunk_index,
            chunk_count=args.chunk_count,
            recommend_only=args.recommend_only,
            output=args.output,
            second_chance=args.second_chance,
        )
        return

    if args.command == "execute-ranked-trades":
        _execute_ranked_trades(
            args.input_dir,
            dry_run=args.dry_run,
            max_trade_slots=args.max_trade_slots,
            show_all=args.all,
            limit=args.limit,
            second_chance=args.second_chance,
        )
        return

    if args.command == "monitor-positions":
        monitor_positions_once(dry_run=args.dry_run)
        return

    if args.command == "live-loop":
        _live_loop(
            args.universe,
            args.limit,
            show_all=args.all,
            dry_run=args.dry_run,
            cycles=args.cycles,
            scan_interval_minutes=args.scan_interval_minutes,
            monitor_interval_minutes=args.monitor_interval_minutes,
            chunk_index=args.chunk_index,
            chunk_count=args.chunk_count,
        )
        return

    if args.command == "screen":
        _screen(
            args.universe,
            args.limit,
            show_all=args.all,
            chunk_index=args.chunk_index,
            chunk_count=args.chunk_count,
        )
        return

    if args.command == "simulate-weekly-strategy":
        from backtests.weekly_strategy_simulator import simulate_strategy

        tickers = _load_universe(args.universe)
        strategies = None if args.strategy == "all" else [args.strategy]
        summary = simulate_strategy(
            tickers,
            years=args.years,
            strategies=strategies,
            output_dir=args.output_dir,
        )
        print(json.dumps(summary, indent=2, default=str))
        return

    if args.command == "build-options-history-dataset":
        from backtests.options_history_dataset_builder import build_options_history_dataset

        tickers = _load_universe(args.universe)
        summary = build_options_history_dataset(
            tickers,
            years=args.years,
            output_dir=args.output_dir,
        )
        print(json.dumps(summary, indent=2, default=str))
        return

    if args.command == "build-simulation-live-filter":
        from backtests.simulation_live_filter_builder import build_simulation_live_filter

        summary = build_simulation_live_filter(
            input_path=args.input,
            output_path=args.output,
            config_path=args.config_output,
            min_sample_size=args.min_sample_size,
            max_adjustment=args.max_adjustment,
        )
        print(json.dumps(summary, indent=2, default=str))
        return


if __name__ == "__main__":
    main()
