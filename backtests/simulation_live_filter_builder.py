from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationLiveFilterConfig:
    min_sample_size: int = 20
    strong_win_rate_pct: float = 60.0
    weak_win_rate_pct: float = 48.0
    strong_profit_factor: float = 1.25
    weak_profit_factor: float = 0.90
    block_min_sample_size: int = 20
    max_adjustment: int = 10


def _profit_factor(pnls: pd.Series) -> float:
    gains = float(pnls[pnls > 0].sum())
    losses = abs(float(pnls[pnls < 0].sum()))
    if losses == 0:
        return 99.0 if gains > 0 else 0.0
    return gains / losses


def _adjustment(sample: int, win_rate: float, avg_pnl: float, pf: float, cfg: SimulationLiveFilterConfig) -> int:
    if sample < cfg.min_sample_size:
        return 0

    score = 0
    if win_rate >= 70:
        score += 5
    elif win_rate >= cfg.strong_win_rate_pct:
        score += 3
    elif win_rate < cfg.weak_win_rate_pct:
        score -= 4
    elif win_rate < 52:
        score -= 2

    if avg_pnl >= 1.5:
        score += 3
    elif avg_pnl > 0.25:
        score += 2
    elif avg_pnl < -0.25:
        score -= 3
    elif avg_pnl < 0:
        score -= 1

    if pf >= 2.0:
        score += 2
    elif pf >= cfg.strong_profit_factor:
        score += 1
    elif pf < cfg.weak_profit_factor:
        score -= 2

    if sample >= 75:
        score += 1 if score > 0 else 0
    elif sample < 30 and score > 0:
        score -= 1

    return int(max(-cfg.max_adjustment, min(cfg.max_adjustment, score)))


def _block_trade(sample: int, win_rate: float, avg_pnl: float, pf: float, cfg: SimulationLiveFilterConfig) -> bool:
    if sample < cfg.block_min_sample_size:
        return False
    if win_rate < cfg.weak_win_rate_pct and avg_pnl < 0:
        return True
    if avg_pnl <= -0.75 and pf < 1.0:
        return True
    return False


def _summarize_group(df: pd.DataFrame, keys: list[str], match_key: str, cfg: SimulationLiveFilterConfig) -> dict[str, Any]:
    pnl = pd.to_numeric(df["pnl_pct"], errors="coerce").fillna(0.0)
    fav = pd.to_numeric(df.get("max_favorable_pct", 0.0), errors="coerce").fillna(0.0)
    adv = pd.to_numeric(df.get("max_adverse_pct", 0.0), errors="coerce").fillna(0.0)
    sample = int(len(df))
    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())
    win_rate = (wins / sample * 100.0) if sample else 0.0
    avg_pnl = float(pnl.mean()) if sample else 0.0
    pf = _profit_factor(pnl)
    adj = _adjustment(sample, win_rate, avg_pnl, pf, cfg)
    block = _block_trade(sample, win_rate, avg_pnl, pf, cfg)

    row: dict[str, Any] = {key: str(df.iloc[0][key]).upper().strip() for key in keys}
    row.update(
        {
            "match_key": match_key,
            "sample_size": sample,
            "winning_trades": wins,
            "losing_trades": losses,
            "win_rate_pct": round(win_rate, 4),
            "avg_pnl_pct": round(avg_pnl, 4),
            "median_pnl_pct": round(float(pnl.median()) if sample else 0.0, 4),
            "avg_max_favorable_pct": round(float(fav.mean()) if sample else 0.0, 4),
            "avg_max_adverse_pct": round(float(adv.mean()) if sample else 0.0, 4),
            "profit_factor": round(float(pf), 4),
            "recommended_score_adjustment": adj,
            "block_trade": block,
        }
    )
    return row


def build_simulation_live_filter(
    input_path: str | Path,
    output_path: str | Path = "data/simulation_live_filter_summary.csv",
    config_path: str | Path | None = "data/simulation_live_filter_config.json",
    min_sample_size: int = 20,
    max_adjustment: int = 10,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    cfg = SimulationLiveFilterConfig(min_sample_size=int(min_sample_size), max_adjustment=int(max_adjustment))

    if not input_path.exists():
        raise FileNotFoundError(f"Simulation trades CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    required = {"ticker", "setup_type", "direction", "entry_day", "pnl_pct"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Simulation trades file is missing required columns: {sorted(missing)}")

    df = df.copy()
    for col in ["ticker", "setup_type", "direction", "entry_day"]:
        df[col] = df[col].fillna("UNKNOWN").astype(str).str.upper().str.strip()
    df["pnl_pct"] = pd.to_numeric(df["pnl_pct"], errors="coerce")
    df = df.dropna(subset=["pnl_pct"])

    rows: list[dict[str, Any]] = []
    group_specs = [
        (["ticker", "setup_type", "direction", "entry_day"], "ticker_setup_direction_day"),
        (["setup_type", "direction", "entry_day"], "setup_direction_day"),
        (["ticker", "setup_type", "direction"], "ticker_setup_direction_any_day"),
        (["setup_type", "direction"], "setup_direction_any_day"),
    ]

    for keys, match_key in group_specs:
        for _, group in df.groupby(keys, dropna=False):
            row = _summarize_group(group, keys, match_key, cfg)
            for fill_key in ["ticker", "setup_type", "direction", "entry_day"]:
                row.setdefault(fill_key, "ANY")
            if match_key.endswith("any_day"):
                row["entry_day"] = "ANY"
            if match_key in {"setup_direction_day", "setup_direction_any_day"}:
                row["ticker"] = "ANY"
            rows.append(row)

    out = pd.DataFrame(rows)
    column_order = [
        "match_key",
        "ticker",
        "setup_type",
        "direction",
        "entry_day",
        "sample_size",
        "winning_trades",
        "losing_trades",
        "win_rate_pct",
        "avg_pnl_pct",
        "median_pnl_pct",
        "avg_max_favorable_pct",
        "avg_max_adverse_pct",
        "profit_factor",
        "recommended_score_adjustment",
        "block_trade",
    ]
    out = out[column_order]
    out = out.sort_values(
        ["match_key", "recommended_score_adjustment", "profit_factor", "sample_size"],
        ascending=[True, False, False, False],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows": int(len(out)),
        "source_trades": int(len(df)),
        "min_sample_size": cfg.min_sample_size,
        "max_adjustment": cfg.max_adjustment,
        "positive_adjustment_rows": int((out["recommended_score_adjustment"] > 0).sum()),
        "negative_adjustment_rows": int((out["recommended_score_adjustment"] < 0).sum()),
        "block_rows": int(out["block_trade"].sum()),
    }

    if config_path:
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(summary, indent=2))

    return summary
