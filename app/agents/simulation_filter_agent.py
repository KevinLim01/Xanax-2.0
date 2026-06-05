from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.agents.agent_common import AgentSignal, clamp
from app.config import settings


DAY_NAMES = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]


@dataclass(frozen=True)
class SimulationFilterResult:
    matched: bool
    match_level: str
    sample_size: int
    win_rate_pct: float
    avg_pnl_pct: float
    avg_max_favorable_pct: float
    avg_max_adverse_pct: float
    profit_factor: float
    recommended_score_adjustment: int
    block_trade: bool
    reason: str


class SimulationFilterAgent:
    """Uses the no-weekend Xanax simulation summary as a live guardrail.

    This agent does not predict by itself. It checks whether the current
    ticker/setup/day combination behaved well in the prior two-year simulation
    and gives a small conviction adjustment or block recommendation.
    """

    name = "simulation_filter_agent"

    REQUIRED_COLUMNS = {
        "ticker",
        "setup_type",
        "direction",
        "entry_day",
        "sample_size",
        "win_rate_pct",
        "avg_pnl_pct",
        "avg_max_favorable_pct",
        "avg_max_adverse_pct",
        "profit_factor",
        "recommended_score_adjustment",
        "block_trade",
    }

    def __init__(self, summary_path: str | Path | None = None) -> None:
        self.summary_path = Path(summary_path or settings.simulation_filter_summary_path)
        if not self.summary_path.is_absolute():
            self.summary_path = Path(__file__).resolve().parents[2] / self.summary_path
        self._summary_df: pd.DataFrame | None = None

    def load(self, force_reload: bool = False) -> pd.DataFrame:
        if self._summary_df is not None and not force_reload:
            return self._summary_df

        if not self.summary_path.exists():
            self._summary_df = pd.DataFrame()
            return self._summary_df

        df = pd.read_csv(self.summary_path)
        missing = self.REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            raise ValueError(
                f"Simulation filter summary is missing columns: {sorted(missing)}. "
                "Build it with: python main.py build-simulation-live-filter --input data/xanax_no_weekend_2y_simulation_trades.csv"
            )

        for col in ["ticker", "setup_type", "direction", "entry_day", "match_key"]:
            if col in df.columns:
                df[col] = df[col].fillna("ANY").astype(str).str.upper().str.strip()

        numeric_cols = [
            "sample_size",
            "win_rate_pct",
            "avg_pnl_pct",
            "avg_max_favorable_pct",
            "avg_max_adverse_pct",
            "profit_factor",
            "recommended_score_adjustment",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["block_trade"] = df["block_trade"].astype(str).str.lower().isin({"1", "true", "yes", "y", "block"})
        df = df.dropna(subset=["sample_size", "win_rate_pct", "avg_pnl_pct"])
        df["sample_size"] = df["sample_size"].astype(int)
        self._summary_df = df
        return self._summary_df

    def evaluate(
        self,
        ticker: str,
        setup_type: str | None,
        forecast_direction: str | None,
        entry_day: str | None = None,
        conviction_score: int | float | None = None,
    ) -> AgentSignal:
        result = self.lookup(
            ticker=ticker,
            setup_type=setup_type,
            forecast_direction=forecast_direction,
            entry_day=entry_day,
            conviction_score=conviction_score,
        )

        if not result.matched:
            return AgentSignal(
                agent_name=self.name,
                ticker=ticker,
                score=0.0,
                direction="NEUTRAL",
                confidence=0.0,
                reason=result.reason,
                metrics=result.__dict__,
            )

        direction = (forecast_direction or "NEUTRAL").upper()
        signed = 1.0 if direction == "UP" else -1.0 if direction == "DOWN" else 0.0
        confidence = self._confidence(result.sample_size, result.win_rate_pct, result.profit_factor)
        strength = clamp(abs(result.recommended_score_adjustment) / max(1, settings.simulation_filter_max_adjustment), 0.0, 1.0)
        score = signed * strength
        if result.recommended_score_adjustment < 0 or result.block_trade:
            score *= -1.0

        return AgentSignal(
            agent_name=self.name,
            ticker=ticker,
            score=clamp(score, -1.0, 1.0),
            direction="UP" if score > 0.10 else "DOWN" if score < -0.10 else "NEUTRAL",
            confidence=confidence,
            risk_level="HIGH" if result.block_trade else "LOW" if confidence >= 0.65 else "MEDIUM",
            reason=result.reason,
            metrics=result.__dict__,
        )

    def lookup(
        self,
        ticker: str,
        setup_type: str | None,
        forecast_direction: str | None,
        entry_day: str | None = None,
        conviction_score: int | float | None = None,
    ) -> SimulationFilterResult:
        df = self.load()
        if df.empty:
            return self._no_match("simulation filter summary not found or empty")

        ticker = self._norm(ticker)
        setup = self._norm(setup_type)
        direction = self._norm(forecast_direction)
        day = self._norm(entry_day) or DAY_NAMES[datetime.now().weekday()]
        conviction = self._safe_int(conviction_score, 0)

        if not ticker or not setup or direction not in {"UP", "DOWN"}:
            return self._no_match("missing ticker/setup/direction for simulation filter")

        candidates = [
            ("ticker_setup_direction_day", (df["ticker"] == ticker) & (df["setup_type"] == setup) & (df["direction"] == direction) & (df["entry_day"] == day)),
            ("setup_direction_day", (df["ticker"] == "ANY") & (df["setup_type"] == setup) & (df["direction"] == direction) & (df["entry_day"] == day)),
            ("ticker_setup_direction_any_day", (df["ticker"] == ticker) & (df["setup_type"] == setup) & (df["direction"] == direction) & (df["entry_day"] == "ANY")),
            ("setup_direction_any_day", (df["ticker"] == "ANY") & (df["setup_type"] == setup) & (df["direction"] == direction) & (df["entry_day"] == "ANY")),
        ]

        for level, mask in candidates:
            hit = df.loc[mask].copy()
            if hit.empty:
                continue
            hit = hit.sort_values(["sample_size", "profit_factor", "win_rate_pct"], ascending=[False, False, False])
            row = hit.iloc[0]
            sample = int(row["sample_size"])
            if sample < settings.simulation_filter_min_sample_size:
                continue
            return self._row_to_result(row, level, conviction)

        return self._no_match(f"no simulation match for {ticker}/{setup}/{direction}/{day}")

    def _row_to_result(self, row: pd.Series, level: str, conviction: int) -> SimulationFilterResult:
        sample = int(row["sample_size"])
        win_rate = self._safe_float(row.get("win_rate_pct"), 0.0)
        avg_pnl = self._safe_float(row.get("avg_pnl_pct"), 0.0)
        max_fav = self._safe_float(row.get("avg_max_favorable_pct"), 0.0)
        max_adv = self._safe_float(row.get("avg_max_adverse_pct"), 0.0)
        pf = self._safe_float(row.get("profit_factor"), 0.0)
        adjustment = max(-settings.simulation_filter_max_adjustment, min(settings.simulation_filter_max_adjustment, int(round(self._safe_float(row.get("recommended_score_adjustment"), 0.0)))))
        block = bool(row.get("block_trade", False))

        if conviction >= settings.simulation_filter_high_conviction_override:
            block = False
            if adjustment < 0:
                adjustment = max(adjustment, -3)

        reason = (
            f"simulation {level}: sample={sample}, win={win_rate:.1f}%, avg={avg_pnl:+.2f}%, "
            f"pf={pf:.2f}, adj={adjustment:+d}, block={block}"
        )

        return SimulationFilterResult(
            matched=True,
            match_level=level,
            sample_size=sample,
            win_rate_pct=win_rate,
            avg_pnl_pct=avg_pnl,
            avg_max_favorable_pct=max_fav,
            avg_max_adverse_pct=max_adv,
            profit_factor=pf,
            recommended_score_adjustment=adjustment,
            block_trade=block,
            reason=reason,
        )

    @staticmethod
    def _confidence(sample_size: int, win_rate_pct: float, profit_factor: float) -> float:
        sample_component = min(1.0, sample_size / 80.0)
        win_component = min(1.0, max(0.0, (win_rate_pct - 50.0) / 25.0))
        pf_component = min(1.0, max(0.0, (profit_factor - 1.0) / 1.0)) if profit_factor > 0 else 0.0
        return clamp(0.45 * sample_component + 0.35 * win_component + 0.20 * pf_component, 0.0, 1.0)

    @staticmethod
    def _norm(value: Any) -> str:
        return str(value or "").upper().strip()

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
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
    def _no_match(reason: str) -> SimulationFilterResult:
        return SimulationFilterResult(
            matched=False,
            match_level="none",
            sample_size=0,
            win_rate_pct=0.0,
            avg_pnl_pct=0.0,
            avg_max_favorable_pct=0.0,
            avg_max_adverse_pct=0.0,
            profit_factor=0.0,
            recommended_score_adjustment=0,
            block_trade=False,
            reason=reason,
        )
