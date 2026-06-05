from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.agents.agent_common import AgentSignal, clamp


@dataclass(frozen=True)
class HistoryLookupConfig:
    """Config for the saved-history lookup agent.

    This agent does NOT run backtests. It only reads the summary created by:
      python main.py build-history-summary --universe custom
    """

    summary_path: str = "data/history_setup_summary.csv"
    min_sample_size: int = 50
    strong_sample_size: int = 250
    neutral_success_rate: float = 55.0
    strong_success_rate: float = 65.0
    max_conviction_delta: int = 10


@dataclass(frozen=True)
class HistoryLookupResult:
    ticker: str
    matched: bool
    match_level: str
    setup_type: str | None
    forecast_direction: str | None
    primary_regime: str | None
    ticker_archetype: str | None
    sample_size: int
    true_during_week_rate: float | None
    average_best_correct_return_pct: float | None
    average_adverse_move_pct: float | None
    recommended_score_adjustment: int
    reason: str


class HistoryLookupAgent:
    """Looks up how similar historical setups performed.

    Use this inside the live scan after the current model has produced a setup_type,
    direction, regime, and ticker_archetype.

    It is intentionally lightweight:
      - no yfinance
      - no news
      - no LLM
      - no Alpaca
      - no simulation
    """

    name = "history_lookup_agent"

    REQUIRED_COLUMNS = {
        "setup_type",
        "forecast_direction",
        "primary_regime",
        "ticker_archetype",
        "sample_size",
        "true_during_week_rate",
        "average_best_correct_return_pct",
        "average_adverse_move_pct",
        "recommended_score_adjustment",
    }

    def __init__(self, config: HistoryLookupConfig | None = None) -> None:
        self.config = config or HistoryLookupConfig()
        self._summary_df: pd.DataFrame | None = None

    def load(self, force_reload: bool = False) -> pd.DataFrame:
        if self._summary_df is not None and not force_reload:
            return self._summary_df

        path = Path(self.config.summary_path)
        if not path.exists():
            self._summary_df = pd.DataFrame()
            return self._summary_df

        df = pd.read_csv(path)
        missing = self.REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            raise ValueError(
                f"History summary is missing required columns: {sorted(missing)}. "
                f"Rebuild it with: python main.py build-history-summary --universe custom"
            )

        for col in ["setup_type", "forecast_direction", "primary_regime", "ticker_archetype"]:
            df[col] = df[col].fillna("UNKNOWN").astype(str).str.upper().str.strip()

        numeric_cols = [
            "sample_size",
            "true_during_week_rate",
            "average_best_correct_return_pct",
            "average_adverse_move_pct",
            "recommended_score_adjustment",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["sample_size", "true_during_week_rate"])
        df["sample_size"] = df["sample_size"].astype(int)
        self._summary_df = df
        return self._summary_df

    def evaluate(
        self,
        ticker: str,
        setup_type: str | None,
        forecast_direction: str | None,
        primary_regime: str | None = "NORMAL",
        ticker_archetype: str | None = "NORMAL",
    ) -> AgentSignal:
        result = self.lookup(
            ticker=ticker,
            setup_type=setup_type,
            forecast_direction=forecast_direction,
            primary_regime=primary_regime,
            ticker_archetype=ticker_archetype,
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
        confidence = self._confidence(result.sample_size, result.true_during_week_rate)
        score_strength = clamp(abs(result.recommended_score_adjustment) / max(1, self.config.max_conviction_delta), 0.0, 1.0)
        score = signed * score_strength

        # If the historical adjustment is negative, oppose the proposed direction.
        if result.recommended_score_adjustment < 0:
            score *= -1.0

        return AgentSignal(
            agent_name=self.name,
            ticker=ticker,
            score=clamp(score, -1.0, 1.0),
            direction="UP" if score > 0.10 else "DOWN" if score < -0.10 else "NEUTRAL",
            confidence=confidence,
            risk_level="LOW" if confidence >= 0.65 else "MEDIUM",
            reason=result.reason,
            metrics=result.__dict__,
        )

    def lookup(
        self,
        ticker: str,
        setup_type: str | None,
        forecast_direction: str | None,
        primary_regime: str | None = "NORMAL",
        ticker_archetype: str | None = "NORMAL",
    ) -> HistoryLookupResult:
        df = self.load()
        if df.empty:
            return self._no_match(ticker, "History summary file was not found or is empty.")

        setup = self._norm(setup_type)
        direction = self._norm(forecast_direction)
        regime = self._norm(primary_regime or "NORMAL")
        archetype = self._norm(ticker_archetype or "NORMAL")

        if not setup or direction not in {"UP", "DOWN"}:
            return self._no_match(ticker, "History lookup skipped because setup or direction is missing.")

        candidates = [
            (
                "exact_setup_direction_regime_archetype",
                (df["setup_type"] == setup)
                & (df["forecast_direction"] == direction)
                & (df["primary_regime"] == regime)
                & (df["ticker_archetype"] == archetype),
            ),
            (
                "setup_direction_regime_normal_archetype",
                (df["setup_type"] == setup)
                & (df["forecast_direction"] == direction)
                & (df["primary_regime"] == regime)
                & (df["ticker_archetype"] == "NORMAL"),
            ),
            (
                "setup_direction_regime_any_archetype",
                (df["setup_type"] == setup)
                & (df["forecast_direction"] == direction)
                & (df["primary_regime"] == regime),
            ),
            (
                "setup_direction_any_regime_any_archetype",
                (df["setup_type"] == setup) & (df["forecast_direction"] == direction),
            ),
        ]

        for level, mask in candidates:
            hit = df.loc[mask].copy()
            if hit.empty:
                continue
            row = self._pick_best_row(hit, prefer_archetype=archetype)
            if int(row["sample_size"]) < self.config.min_sample_size:
                continue
            return self._row_to_result(ticker, row, level)

        return self._no_match(
            ticker,
            f"No reliable history match for {setup}/{direction}/{regime}/{archetype} with sample size >= {self.config.min_sample_size}.",
        )

    def _pick_best_row(self, rows: pd.DataFrame, prefer_archetype: str) -> pd.Series:
        rows = rows.copy()
        rows["_preferred"] = (rows["ticker_archetype"] == prefer_archetype).astype(int)
        rows = rows.sort_values(
            by=["_preferred", "sample_size", "true_during_week_rate"],
            ascending=[False, False, False],
        )
        return rows.iloc[0]

    def _row_to_result(self, ticker: str, row: pd.Series, match_level: str) -> HistoryLookupResult:
        sample_size = int(row["sample_size"])
        success_rate = float(row["true_during_week_rate"])
        adjustment = int(row["recommended_score_adjustment"])
        setup = str(row["setup_type"])
        direction = str(row["forecast_direction"])
        regime = str(row["primary_regime"])
        archetype = str(row["ticker_archetype"])
        best = self._maybe_float(row.get("average_best_correct_return_pct"))
        adverse = self._maybe_float(row.get("average_adverse_move_pct"))

        if adjustment > 0:
            adj_text = f"adds +{adjustment} conviction"
        elif adjustment < 0:
            adj_text = f"subtracts {abs(adjustment)} conviction"
        else:
            adj_text = "adds no conviction"

        reason = (
            f"History match {match_level}: {setup}/{direction}/{regime}/{archetype} had "
            f"{sample_size:,} past cases and a {success_rate:.2f}% true-during-week rate; {adj_text}."
        )

        return HistoryLookupResult(
            ticker=ticker,
            matched=True,
            match_level=match_level,
            setup_type=setup,
            forecast_direction=direction,
            primary_regime=regime,
            ticker_archetype=archetype,
            sample_size=sample_size,
            true_during_week_rate=success_rate,
            average_best_correct_return_pct=best,
            average_adverse_move_pct=adverse,
            recommended_score_adjustment=adjustment,
            reason=reason,
        )

    def _confidence(self, sample_size: int, success_rate: float | None) -> float:
        if success_rate is None:
            return 0.0
        sample_conf = clamp(sample_size / max(1, self.config.strong_sample_size), 0.0, 1.0)
        edge_conf = clamp(abs(float(success_rate) - self.config.neutral_success_rate) / 20.0, 0.0, 1.0)
        return clamp((0.65 * sample_conf) + (0.35 * edge_conf), 0.0, 1.0)

    def _no_match(self, ticker: str, reason: str) -> HistoryLookupResult:
        return HistoryLookupResult(
            ticker=ticker,
            matched=False,
            match_level="none",
            setup_type=None,
            forecast_direction=None,
            primary_regime=None,
            ticker_archetype=None,
            sample_size=0,
            true_during_week_rate=None,
            average_best_correct_return_pct=None,
            average_adverse_move_pct=None,
            recommended_score_adjustment=0,
            reason=reason,
        )

    @staticmethod
    def _norm(value: str | None) -> str:
        return str(value or "").upper().strip()

    @staticmethod
    def _maybe_float(value: Any) -> float | None:
        try:
            if pd.isna(value):
                return None
            return float(value)
        except Exception:
            return None
