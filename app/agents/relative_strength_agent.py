from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from app.agents.agent_common import AgentSignal, clamp, direction_from_score


DEFAULT_SECTOR_ETFS: dict[str, str] = {
    # Broad market / fallback
    "DEFAULT": "SPY",
    "TECH": "QQQ",
    "SEMIS": "SMH",
    "FINANCIALS": "XLF",
    "ENERGY": "XLE",
    "INDUSTRIALS": "XLI",
    "HEALTHCARE": "XLV",
    "CONSUMER_DISCRETIONARY": "XLY",
    "CONSUMER_STAPLES": "XLP",
    "UTILITIES": "XLU",
    "REAL_ESTATE": "XLRE",
    "MATERIALS": "XLB",
    "COMMUNICATIONS": "XLC",
    "SMALL_CAP": "IWM",
}


@dataclass(frozen=True)
class RelativeStrengthConfig:
    lookback_bars: int = 78  # roughly one trading day with 5-min bars
    strong_threshold_pct: float = 1.00
    weak_threshold_pct: float = 0.25
    min_bars: int = 10


class RelativeStrengthAgent:
    """Compares a stock against SPY/QQQ/sector ETF.

    Use this to avoid confusing market-wide movement with stock-specific strength.

    Expected input:
      prices[ticker] = DataFrame with a 'close' column, indexed by time or ordered rows.
      prices[benchmark] = same shape.
    """

    name = "relative_strength_agent"

    def __init__(
        self,
        config: RelativeStrengthConfig | None = None,
        sector_etfs: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config or RelativeStrengthConfig()
        self.sector_etfs = dict(sector_etfs or DEFAULT_SECTOR_ETFS)

    def evaluate(
        self,
        ticker: str,
        prices: Mapping[str, pd.DataFrame],
        sector_key: str | None = None,
        benchmark: str | None = None,
    ) -> AgentSignal:
        benchmark = benchmark or self._benchmark_for_sector(sector_key)

        stock_df = prices.get(ticker)
        bench_df = prices.get(benchmark)

        if stock_df is None or bench_df is None:
            return AgentSignal(
                agent_name=self.name,
                ticker=ticker,
                score=0.0,
                direction="NEUTRAL",
                confidence=0.0,
                reason=f"Missing price data for {ticker} or benchmark {benchmark}.",
                metrics={"benchmark": benchmark},
            )

        stock_ret = self._return_pct(stock_df)
        bench_ret = self._return_pct(bench_df)

        if stock_ret is None or bench_ret is None:
            return AgentSignal(
                agent_name=self.name,
                ticker=ticker,
                score=0.0,
                direction="NEUTRAL",
                confidence=0.0,
                reason="Not enough bars to calculate relative strength.",
                metrics={"benchmark": benchmark},
            )

        relative_strength_pct = stock_ret - bench_ret

        # Convert relative strength to -1..+1 score.
        score = clamp(relative_strength_pct / self.config.strong_threshold_pct, -1.0, 1.0)
        confidence = clamp(abs(relative_strength_pct) / self.config.strong_threshold_pct, 0.0, 1.0)

        if abs(relative_strength_pct) < self.config.weak_threshold_pct:
            score *= 0.35
            confidence *= 0.50

        direction = direction_from_score(score)
        reason = (
            f"{ticker} returned {stock_ret:.2f}% vs {benchmark} {bench_ret:.2f}% "
            f"over the lookback; relative strength is {relative_strength_pct:.2f}%."
        )

        return AgentSignal(
            agent_name=self.name,
            ticker=ticker,
            score=score,
            direction=direction,
            confidence=confidence,
            risk_level="LOW" if confidence >= 0.65 else "MEDIUM",
            reason=reason,
            metrics={
                "benchmark": benchmark,
                "stock_return_pct": stock_ret,
                "benchmark_return_pct": bench_ret,
                "relative_strength_pct": relative_strength_pct,
                "lookback_bars": self.config.lookback_bars,
            },
        )

    def _benchmark_for_sector(self, sector_key: str | None) -> str:
        if not sector_key:
            return self.sector_etfs["DEFAULT"]
        return self.sector_etfs.get(sector_key.upper(), self.sector_etfs["DEFAULT"])

    def _return_pct(self, df: pd.DataFrame) -> float | None:
        if "close" not in df.columns:
            return None
        clean = df["close"].dropna().astype(float)
        if len(clean) < self.config.min_bars:
            return None
        window = clean.tail(self.config.lookback_bars)
        start = float(window.iloc[0])
        end = float(window.iloc[-1])
        if start <= 0:
            return None
        return ((end - start) / start) * 100.0
