from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from app.agents.agent_common import AgentSignal, clamp, direction_from_score

TradeBias = Literal["BUY", "SELL", "WATCH", "UP", "DOWN", "NEUTRAL"]


@dataclass(frozen=True)
class IntradayConfirmationConfig:
    lookback_bars: int = 12  # 12 x 5-minute bars = 60 minutes
    min_bars: int = 20
    momentum_confirm_pct: float = 0.50
    vwap_band_pct: float = 0.10


class IntradayConfirmationAgent:
    """Checks whether the live intraday tape agrees with the model's trade direction.

    Expected DataFrame columns:
      close is required.
      high, low, volume are recommended for VWAP.
    """

    name = "intraday_confirmation_agent"

    def __init__(self, config: IntradayConfirmationConfig | None = None) -> None:
        self.config = config or IntradayConfirmationConfig()

    def evaluate(self, ticker: str, bars: pd.DataFrame, intended_bias: TradeBias = "NEUTRAL") -> AgentSignal:
        if bars is None or bars.empty or "close" not in bars.columns:
            return AgentSignal(
                agent_name=self.name,
                ticker=ticker,
                score=0.0,
                direction="NEUTRAL",
                confidence=0.0,
                reason="Missing intraday bars.",
            )

        clean = bars.dropna(subset=["close"]).copy()
        if len(clean) < self.config.min_bars:
            return AgentSignal(
                agent_name=self.name,
                ticker=ticker,
                score=0.0,
                direction="NEUTRAL",
                confidence=0.0,
                reason="Not enough intraday bars for confirmation.",
                metrics={"bar_count": len(clean)},
            )

        close = clean["close"].astype(float)
        last_price = float(close.iloc[-1])
        lookback = close.tail(self.config.lookback_bars)
        start = float(lookback.iloc[0])
        momentum_pct = ((last_price - start) / start) * 100.0 if start > 0 else 0.0

        vwap = self._vwap(clean)
        vwap_distance_pct = ((last_price - vwap) / vwap) * 100.0 if vwap and vwap > 0 else 0.0

        momentum_score = clamp(momentum_pct / self.config.momentum_confirm_pct, -1.0, 1.0)
        vwap_score = clamp(vwap_distance_pct / self.config.vwap_band_pct, -1.0, 1.0)
        raw_score = (0.60 * momentum_score) + (0.40 * vwap_score)
        score = clamp(raw_score, -1.0, 1.0)

        direction = direction_from_score(score)
        confidence = clamp((abs(momentum_score) + abs(vwap_score)) / 2.0, 0.0, 1.0)

        bias = intended_bias.upper()
        confirms_bias = (
            (bias in {"BUY", "UP"} and score > 0.10)
            or (bias in {"SELL", "DOWN"} and score < -0.10)
            or (bias in {"WATCH", "NEUTRAL"})
        )

        reason = (
            f"Intraday momentum is {momentum_pct:.2f}% over the lookback and price is "
            f"{vwap_distance_pct:.2f}% from VWAP."
        )
        if bias not in {"WATCH", "NEUTRAL"}:
            reason += " Confirms intended trade." if confirms_bias else " Conflicts with intended trade."

        return AgentSignal(
            agent_name=self.name,
            ticker=ticker,
            score=score,
            direction=direction,
            confidence=confidence,
            risk_level="LOW" if confirms_bias and confidence >= 0.50 else "MEDIUM",
            reason=reason,
            metrics={
                "intended_bias": intended_bias,
                "last_price": last_price,
                "momentum_pct": momentum_pct,
                "vwap": vwap,
                "vwap_distance_pct": vwap_distance_pct,
                "confirms_bias": confirms_bias,
                "lookback_bars": self.config.lookback_bars,
            },
        )

    @staticmethod
    def _vwap(df: pd.DataFrame) -> float | None:
        if "volume" not in df.columns:
            return None
        volume = df["volume"].fillna(0).astype(float)
        if volume.sum() <= 0:
            return None

        if {"high", "low", "close"}.issubset(df.columns):
            typical = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0
        else:
            typical = df["close"].astype(float)

        return float((typical * volume).sum() / volume.sum())
