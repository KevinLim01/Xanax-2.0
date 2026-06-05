from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.agents.agent_common import AgentSignal, clamp, direction_from_score


@dataclass(frozen=True)
class BlendedDecision:
    ticker: str
    base_score: float
    adjusted_score: float
    adjusted_direction: str
    conviction_delta: int
    reasons: list[str]
    agent_metrics: dict[str, dict]


DEFAULT_AGENT_WEIGHTS: dict[str, float] = {
    "relative_strength_agent": 0.35,
    "intraday_confirmation_agent": 0.30,
    "liquidity_agent": 0.20,
}


class FinalScoreBlender:
    """Small adapter to blend the new agents into your existing final score.

    base_score should be your current final raw score in the -1..+1 range.
    This returns an adjusted score and a conviction_delta you can add/subtract.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = dict(weights or DEFAULT_AGENT_WEIGHTS)

    def blend(self, ticker: str, base_score: float, signals: Iterable[AgentSignal]) -> BlendedDecision:
        base_score = clamp(float(base_score), -1.0, 1.0)
        adjustment = 0.0
        reasons: list[str] = []
        agent_metrics: dict[str, dict] = {}

        for signal in signals:
            weight = self.weights.get(signal.agent_name, 0.10)

            # Liquidity is a trade-quality gate, not a directional predictor.
            # Poor liquidity reduces confidence by pulling the score toward zero.
            if signal.agent_name == "liquidity_agent" and signal.score < 0.20:
                adjustment -= abs(base_score) * min(0.35, weight + 0.10)
            else:
                adjustment += signal.score * signal.confidence * weight

            if signal.reason:
                reasons.append(f"{signal.agent_name}: {signal.reason}")
            agent_metrics[signal.agent_name] = signal.metrics

        adjusted = clamp(base_score + adjustment, -1.0, 1.0)
        conviction_delta = int(round((abs(adjusted) - abs(base_score)) * 20))

        return BlendedDecision(
            ticker=ticker,
            base_score=base_score,
            adjusted_score=adjusted,
            adjusted_direction=direction_from_score(adjusted),
            conviction_delta=conviction_delta,
            reasons=reasons,
            agent_metrics=agent_metrics,
        )
