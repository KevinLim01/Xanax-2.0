from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Direction = Literal["UP", "DOWN", "NEUTRAL"]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]


@dataclass(frozen=True)
class AgentSignal:
    """Common output shape for plug-in agents.

    score range:
      +1.00 = strongly bullish / supportive of BUY-UP
       0.00 = neutral
      -1.00 = strongly bearish / supportive of SELL-DOWN

    confidence range:
      0.00 = weak evidence
      1.00 = strong evidence
    """

    agent_name: str
    ticker: str
    score: float
    direction: Direction
    confidence: float
    risk_level: RiskLevel = "MEDIUM"
    reason: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def direction_from_score(score: float, neutral_band: float = 0.10) -> Direction:
    if score > neutral_band:
        return "UP"
    if score < -neutral_band:
        return "DOWN"
    return "NEUTRAL"
