from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agents.agent_common import AgentSignal, clamp


@dataclass(frozen=True)
class LiquidityConfig:
    max_good_spread_pct: float = 0.10
    max_acceptable_spread_pct: float = 0.35
    min_dollar_volume: float = 25_000_000
    min_volume: float = 500_000


class LiquidityAgent:
    """Scores whether a stock is safe/clean enough to trade.

    This is not bullish or bearish. It is a trade-quality filter.

    Positive score means liquidity supports taking the trade.
    Negative score means spread/volume/slippage risk should reduce conviction or block trade.
    """

    name = "liquidity_agent"

    def __init__(self, config: LiquidityConfig | None = None) -> None:
        self.config = config or LiquidityConfig()

    def evaluate(
        self,
        ticker: str,
        bid: float | None = None,
        ask: float | None = None,
        last_price: float | None = None,
        avg_daily_volume: float | None = None,
        dollar_volume: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> AgentSignal:
        spread_pct = self._spread_pct(bid, ask)

        penalties: list[str] = []
        score = 1.0

        if spread_pct is None:
            score -= 0.25
            penalties.append("missing bid/ask spread")
        elif spread_pct <= self.config.max_good_spread_pct:
            pass
        elif spread_pct <= self.config.max_acceptable_spread_pct:
            score -= 0.35
            penalties.append(f"spread is moderate at {spread_pct:.3f}%")
        else:
            score -= 0.85
            penalties.append(f"spread is wide at {spread_pct:.3f}%")

        if dollar_volume is None and last_price is not None and avg_daily_volume is not None:
            dollar_volume = float(last_price) * float(avg_daily_volume)

        if dollar_volume is None:
            score -= 0.15
            penalties.append("missing dollar volume")
        elif dollar_volume < self.config.min_dollar_volume:
            score -= 0.45
            penalties.append(f"low dollar volume ${dollar_volume:,.0f}")

        if avg_daily_volume is None:
            score -= 0.10
            penalties.append("missing volume")
        elif avg_daily_volume < self.config.min_volume:
            score -= 0.35
            penalties.append(f"low share volume {avg_daily_volume:,.0f}")

        score = clamp(score, -1.0, 1.0)
        confidence = clamp(abs(score), 0.0, 1.0)

        if score >= 0.65:
            direction = "UP"
            risk = "LOW"
            reason = "Liquidity is acceptable for trading."
        elif score >= 0.20:
            direction = "NEUTRAL"
            risk = "MEDIUM"
            reason = "Liquidity is usable but has some execution risk."
        else:
            direction = "DOWN"
            risk = "HIGH"
            reason = "Liquidity risk is high; trade should be reduced or avoided."

        if penalties:
            reason += " Issues: " + "; ".join(penalties) + "."

        return AgentSignal(
            agent_name=self.name,
            ticker=ticker,
            score=score,
            direction=direction,
            confidence=confidence,
            risk_level=risk,
            reason=reason,
            metrics={
                "bid": bid,
                "ask": ask,
                "last_price": last_price,
                "spread_pct": spread_pct,
                "avg_daily_volume": avg_daily_volume,
                "dollar_volume": dollar_volume,
                **(extra or {}),
            },
        )

    @staticmethod
    def _spread_pct(bid: float | None, ask: float | None) -> float | None:
        if bid is None or ask is None:
            return None
        bid = float(bid)
        ask = float(ask)
        mid = (bid + ask) / 2.0
        if bid <= 0 or ask <= 0 or mid <= 0 or ask < bid:
            return None
        return ((ask - bid) / mid) * 100.0
