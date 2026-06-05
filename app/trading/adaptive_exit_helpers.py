from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptiveExitTarget:
    take_profit_pct: float
    stop_loss_pct: float
    reason: str


def adaptive_take_profit_pct(
    expected_weekly_move_pct: float | None,
    default_take_profit_pct: float = 3.0,
    floor_pct: float = 1.25,
    capture_ratio: float = 0.70,
) -> float:
    """Volatility-based take profit target.

    This is better than lowering profit targets just because a stock has a high share price.
    """

    if expected_weekly_move_pct is None or expected_weekly_move_pct <= 0:
        return float(default_take_profit_pct)

    adaptive = float(expected_weekly_move_pct) * float(capture_ratio)
    return max(float(floor_pct), min(float(default_take_profit_pct), adaptive))


def build_adaptive_exit_target(
    expected_weekly_move_pct: float | None,
    default_take_profit_pct: float = 3.0,
    default_stop_loss_pct: float = 2.0,
) -> AdaptiveExitTarget:
    take_profit = adaptive_take_profit_pct(
        expected_weekly_move_pct=expected_weekly_move_pct,
        default_take_profit_pct=default_take_profit_pct,
    )

    reason = (
        f"Adaptive take profit set to {take_profit:.2f}% based on expected weekly move "
        f"{expected_weekly_move_pct if expected_weekly_move_pct is not None else 'unknown'}%."
    )

    return AdaptiveExitTarget(
        take_profit_pct=take_profit,
        stop_loss_pct=float(default_stop_loss_pct),
        reason=reason,
    )
