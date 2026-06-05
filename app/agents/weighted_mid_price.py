from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WeightedMidPrice:
    bid: float
    ask: float
    bid_size: float | None = None
    ask_size: float | None = None
    mid: float | None = None
    weighted_mid: float | None = None
    imbalance: float | None = None
    spread_pct: float | None = None


def calculate_weighted_mid(
    bid: float | None,
    ask: float | None,
    bid_size: float | None = None,
    ask_size: float | None = None,
) -> WeightedMidPrice | None:
    """Order-book-aware fair price helper.

    If sizes are available, this leans price toward the side with weaker liquidity.
    If sizes are missing, it returns the normal mid-price.
    """

    if bid is None or ask is None:
        return None

    bid = float(bid)
    ask = float(ask)
    if bid <= 0 or ask <= 0 or ask < bid:
        return None

    mid = (bid + ask) / 2.0
    spread_pct = ((ask - bid) / mid) * 100.0 if mid > 0 else None

    imbalance = None
    weighted_mid = mid

    if bid_size is not None and ask_size is not None:
        bid_size = float(bid_size)
        ask_size = float(ask_size)
        total = bid_size + ask_size
        if total > 0:
            imbalance = bid_size / total
            weighted_mid = (ask * imbalance) + (bid * (1.0 - imbalance))

    return WeightedMidPrice(
        bid=bid,
        ask=ask,
        bid_size=bid_size,
        ask_size=ask_size,
        mid=mid,
        weighted_mid=weighted_mid,
        imbalance=imbalance,
        spread_pct=spread_pct,
    )
