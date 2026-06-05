from __future__ import annotations

import math
from typing import Any

import requests

from app.llm_client import GeminiJSONClient
from app.models import ModifierSignal

GAMMA_URL = "https://gamma-api.polymarket.com/markets"

LLM_SYSTEM = """
You evaluate whether a Polymarket question is genuinely relevant to a stock's weekly direction. Return JSON with keys: ticker_relevance (0-1), sector_relevance (0-1), weekly_direction_bias (-1 to 1), explanation.
"""


def fetch_polymarket_modifier(ticker: str) -> ModifierSignal:
    params = {"active": "true", "closed": "false", "limit": 80}
    try:
        response = requests.get(GAMMA_URL, params=params, timeout=12)
        response.raise_for_status()
        markets = response.json()
    except Exception:
        return ModifierSignal(name="polymarket", bias=0.0, validity=0.0, details=["Polymarket fetch failed"], metadata={"count": 0})

    matches: list[dict[str, Any]] = []
    ticker_lower = ticker.lower()
    llm = GeminiJSONClient()
    for market in markets:
        question = str(market.get("question", "") or "").strip()
        if not question:
            continue
        lower = question.lower()
        match_quality = 0.0
        if ticker_lower in lower:
            match_quality += 1.0
        elif any(token in lower for token in [ticker_lower.replace(".", ""), "stock", "shares", "earnings", "guidance"]):
            match_quality += 0.4
        llm_out = None
        if match_quality > 0 or any(k in lower for k in ["semiconductor", "ai", "bank", "oil", "healthcare"]):
            llm_out = llm.classify(LLM_SYSTEM, {"ticker": ticker, "question": question})
            if llm_out:
                match_quality += 0.6 * float(llm_out.get("ticker_relevance", 0.0)) + 0.35 * float(llm_out.get("sector_relevance", 0.0))

        if match_quality <= 0.35:
            continue

        try:
            volume = float(market.get("volume", 0) or 0)
            one_day = float(market.get("oneDayPriceChange", 0) or 0)
        except Exception:
            volume, one_day = 0.0, 0.0

        probs = None
        try:
            outcomes = market.get("outcomePrices")
            if isinstance(outcomes, list) and len(outcomes) >= 2:
                probs = [float(outcomes[0]), float(outcomes[1])]
            elif isinstance(outcomes, str):
                cleaned = outcomes.strip("[]")
                bits = [float(x) for x in cleaned.split(",")[:2]]
                if len(bits) == 2:
                    probs = bits
        except Exception:
            probs = None

        matches.append({
            "question": question,
            "match_quality": match_quality,
            "volume": volume,
            "one_day": one_day,
            "probs": probs,
            "llm": llm_out or {},
        })

    if not matches:
        return ModifierSignal(name="polymarket", bias=0.0, validity=0.0, details=["No relevant Polymarket markets"], metadata={"count": 0})

    matches.sort(key=lambda m: (m["match_quality"], m["volume"]), reverse=True)
    top = matches[:3]
    best = top[0]
    probs = best.get("probs")
    if probs and len(probs) == 2:
        p_up, p_down = float(probs[0]), float(probs[1])
        tilt = p_up - p_down
    else:
        p_up = p_down = 0.5
        tilt = 0.0

    llm_bias = float(best.get("llm", {}).get("weekly_direction_bias", 0.0))
    tilt = 0.65 * tilt + 0.35 * llm_bias
    if abs(tilt) < 0.07:
        tilt = 0.0

    avg_quality = sum(m["match_quality"] for m in top) / len(top)
    avg_volume = sum(min(1.0, math.log10(max(1.0, m["volume"] + 1)) / 4.0) for m in top) / len(top)
    validity = min(1.0, 0.10 + 0.50 * avg_quality + 0.40 * avg_volume)
    details = [m["question"] for m in top]
    return ModifierSignal(
        name="polymarket",
        bias=round(max(-1.0, min(1.0, tilt)), 4),
        validity=round(validity, 4),
        details=details,
        metadata={"count": len(matches), "top": top},
    )
