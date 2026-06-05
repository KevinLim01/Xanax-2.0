from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import feedparser

from app.domain_maps import peer_relevance
from app.llm_client import GeminiJSONClient
from app.models import ModifierSignal

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"


@dataclass
class PeerEvent:
    peer: str
    sector: str
    headline: str
    source: str
    published_at: str
    sentiment: float
    relevance: float
    reason: str


SECTOR_PEERS: dict[str, list[str]] = {
    "semis": ["AMD", "NVDA", "AVGO", "QCOM", "TXN", "AMAT", "INTC", "MU", "TSM"],
    "mega_tech": ["AAPL", "MSFT", "AMZN", "META", "GOOG", "GOOGL", "ORCL", "NFLX"],
    "banks": ["JPM", "GS", "BAC", "WFC", "MS", "C"],
    "energy": ["XOM", "CVX", "COP", "SLB"],
    "healthcare": ["LLY", "JNJ", "ABBV", "MRK", "UNH", "ABT", "TMO"],
    "consumer_def": ["KO", "PEP", "PM", "MCD", "WMT", "COST"],
    "payments": ["V", "MA", "AXP", "PYPL"],
    "software": ["ADBE", "CRM", "NOW", "INTU", "ACN", "IBM"],
    "industrial": ["CAT", "GE", "LIN", "HON"],
}

TICKER_TO_SECTOR: dict[str, str] = {}
for sector_name, tickers in SECTOR_PEERS.items():
    for t in tickers:
        TICKER_TO_SECTOR[t] = sector_name

HIGH_QUALITY = {
    "Reuters",
    "Bloomberg",
    "CNBC",
    "Yahoo Finance",
    "Barron's",
    "The Wall Street Journal",
    "Financial Times",
    "Investor's Business Daily",
    "MarketWatch",
}

LOW_QUALITY_SOURCE_SUBSTRINGS = ["marketbeat", "invezz", "simplywall", "aastocks", "mexc", "tradingkey"]

POSITIVE_TERMS = [
    "beats", "strong demand", "raises guidance", "higher outlook",
    "upbeat forecast", "record", "surges", "jumps", "rebound",
]
NEGATIVE_TERMS = [
    "misses", "cuts guidance", "lower outlook", "warning",
    "falls", "slides", "drops", "weak demand",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(entry) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        parsed = entry.get(key)
        if parsed:
            return datetime(*parsed[:6], tzinfo=timezone.utc)
    return None


def _source_quality(source: str) -> float:
    if source in HIGH_QUALITY:
        return 1.0
    low = source.lower()
    if any(token in low for token in LOW_QUALITY_SOURCE_SUBSTRINGS):
        return 0.10
    return 0.45


def _headline_sentiment(text: str) -> float:
    lower = text.lower()
    score = 0.0
    for term in POSITIVE_TERMS:
        if term in lower:
            score += 0.30
    for term in NEGATIVE_TERMS:
        if term in lower:
            score -= 0.30
    return max(-1.0, min(1.0, score))


def _fetch_peer_events(target_ticker: str, sector: str, lookback_days: int = 4) -> list[PeerEvent]:
    peers = [p for p in SECTOR_PEERS.get(sector, []) if p != target_ticker]
    if not peers:
        return []

    cutoff = _utc_now() - timedelta(days=lookback_days)
    events: list[PeerEvent] = []

    for peer in peers[:6]:
        query = f"{peer} earnings OR {peer} guidance OR {peer} demand OR {peer} forecast"
        url = f"{GOOGLE_NEWS_RSS}?q={query.replace(' ', '%20')}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)

        for entry in feed.entries[:5]:
            dt = _parse_dt(entry)
            if dt is None or dt < cutoff:
                continue

            headline = str(entry.get("title", "")).strip()
            if not headline:
                continue

            source = "Unknown"
            source_obj = entry.get("source")
            if isinstance(source_obj, dict):
                source = str(source_obj.get("title", "Unknown")).strip() or "Unknown"

            src_q = _source_quality(source)
            sentiment = _headline_sentiment(headline)
            relevance = 0.15 + 0.25 * src_q + 0.55 * peer_relevance(target_ticker, peer, headline)

            low = headline.lower()
            if any(k in low for k in ["earnings", "guidance", "forecast", "demand"]):
                relevance += 0.18

            if abs(sentiment) < 0.12 or relevance < 0.45:
                continue

            reason = "Sector read-through"
            if "guidance" in low or "forecast" in low:
                reason = "Guidance read-through"
            elif "demand" in low:
                reason = "Demand read-through"
            elif "earnings" in low:
                reason = "Earnings read-through"

            events.append(
                PeerEvent(
                    peer=peer,
                    sector=sector,
                    headline=headline,
                    source=source,
                    published_at=dt.isoformat(),
                    sentiment=sentiment,
                    relevance=min(1.0, relevance),
                    reason=reason,
                )
            )

    events.sort(key=lambda x: (x.relevance, x.published_at), reverse=True)
    return events[:6]


def _gemini_sector_readthrough(ticker: str, sector: str, events: list[dict[str, Any]]) -> dict[str, Any] | None:
    client = GeminiJSONClient()
    if not client.enabled:
        return None

    payload = {"ticker": ticker, "sector": sector, "peer_events": events[:6]}
    system = """
You are evaluating sector peer read-through for a weekly stock model.

Return only JSON:
{
  "bias": float,
  "validity": float,
  "details": ["string", "string", "string"]
}

Rules:
- bias in [-1, 1]
- validity in [0, 1]
- do not overreact just because the sector is active
- reward direct business-model read-through
- penalize weak generic sector sympathy
- be conservative
""".strip()
    return client.classify(system, payload)


def fetch_sector_readthrough_modifier(ticker: str) -> ModifierSignal:
    target = ticker.upper()
    sector = TICKER_TO_SECTOR.get(target)
    if not sector:
        return ModifierSignal(
            name="sector_readthrough",
            bias=0.0,
            validity=0.0,
            details=["No sector mapping available."],
            metadata={"sector": None, "readthrough_quality": 0.0},
        )

    events = _fetch_peer_events(target, sector)
    if not events:
        return ModifierSignal(
            name="sector_readthrough",
            bias=0.0,
            validity=0.0,
            details=[f"No recent peer read-through found for sector={sector}."],
            metadata={"sector": sector, "readthrough_quality": 0.0},
        )

    weighted_sum = 0.0
    weight_sum = 0.0
    details = []

    for ev in events:
        w = ev.relevance
        weighted_sum += ev.sentiment * w
        weight_sum += w
        details.append(f"{ev.peer}: {ev.reason} | {ev.headline}")

    bias = weighted_sum / weight_sum if weight_sum else 0.0
    quality = min(1.0, 0.22 + 0.12 * len(events) + 0.25 * (sum(ev.relevance for ev in events) / len(events)))

    gemini = _gemini_sector_readthrough(
        target,
        sector,
        [
            {
                "peer": ev.peer,
                "headline": ev.headline,
                "source": ev.source,
                "relevance": ev.relevance,
                "sentiment": ev.sentiment,
            }
            for ev in events
        ],
    )

    if gemini:
        bias = 0.72 * bias + 0.28 * float(gemini.get("bias", 0.0))
        quality = 0.80 * quality + 0.20 * float(gemini.get("validity", 0.0))
        if isinstance(gemini.get("details"), list) and gemini["details"]:
            details = [str(x) for x in gemini["details"]][:3]

    return ModifierSignal(
        name="sector_readthrough",
        bias=round(float(max(-1.0, min(1.0, bias))), 4),
        validity=round(float(max(0.0, min(1.0, quality))), 4),
        details=details[:3],
        metadata={"sector": sector, "readthrough_quality": round(float(quality), 4)},
    )