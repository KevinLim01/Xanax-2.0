from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.config import settings
from app.models import ModifierSignal

try:
    from app.llm_client import GeminiTextClient
except Exception:
    GeminiTextClient = None


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT_DIR / "data" / "truthsocial_feed.json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _last_friday_to_monday_window(now: datetime) -> tuple[datetime, datetime]:
    weekday = now.weekday()  # Monday=0
    this_monday = (now - timedelta(days=weekday)).replace(hour=0, minute=0, second=0, microsecond=0)
    last_friday = this_monday - timedelta(days=3)
    return last_friday, this_monday


def _load_feed() -> list[dict[str, Any]]:
    if not DATA_FILE.exists():
        return []
    raw = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    return []


def _filter_window(posts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = _utc_now()
    start_ts, end_ts = _last_friday_to_monday_window(now)
    out = []

    for post in posts:
        dt = _safe_parse_dt(post.get("created_at"))
        if dt is None:
            continue
        if start_ts <= dt < end_ts:
            item = dict(post)
            item["_parsed_dt"] = dt
            out.append(item)

    out.sort(key=lambda x: x["_parsed_dt"], reverse=True)
    return out


def _rule_based_truthsocial(ticker: str, posts: list[dict[str, Any]]) -> ModifierSignal:
    if not posts:
        return ModifierSignal(
            name="truthsocial",
            bias=0.0,
            validity=0.0,
            details=["No qualifying Truth Social posts found in the Friday-to-Monday window."],
        )

    ticker_upper = ticker.upper()
    positive_terms = ["deal", "growth", "boom", "great", "strong", "win", "lower rates"]
    negative_terms = ["tariff", "war", "sanction", "weak", "bad", "cut", "crisis", "china"]

    score = 0.0
    used = 0
    details: list[str] = []

    for post in posts[:8]:
        text = str(post.get("text", "")).strip()
        lower = text.lower()
        if not text:
            continue

        relevance = 0.2
        if ticker_upper.lower() in lower:
            relevance += 0.5

        sector_hits = {
            "AMD": ["chip", "semiconductor", "ai", "server", "china"],
            "NVDA": ["chip", "semiconductor", "ai", "server", "china"],
            "AAPL": ["iphone", "apple", "china", "tariff", "consumer electronics"],
            "XOM": ["oil", "energy", "iran", "middle east"],
            "CVX": ["oil", "energy", "iran", "middle east"],
            "JPM": ["bank", "rate", "fed", "financial"],
            "GS": ["bank", "rate", "fed", "financial"],
        }.get(ticker_upper, [])

        if any(k in lower for k in sector_hits):
            relevance += 0.35

        sentiment = 0.0
        sentiment += sum(0.25 for term in positive_terms if term in lower)
        sentiment -= sum(0.25 for term in negative_terms if term in lower)

        if abs(sentiment) < 0.05 and relevance < 0.5:
            continue

        score += sentiment * relevance
        used += 1
        details.append(text[:140])

    if used == 0:
        return ModifierSignal(
            name="truthsocial",
            bias=0.0,
            validity=0.15,
            details=["Posts found, but none looked strongly relevant to this ticker."],
        )

    bias = max(-1.0, min(1.0, score / used))
    validity = min(1.0, 0.30 + 0.12 * used)

    return ModifierSignal(
        name="truthsocial",
        bias=round(float(bias), 4),
        validity=round(float(validity), 4),
        details=details[:3],
    )


def _gemini_truthsocial(ticker: str, posts: list[dict[str, Any]]) -> ModifierSignal:
    if not GeminiTextClient:
        return _rule_based_truthsocial(ticker, posts)

    payload = []
    for post in posts[:8]:
        payload.append(
            {
                "created_at": post.get("created_at"),
                "text": post.get("text", ""),
            }
        )

    prompt = f"""
You are evaluating whether Trump Truth Social posts from last Friday to Monday matter for the stock {ticker} this week.

Return JSON only with this schema:
{{
  "bias": float,
  "validity": float,
  "details": ["string", "string", "string"]
}}

Rules:
- bias ranges from -1.0 to 1.0
- validity ranges from 0.0 to 1.0
- positive bias means bullish for {ticker}
- negative bias means bearish for {ticker}
- think in terms of tariffs, trade, sanctions, AI demand, chips, oil, banks, healthcare, sector spillover
- if posts are irrelevant, set both low
- be conservative
- details should be short explanations, not long paragraphs

Posts:
{json.dumps(payload, ensure_ascii=False)}
""".strip()

    try:
        client = GeminiTextClient()
        raw = client.generate_json(prompt)
        return ModifierSignal(
            name="truthsocial",
            bias=round(float(raw.get("bias", 0.0)), 4),
            validity=round(float(raw.get("validity", 0.0)), 4),
            details=[str(x) for x in raw.get("details", [])][:3],
        )
    except Exception:
        return _rule_based_truthsocial(ticker, posts)


def fetch_truthsocial_modifier(ticker: str) -> ModifierSignal:
    posts = _filter_window(_load_feed())
    if not posts:
        return ModifierSignal(
            name="truthsocial",
            bias=0.0,
            validity=0.0,
            details=["No qualifying Truth Social posts found in the Friday-to-Monday window."],
        )

    if settings.use_gemini_text_agents and settings.gemini_api_key:
        return _gemini_truthsocial(ticker, posts)

    return _rule_based_truthsocial(ticker, posts)
