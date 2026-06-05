from __future__ import annotations

import html
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import feedparser

from app.domain_maps import (
    any_exposure,
    is_high_beta_name,
    is_megacap_tech_name,
    is_strict_sector_name,
    keyword_exposure_hits,
)
from app.llm_client import GeminiJSONClient
from app.models import ModifierSignal

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"

HIGH_QUALITY_SOURCES = {
    "Reuters",
    "Bloomberg",
    "CNBC",
    "The Wall Street Journal",
    "Financial Times",
    "Associated Press",
    "AP News",
    "Barron's",
    "MarketWatch",
    "Investor's Business Daily",
    "Yahoo Finance",
    "The Information",
}

LOW_QUALITY_SOURCES = {
    "Stocktwits",
    "Traders Union",
    "Benzinga",
    "thestreet.com",
    "Quiver Quantitative",
}

LOW_QUALITY_SOURCE_SUBSTRINGS = [
    "marketbeat",
    "invezz",
    "simplywall",
    "aastocks",
    "mexc",
    "tradingkey",
    "247 wall st",
    "financialcontent",
    "stocktwits",
    "traders union",
]

LOW_QUALITY_HEADLINE_PATTERNS = [
    "stock quote",
    "forecast",
    "price prediction",
    "shares bought by",
    "holdings lowered by",
    "analyst upgrades/downgrades",
    "here is why",
    "here's why",
]

GUIDANCE_POS_PATTERNS = [
    r"\braises? guidance\b",
    r"\bboosts? outlook\b",
    r"\bupbeat forecast\b",
    r"\bstrong demand\b",
    r"\bhigher outlook\b",
    r"\bbetter-than-expected outlook\b",
    r"\braised full-year\b",
]

GUIDANCE_NEG_PATTERNS = [
    r"\bcuts? guidance\b",
    r"\blower outlook\b",
    r"\bweak demand\b",
    r"\bsoft demand\b",
    r"\bwarns?\b",
    r"\bslowing demand\b",
    r"\bsees weakness\b",
]

EARNINGS_PATTERNS = [
    r"\bearnings\b",
    r"\bresults\b",
    r"\bquarter\b",
    r"\bq[1-4]\b",
    r"\bfiscal\b",
]

RELIEF_PATTERNS = [
    r"\bbetter than feared\b",
    r"\bless bad than expected\b",
    r"\brelief rally\b",
    r"\bafter selloff\b",
    r"\bafter slump\b",
]

MANAGEMENT_PATTERNS = [
    r"\bceo\b",
    r"\bchief executive\b",
    r"\bexecutive chairman\b",
    r"\bsteps down\b",
    r"\bappointed\b",
    r"\bnamed\b",
    r"\bsuccessor\b",
    r"\belon musk\b",
    r"\bmusk comments\b",
    r"\bmanagement comments\b",
]

REGULATORY_PATTERNS = [
    r"\bdoj\b",
    r"\bsec\b",
    r"\beu\b",
    r"\bprobe\b",
    r"\blawsuit\b",
    r"\bcourt\b",
    r"\btariff\b",
    r"\bsanction\b",
]

NEGATIVE_REACTION_PATTERNS = [
    r"\bstock falls\b",
    r"\bshares fall\b",
    r"\bstock drops\b",
    r"\bshares drop\b",
    r"\bfalls on\b",
    r"\bdrops on\b",
    r"\bslides on\b",
    r"\bbeat, but\b",
    r"\bresults beat, but\b",
    r"\bbut .* stock falls\b",
    r"\bbut .* shares fall\b",
    r"\bbut .* drops\b",
    r"\bdoes not have the capability\b",
    r"\bnot have the capability\b",
]

POSITIVE_WORDS = [
    "beats", "beat", "strong", "growth", "upside", "jumps", "surges", "higher",
    "bullish", "record", "rebound", "relief",
]
NEGATIVE_WORDS = [
    "misses", "miss", "cuts", "falls", "slides", "drops", "warning", "weak", "soft", "bearish",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(entry: feedparser.FeedParserDict) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        parsed = entry.get(key)
        if parsed:
            return datetime(*parsed[:6], tzinfo=timezone.utc)
    return None


def _extract_source(entry: feedparser.FeedParserDict, headline: str) -> str:
    source_obj = entry.get("source")
    if isinstance(source_obj, dict):
        title = str(source_obj.get("title", "")).strip()
        if title:
            return title
    if " - " in headline:
        return headline.rsplit(" - ", 1)[-1].strip()
    return "Unknown"


def _is_low_quality_source(source: str) -> bool:
    s = source.lower()
    if source in LOW_QUALITY_SOURCES:
        return True
    return any(token in s for token in LOW_QUALITY_SOURCE_SUBSTRINGS)


def _is_low_quality_headline(headline: str) -> bool:
    h = headline.lower()
    return any(token in h for token in LOW_QUALITY_HEADLINE_PATTERNS)


def _sentiment_score(text: str) -> float:
    t = text.lower()
    score = 0.0
    score += 0.18 * sum(1 for w in POSITIVE_WORDS if w in t)
    score -= 0.18 * sum(1 for w in NEGATIVE_WORDS if w in t)
    return max(-1.0, min(1.0, score))


def _count_matches(text: str, patterns: list[str]) -> int:
    low = text.lower()
    return sum(1 for p in patterns if re.search(p, low))


def _source_quality(source: str) -> float:
    if source in HIGH_QUALITY_SOURCES:
        return 1.0
    if _is_low_quality_source(source):
        return 0.05
    return 0.35


def _sanitize_exposure_hits(ticker: str, hits: set[str], headline: str) -> set[str]:
    cleaned = set(hits)
    low = headline.lower()

    # Do not tag random Elon mentions as EV exposure for non-EV names
    if "ev" in cleaned and not any_exposure(ticker, {"ev", "autonomy"}):
        if "elon musk" in low or "musk" in low:
            cleaned.discard("ev")
            cleaned.discard("autonomy")

    return cleaned


def _classify_items_with_gemini(ticker: str, items: list[dict[str, Any]]) -> dict[str, Any] | None:
    client = GeminiJSONClient()
    if not client.enabled:
        return None

    payload = {"ticker": ticker, "items": items[:8]}
    system = """
You are classifying stock news for a weekly Monday-close to Friday-close stock model.

Return only JSON:
{
  "overall_bias": float,
  "catalyst_quality": float,
  "mixed_news": bool,
  "relief_rally_signal": float,
  "guidance_signal": float,
  "details": ["string", "string", "string"]
}

Rules:
- overall_bias in [-1, 1]
- catalyst_quality in [0, 1]
- mixed_news true if the news flow is contradictory or noisy
- relief_rally_signal in [-1, 1]
- guidance_signal in [-1, 1]
- reward forward guidance, demand commentary, regulatory decisions, clear management changes
- penalize generic analyst chatter, quote pages, weak finance spam
- specifically penalize headlines like "beat, but stock falls" or negative management commentary
- for megacap tech, low-quality source bearishness should count much less
- be conservative
""".strip()
    return client.classify(system, payload)


def fetch_news_modifier(ticker: str) -> ModifierSignal:
    query = f"{ticker} earnings OR {ticker} guidance OR {ticker} outlook OR {ticker} demand OR {ticker} CEO"
    url = f"{GOOGLE_NEWS_RSS}?q={query.replace(' ', '%20')}&hl=en-US&gl=US&ceid=US:en"

    feed = feedparser.parse(url)
    cutoff = _utc_now() - timedelta(days=5)

    raw_items: list[dict[str, Any]] = []
    for entry in feed.entries[:24]:
        headline = html.unescape(str(entry.get("title", "")).strip())
        if not headline:
            continue
        dt = _parse_dt(entry)
        if dt is None or dt < cutoff:
            continue
        source = _extract_source(entry, headline)
        raw_items.append(
            {
                "headline": headline,
                "source": source,
                "published_at": dt.isoformat(),
            }
        )

    if not raw_items:
        return ModifierSignal(
            name="news",
            bias=0.0,
            validity=0.0,
            details=["No recent qualifying news found."],
            metadata={
                "catalyst_quality": 0.0,
                "mixed_news": False,
                "guidance_signal": 0.0,
                "relief_rally_signal": 0.0,
                "exposure_hits": [],
                "negative_reaction_signal": 0.0,
            },
        )

    scored_items: list[dict[str, Any]] = []
    all_exposure_hits: set[str] = set()

    for item in raw_items:
        headline = item["headline"]
        source = item["source"]

        src_quality = _source_quality(source)
        sentiment = _sentiment_score(headline)

        guidance_pos = _count_matches(headline, GUIDANCE_POS_PATTERNS)
        guidance_neg = _count_matches(headline, GUIDANCE_NEG_PATTERNS)
        earnings_hits = _count_matches(headline, EARNINGS_PATTERNS)
        relief_hits = _count_matches(headline, RELIEF_PATTERNS)
        mgmt_hits = _count_matches(headline, MANAGEMENT_PATTERNS)
        reg_hits = _count_matches(headline, REGULATORY_PATTERNS)
        neg_reaction_hits = _count_matches(headline, NEGATIVE_REACTION_PATTERNS)

        exposure_hits = _sanitize_exposure_hits(ticker, keyword_exposure_hits(headline), headline)
        all_exposure_hits |= exposure_hits

        catalyst_quality = 0.12
        catalyst_quality += 0.28 * min(1, guidance_pos + guidance_neg)
        catalyst_quality += 0.12 * min(1, earnings_hits)
        catalyst_quality += 0.08 * min(1, relief_hits)
        catalyst_quality += 0.10 * min(1, mgmt_hits)
        catalyst_quality += 0.10 * min(1, reg_hits)
        catalyst_quality += 0.22 * src_quality
        catalyst_quality += 0.10 * min(1, len(exposure_hits))

        if neg_reaction_hits:
            catalyst_quality += 0.08

        if _is_low_quality_headline(headline):
            catalyst_quality -= 0.24
        if _is_low_quality_source(source):
            catalyst_quality -= 0.22

        neg_signal = -0.85 * neg_reaction_hits
        if "elon musk" in headline.lower() or "musk comments" in headline.lower():
            if any_exposure(ticker, {"ev", "autonomy"}):
                neg_signal -= 0.35

        # Megacap tech: low-quality source bearishness counts much less
        if is_megacap_tech_name(ticker) and _is_low_quality_source(source):
            sentiment *= 0.35
            neg_signal *= 0.20

        item["source_quality"] = round(src_quality, 4)
        item["sentiment"] = round(sentiment, 4)
        item["guidance_signal"] = round(0.50 * guidance_pos - 0.50 * guidance_neg, 4)
        item["relief_signal"] = round(0.28 * relief_hits, 4)
        item["negative_reaction_signal"] = round(neg_signal, 4)
        item["catalyst_quality"] = round(max(0.0, min(1.0, catalyst_quality)), 4)
        item["exposure_hits"] = sorted(exposure_hits)
        scored_items.append(item)

    gemini = _classify_items_with_gemini(ticker, scored_items)

    avg_bias = sum(x["sentiment"] for x in scored_items) / len(scored_items)
    avg_quality = sum(x["catalyst_quality"] for x in scored_items) / len(scored_items)
    guidance_signal = sum(x["guidance_signal"] for x in scored_items) / len(scored_items)
    relief_signal = sum(x["relief_signal"] for x in scored_items) / len(scored_items)
    negative_reaction_signal = sum(x["negative_reaction_signal"] for x in scored_items) / len(scored_items)

    strong_pos = sum(1 for x in scored_items if x["sentiment"] > 0.10)
    strong_neg = sum(1 for x in scored_items if x["sentiment"] < -0.10)
    mixed_news = strong_pos > 0 and strong_neg > 0

    if gemini:
        avg_bias = 0.60 * avg_bias + 0.40 * float(gemini.get("overall_bias", 0.0))
        avg_quality = 0.55 * avg_quality + 0.45 * float(gemini.get("catalyst_quality", 0.0))
        guidance_signal = 0.65 * guidance_signal + 0.35 * float(gemini.get("guidance_signal", 0.0))
        relief_signal = 0.70 * relief_signal + 0.30 * float(gemini.get("relief_rally_signal", 0.0))
        mixed_news = bool(gemini.get("mixed_news", mixed_news))

    avg_bias += 0.42 * guidance_signal + 0.22 * relief_signal + 1.10 * negative_reaction_signal

    if mixed_news:
        avg_bias *= 0.66
        avg_quality *= 0.84

    if is_strict_sector_name(ticker):
        if avg_quality < 0.70:
            avg_bias *= 0.58
        if avg_quality < 0.58:
            avg_bias *= 0.55
        if mixed_news:
            avg_bias *= 0.80

    if is_high_beta_name(ticker) and any_exposure(ticker, {"ev", "high_beta"}):
        if negative_reaction_signal < 0 and any_exposure(ticker, {"ev", "autonomy"}):
            avg_bias += 1.60 * negative_reaction_signal
        if avg_quality < 0.60 and mixed_news:
            avg_bias *= 0.72

    if avg_quality < 0.55:
        avg_bias *= 0.50
    if avg_quality < 0.42:
        avg_bias *= 0.32

    details = []
    if gemini and isinstance(gemini.get("details"), list):
        details = [str(x) for x in gemini["details"]][:3]
    if not details:
        details = [f"{x['source']}: {x['headline']}" for x in scored_items[:3]]

    validity = min(1.0, 0.18 + 0.60 * avg_quality + 0.08 * min(1.0, len(scored_items) / 4))
    if mixed_news:
        validity *= 0.86

    return ModifierSignal(
        name="news",
        bias=round(float(max(-1.0, min(1.0, avg_bias))), 4),
        validity=round(float(max(0.0, min(1.0, validity))), 4),
        details=details[:3],
        metadata={
            "catalyst_quality": round(float(max(0.0, min(1.0, avg_quality))), 4),
            "mixed_news": mixed_news,
            "guidance_signal": round(float(guidance_signal), 4),
            "relief_rally_signal": round(float(relief_signal), 4),
            "negative_reaction_signal": round(float(negative_reaction_signal), 4),
            "exposure_hits": sorted(all_exposure_hits),
        },
    )