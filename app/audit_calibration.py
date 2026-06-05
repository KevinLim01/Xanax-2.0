from __future__ import annotations

# Audit-informed calibration from the week where the model missed TXN, UNH, AMAT,
# AMZN on the upside and TMO on the downside while AMD/NVDA/GS highlighted the
# importance of peer read-through and guidance.
AUDIT_IMPORTANCE = {
    "guidance_multiplier": 1.8,
    "sector_readthrough_multiplier": 1.6,
    "relief_rally_multiplier": 1.45,
    "eventful_regime_multiplier": 1.35,
    "technical_trend_penalty_eventful_multiplier": 0.45,
    "weak_truthsocial_multiplier": 0.15,
    "baseline_news_floor": 0.10,
}
