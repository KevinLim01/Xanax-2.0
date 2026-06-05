from __future__ import annotations

from typing import Iterable

STRICT_SECTORS = {
    "JPM", "GS", "BAC", "WFC",
    "MA", "V",
    "UNH", "LLY", "ABBV", "JNJ", "ABT", "TMO", "MRK",
    "KO", "PEP", "PM", "MCD", "WMT", "COST",
}

HIGH_BETA_NAMES = {
    "TSLA", "PLTR", "AMD", "NVDA", "META", "AMZN", "NFLX"
}

MEGACAP_TECH_NAMES = {
    "AAPL", "MSFT", "AMZN", "META", "GOOG", "GOOGL", "NVDA"
}

FINANCIAL_NAMES = {
    "JPM", "GS", "BAC", "WFC", "MS", "C"
}

EXPOSURE_MAP: dict[str, set[str]] = {
    "NVDA": {"ai_compute", "semi", "gpu", "data_center", "high_beta"},
    "AMD": {"ai_compute", "semi", "server_cpu", "data_center", "high_beta"},
    "AVGO": {"ai_compute", "semi", "networking", "data_center"},
    "QCOM": {"semi", "consumer_device", "wireless"},
    "TXN": {"semi", "analog", "industrial_capex"},
    "AMAT": {"semi_equipment", "industrial_capex", "fab_capex"},
    "INTC": {"semi", "server_cpu", "pc_cycle"},
    "MU": {"memory", "semi", "data_center"},
    "TSM": {"foundry", "semi", "fab_capex"},
    "MSFT": {"cloud_capex", "ai_compute", "enterprise_software"},
    "AMZN": {"cloud_capex", "consumer_discretionary", "ecommerce", "high_beta"},
    "GOOG": {"cloud_capex", "ad_market", "ai_compute"},
    "GOOGL": {"cloud_capex", "ad_market", "ai_compute"},
    "META": {"ad_market", "ai_compute", "high_beta"},
    "ORCL": {"cloud_capex", "enterprise_software", "data_center"},
    "NFLX": {"consumer_discretionary", "streaming", "high_beta"},
    "ADBE": {"enterprise_software", "creative_software"},
    "CRM": {"enterprise_software", "cloud_capex"},
    "NOW": {"enterprise_software", "cloud_capex"},
    "JPM": {"banks", "rates_sensitive", "credit_cycle"},
    "GS": {"banks", "markets_revenue", "rates_sensitive"},
    "BAC": {"banks", "rates_sensitive", "credit_cycle"},
    "WFC": {"banks", "rates_sensitive", "credit_cycle"},
    "MA": {"payments", "consumer_spend", "cross_border"},
    "V": {"payments", "consumer_spend", "cross_border"},
    "UNH": {"healthcare_defensive", "managed_care", "reimbursement"},
    "LLY": {"healthcare_growth", "drug_cycle"},
    "ABBV": {"healthcare_defensive", "drug_cycle"},
    "JNJ": {"healthcare_defensive", "drug_cycle"},
    "ABT": {"healthcare_defensive", "medtech"},
    "TMO": {"healthcare_tools", "biopharma_spend"},
    "MRK": {"healthcare_defensive", "drug_cycle"},
    "KO": {"defensive_consumer", "consumer_spend"},
    "PEP": {"defensive_consumer", "consumer_spend"},
    "PM": {"defensive_consumer"},
    "MCD": {"defensive_consumer", "consumer_spend"},
    "WMT": {"defensive_consumer", "consumer_spend"},
    "COST": {"defensive_consumer", "consumer_spend"},
    "CAT": {"industrial_capex", "cyclical_industrial"},
    "GE": {"industrial_capex", "cyclical_industrial"},
    "LIN": {"industrial_capex", "cyclical_industrial"},
    "XOM": {"energy", "oil_price"},
    "CVX": {"energy", "oil_price"},
    "TSLA": {"high_beta", "ev", "consumer_discretionary", "autonomy"},
    "PLTR": {"high_beta", "defense_software", "government_spend"},
}

EXPOSURE_KEYWORDS: dict[str, set[str]] = {
    "ai_compute": {"ai", "accelerator", "gpu", "inference", "training"},
    "server_cpu": {"server", "cpu", "epyc", "xeon", "data center"},
    "cloud_capex": {"cloud", "data center", "datacenter", "capex", "infrastructure"},
    "semi_equipment": {"wafer", "fab", "equipment", "foundry", "capex"},
    "memory": {"dram", "nand", "memory"},
    "banks": {"deposit", "loan", "credit", "nii", "net interest income", "bank"},
    "payments": {"payments", "spend", "cross-border", "card"},
    "managed_care": {"medical cost", "medicare", "membership", "utilization"},
    "reimbursement": {"reimbursement", "medicaid", "medicare"},
    "defensive_consumer": {"consumer", "traffic", "pricing", "volume"},
    "industrial_capex": {"orders", "capex", "industrial", "backlog"},
    "energy": {"oil", "gas", "energy", "crude"},
    "ev": {"ev", "electric vehicle", "vehicle", "automotive"},
    "autonomy": {"autonomy", "robotaxi", "fsd", "self-driving"},
    "ad_market": {"advertising", "ad spend", "ads"},
}


def get_exposures(ticker: str) -> set[str]:
    return EXPOSURE_MAP.get(ticker.upper(), set())


def has_exposure(ticker: str, exposure: str) -> bool:
    return exposure in get_exposures(ticker)


def exposure_overlap(a: str, b: str) -> float:
    ea = get_exposures(a)
    eb = get_exposures(b)
    if not ea or not eb:
        return 0.0
    inter = len(ea & eb)
    union = len(ea | eb)
    return inter / union if union else 0.0


def keyword_exposure_hits(text: str) -> set[str]:
    low = text.lower()
    hits: set[str] = set()
    for exposure, keywords in EXPOSURE_KEYWORDS.items():
        if any(k in low for k in keywords):
            hits.add(exposure)
    return hits


def peer_relevance(target: str, peer: str, headline: str) -> float:
    base = exposure_overlap(target, peer)
    target_exp = get_exposures(target)
    peer_exp = get_exposures(peer)
    hits = keyword_exposure_hits(headline)

    match_bonus = 0.0
    if hits & target_exp:
        match_bonus += 0.28
    if hits & peer_exp:
        match_bonus += 0.18
    if {"ai_compute", "server_cpu", "cloud_capex"} & hits and {"ai_compute", "server_cpu", "cloud_capex"} & target_exp:
        match_bonus += 0.15
    if {"banks", "payments", "managed_care", "reimbursement", "defensive_consumer"} & hits and not (hits & target_exp):
        match_bonus -= 0.10

    return max(0.0, min(1.0, base + match_bonus))


def is_strict_sector_name(ticker: str) -> bool:
    return ticker.upper() in STRICT_SECTORS


def is_high_beta_name(ticker: str) -> bool:
    return ticker.upper() in HIGH_BETA_NAMES


def is_megacap_tech_name(ticker: str) -> bool:
    return ticker.upper() in MEGACAP_TECH_NAMES


def is_financial_name(ticker: str) -> bool:
    return ticker.upper() in FINANCIAL_NAMES


def any_exposure(ticker: str, exposures: Iterable[str]) -> bool:
    exp = get_exposures(ticker)
    return any(x in exp for x in exposures)