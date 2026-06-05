from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


def _bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def _int(name: str, default: str) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return int(default)


def _float(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    gemini_api_version: str = os.getenv("GEMINI_API_VERSION", "v1beta")
    use_gemini_text_agents: bool = _bool("USE_GEMINI_TEXT_AGENTS", "true")

    default_ticker: str = os.getenv("DEFAULT_TICKER", "AAPL")
    database_path: str = os.getenv("DATABASE_PATH", "data/stock_signals.db")
    log_level: str = os.getenv("LOG_LEVEL", "WARNING").upper()

    training_years: int = _int("TRAINING_YEARS", "8")
    neutral_band: float = _float("NEUTRAL_BAND", "0.006")
    opportunity_threshold_pct: float = _float("OPPORTUNITY_THRESHOLD_PCT", "1.0")
    monday_min_tradeable_threshold_pct: float = _float("MONDAY_MIN_TRADEABLE_THRESHOLD_PCT", "1.5")
    min_training_rows: int = _int("MIN_TRAINING_ROWS", "60")

    news_enabled: bool = _bool("USE_NEWS_MODIFIER", "true")
    polymarket_enabled: bool = _bool("USE_POLYMARKET_MODIFIER", "true")
    truthsocial_enabled: bool = _bool("USE_TRUTHSOCIAL_MODIFIER", "true")
    sector_readthrough_enabled: bool = _bool("USE_SECTOR_READTHROUGH_MODIFIER", "true")

    news_weight: float = _float("NEWS_WEIGHT", "0.22")
    polymarket_weight: float = _float("POLYMARKET_WEIGHT", "0.08")
    truthsocial_weight: float = _float("TRUTHSOCIAL_WEIGHT", "0.06")
    sector_readthrough_weight: float = _float("SECTOR_READTHROUGH_WEIGHT", "0.20")

    # New agent toggles.
    use_relative_strength_agent: bool = _bool("USE_RELATIVE_STRENGTH_AGENT", "true")
    use_intraday_confirmation_agent: bool = _bool("USE_INTRADAY_CONFIRMATION_AGENT", "true")
    use_liquidity_agent: bool = _bool("USE_LIQUIDITY_AGENT", "true")

    # Relative-strength agent settings.
    relative_strength_lookback_days: int = _int("RELATIVE_STRENGTH_LOOKBACK_DAYS", "5")
    relative_strength_min_abs_score: float = _float("RELATIVE_STRENGTH_MIN_ABS_SCORE", "0.05")
    relative_strength_weight: float = _float("RELATIVE_STRENGTH_WEIGHT", "0.20")

    # Intraday-confirmation agent settings.
    intraday_confirmation_period: str = os.getenv("INTRADAY_CONFIRMATION_PERIOD", "1d")
    intraday_confirmation_interval: str = os.getenv("INTRADAY_CONFIRMATION_INTERVAL", "5m")
    intraday_confirmation_weight: float = _float("INTRADAY_CONFIRMATION_WEIGHT", "0.15")
    intraday_vwap_weight: float = _float("INTRADAY_VWAP_WEIGHT", "0.45")
    intraday_momentum_weight: float = _float("INTRADAY_MOMENTUM_WEIGHT", "0.35")
    intraday_volume_weight: float = _float("INTRADAY_VOLUME_WEIGHT", "0.20")

    # Liquidity agent settings.
    liquidity_weight: float = _float("LIQUIDITY_WEIGHT", "0.05")
    min_dollar_volume: float = _float("MIN_DOLLAR_VOLUME", "10000000")
    max_spread_pct: float = _float("MAX_SPREAD_PCT", "0.50")
    liquidity_warning_spread_pct: float = _float("LIQUIDITY_WARNING_SPREAD_PCT", "0.25")
    liquidity_hard_block_spread_pct: float = _float("LIQUIDITY_HARD_BLOCK_SPREAD_PCT", "0.50")
    liquidity_min_avg_volume: float = _float("LIQUIDITY_MIN_AVG_VOLUME", "500000")

    # Final score blending settings.
    final_prediction_weight: float = _float("FINAL_PREDICTION_WEIGHT", "0.25")
    final_technical_weight: float = _float("FINAL_TECHNICAL_WEIGHT", "0.20")
    final_relative_strength_weight: float = _float("FINAL_RELATIVE_STRENGTH_WEIGHT", "0.20")
    final_intraday_weight: float = _float("FINAL_INTRADAY_WEIGHT", "0.15")
    final_macro_weight: float = _float("FINAL_MACRO_WEIGHT", "0.10")
    final_news_weight: float = _float("FINAL_NEWS_WEIGHT", "0.05")
    final_liquidity_weight: float = _float("FINAL_LIQUIDITY_WEIGHT", "0.05")

    # Agent conviction adjustments.
    relative_strength_agree_boost: int = _int("RELATIVE_STRENGTH_AGREE_BOOST", "7")
    relative_strength_conflict_penalty: int = _int("RELATIVE_STRENGTH_CONFLICT_PENALTY", "8")
    intraday_agree_boost: int = _int("INTRADAY_AGREE_BOOST", "6")
    intraday_conflict_penalty: int = _int("INTRADAY_CONFLICT_PENALTY", "10")
    weak_liquidity_penalty: int = _int("WEAK_LIQUIDITY_PENALTY", "8")
    bad_liquidity_penalty: int = _int("BAD_LIQUIDITY_PENALTY", "15")
    bad_liquidity_conviction_cap: int = _int("BAD_LIQUIDITY_CONVICTION_CAP", "45")
    conflict_conviction_cap: int = _int("CONFLICT_CONVICTION_CAP", "65")

    buy_score_min: float = _float("BUY_SCORE_MIN", "0.18")
    sell_score_min: float = _float("SELL_SCORE_MIN", "0.18")
    neutral_score_band: float = _float("NEUTRAL_SCORE_BAND", "0.07")
    buy_expected_move_min: float = _float("BUY_EXPECTED_MOVE_MIN", "0.006")
    sell_expected_move_min: float = _float("SELL_EXPECTED_MOVE_MIN", "0.006")

    momentum_sell_confirmation_min: int = _int("MOMENTUM_SELL_CONFIRMATION_MIN", "2")
    reversal_conviction_cap: int = _int("REVERSAL_CONVICTION_CAP", "60")
    dangerous_short_conviction_penalty: int = _int("DANGEROUS_SHORT_CONVICTION_PENALTY", "15")
    enable_momentum_override: bool = _bool("ENABLE_MOMENTUM_OVERRIDE", "true")

    screen_buy_quantile: float = _float("SCREEN_BUY_QUANTILE", "0.82")
    screen_sell_quantile: float = _float("SCREEN_SELL_QUANTILE", "0.18")
    screen_min_abs_score: float = _float("SCREEN_MIN_ABS_SCORE", "0.07")

    # Alpaca paper-trading add-on. Defaults are deliberately conservative.
    alpaca_trading_mode: str = os.getenv("ALPACA_TRADING_MODE", "paper")
    allow_live_trading: bool = _bool("ALLOW_LIVE_TRADING", "false")
    alpaca_api_key: str = os.getenv("ALPACA_API_KEY", "")
    alpaca_secret_key: str = os.getenv("ALPACA_SECRET_KEY", "")

    auto_trade_enabled: bool = _bool("AUTO_TRADE_ENABLED", "false")
    auto_trade_allow_shorts: bool = _bool("AUTO_TRADE_ALLOW_SHORTS", "false")
    auto_trade_instrument: str = os.getenv("AUTO_TRADE_INSTRUMENT", "stock").strip().lower()
    auto_trade_allow_options: bool = _bool("AUTO_TRADE_ALLOW_OPTIONS", "false")
    auto_trade_options_contracts_per_trade: int = _int("AUTO_TRADE_OPTIONS_CONTRACTS_PER_TRADE", "1")
    auto_trade_options_min_dte: int = _int("AUTO_TRADE_OPTIONS_MIN_DTE", "7")
    auto_trade_options_max_dte: int = _int("AUTO_TRADE_OPTIONS_MAX_DTE", "21")
    auto_trade_options_call_strike_offset_pct: float = _float("AUTO_TRADE_OPTIONS_CALL_STRIKE_OFFSET_PCT", "0.0")
    auto_trade_options_put_strike_offset_pct: float = _float("AUTO_TRADE_OPTIONS_PUT_STRIKE_OFFSET_PCT", "0.0")
    auto_trade_options_max_contract_price: float = _float("AUTO_TRADE_OPTIONS_MAX_CONTRACT_PRICE", "5.00")
    auto_trade_options_close_expiring_within_days: int = _int("AUTO_TRADE_OPTIONS_CLOSE_EXPIRING_WITHIN_DAYS", "2")
    auto_trade_require_market_open: bool = _bool("AUTO_TRADE_REQUIRE_MARKET_OPEN", "true")
    auto_trade_require_moderate_edge: bool = _bool("AUTO_TRADE_REQUIRE_MODERATE_EDGE", "true")
    auto_trade_min_conviction: int = _int("AUTO_TRADE_MIN_CONVICTION", "55")
    auto_trade_max_trades_per_run: int = _int("AUTO_TRADE_MAX_TRADES_PER_RUN", "5")
    auto_trade_max_active_positions: int = _int("AUTO_TRADE_MAX_ACTIVE_POSITIONS", "5")
    auto_trade_max_position_size_usd: float = _float("AUTO_TRADE_MAX_POSITION_SIZE_USD", "1000")
    auto_trade_max_total_exposure_usd: float = _float("AUTO_TRADE_MAX_TOTAL_EXPOSURE_USD", "5000")

    # Live paper reinvestment / capital allocation.
    # Set AUTO_TRADE_STARTING_ACCOUNT_EQUITY_USD to your Alpaca paper equity at the moment you start this model.
    # Example: paper equity starts at $100000, bot base capital is $5000. If paper equity rises to $101000,
    # the bot's allowed exposure becomes about $6000.
    auto_trade_reinvest_enabled: bool = _bool("AUTO_TRADE_REINVEST_ENABLED", "true")
    auto_trade_base_capital_usd: float = _float("AUTO_TRADE_BASE_CAPITAL_USD", "5000")
    auto_trade_starting_account_equity_usd: float = _float("AUTO_TRADE_STARTING_ACCOUNT_EQUITY_USD", "0")
    auto_trade_reinvest_min_total_exposure_usd: float = _float("AUTO_TRADE_REINVEST_MIN_TOTAL_EXPOSURE_USD", "1000")
    auto_trade_reinvest_max_total_exposure_usd: float = _float("AUTO_TRADE_REINVEST_MAX_TOTAL_EXPOSURE_USD", "25000")
    auto_trade_reinvest_position_fraction: float = _float("AUTO_TRADE_REINVEST_POSITION_FRACTION", "0.20")
    auto_trade_reinvest_min_position_size_usd: float = _float("AUTO_TRADE_REINVEST_MIN_POSITION_SIZE_USD", "50")
    auto_trade_reinvest_max_position_size_usd: float = _float("AUTO_TRADE_REINVEST_MAX_POSITION_SIZE_USD", "5000")

    # Old exit-engine defaults are still present for emergency/friday logic.
    # The new monitor ignores old fixed profit exits unless you disable the model/history monitor below.
    auto_trade_take_profit_pct: float = _float("AUTO_TRADE_TAKE_PROFIT_PCT", "3.0")
    auto_trade_stop_loss_pct: float = _float("AUTO_TRADE_STOP_LOSS_PCT", "8.0")
    auto_trade_trailing_stop_pct: float = _float("AUTO_TRADE_TRAILING_STOP_PCT", "1.25")
    auto_trade_force_exit_friday_hour: int = _int("AUTO_TRADE_FORCE_EXIT_FRIDAY_HOUR", "15")
    auto_trade_force_exit_friday_minute: int = _int("AUTO_TRADE_FORCE_EXIT_FRIDAY_MINUTE", "45")

    # Adaptive exit logic from older version. Kept for compatibility.
    adaptive_take_profit_enabled: bool = _bool("ADAPTIVE_TAKE_PROFIT_ENABLED", "true")
    adaptive_take_profit_floor_pct: float = _float("ADAPTIVE_TAKE_PROFIT_FLOOR_PCT", "1.25")
    adaptive_take_profit_cap_pct: float = _float("ADAPTIVE_TAKE_PROFIT_CAP_PCT", "3.0")
    adaptive_take_profit_move_fraction: float = _float("ADAPTIVE_TAKE_PROFIT_MOVE_FRACTION", "0.70")

    no_same_day_profit_exit: bool = _bool("NO_SAME_DAY_PROFIT_EXIT", "true")
    allow_same_day_stop_loss: bool = _bool("ALLOW_SAME_DAY_STOP_LOSS", "true")
    allow_same_day_friday_exit: bool = _bool("ALLOW_SAME_DAY_FRIDAY_EXIT", "true")
    allow_same_day_option_expiration_exit: bool = _bool("ALLOW_SAME_DAY_OPTION_EXPIRATION_EXIT", "true")

    # New model/history monitor selling logic.
    monitor_use_model_rerun: bool = _bool("MONITOR_USE_MODEL_RERUN", "true")
    monitor_use_history_exit: bool = _bool("MONITOR_USE_HISTORY_EXIT", "true")
    monitor_exit_on_signal_flip: bool = _bool("MONITOR_EXIT_ON_SIGNAL_FLIP", "true")
    monitor_exit_on_low_conviction: bool = _bool("MONITOR_EXIT_ON_LOW_CONVICTION", "true")
    monitor_low_conviction_exit_threshold: int = _int("MONITOR_LOW_CONVICTION_EXIT_THRESHOLD", "50")
    monitor_history_profit_capture_ratio: float = _float("MONITOR_HISTORY_PROFIT_CAPTURE_RATIO", "0.85")
    monitor_history_profit_capture_ratio_mid: float = _float("MONITOR_HISTORY_PROFIT_CAPTURE_RATIO_MID", "0.90")
    monitor_history_profit_capture_ratio_high: float = _float("MONITOR_HISTORY_PROFIT_CAPTURE_RATIO_HIGH", "0.95")
    monitor_history_mid_conviction: int = _int("MONITOR_HISTORY_MID_CONVICTION", "65")
    monitor_history_high_conviction: int = _int("MONITOR_HISTORY_HIGH_CONVICTION", "75")
    monitor_history_min_sample_size: int = _int("MONITOR_HISTORY_MIN_SAMPLE_SIZE", "50")

    # Profit protection: keep winners from round-tripping after they have moved green.
    monitor_profit_protection_enabled: bool = _bool("MONITOR_PROFIT_PROTECTION_ENABLED", "true")
    monitor_profit_protection_activation_pct: float = _float("MONITOR_PROFIT_PROTECTION_ACTIVATION_PCT", "2.0")
    monitor_profit_protection_floor_pct: float = _float("MONITOR_PROFIT_PROTECTION_FLOOR_PCT", "0.5")
    monitor_profit_protection_state_path: str = os.getenv("MONITOR_PROFIT_PROTECTION_STATE_PATH", "data/profit_protection_state.json")

    # Concentrated top-5 weekly model filters.
    day_min_conviction_monday: int = _int("DAY_MIN_CONVICTION_MONDAY", "55")
    day_min_conviction_tuesday: int = _int("DAY_MIN_CONVICTION_TUESDAY", "60")
    day_min_conviction_wednesday: int = _int("DAY_MIN_CONVICTION_WEDNESDAY", "65")
    day_min_conviction_thursday: int = _int("DAY_MIN_CONVICTION_THURSDAY", "70")
    day_min_conviction_friday: int = _int("DAY_MIN_CONVICTION_FRIDAY", "999")

    already_ran_filter_enabled: bool = _bool("ALREADY_RAN_FILTER_ENABLED", "true")
    already_ran_capture_ratio: float = _float("ALREADY_RAN_CAPTURE_RATIO", "0.90")
    already_ran_min_history_sample_size: int = _int("ALREADY_RAN_MIN_HISTORY_SAMPLE_SIZE", "50")

    ticker_penalty_enabled: bool = _bool("TICKER_PENALTY_ENABLED", "true")
    ticker_penalty_summary_path: str = os.getenv("TICKER_PENALTY_SUMMARY_PATH", "data/history_ticker_summary.csv")
    ticker_penalty_min_sample_size: int = _int("TICKER_PENALTY_MIN_SAMPLE_SIZE", "50")
    ticker_penalty_min_success_rate: float = _float("TICKER_PENALTY_MIN_SUCCESS_RATE", "55.0")
    ticker_penalty_max_adverse_move_pct: float = _float("TICKER_PENALTY_MAX_ADVERSE_MOVE_PCT", "6.0")
    ticker_penalty_block_conviction_below: int = _int("TICKER_PENALTY_BLOCK_CONVICTION_BELOW", "75")

    gap_risk_filter_enabled: bool = _bool("GAP_RISK_FILTER_ENABLED", "true")
    gap_risk_max_overnight_gap_pct: float = _float("GAP_RISK_MAX_OVERNIGHT_GAP_PCT", "3.0")
    gap_risk_block_conviction_below: int = _int("GAP_RISK_BLOCK_CONVICTION_BELOW", "75")

    rank_stability_filter_enabled: bool = _bool("RANK_STABILITY_FILTER_ENABLED", "false")
    rank_stability_min_seen_count: int = _int("RANK_STABILITY_MIN_SEEN_COUNT", "2")

    # Simulation-backed live guardrail built from the two-year no-weekend Xanax simulation.
    simulation_filter_enabled: bool = _bool("SIMULATION_FILTER_ENABLED", "true")
    simulation_filter_summary_path: str = os.getenv("SIMULATION_FILTER_SUMMARY_PATH", "data/simulation_live_filter_summary.csv")
    simulation_filter_min_sample_size: int = _int("SIMULATION_FILTER_MIN_SAMPLE_SIZE", "20")
    simulation_filter_block_enabled: bool = _bool("SIMULATION_FILTER_BLOCK_ENABLED", "true")
    simulation_filter_max_adjustment: int = _int("SIMULATION_FILTER_MAX_ADJUSTMENT", "10")
    simulation_filter_high_conviction_override: int = _int("SIMULATION_FILTER_HIGH_CONVICTION_OVERRIDE", "90")

    earnings_risk_filter_enabled: bool = _bool("EARNINGS_RISK_FILTER_ENABLED", "true")
    earnings_risk_days: int = _int("EARNINGS_RISK_DAYS", "2")
    earnings_risk_min_conviction: int = _int("EARNINGS_RISK_MIN_CONVICTION", "85")

    # Tuesday/Wednesday second-chance buying. This is stricter than Monday buying.
    second_chance_min_conviction: int = _int("SECOND_CHANCE_MIN_CONVICTION", "70")
    second_chance_min_history_rate: float = _float("SECOND_CHANCE_MIN_HISTORY_RATE", "70.0")
    second_chance_min_history_sample_size: int = _int("SECOND_CHANCE_MIN_HISTORY_SAMPLE_SIZE", "50")
    second_chance_max_spread_pct: float = _float("SECOND_CHANCE_MAX_SPREAD_PCT", "0.25")
    second_chance_require_intraday_confirmation: bool = _bool("SECOND_CHANCE_REQUIRE_INTRADAY_CONFIRMATION", "true")
    second_chance_require_moderate_edge: bool = _bool("SECOND_CHANCE_REQUIRE_MODERATE_EDGE", "true")



    # Tuned weekly strategy filters.
    tuned_prefer_long_up: bool = _bool("TUNED_PREFER_LONG_UP", "true")
    tuned_block_weak_shorts: bool = _bool("TUNED_BLOCK_WEAK_SHORTS", "true")
    tuned_long_momentum_min_history_rate: float = _float("TUNED_LONG_MOMENTUM_MIN_HISTORY_RATE", "60.0")
    tuned_up_opportunity_min_conviction: int = _int("TUNED_UP_OPPORTUNITY_MIN_CONVICTION", "82")
    tuned_up_opportunity_min_history_rate: float = _float("TUNED_UP_OPPORTUNITY_MIN_HISTORY_RATE", "75.0")
    tuned_short_min_conviction: int = _int("TUNED_SHORT_MIN_CONVICTION", "88")
    tuned_short_min_history_rate: float = _float("TUNED_SHORT_MIN_HISTORY_RATE", "78.0")
    tuned_short_min_history_sample_size: int = _int("TUNED_SHORT_MIN_HISTORY_SAMPLE_SIZE", "100")
    tuned_short_max_spread_pct: float = _float("TUNED_SHORT_MAX_SPREAD_PCT", "0.20")
    tuned_short_require_strong_edge: bool = _bool("TUNED_SHORT_REQUIRE_STRONG_EDGE", "true")

    # Options are allowed only for high-quality LONG/UP momentum setups.
    options_long_up_only: bool = _bool("OPTIONS_LONG_UP_ONLY", "true")
    options_long_up_min_conviction: int = _int("OPTIONS_LONG_UP_MIN_CONVICTION", "78")
    options_long_up_min_history_rate: float = _float("OPTIONS_LONG_UP_MIN_HISTORY_RATE", "72.0")
    options_long_up_min_history_sample_size: int = _int("OPTIONS_LONG_UP_MIN_HISTORY_SAMPLE_SIZE", "100")
    options_long_up_allowed_setup: str = os.getenv("OPTIONS_LONG_UP_ALLOWED_SETUP", "MOMENTUM_CONTINUATION").strip().upper()
    options_both_prefers_option_for_qualified_longs: bool = _bool("OPTIONS_BOTH_PREFERS_OPTION_FOR_QUALIFIED_LONGS", "true")

    # Keep this true. It prevents Monday same-day exits unless the old exit engine reports emergency/friday risk.
    monitor_block_same_day_exit: bool = _bool("MONITOR_BLOCK_SAME_DAY_EXIT", "true")

    # Continuous paper-trading loop. Keep disabled until you intentionally run live-loop.
    live_loop_enabled: bool = _bool("LIVE_LOOP_ENABLED", "false")
    live_loop_model_scan_interval_minutes: int = _int("LIVE_LOOP_MODEL_SCAN_INTERVAL_MINUTES", "60")
    live_loop_position_monitor_interval_minutes: int = _int("LIVE_LOOP_POSITION_MONITOR_INTERVAL_MINUTES", "5")
    live_loop_no_new_trades_after_hour: int = _int("LIVE_LOOP_NO_NEW_TRADES_AFTER_HOUR", "15")
    live_loop_no_new_trades_after_minute: int = _int("LIVE_LOOP_NO_NEW_TRADES_AFTER_MINUTE", "0")
    live_loop_max_new_trades_per_day: int = _int("LIVE_LOOP_MAX_NEW_TRADES_PER_DAY", "5")
    live_loop_require_market_open: bool = _bool("LIVE_LOOP_REQUIRE_MARKET_OPEN", "true")

    @property
    def db_abspath(self) -> Path:
        path = Path(self.database_path)
        if path.is_absolute():
            return path
        return ROOT_DIR / path


settings = Settings()
