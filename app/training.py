from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression

from app.config import settings
from app.features import build_training_rows, current_week_feature_row, prepare_feature_frame
from app.fetchers.market_data import fetch_daily_history, fetch_vix_history

FEATURE_COLUMNS = [
    "prev_1d_return",
    "prev_5d_return",
    "prev_20d_return",
    "rsi_14",
    "zscore_20",
    "ma_gap_5_20",
    "ma_gap_10_50",
    "realized_vol_20",
    "vix_close",
    "vix_zscore_20",
    "macd_hist",
]


@dataclass
class TrainedBundle:
    logistic: LogisticRegression
    gbt_classifier: HistGradientBoostingClassifier
    rf_classifier: RandomForestClassifier
    regressor: HistGradientBoostingRegressor
    current_row: dict
    validation: dict
    training_rows: int


def _directional_accuracy(y_true: pd.Series, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def _majority_vote(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    out = []
    for x, y, z in zip(a, b, c):
        votes = [x, y, z]
        counts = Counter(votes)
        top_count = max(counts.values())
        winners = [label for label, count in counts.items() if count == top_count]
        out.append(x if len(winners) > 1 else winners[0])
    return np.asarray(out)


def train_for_ticker(ticker: str) -> TrainedBundle:
    price_df = fetch_daily_history(ticker, years=settings.training_years)
    vix_df = fetch_vix_history(years=settings.training_years)
    feat_df = prepare_feature_frame(price_df, vix_df)
    training = build_training_rows(price_df, feat_df, settings.opportunity_threshold_pct / 100.0)

    if len(training) < settings.min_training_rows:
        raise ValueError(f"Not enough training rows for {ticker}: {len(training)}")

    split = max(settings.min_training_rows, int(len(training) * 0.80))
    split = min(split, len(training) - 1) if len(training) > 1 else len(training)

    train_df = training.iloc[:split].copy()
    valid_df = training.iloc[split:].copy()
    if len(valid_df) == 0:
        valid_df = training.iloc[-12:].copy()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["target_class"]
    y_ret = train_df["target_return"]

    logistic = LogisticRegression(
        max_iter=1500,
        C=0.7,
        class_weight="balanced",
        solver="lbfgs",
    )
    gbt = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=4,
        max_iter=300,
        l2_regularization=0.12,
        random_state=42,
    )
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced_subsample",
    )
    reg = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=3,
        max_iter=250,
        l2_regularization=0.08,
        random_state=42,
    )

    logistic.fit(X_train, y_train)
    gbt.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    reg.fit(X_train, y_ret)

    X_valid = valid_df[FEATURE_COLUMNS]
    log_pred = logistic.predict(X_valid)
    gbt_pred = gbt.predict(X_valid)
    rf_pred = rf.predict(X_valid)
    ensemble_pred = _majority_vote(log_pred, gbt_pred, rf_pred)

    validation = {
        "rows": int(len(valid_df)),
        "logistic_accuracy": round(_directional_accuracy(valid_df["target_class"], log_pred), 4),
        "gbt_accuracy": round(_directional_accuracy(valid_df["target_class"], gbt_pred), 4),
        "rf_accuracy": round(_directional_accuracy(valid_df["target_class"], rf_pred), 4),
        "ensemble_accuracy": round(_directional_accuracy(valid_df["target_class"], ensemble_pred), 4),
    }

    current = current_week_feature_row(price_df, feat_df, ticker)
    row_dict = current.__dict__.copy()
    idx = pd.Timestamp(current.asof_date)
    row_dict["macd_hist"] = float(feat_df.loc[idx, "macd_hist"]) if idx in feat_df.index else float(feat_df.iloc[-1]["macd_hist"])

    return TrainedBundle(
        logistic=logistic,
        gbt_classifier=gbt,
        rf_classifier=rf,
        regressor=reg,
        current_row=row_dict,
        validation=validation,
        training_rows=int(len(training)),
    )