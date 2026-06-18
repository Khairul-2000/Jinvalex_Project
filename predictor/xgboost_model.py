"""
Global XGBoost forecaster.

Unlike Prophet (one model per series), this trains ONE model on features from
every (card, field) series. At inference it forecasts 7 days recursively:
predict day+1, feed that back in as the newest lag, predict day+2, etc.

Public surface mirrors prophet_model.py so the pipeline/ensemble treat models
uniformly:
    train_global(...)        -> fit + persist one model over all series
    forecast_series(...)     -> 7-day recursive forecast for one series
    load_model() / save      -> persistence
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .config import FORECAST_HORIZON_DAYS, PRICE_FIELDS
from .features import (
    FEATURE_COLS,
    LAG_DAYS,
    PCT_WINDOWS,
    ROLL_WINDOWS,
    build_features_for_series,
)

MODEL_NAME = "xgboost"
MODEL_PATH = os.environ.get(
    "TCG_XGB_PATH", os.path.join(os.path.dirname(__file__), "xgb_model.json")
)

# Stable integer code per price field so the model can tell series apart.
FIELD_CODE = {f: i for i, f in enumerate(PRICE_FIELDS)}


def _new_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def assemble_training_table(series_iter) -> pd.DataFrame:
    """
    series_iter yields (series_df, card_id, field_name) tuples.
    Returns one stacked feature table across all of them.
    """
    frames = []
    for series, card_id, field in series_iter:
        code = FIELD_CODE.get(field)
        if code is None:
            continue
        feats = build_features_for_series(series, card_id, code)
        if not feats.empty:
            frames.append(feats)
    if not frames:
        return pd.DataFrame(columns=FEATURE_COLS + ["y"])
    return pd.concat(frames, ignore_index=True)


def train_global(training_table: pd.DataFrame, save: bool = True) -> XGBRegressor:
    """
    Fit one model on the stacked table. Uses a time-agnostic fit (rows already
    encode their own history via lags), evaluated separately by backtest.
    """
    if training_table.empty:
        raise ValueError("empty training table — no series produced features")

    X = training_table[FEATURE_COLS]
    y = training_table["y"]

    model = _new_model()
    model.fit(X, y)

    if save:
        model.save_model(MODEL_PATH)
    return model


def load_model() -> XGBRegressor | None:
    if not os.path.exists(MODEL_PATH):
        return None
    model = _new_model()
    model.load_model(MODEL_PATH)
    return model


# ---------------------------------------------------------------------------
# Recursive multi-step forecasting
# ---------------------------------------------------------------------------

def _features_from_history(hist_y: list[float], next_ds: pd.Timestamp,
                           start_ds: pd.Timestamp, card_id: int,
                           field_code: int) -> pd.DataFrame | None:
    """
    Build a single feature row for `next_ds` given a running list of past y
    values (most recent last). Returns a 1-row DataFrame in FEATURE_COLS order,
    or None if history is too short for a complete row.
    """
    n = len(hist_y)
    row = {}

    for d in LAG_DAYS:
        if n - d < 0:
            return None
        row[f"lag_{d}"] = hist_y[n - d]

    arr = np.array(hist_y, dtype=float)
    for w in ROLL_WINDOWS:
        if n < w:
            return None
        window = arr[n - w:]
        row[f"roll_mean_{w}"] = window.mean()
        row[f"roll_std_{w}"] = window.std(ddof=1) if w > 1 else 0.0
        row[f"roll_min_{w}"] = window.min()
        row[f"roll_max_{w}"] = window.max()

    for w in PCT_WINDOWS:
        if n - w - 1 < 0 or arr[n - w - 1] == 0:
            return None
        row[f"pct_{w}"] = (arr[-1] - arr[n - w - 1]) / arr[n - w - 1]

    row["dow"] = next_ds.dayofweek
    row["dom"] = next_ds.day
    row["month"] = next_ds.month
    row["series_age"] = (next_ds - start_ds).days
    row["field_code"] = field_code
    row["card_id"] = card_id

    return pd.DataFrame([row])[FEATURE_COLS]


def forecast_series(
    model: XGBRegressor,
    series: pd.DataFrame,
    card_id: int,
    field: str,
    horizon: int = FORECAST_HORIZON_DAYS,
) -> pd.DataFrame | None:
    """
    Recursively forecast `horizon` future days for one series.

    Returns columns ds, yhat, yhat_lower, yhat_upper (bounds derived from the
    series' recent residual scale, since a point regressor has no native
    interval). None if the series can't seed a forecast.
    """
    code = FIELD_CODE.get(field)
    if code is None:
        return None

    df = series.sort_values("ds").reset_index(drop=True)
    start_ds = df["ds"].iloc[0]
    hist_y = df["y"].tolist()
    last_ds = df["ds"].iloc[-1]

    # Rough uncertainty: recent volatility of day-to-day changes.
    diffs = np.diff(np.array(hist_y[-30:], dtype=float))
    sigma = float(np.std(diffs)) if len(diffs) > 1 else 0.0

    preds = []
    for step in range(1, horizon + 1):
        next_ds = last_ds + pd.Timedelta(days=step)
        feat = _features_from_history(hist_y, next_ds, start_ds, card_id, code)
        if feat is None:
            return None
        yhat = float(model.predict(feat)[0])
        yhat = max(yhat, 0.0)
        # widen the band as we step further out (uncertainty compounds).
        band = sigma * np.sqrt(step) * 1.28  # ~80% if changes ~normal
        preds.append(
            {
                "ds": next_ds,
                "yhat": round(yhat, 2),
                "yhat_lower": round(max(yhat - band, 0.0), 2),
                "yhat_upper": round(yhat + band, 2),
            }
        )
        hist_y.append(yhat)  # feed prediction back in for the next step

    return pd.DataFrame(preds)


def backtest_series(
    model: XGBRegressor, series: pd.DataFrame, card_id: int, field: str,
    holdout: int = FORECAST_HORIZON_DAYS,
) -> dict | None:
    """
    Hold out the last `holdout` days, recursively forecast them from the
    truncated history, and compare to actuals. NOTE: for an honest backtest the
    model should have been trained without these days; this is a quick sanity
    metric, not leakage-free CV.
    """
    if len(series) <= holdout + max(LAG_DAYS) + max(ROLL_WINDOWS):
        return None
    train = series.iloc[:-holdout]
    test = series.iloc[-holdout:]

    fc = forecast_series(model, train, card_id, field, horizon=holdout)
    if fc is None:
        return None
    merged = test.merge(fc, on="ds", how="inner")
    if merged.empty:
        return None

    err = merged["y"].values - merged["yhat"].values
    mae = float(np.mean(np.abs(err)))
    denom = merged["y"].replace(0, np.nan).values
    mape = float(np.nanmean(np.abs(err) / denom) * 100)
    return {"mae": round(mae, 4), "mape": round(mape, 2), "n": len(merged)}
