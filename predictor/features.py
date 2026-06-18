"""
Feature engineering for the global XGBoost model.

Prophet fits one model per series. XGBoost is the opposite: we flatten EVERY
(card, field) series into rows of [features -> next-day price], stack them all
into one table, and train a single model across the whole dataset.

Each row answers: "given the recent behaviour of this series (and which series
it is), what is the price on the target day?"

Feature groups
--------------
  lags        : price 1, 2, 3, 7, 14, 30 days ago
  rolling     : mean / std / min / max over 7 and 30 day windows
  momentum    : pct change over 1, 7, 14 days
  calendar    : day-of-week, day-of-month, month
  identity    : card_id, price_field (encoded) -> lets one model serve all series
  meta        : days since the series started (age/maturity signal)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Which past offsets to expose as raw lag features.
LAG_DAYS = [1, 2, 3, 7, 14, 30]
ROLL_WINDOWS = [7, 30]
PCT_WINDOWS = [1, 7, 14]

# The feature columns the model trains on (identity cols added separately).
LAG_COLS = [f"lag_{d}" for d in LAG_DAYS]
ROLL_COLS = (
    [f"roll_mean_{w}" for w in ROLL_WINDOWS]
    + [f"roll_std_{w}" for w in ROLL_WINDOWS]
    + [f"roll_min_{w}" for w in ROLL_WINDOWS]
    + [f"roll_max_{w}" for w in ROLL_WINDOWS]
)
PCT_COLS = [f"pct_{w}" for w in PCT_WINDOWS]
CAL_COLS = ["dow", "dom", "month", "series_age"]

FEATURE_COLS = LAG_COLS + ROLL_COLS + PCT_COLS + CAL_COLS + ["field_code", "card_id"]


def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    ds = df["ds"]
    df["dow"] = ds.dt.dayofweek
    df["dom"] = ds.dt.day
    df["month"] = ds.dt.month
    df["series_age"] = (ds - ds.min()).dt.days
    return df


def build_features_for_series(
    series: pd.DataFrame, card_id: int, field_code: int
) -> pd.DataFrame:
    """
    Given one clean (ds, y) series (already regularised by preprocess.py),
    produce a feature table where each row's target `y` is that day's price
    and the features describe the days leading up to it.

    Returns an empty frame if the series is too short to form any complete row.
    """
    df = series.copy().sort_values("ds").reset_index(drop=True)

    for d in LAG_DAYS:
        df[f"lag_{d}"] = df["y"].shift(d)

    for w in ROLL_WINDOWS:
        # shift(1) so the window uses only PAST values, never the target day.
        past = df["y"].shift(1)
        df[f"roll_mean_{w}"] = past.rolling(w).mean()
        df[f"roll_std_{w}"] = past.rolling(w).std()
        df[f"roll_min_{w}"] = past.rolling(w).min()
        df[f"roll_max_{w}"] = past.rolling(w).max()

    for w in PCT_WINDOWS:
        df[f"pct_{w}"] = df["y"].pct_change(w)

    df = _add_calendar(df)
    df["field_code"] = field_code
    df["card_id"] = card_id

    # Drop rows that don't have a full lag/rolling history yet.
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS + ["y"])
    return df


def latest_feature_row(
    series: pd.DataFrame, card_id: int, field_code: int
) -> pd.Series | None:
    """
    Build the single feature row representing "the day AFTER the last observation"
    — i.e. the first day we want to forecast. Used to kick off recursive
    multi-step forecasting at inference time.
    """
    df = series.copy().sort_values("ds").reset_index(drop=True)
    next_ds = df["ds"].iloc[-1] + pd.Timedelta(days=1)

    # Append a placeholder row for next_ds so lags/rolls compute against history.
    df = pd.concat(
        [df, pd.DataFrame({"ds": [next_ds], "y": [np.nan]})], ignore_index=True
    )

    for d in LAG_DAYS:
        df[f"lag_{d}"] = df["y"].shift(d)
    for w in ROLL_WINDOWS:
        past = df["y"].shift(1)
        df[f"roll_mean_{w}"] = past.rolling(w).mean()
        df[f"roll_std_{w}"] = past.rolling(w).std()
        df[f"roll_min_{w}"] = past.rolling(w).min()
        df[f"roll_max_{w}"] = past.rolling(w).max()
    for w in PCT_WINDOWS:
        df[f"pct_{w}"] = df["y"].pct_change(w)

    df = _add_calendar(df)
    df["field_code"] = field_code
    df["card_id"] = card_id

    row = df.iloc[-1]
    if row[FEATURE_COLS].isna().any():
        return None
    return row
