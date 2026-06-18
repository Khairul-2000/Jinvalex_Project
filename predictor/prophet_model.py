"""
Prophet wrapper for a single (card, field) series.

Kept deliberately thin so the same interface can later be reused for
SARIMA / XGBoost / Monte Carlo: each just needs fit_forecast() and backtest().
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from prophet import Prophet

from .config import BACKTEST_DAYS, FORECAST_HORIZON_DAYS

# Prophet/cmdstanpy are very chatty; silence the per-fit logging.
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


def _build_model() -> Prophet:
    """
    Daily TCG prices: weekly seasonality is plausible (weekend buying),
    yearly only matters with >1yr history (Prophet auto-disables if short).
    multiplicative seasonality suits prices that swing proportionally to level.
    """
    return Prophet(
        weekly_seasonality=True,
        yearly_seasonality="auto",
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        interval_width=0.80,  # 80% prediction interval -> lower/upper bounds
    )


def fit_forecast(series: pd.DataFrame, horizon: int = FORECAST_HORIZON_DAYS) -> pd.DataFrame:
    """
    Fit on the full series and forecast `horizon` future days.

    Returns only the FUTURE rows with columns:
        ds, yhat, yhat_lower, yhat_upper
    Negative forecasts are clipped to 0 (a price can't be negative).
    """
    model = _build_model()
    model.fit(series)

    future = model.make_future_dataframe(periods=horizon, freq="D")
    fc = model.predict(future)

    future_only = fc.tail(horizon)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        future_only[col] = future_only[col].clip(lower=0).round(2)
    return future_only


def backtest(series: pd.DataFrame, holdout: int = BACKTEST_DAYS) -> dict | None:
    """
    Train on everything except the last `holdout` days, predict them,
    and compare to actuals. Returns error metrics or None if too short.

    MAPE is the headline number you store to decide how much to trust
    this series' forecast (and later, which model wins per series).
    """
    if len(series) <= holdout + 5:
        return None

    train = series.iloc[:-holdout]
    test = series.iloc[-holdout:]

    try:
        fc = fit_forecast(train, horizon=holdout)
    except Exception:
        return None

    merged = test.merge(fc, on="ds", how="inner")
    if merged.empty:
        return None

    err = merged["y"] - merged["yhat"]
    mae = float(np.mean(np.abs(err)))
    # guard against divide-by-zero on near-zero prices
    mape = float(np.mean(np.abs(err) / merged["y"].replace(0, np.nan)) * 100)

    return {"mae": round(mae, 4), "mape": round(mape, 2), "n": len(merged)}
