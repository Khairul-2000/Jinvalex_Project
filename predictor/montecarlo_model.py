"""
Monte Carlo forecaster for a single (card, field) series.

Unlike the other three models there is NO training and NO fitting. We:
  1. measure the historical daily log-returns of the series,
  2. simulate thousands of random future price paths from that distribution
     (Geometric Brownian Motion — prices stay positive, moves scale with level),
  3. read the forecast and interval straight off the simulated distribution:
       yhat        = median path value per day
       lower/upper = 10th / 90th percentile  (-> ~80% band, matching others)

Strengths: dirt cheap, no model to persist, gives a full distribution and
naturally widening uncertainty. Weakness: it has no concept of trend direction
beyond the average drift, so it's best as an uncertainty/ensemble component.

Interface mirrors the other models:
    fit_forecast(series, horizon) -> DataFrame[ds, yhat, yhat_lower, yhat_upper]
    backtest(series, holdout)     -> {mae, mape, n} | None
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BACKTEST_DAYS, FORECAST_HORIZON_DAYS

# Number of simulated paths. More = smoother percentiles, slightly slower.
N_SIMS = 2000

# Percentiles for the prediction interval (10/90 -> 80% band).
LOWER_PCT = 10
UPPER_PCT = 90

# Cap per-step drift so a short noisy history can't extrapolate to the moon.
MAX_DAILY_DRIFT = 0.05  # +/-5% per day


def _log_returns(y: np.ndarray) -> np.ndarray:
    """Daily log-returns, with non-finite values removed."""
    y = y[y > 0]
    lr = np.diff(np.log(y))
    return lr[np.isfinite(lr)]


def fit_forecast(
    series: pd.DataFrame,
    horizon: int = FORECAST_HORIZON_DAYS,
    n_sims: int = N_SIMS,
    seed: int | None = 42,
) -> pd.DataFrame | None:
    """
    Simulate `n_sims` GBM paths `horizon` days forward and summarise them.
    Returns ds, yhat (median), yhat_lower (P10), yhat_upper (P90), or None.
    """
    series = series.sort_values("ds").reset_index(drop=True)
    y = series["y"].to_numpy(dtype=float)
    last_price = y[-1]
    last_ds = series["ds"].iloc[-1]

    lr = _log_returns(y)
    if len(lr) < 5 or last_price <= 0:
        return None

    mu = float(np.clip(np.mean(lr), -MAX_DAILY_DRIFT, MAX_DAILY_DRIFT))
    sigma = float(np.std(lr))
    if sigma == 0:
        sigma = 1e-6  # degenerate flat series; keep paths from collapsing

    rng = np.random.default_rng(seed)
    # shocks: shape (n_sims, horizon)
    shocks = rng.normal(loc=mu, scale=sigma, size=(n_sims, horizon))
    # cumulative log-return per path, then back to price level
    cum = np.cumsum(shocks, axis=1)
    paths = last_price * np.exp(cum)  # (n_sims, horizon)

    median = np.median(paths, axis=0)
    lower = np.percentile(paths, LOWER_PCT, axis=0)
    upper = np.percentile(paths, UPPER_PCT, axis=0)

    future_ds = pd.date_range(
        last_ds + pd.Timedelta(days=1), periods=horizon, freq="D"
    )
    return pd.DataFrame(
        {
            "ds": future_ds,
            "yhat": np.clip(median, 0, None).round(2),
            "yhat_lower": np.clip(lower, 0, None).round(2),
            "yhat_upper": np.clip(upper, 0, None).round(2),
        }
    )


def backtest(
    series: pd.DataFrame, holdout: int = BACKTEST_DAYS
) -> dict | None:
    """
    Hold out the last `holdout` days, simulate them from the truncated history,
    and compare the median path to actuals. Honest: the holdout is excluded from
    the return-distribution estimate.
    """
    if len(series) <= holdout + 6:
        return None
    train = series.iloc[:-holdout]
    test = series.iloc[-holdout:]

    fc = fit_forecast(train, horizon=holdout)
    if fc is None:
        return None
    merged = test.merge(fc, on="ds", how="inner")
    if merged.empty:
        return None

    err = merged["y"].to_numpy() - merged["yhat"].to_numpy()
    mae = float(np.mean(np.abs(err)))
    denom = merged["y"].replace(0, np.nan).to_numpy()
    mape = float(np.nanmean(np.abs(err) / denom) * 100)
    return {"mae": round(mae, 4), "mape": round(mape, 2), "n": len(merged)}
