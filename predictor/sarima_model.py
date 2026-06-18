"""
SARIMA wrapper for a single (card, field) series.

Per-series like Prophet, but classical: it models autocorrelation (AR),
differencing for trend (I), moving-average errors (MA), plus a seasonal
counterpart (weekly, s=7). Orders are chosen automatically per series with
pmdarima.auto_arima, because hand-tuning thousands of series is infeasible.

Interface mirrors prophet_model.py:
    fit_forecast(series, horizon) -> DataFrame[ds, yhat, yhat_lower, yhat_upper]
    backtest(series, holdout)     -> {mae, mape, n} | None
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# pmdarima/statsmodels emit many convergence/runtime warnings on short or
# flat series. They're expected during the order search; silence them.
warnings.filterwarnings("ignore")

import pmdarima as pm  # noqa: E402

from .config import BACKTEST_DAYS, FORECAST_HORIZON_DAYS  # noqa: E402

# Weekly seasonality. Daily TCG prices plausibly have a 7-day cycle (weekend
# buying). If a series is shorter than 2 full seasons auto_arima drops seasonality.
SEASONAL_PERIOD = 7
...

# 80% interval to match the bounds produced by the other models.
INTERVAL_ALPHA = 0.20


def _fit_auto(y: np.ndarray):
    """
    Run auto_arima with a bounded search so it stays fast across many series.
    Seasonality is only attempted when there's enough data for it.
    """
    use_seasonal = len(y) >= 2 * SEASONAL_PERIOD
    return pm.auto_arima(
        y,
        seasonal=use_seasonal,
        m=SEASONAL_PERIOD if use_seasonal else 1,
        start_p=0, start_q=0, max_p=3, max_q=3,
        max_P=1, max_Q=1, max_d=2, max_D=1,
        stepwise=True,               # fast heuristic search, not full grid
        suppress_warnings=True,
        error_action="ignore",       # skip orders that fail to fit
        information_criterion="aic",
    )


def fit_forecast(
    series: pd.DataFrame, horizon: int = FORECAST_HORIZON_DAYS
) -> pd.DataFrame | None:
    """
    Fit SARIMA on the full series and forecast `horizon` future days.

    Returns future rows with ds, yhat, yhat_lower, yhat_upper (clipped >= 0),
    or None if the model can't be fit.
    """
    series = series.sort_values("ds").reset_index(drop=True)
    y = series["y"].to_numpy(dtype=float)
    last_ds = series["ds"].iloc[-1]

    try:
        model = _fit_auto(y)
        mean, conf = model.predict(
            n_periods=horizon, return_conf_int=True, alpha=INTERVAL_ALPHA
        )
    except Exception:
        return None

    future_ds = pd.date_range(
        last_ds + pd.Timedelta(days=1), periods=horizon, freq="D"
    )
    out = pd.DataFrame(
        {
            "ds": future_ds,
            "yhat": np.clip(mean, 0, None).round(2),
            "yhat_lower": np.clip(conf[:, 0], 0, None).round(2),
            "yhat_upper": np.clip(conf[:, 1], 0, None).round(2),
        }
    )
    return out


def backtest(
    series: pd.DataFrame, holdout: int = BACKTEST_DAYS
) -> dict | None:
    """
    Train on all but the last `holdout` days, forecast them, compare to actuals.
    Honest here (unlike the global XGBoost backtest) because each SARIMA fit is
    per-series and the holdout is genuinely excluded from this fit.
    """
    if len(series) <= holdout + 2 * SEASONAL_PERIOD:
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
