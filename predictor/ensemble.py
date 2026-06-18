"""
Ensemble — combine the four models into one final prediction per
(card, field, date), written back as model_used='ensemble'.

This is what your app should actually display: a single number per card/field
with one uncertainty band, instead of four competing numbers.

Weighting strategy
------------------
A flat average treats a great model and a poor one equally. Instead we weight
each model by how accurate it has been *on this specific series*, measured by
its backtest MAPE (lower error -> higher weight). The best model differs by
series — Prophet on seasonal cards, Monte Carlo on volatile ones — so per-series
weighting beats any single global choice.

  weight_m = 1 / (mape_m + eps)        (inverse-error)
  yhat     = sum(weight_m * yhat_m) / sum(weight_m)

If no backtest scores are available (e.g. too little history), we fall back to
an equal-weight average of whatever models produced a prediction.

The combined band is the weighted average of the per-model bands, then widened
to at least span the disagreement between models (if models diverge, that itself
is uncertainty worth showing).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-3  # avoids divide-by-zero when a model has ~0% error


def _model_weights(scores: dict[str, float] | None, models: list[str]) -> dict[str, float]:
    """
    Turn per-model MAPE scores into normalised weights for the models present.
    Falls back to equal weights when scores are missing.
    """
    if not scores:
        w = {m: 1.0 for m in models}
    else:
        w = {}
        for m in models:
            mape = scores.get(m)
            if mape is None or not np.isfinite(mape):
                # model present but un-scored -> give it the average-ish weight
                w[m] = 1.0
            else:
                w[m] = 1.0 / (mape + EPS)
    total = sum(w.values()) or 1.0
    return {m: v / total for m, v in w.items()}


def ensemble_for_card(
    preds: pd.DataFrame,
    scores_by_field: dict[str, dict[str, float]] | None = None,
) -> list[dict]:
    """
    preds: all per-model rows for ONE card (from db.load_model_predictions).
    scores_by_field: optional {price_field: {model_used: mape}} to weight by.

    Returns a list of ensemble row dicts ready for db.save_predictions.
    """
    if preds.empty:
        return []

    rows: list[dict] = []

    # one ensemble value per (field, date) across whatever models exist there
    for (field, date), grp in preds.groupby(["price_field", "predict_date"]):
        models = grp["model_used"].tolist()
        field_scores = (scores_by_field or {}).get(field)
        weights = _model_weights(field_scores, models)

        w = grp["model_used"].map(weights).to_numpy()
        yhat_m = grp["predicted_value"].to_numpy(dtype=float)
        lo_m = grp["lower_bound"].to_numpy(dtype=float)
        hi_m = grp["upper_bound"].to_numpy(dtype=float)

        wsum = w.sum() or 1.0
        yhat = float(np.sum(w * yhat_m) / wsum)
        lower = float(np.sum(w * lo_m) / wsum)
        upper = float(np.sum(w * hi_m) / wsum)

        # widen the band to also cover model disagreement on the point estimate
        lower = min(lower, float(np.min(yhat_m)))
        upper = max(upper, float(np.max(yhat_m)))

        rows.append(
            {
                "card_id": int(grp["card_id"].iloc[0]),
                "price_field": field,
                "predict_date": date.date() if hasattr(date, "date") else date,
                "predicted_value": round(max(yhat, 0.0), 2),
                "lower_bound": round(max(lower, 0.0), 2),
                "upper_bound": round(max(upper, 0.0), 2),
                "model_used": "ensemble",
            }
        )
    return rows
