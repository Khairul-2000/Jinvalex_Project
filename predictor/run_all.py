"""
Master entrypoint — run the whole prediction pipeline end to end.

This is the single command your production scheduler calls:

    python -m predictor.run_all                 # everything, all cards
    python -m predictor.run_all --card 1142912  # everything, one card
    python -m predictor.run_all --skip xgboost  # skip a model (e.g. when thin data)

Order:
    1. Prophet      (per-series)      -> model_used='prophet'
    2. SARIMA       (per-series)      -> model_used='sarima'
    3. Monte Carlo  (per-series)      -> model_used='montecarlo'
    4. XGBoost      (global train+predict) -> model_used='xgboost'
    5. Ensemble     (combine the above)     -> model_used='ensemble'

Each model writes its own rows; the ensemble reads them back and writes the
final blended row your app displays.

Backtest scores from each per-series model are collected during their runs and
fed to the ensemble as weights, so more-accurate models count for more.
"""
from __future__ import annotations

import argparse
import logging
import sys

from . import db
from .config import FORECAST_HORIZON_DAYS, PRICE_FIELDS
from .preprocess import prepare_series
from .ensemble import ensemble_for_card

# per-series models share one interface (fit_forecast + backtest)
from . import prophet_model, sarima_model, montecarlo_model
from . import xgboost_model
from .run_xgboost import do_train as xgb_train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_all")

# name -> (module, model_used string)
PER_SERIES_MODELS = {
    "prophet": prophet_model,
    "sarima": sarima_model,
    "montecarlo": montecarlo_model,
}


def _run_per_series_models(card_id, series, field, skip, scores):
    """Run each per-series model on one series, collecting rows + backtest scores."""
    rows = []
    for name, mod in PER_SERIES_MODELS.items():
        if name in skip:
            continue
        m = mod.backtest(series)
        if m:
            scores.setdefault(field, {})[name] = m["mape"]
        fc = mod.fit_forecast(series, horizon=FORECAST_HORIZON_DAYS)
        if fc is None:
            continue
        for _, r in fc.iterrows():
            rows.append({
                "card_id": card_id, "price_field": field,
                "predict_date": r["ds"].date(),
                "predicted_value": float(r["yhat"]),
                "lower_bound": float(r["yhat_lower"]),
                "upper_bound": float(r["yhat_upper"]),
                "model_used": name,
            })
    return rows


def run(card_id=None, skip=None):
    skip = set(skip or [])
    db.ensure_predictions_table()

    # XGBoost trains once globally before predicting (unless skipped).
    xgb_model = None
    if "xgboost" not in skip:
        log.info("training global XGBoost model...")
        if xgb_train():
            xgb_model = xgboost_model.load_model()
        else:
            log.warning("XGBoost training produced no model; skipping it.")

    card_ids = [card_id] if card_id else db.load_eligible_card_ids()
    log.info("running full pipeline on %d card(s)", len(card_ids))

    total = 0
    for i, cid in enumerate(card_ids, 1):
        history = db.load_history(cid)
        if history.empty:
            continue

        scores: dict[str, dict[str, float]] = {}
        all_rows = []

        for field in PRICE_FIELDS:
            series = prepare_series(history, field)
            if series is None:
                continue

            # per-series models
            all_rows += _run_per_series_models(cid, series, field, skip, scores)

            # global xgboost
            if xgb_model is not None:
                bt = xgboost_model.backtest_series(xgb_model, series, cid, field)
                if bt:
                    scores.setdefault(field, {})["xgboost"] = bt["mape"]
                fc = xgboost_model.forecast_series(
                    xgb_model, series, cid, field, horizon=FORECAST_HORIZON_DAYS
                )
                if fc is not None:
                    for _, r in fc.iterrows():
                        all_rows.append({
                            "card_id": cid, "price_field": field,
                            "predict_date": r["ds"].date(),
                            "predicted_value": float(r["yhat"]),
                            "lower_bound": float(r["yhat_lower"]),
                            "upper_bound": float(r["yhat_upper"]),
                            "model_used": "xgboost",
                        })

        # write per-model rows first, then read them back for the ensemble
        db.save_predictions(all_rows)
        per_model = db.load_model_predictions(cid)
        ens_rows = ensemble_for_card(per_model, scores_by_field=scores)
        db.save_predictions(ens_rows)

        total += len(all_rows) + len(ens_rows)
        if i % 50 == 0:
            log.info("...%d/%d cards done", i, len(card_ids))

    log.info("FULL PIPELINE DONE. wrote %d rows total.", total)


def main(argv=None):
    p = argparse.ArgumentParser(description="Run all models + ensemble")
    p.add_argument("--card", type=int)
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["prophet", "sarima", "montecarlo", "xgboost"],
                   help="models to skip this run")
    args = p.parse_args(argv)
    run(card_id=args.card, skip=args.skip)
    return 0


if __name__ == "__main__":
    sys.exit(main())
