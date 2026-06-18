"""
XGBoost orchestration — two phases.

  TRAIN:   one pass over every card/field to assemble features, fit ONE model,
           persist it to disk.
  PREDICT: load the model, recursively forecast 7 days per series, upsert into
           card_predictions with model_used='xgboost'.

Usage:
    python -m predictor.run_xgboost --train          # build/refresh the model
    python -m predictor.run_xgboost --predict         # forecast all cards
    python -m predictor.run_xgboost --train --predict # both (typical nightly)
    python -m predictor.run_xgboost --predict --card 1142912   # one card
"""
from __future__ import annotations

import argparse
import logging
import sys

from .config import FORECAST_HORIZON_DAYS, PRICE_FIELDS
from . import db
from .preprocess import prepare_series
from .xgboost_model import (
    MODEL_NAME,
    assemble_training_table,
    backtest_series,
    forecast_series,
    load_model,
    train_global,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("xgb")


def _iter_series(card_ids):
    """Yield (series, card_id, field) for every usable card/field combo."""
    for cid in card_ids:
        history = db.load_history(cid)
        if history.empty:
            continue
        for field in PRICE_FIELDS:
            series = prepare_series(history, field)
            if series is not None:
                yield series, cid, field


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def do_train():
    card_ids = db.load_eligible_card_ids()
    log.info("TRAIN: scanning %d cards for features...", len(card_ids))

    table = assemble_training_table(_iter_series(card_ids))
    log.info("TRAIN: assembled %d feature rows from all series", len(table))

    if table.empty:
        log.error("No features produced — need more history. Aborting train.")
        return False

    train_global(table, save=True)
    log.info("TRAIN: model fit and saved.")
    return True


def do_predict(card_id: int | None = None, do_backtest: bool = True):
    model = load_model()
    if model is None:
        log.error("No saved model found. Run with --train first.")
        return

    db.ensure_predictions_table()
    card_ids = [card_id] if card_id else db.load_eligible_card_ids()
    log.info("PREDICT: forecasting %d card(s)...", len(card_ids))

    total = 0
    for i, cid in enumerate(card_ids, 1):
        history = db.load_history(cid)
        if history.empty:
            continue

        rows = []
        for field in PRICE_FIELDS:
            series = prepare_series(history, field)
            if series is None:
                continue

            if do_backtest:
                m = backtest_series(model, series, cid, field)
                if m:
                    log.info("card %s / %-16s MAPE=%5.1f%% (n=%d)",
                             cid, field, m["mape"], m["n"])

            fc = forecast_series(model, series, cid, field,
                                 horizon=FORECAST_HORIZON_DAYS)
            if fc is None:
                continue
            for _, r in fc.iterrows():
                rows.append({
                    "card_id": cid,
                    "price_field": field,
                    "predict_date": r["ds"].date(),
                    "predicted_value": float(r["yhat"]),
                    "lower_bound": float(r["yhat_lower"]),
                    "upper_bound": float(r["yhat_upper"]),
                    "model_used": MODEL_NAME,
                })

        db.save_predictions(rows)
        total += len(rows)
        if i % 100 == 0:
            log.info("...%d/%d cards done", i, len(card_ids))

    log.info("PREDICT: DONE. wrote %d rows (model_used='%s').", total, MODEL_NAME)


def main(argv=None):
    p = argparse.ArgumentParser(description="Global XGBoost price forecaster")
    p.add_argument("--train", action="store_true", help="build/refresh the model")
    p.add_argument("--predict", action="store_true", help="forecast and write rows")
    p.add_argument("--card", type=int, help="predict a single card_id")
    p.add_argument("--no-backtest", action="store_true")
    args = p.parse_args(argv)

    if not (args.train or args.predict):
        p.error("specify --train and/or --predict")

    if args.train:
        ok = do_train()
        if not ok and args.predict:
            return 1
    if args.predict:
        do_predict(card_id=args.card, do_backtest=not args.no_backtest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
