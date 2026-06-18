"""
Monte Carlo orchestration — per-series, mirrors the Prophet/SARIMA runners.

    python -m predictor.run_montecarlo                 # all cards
    python -m predictor.run_montecarlo --card 1142912  # one card
    python -m predictor.run_montecarlo --no-backtest

Writes rows with model_used='montecarlo' into the shared card_predictions table.
This is by far the fastest model (pure simulation, no fitting), so it's cheap to
run as often as you like.
"""
from __future__ import annotations

import argparse
import logging
import sys

from .config import FORECAST_HORIZON_DAYS, PRICE_FIELDS
from . import db
from .preprocess import prepare_series
from .montecarlo_model import backtest, fit_forecast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("montecarlo")

MODEL_NAME = "montecarlo"


def process_card(card_id: int, do_backtest: bool = True) -> list[dict]:
    history = db.load_history(card_id)
    if history.empty:
        return []

    rows: list[dict] = []
    for field in PRICE_FIELDS:
        series = prepare_series(history, field)
        if series is None:
            continue

        if do_backtest:
            metrics = backtest(series)
            if metrics:
                log.info(
                    "card %s / %-16s  MAPE=%5.1f%%  (n=%d)",
                    card_id, field, metrics["mape"], metrics["n"],
                )

        fc = fit_forecast(series, horizon=FORECAST_HORIZON_DAYS)
        if fc is None:
            continue

        for _, r in fc.iterrows():
            rows.append(
                {
                    "card_id": card_id,
                    "price_field": field,
                    "predict_date": r["ds"].date(),
                    "predicted_value": float(r["yhat"]),
                    "lower_bound": float(r["yhat_lower"]),
                    "upper_bound": float(r["yhat_upper"]),
                    "model_used": MODEL_NAME,
                }
            )
    return rows


def run(card_id: int | None = None, do_backtest: bool = True):
    db.ensure_predictions_table()
    card_ids = [card_id] if card_id else db.load_eligible_card_ids()
    log.info("processing %d card(s) with Monte Carlo", len(card_ids))

    total = 0
    for i, cid in enumerate(card_ids, 1):
        rows = process_card(cid, do_backtest=do_backtest)
        db.save_predictions(rows)
        total += len(rows)
        if i % 100 == 0:
            log.info("...%d/%d cards done", i, len(card_ids))

    log.info("DONE. wrote %d rows (model_used='%s').", total, MODEL_NAME)


def main(argv=None):
    p = argparse.ArgumentParser(description="Monte Carlo price forecaster")
    p.add_argument("--card", type=int, help="forecast a single card_id")
    p.add_argument("--no-backtest", action="store_true")
    args = p.parse_args(argv)
    run(card_id=args.card, do_backtest=not args.no_backtest)


if __name__ == "__main__":
    sys.exit(main())
