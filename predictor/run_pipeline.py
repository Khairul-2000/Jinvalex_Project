"""
Orchestration: tie everything together.

This is what your nightly cron runs:
    python -m predictor.run_pipeline            # all cards
    python -m predictor.run_pipeline --card 1142912   # one card (testing)

Flow per (card, field):
    load history -> prepare series -> backtest (optional) -> forecast -> collect rows
Then bulk-upsert all rows into card_predictions.
"""
from __future__ import annotations

import argparse
import logging
import sys


from .config import FORECAST_HORIZON_DAYS, PRICE_FIELDS
from . import db
from .preprocess import prepare_series
from .prophet_model import backtest, fit_forecast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

MODEL_NAME = "prophet"


def process_card(card_id: int, do_backtest: bool = True) -> list[dict]:
    """Forecast every usable price field for one card. Returns prediction rows."""
    history = db.load_history(card_id)
    if history.empty:
        return []

    rows: list[dict] = []
    for field in PRICE_FIELDS:
        series = prepare_series(history, field)
        if series is None:
            continue  # not enough data for this grade on this card

        if do_backtest:
            metrics = backtest(series)
            if metrics:
                log.info(
                    "card %s / %-16s  MAPE=%5.1f%%  (n=%d)",
                    card_id, field, metrics["mape"], metrics["n"],
                )

        try:
            fc = fit_forecast(series, horizon=FORECAST_HORIZON_DAYS)
        except Exception as exc:
            log.warning("card %s / %s forecast failed: %s", card_id, field, exc)
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
    log.info("processing %d card(s)", len(card_ids))

    total = 0
    for i, cid in enumerate(card_ids, 1):
        rows = process_card(cid, do_backtest=do_backtest)
        db.save_predictions(rows)
        total += len(rows)
        if i % 100 == 0:
            log.info("...%d/%d cards done", i, len(card_ids))

    log.info("DONE. wrote %d prediction rows.", total)


def main(argv=None):
    p = argparse.ArgumentParser(description="Prophet price forecaster")
    p.add_argument("--card", type=int, help="forecast a single card_id")
    p.add_argument("--no-backtest", action="store_true", help="skip backtest metrics")
    args = p.parse_args(argv)
    run(card_id=args.card, do_backtest=not args.no_backtest)


if __name__ == "__main__":
    sys.exit(main())
