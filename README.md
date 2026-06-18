# TCG Prophet Price Predictor

A self-contained Prophet forecasting pipeline that reads your `card_history`
and writes 7-day forecasts into a new `card_predictions` table. Your existing
backend only needs to **read** from `card_predictions`.

## Structure

```
tcg_predictor/
├── requirements.txt
└── predictor/
    ├── config.py          # DB URL, price fields, horizon, thresholds
    ├── db.py              # engine, predictions table, read/write helpers
    ├── preprocess.py      # raw history -> clean regular (ds, y) series
    ├── prophet_model.py   # fit / forecast / backtest one series
    └── run_pipeline.py    # orchestrator (the cron entrypoint)
```

## Setup

```bash
cd tcg_predictor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export TCG_DATABASE_URL="postgresql+psycopg2://USER:PASS@HOST:5432/DBNAME"
```

## Run

```bash
# test on one card first
python -m predictor.run_pipeline --card 1142912

# full run (what cron executes)
python -m predictor.run_pipeline
```

## How it works

1. **One series per (card, price field).** Each grade column is forecast
   independently. Fields that are mostly NULL/0 for a card are skipped.
2. **Preprocess** pivots the field, drops empty rows, puts it on a regular
   daily grid, and forward-fills short gaps only.
3. **Prophet** fits each series and predicts the next 7 days with an 80%
   interval (`lower_bound` / `upper_bound`).
4. **Backtest** holds out the last 7 days and logs MAPE so you know which
   forecasts to trust. (Doesn't block writing — it's a quality signal.)
5. **Upsert** into `card_predictions`. Re-running overwrites the same
   card/field/date/model, so it's safe to run nightly.

## The output table

```
card_predictions(
  card_id, price_field, predict_date,
  predicted_value, lower_bound, upper_bound,
  model_used, generated_at
)
```

## Backend integration

Your API joins current price (from `cards`) with the forecast:

```sql
SELECT predict_date, predicted_value, lower_bound, upper_bound
FROM card_predictions
WHERE card_id = :card_id
  AND price_field = 'psa_10_price'
  AND model_used  = 'prophet'
ORDER BY predict_date;
```

No ML runs in the request path — predictions are precomputed by cron.

## Schedule (nightly, after your scrape finishes)

```cron
30 11 * * *  cd /path/tcg_predictor && ./.venv/bin/python -m predictor.run_pipeline >> /var/log/tcg_predict.log 2>&1
```

## Adding the other 3 models later

`model_used` already distinguishes models in the same table. Implement a
`sarima_model.py` / `xgboost_model.py` / `montecarlo_model.py` exposing the
same `fit_forecast()` shape, write rows with their own `model_used`, then add
an ensemble step that reads all models per card/field and writes a final
`model_used='ensemble'` row.
```
```
