# TCG Price Predictor — Production Guide

A batch ML pipeline that forecasts trading-card prices 7 days ahead. It reads
your price history from the `public` schema and writes forecasts to a table in
the `ml` schema. Four models plus an ensemble. Your backend only ever **reads**
the results — no ML runs in the request path.

---

## 1. How it fits your system

```
  [your scraper]                [THIS PIPELINE]                  [your backend]
 apify / pricecharting   →   runs on a schedule, reads     →   reads ml.card_predictions
 writes public.cards_         public.cards_pricehistory,        serves forecasts to the app
 pricehistory                 writes ml.card_predictions

      (nightly)                     (nightly, after scrape)          (every request)
```

The ML never executes when a user opens a card. Forecasts are precomputed and
sit in `ml.card_predictions` ready to read.

### Your access constraints (already handled)
- **Read-only** on `public` (your data: `cards_pricehistory`, `cards_card`, ...).
- **Read/write** only on the `ml` schema.
- The pipeline reads from `public.cards_pricehistory` and creates/writes
  `ml.card_predictions`. It never writes to `public`. The backend developer's
  `public.predictions_prediction` is never touched.

---

## 2. Project layout

Files MUST live inside a `predictor/` package folder (the code uses package
imports like `from . import db`). Place it next to your project:

```
Jinvalex_Project/
└── predictor/
    ├── __init__.py
    ├── config.py            # all schema/table/column names + tunables   [EDIT THIS]
    ├── db.py                # engine, ml.card_predictions, read/write     [shared]
    ├── preprocess.py        # raw history -> clean (ds, y) series         [shared]
    ├── prophet_model.py     + run_pipeline.py     -> model_used='prophet'
    ├── sarima_model.py      + run_sarima.py       -> model_used='sarima'
    ├── montecarlo_model.py  + run_montecarlo.py   -> model_used='montecarlo'
    ├── features.py, xgboost_model.py + run_xgboost.py -> model_used='xgboost'
    ├── ensemble.py                                -> model_used='ensemble'
    └── run_all.py           # master entrypoint: all models + ensemble
```

Run everything from `Jinvalex_Project/` (the folder ABOVE `predictor/`):

```bash
python -m predictor.run_all            # NOT  python -m run_all
```

---

## 3. One-time setup

```bash
cd Jinvalex_Project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# point at your Postgres (store as a secret in prod, don't hardcode)
export TCG_DATABASE_URL="postgresql+psycopg2://USER:PASS@HOST:5432/jinvalex_database"
```

### Confirm your column names BEFORE the first real run
The pipeline forecasts the price columns listed in `config.py` -> `PRICE_FIELDS`.
Verify they match `cards_pricehistory` exactly:

```sql
SELECT column_name FROM information_schema.columns
WHERE table_schema = 'public' AND table_name = 'cards_pricehistory'
ORDER BY ordinal_position;
```

Edit `config.py` if needed:
- `PRICE_FIELDS` — the price columns to forecast
- `HISTORY_DATE_COL` / `HISTORY_CARD_ID_COL` — if the date / card-id columns differ
- `READ_SCHEMA` / `WRITE_SCHEMA` — only if your schemas are named differently

Columns not present in the table are skipped automatically (no crash), so a
stray name won't break the run — it just won't be forecast.

---

## 4. Running

```bash
# test ONE card first — confirms DB access, table creation, and a real forecast
python -m predictor.run_all --card <some_card_id>

# full run (the production command)
python -m predictor.run_all

# skip a model when history is still thin (XGBoost needs ~30 days)
python -m predictor.run_all --skip xgboost

# or run a single model on its own
python -m predictor.run_pipeline        # prophet
python -m predictor.run_sarima
python -m predictor.run_montecarlo
python -m predictor.run_xgboost --train --predict
```

After a run, confirm rows landed:

```sql
SELECT model_used, COUNT(*) FROM ml.card_predictions GROUP BY model_used;
```

---

## 5. The output table: `ml.card_predictions`

```
id, card_id, price_field, predict_date,
predicted_value, lower_bound, upper_bound,
model_used, generated_at
UNIQUE (card_id, price_field, predict_date, model_used)
```

All five model types (prophet, sarima, montecarlo, xgboost, ensemble) coexist
as separate rows. Re-runs upsert (overwrite same key), so it is safe to run
repeatedly. The pipeline creates this table on first run; or create it yourself
with the SQL in `START_HERE.md`.

---

## 6. The four models + ensemble

| Model       | Type             | Trains?           | Speed     | Min history |
|-------------|------------------|-------------------|-----------|-------------|
| Monte Carlo | simulation       | none              | fastest   | ~6 days     |
| Prophet     | per-series       | per-series fit    | medium    | ~20 days    |
| SARIMA      | per-series       | per-series search | slow      | ~14 days    |
| XGBoost     | one global model | yes (persisted)   | fast pred | ~30 days    |

**Ensemble** reads all four models' rows per (card, field, date) and blends
them into one `model_used='ensemble'` row — weighted by each model's backtest
accuracy on that series (the best model for a card counts most). This is the
row your app should display.

---

## 7. Backend integration

Your API reads the ensemble row and serves it next to the live price:

```sql
SELECT predict_date, predicted_value, lower_bound, upper_bound
FROM ml.card_predictions
WHERE card_id = :id
  AND price_field = 'psa_10_price'
  AND model_used  = 'ensemble'
ORDER BY predict_date;
```

To show a single "predicted price" for a card, take tomorrow's ensemble row;
for a chart, take all 7 days with the lower/upper band.

---

## 8. Production scheduling

This is a batch job. Run `run_all` after your nightly scrape finishes.

### Option A — cron on your existing server (start here)
```cron
# 11:30 daily, after the scrape. Adjust path + time.
30 11 * * *  cd /opt/Jinvalex_Project && \
  TCG_DATABASE_URL="postgresql+psycopg2://user:pass@host:5432/jinvalex_database" \
  ./.venv/bin/python -m predictor.run_all >> /var/log/tcg_predict.log 2>&1
```

### Option B — managed scheduled container
Use the included `Dockerfile` with Render / Railway / Fly cron jobs, AWS ECS
Scheduled Task, or GCP Cloud Run Job + Scheduler. Inject `TCG_DATABASE_URL` as
a secret. Command: `python -m predictor.run_all`.

### Recommended cadence (large catalogues)
| Task                | Cadence           | Why                          |
|---------------------|-------------------|------------------------------|
| Monte Carlo         | nightly           | instant                      |
| Prophet             | nightly           | cheap per-series             |
| SARIMA              | weekly            | slowest model                |
| XGBoost **train**   | weekly            | patterns drift slowly        |
| XGBoost **predict** | nightly           | reuse weekly-trained model   |
| Ensemble            | nightly           | after the others write       |

`run_all` does the whole thing in one shot. For the split cadence, call the
individual runners on their own schedules and run the ensemble last. SARIMA is
the bottleneck at scale — drop it to weekly or shard card ids across processes.

---

## 9. Data-readiness note

Your history starts around late May 2026. Until daily scrapes accumulate,
expect sparse output: Monte Carlo first, then Prophet/SARIMA (~2-3 weeks),
XGBoost last (~30 days). Series with too little data or all-NULL/zero grades
are skipped automatically. This is expected, not an error — coverage grows as
history builds.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No module named predictor` | files not in a `predictor/` folder, or ran `-m run_pipeline` | put files in `predictor/`, run `-m predictor.run_all` |
| `permission denied for schema public` | tried to write to public | confirm `WRITE_SCHEMA='ml'` in config; pipeline writes only to ml |
| `relation cards_pricehistory does not exist` | wrong table/schema name | check `HISTORY_TABLE` / `READ_SCHEMA` in config |
| `column ... does not exist` | `PRICE_FIELDS` doesn't match real columns | run the information_schema query, fix the list |
| few/no rows written | not enough history yet | expected early on; try `--skip xgboost` |
| run is slow | SARIMA per-series search | move SARIMA to weekly, or shard card ids |

See `START_HERE.md` for the original error fixes and the manual table-creation
SQL. See `PRODUCTION.md` for the deeper deployment discussion.
