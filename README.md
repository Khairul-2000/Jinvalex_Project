# TCG Card Price Predictor

A batch machine-learning pipeline that forecasts trading-card prices **7 days
ahead**. It reads price history from your `public` schema, runs four forecasting
models plus an ensemble, and writes the results to `ml.card_predictions`. Your
backend only ever **reads** that table — no ML runs in the request path.

---

## 1. What it does (the mechanism)

```
   [your scraper]                 [THIS PIPELINE]                    [your backend]
 apify / pricecharting   →   runs on a schedule (nightly):    →   reads ml.card_predictions
 writes                       1. reads public.cards_pricehistory    serves forecasts to the app
 public.cards_pricehistory    2. forecasts each card/grade
                              3. writes ml.card_predictions

      (nightly)                    (nightly, after the scrape)          (every request)
```

The forecasting is **precomputed in a batch job**. When a user opens a card, the
backend just reads the latest rows from `ml.card_predictions` — fast, with no
model running live.

### Access model (important)
- **Read-only** on the `public` schema (your data: `cards_pricehistory`, …).
- **Read/write** only on the `ml` schema.
- The pipeline reads `public.cards_pricehistory` and creates/writes
  `ml.card_predictions`. It **never** writes to `public`.

### Per-series forecasting
Each **(card, price field)** pair is forecast as its own independent daily time
series — e.g. card `1384245`'s `psa_10_price` is a separate series from its
`ungraded_price`. There are 9 price fields (grades); a card is forecast only for
the fields that actually have enough price history (see §6).

### Four models + an ensemble
| Model        | Type                | Trains?            | Speed      | Min history |
|--------------|---------------------|--------------------|------------|-------------|
| Monte Carlo  | GBM simulation      | none               | fastest    | ~6 days     |
| Prophet      | per-series fit      | per series         | medium     | ~20 days    |
| SARIMA       | per-series search   | per series         | slow       | ~14 days    |
| XGBoost      | one global model    | yes (persisted)    | fast pred  | ~30 days    |

**Ensemble** combines all four into one final row per (card, field, date). Each
model is weighted by how accurate it has been *on that specific series* (lower
backtest MAPE → higher weight), so the best model for a card counts most. The
combined uncertainty band reflects weighted model disagreement. **This
`model_used='ensemble'` row is the one your app should display.**

---

## 2. Folder structure

The Python files **must** live inside a `predictor/` package (the code uses
package imports like `from . import db`).

```
Jinvalex_Project/
├── README.md                 # this file
├── requirements.txt          # install this one (has all 4 models' deps)
└── predictor/
    ├── __init__.py
    ├── config.py             # DB URL, schemas, price fields, tunables   [EDIT THIS]
    ├── db.py                 # engine, table creation, read/write helpers
    ├── preprocess.py         # raw history -> clean regular (ds, y) series
    ├── features.py           # feature engineering for XGBoost
    │
    ├── prophet_model.py      + run_pipeline.py     -> model_used='prophet'
    ├── sarima_model.py       + run_sarima.py       -> model_used='sarima'
    ├── montecarlo_model.py   + run_montecarlo.py   -> model_used='montecarlo'
    ├── xgboost_model.py      + run_xgboost.py      -> model_used='xgboost'
    ├── ensemble.py                                 -> model_used='ensemble'
    ├── run_all.py            # master entrypoint: all models + ensemble   [RUN THIS]
    │
    ├── xgb_model.json        # the persisted global XGBoost model (auto-written)
    ├── Dockerfile            # for a scheduled container deployment
    ├── PRODUCTION.md         # deeper production / deployment guide
    └── START_HERE.md         # original setup-error fixes + manual table SQL
```

Each model has a paired runner (`run_<model>.py`) so you can run it on its own,
plus `run_all.py` which runs everything end-to-end. They all share one output
table, distinguished by the `model_used` column.

---

## 3. Setup

```bash
cd Jinvalex_Project
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # the ROOT requirements.txt — it has all models' deps
```

Point it at your database. **Use the env var — don't rely on the hardcoded
default in `config.py` (rotate that credential if it is real):**

```bash
export TCG_DATABASE_URL="postgresql+psycopg2://USER:PASS@HOST:5432/jinvalex_database"
```

### Confirm your column names before the first real run
The pipeline forecasts the columns listed in `config.py → PRICE_FIELDS`. Make
sure they match `cards_pricehistory` exactly:

```sql
SELECT column_name FROM information_schema.columns
WHERE table_schema = 'public' AND table_name = 'cards_pricehistory'
ORDER BY ordinal_position;
```

Columns not present in the table are skipped automatically (no crash).

---

## 4. Configuration (`config.py`)

All names and tunables live in one file so you change them in one place:

| Setting | What it controls |
|---|---|
| `READ_SCHEMA` / `WRITE_SCHEMA` | where to read (`public`) and write (`ml`) |
| `HISTORY_TABLE` | source table (`cards_pricehistory`) |
| `HISTORY_CARD_ID_COL` / `HISTORY_DATE_COL` | the card-id and date columns |
| `PRICE_FIELDS` | the price/grade columns to forecast |
| `FORECAST_HORIZON_DAYS` | how many future days to predict (default 7) |
| `MIN_OBSERVATIONS` | min data points before a series is modeled (default 20) |
| `MAX_FFILL_DAYS` | how many days a gap is forward-filled (default 14) |
| `BACKTEST_DAYS` | holdout length for the accuracy backtest (default 7) |

---

## 5. How to run

Run everything from `Jinvalex_Project/` (the folder **above** `predictor/`):

```bash
# Test ONE card first — confirms DB access, table creation, and a real forecast
python -m predictor.run_all --card 1384245

# Full run (the production command): all models + ensemble, all eligible cards
python -m predictor.run_all

# Skip a model (e.g. XGBoost while history is still under ~30 days)
python -m predictor.run_all --skip xgboost
```

Each model can also be run on its own:

```bash
python -m predictor.run_pipeline          # Prophet      (--card N, --no-backtest)
python -m predictor.run_sarima            # SARIMA
python -m predictor.run_montecarlo        # Monte Carlo
python -m predictor.run_xgboost --train --predict   # XGBoost (train then forecast)
```

> Run from the parent folder as `python -m predictor.run_all` — **not**
> `python -m run_all` from inside `predictor/` (that breaks the package imports).

After a run, confirm rows landed:

```sql
SELECT model_used, COUNT(*) FROM ml.card_predictions GROUP BY model_used;
```

---

## 6. Why output can look sparse (two data gates)

Two independent filters decide what gets forecast. This is by design, not a bug.

**Gate 1 — card eligibility** (`db.load_eligible_card_ids`): a card is only
considered if it has **≥ `MIN_OBSERVATIONS` (20) distinct dates** of history.
Early on (history started ~late May 2026) most cards haven't accumulated 20 days
yet, so few cards qualify. Coverage grows automatically as scrapes accumulate.

**Gate 2 — per-field** (`preprocess.prepare_series`): for an eligible card, each
price field needs **≥ 20 non-null, positive** values of its own. Most cards only
trade in a couple of grades (e.g. `ungraded_price` + `psa_10_price`); the other
grade columns are NULL/0 and are skipped. So a card commonly produces forecasts
for **only 1–2 of the 9 fields** — that's expected.

Diagnostic — how many cards clear Gate 1:

```sql
SELECT distinct_days, COUNT(*) AS n_cards FROM (
  SELECT card_id, COUNT(DISTINCT date) AS distinct_days
  FROM public.cards_pricehistory GROUP BY card_id
) t GROUP BY distinct_days ORDER BY distinct_days DESC;
```

To get more cards/fields during early validation, lower `MIN_OBSERVATIONS` in
`config.py` (quality drops on short series; XGBoost still needs ~30 days because
of its 30-day lag feature).

---

## 7. The output table: `ml.card_predictions`

The pipeline creates this on first run (or create it yourself with the SQL in
`START_HERE.md`).

```
id, card_id, price_field, predict_date,
predicted_value, lower_bound, upper_bound,
model_used, generated_at
UNIQUE (card_id, price_field, predict_date, model_used)
```

- All five `model_used` values (`prophet`, `sarima`, `montecarlo`, `xgboost`,
  `ensemble`) coexist as separate rows. The per-model rows are intermediate; the
  **ensemble row is the one to display.**
- Re-runs **upsert** (overwrite the same key), so running nightly is safe and
  idempotent.
- The forecast window is anchored to each series' **last observation date**, so
  if your scrape lags, the first forecast days may already be in the past —
  always filter `predict_date >= CURRENT_DATE` when serving (see §8).

---

## 8. Backend integration

Read the ensemble row and serve it next to the live price. Filter out any
already-past dates:

```sql
SELECT predict_date, predicted_value, lower_bound, upper_bound
FROM ml.card_predictions
WHERE card_id = :id
  AND price_field = 'psa_10_price'
  AND model_used  = 'ensemble'
  AND predict_date >= CURRENT_DATE
ORDER BY predict_date;
```

- For a single "predicted price," take tomorrow's ensemble row.
- For a chart, take all remaining horizon days with the lower/upper band.

No ML runs in the request path — these rows are precomputed by the scheduled job.

---

## 9. Scheduling

Run `run_all` after your nightly scrape finishes.

**Cron:**
```cron
# 11:30 daily, after the scrape. Adjust path + time.
30 11 * * *  cd /opt/Jinvalex_Project && \
  TCG_DATABASE_URL="postgresql+psycopg2://user:pass@host:5432/jinvalex_database" \
  ./.venv/bin/python -m predictor.run_all >> /var/log/tcg_predict.log 2>&1
```

**Container:** the included `predictor/Dockerfile` builds an image whose default
command is `python -m predictor.run_all`. Use it with Render / Railway / Fly
cron, AWS ECS Scheduled Task, or GCP Cloud Run Job + Scheduler, injecting
`TCG_DATABASE_URL` as a secret.

At large scale, SARIMA is the bottleneck (it searches orders per series) — run it
weekly or shard card ids across processes. See `PRODUCTION.md` for the split
cadence and deeper deployment notes.

---

## 10. Recent model improvements

The forecasting logic was tuned after observing bad early output:

- **Prophet** now uses **additive** seasonality and disables weekly seasonality
  until ~4 weeks of data exist (multiplicative + weekly on short series biased
  forecasts ~25% low).
- **XGBoost** recursive forecasting is **clamped to ±10%/day** so its fed-back
  predictions can't compound into a runaway trend.
- **Ensemble weighting** no longer lets un-backtested models dominate, and the
  uncertainty band is now a **weighted** disagreement measure (a poor model
  barely affects it) instead of the raw min/max of model point estimates.
- **`load_model_predictions`** only blends **future-dated** rows
  (`predict_date >= CURRENT_DATE`), so stale per-model rows from earlier runs are
  not re-blended and re-stamped.

> XGBoost's per-series backtest still trains on its holdout (optimistic MAPE) —
> a known limitation to address before fully trusting its ensemble weight.

---

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No module named predictor` | files not in `predictor/`, or ran `-m run_all` | put files in `predictor/`, run `-m predictor.run_all` from the parent folder |
| `permission denied for schema public` | tried to write to public | confirm `WRITE_SCHEMA='ml'` in config |
| `relation cards_pricehistory does not exist` | wrong table/schema name | check `HISTORY_TABLE` / `READ_SCHEMA` in config |
| `column ... does not exist` | `PRICE_FIELDS` ≠ real columns | run the information_schema query, fix the list |
| few / no rows written | not enough history yet (Gate 1/2) | expected early on; lower `MIN_OBSERVATIONS` to validate, or `--skip xgboost` |
| run is slow | SARIMA per-series order search | move SARIMA to weekly, or shard card ids |

See `predictor/START_HERE.md` for original setup-error fixes and the manual
table-creation SQL, and `predictor/PRODUCTION.md` for the deeper deployment guide.
```