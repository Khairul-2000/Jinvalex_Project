# START HERE — fixes for your setup

This addresses the exact errors you hit and adapts the code to YOUR database.

## What was wrong (earlier errors)

1. **SQL crash** (`syntax error at end of input`). The old `db.py` split the
   CREATE TABLE string on `;`, but a `;` inside a `--` comment chopped the
   statement in half. Fixed: statements now run as separate, comment-free
   `text()` objects. No more splitting.

2. **Wrong table names.** The code assumed tables named `cards` / `card_history`.
   Your real tables (from your screenshot) are `cards_card` and
   `cards_pricehistory`. Fixed: the pipeline now reads from `cards_pricehistory`.

3. **Predictions table.** You wanted a clean new one and you can create tables.
   The pipeline now creates and writes to **`ml.card_predictions`** (in the
   `ml` schema — see below), completely separate from the backend developer's
   `public.predictions_prediction` (untouched).

## Schemas: read from `public`, write to `ml`

Your access is split: data lives in the `public` schema, but you can only
create/write tables in the `ml` schema. The code now respects this:

- READS history from `public.cards_pricehistory`
- CREATES + WRITES predictions to `ml.card_predictions`
- never writes to `public`

On first run it also issues `CREATE SCHEMA IF NOT EXISTS ml` (harmless if the
schema already exists). If you'd rather not let it do that, the `ml` schema is
already there in your screenshot, so it'll just be a no-op.

These schema names live at the top of `config.py` as `READ_SCHEMA` and
`WRITE_SCHEMA` — change them there if needed.

## Folder structure matters (this caused your run error)

Your traceback showed files sitting directly in the project root
(`Jinvalex_Project/run_pipeline.py`, `db.py`, ...) and you ran
`python3 -m run_pipeline`. The code uses package imports (`from . import db`),
so the files MUST live inside a `predictor/` folder. Put them like this:

```
Jinvalex_Project/
└── predictor/
    ├── __init__.py
    ├── config.py
    ├── db.py
    ├── preprocess.py
    ├── prophet_model.py
    ├── run_pipeline.py
    └── ... (the rest)
```

Then run from `Jinvalex_Project/` (the folder ABOVE `predictor/`):

```bash
python -m predictor.run_pipeline --card <some_id>   # test one card first
python -m predictor.run_pipeline                    # all cards
```

NOT `python -m run_pipeline`.

## Check your column names

I kept the price-field names from your first message
(`psa_10_price`, `ungraded_price`, `bgs_10_price`, ...). If your real
`cards_pricehistory` columns differ, open `config.py` and edit ONE list:
`PRICE_FIELDS`, plus `HISTORY_DATE_COL` / `HISTORY_CARD_ID_COL` if those differ.
Nothing else needs touching. Fields not present in the table are skipped
automatically, so a wrong/extra name won't crash — it just won't forecast.

## If you'd rather create the table yourself first

You don't have to let the code create it. Run this once in your DB client,
then the pipeline will just use it:

```sql
CREATE SCHEMA IF NOT EXISTS ml;
CREATE TABLE IF NOT EXISTS ml.card_predictions (
    id               BIGSERIAL PRIMARY KEY,
    card_id          BIGINT      NOT NULL,
    price_field      VARCHAR(50) NOT NULL,
    predict_date     DATE        NOT NULL,
    predicted_value  NUMERIC(12, 2),
    lower_bound      NUMERIC(12, 2),
    upper_bound      NUMERIC(12, 2),
    model_used       VARCHAR(50) NOT NULL,
    generated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_ml_card_predictions
        UNIQUE (card_id, price_field, predict_date, model_used)
);
CREATE INDEX IF NOT EXISTS ix_ml_card_predictions_lookup
    ON ml.card_predictions (card_id, price_field, predict_date);
```

## Set your connection string

```bash
export TCG_DATABASE_URL="postgresql+psycopg2://USER:PASS@localhost:5432/jinvalex_database"
```
