"""
Central configuration.

Override the DB connection with an env var so you never hardcode credentials:
    export TCG_DATABASE_URL="postgresql+psycopg2://user:pass@host:5432/dbname"
"""
import os

DATABASE_URL = os.environ.get(
    "TCG_DATABASE_URL",
    "postgresql://ml_dev:rLfblUq78cScD05eaiqdb2sNFVxS@10.10.26.245:5432/jinvalex_database",
)

# ---------------------------------------------------------------------------
# Schema mapping — adjust these to match your actual database.
# All table/column names the pipeline touches live here, so if your real
# columns differ you change them in ONE place and nothing else.
# ---------------------------------------------------------------------------

# Your data tables live in the `public` schema; you can only WRITE to the
# `ml` schema. So we READ history from public and CREATE/WRITE our predictions
# table inside ml. These are kept separate on purpose.
READ_SCHEMA = "public"   # where cards_pricehistory etc. live (read-only for us)
WRITE_SCHEMA = "ml"      # where we are allowed to create/write tables

# The price-history table and its columns (your `cards_pricehistory`).
HISTORY_TABLE = "cards_pricehistory"
HISTORY_CARD_ID_COL = "card_id"   # FK to the card
HISTORY_DATE_COL = "date"         # observation date

# Our own predictions output table, created inside the ml schema. Separate
# from the backend developer's public.predictions_prediction (untouched).
PREDICTIONS_TABLE = "card_predictions"

# Fully-qualified names used in SQL. Don't edit these directly — edit the
# schema/table parts above and these update automatically.
HISTORY_TABLE_FQ = f"{READ_SCHEMA}.{HISTORY_TABLE}"
PREDICTIONS_TABLE_FQ = f"{WRITE_SCHEMA}.{PREDICTIONS_TABLE}"
# A safe identifier for index names (no dots allowed in index names).
PREDICTIONS_INDEX_BASE = f"{WRITE_SCHEMA}_{PREDICTIONS_TABLE}"

# The price columns in the history table that we forecast.
# Each one becomes an independent time series per card.
# If your real column names differ, edit this list to match exactly.
PRICE_FIELDS = [
    "bgs_10_price",
    "grade_9_5_price",
    "grade_7_price",
    "cgc_10_price",
    "sgc_10_price",
    "grade_9_price",
    "ungraded_price",
    "psa_10_price",
    "grade_8_price",
]

# How many days into the future to predict.
FORECAST_HORIZON_DAYS = 7

# A series needs at least this many real observations to be worth modeling.
# Prophet technically runs on fewer, but the forecast is junk below ~15-20.
MIN_OBSERVATIONS = 20

# We resample history onto a regular daily grid. Gaps up to this many days
# get forward-filled (a price is assumed to hold until the next observation).
# Bigger gaps are left as NaN so we don't invent data across long silences.
MAX_FFILL_DAYS = 14

# Backtest: hold out the last N days to measure error before trusting a forecast.
BACKTEST_DAYS = 7
