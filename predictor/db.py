"""
Database access layer.

Schemas (important): your data lives in `public` and you may only write to the
`ml` schema. So we READ from public.cards_pricehistory and CREATE/WRITE our
predictions table as ml.card_predictions. We never write to public.

All table/column/schema names come from config.py — change them there, not here.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text

from .config import (
    DATABASE_URL,
    PRICE_FIELDS,
    HISTORY_TABLE_FQ,
    HISTORY_CARD_ID_COL,
    HISTORY_DATE_COL,
    PREDICTIONS_TABLE_FQ,
    PREDICTIONS_INDEX_BASE,
    WRITE_SCHEMA,
    MIN_OBSERVATIONS,
)

_engine = None


def get_engine():
    """Lazily create and reuse a single engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    return _engine


# ---------------------------------------------------------------------------
# Output table  (created inside the ml schema)
#
# Each statement runs as its own comment-free text() object. We never split a
# SQL string on ';' (that was the original bug — a ';' inside a comment broke
# the statement in half).
# ---------------------------------------------------------------------------

# Harmless if the schema already exists; makes the script self-contained.
_CREATE_SCHEMA_SQL = text(f"CREATE SCHEMA IF NOT EXISTS {WRITE_SCHEMA}")

_CREATE_TABLE_SQL = text(
    f"""
    CREATE TABLE IF NOT EXISTS {PREDICTIONS_TABLE_FQ} (
        id               BIGSERIAL PRIMARY KEY,
        card_id          BIGINT      NOT NULL,
        price_field      VARCHAR(50) NOT NULL,
        predict_date     DATE        NOT NULL,
        predicted_value  NUMERIC(12, 2),
        lower_bound      NUMERIC(12, 2),
        upper_bound      NUMERIC(12, 2),
        model_used       VARCHAR(50) NOT NULL,
        generated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
        CONSTRAINT uq_{PREDICTIONS_INDEX_BASE}
            UNIQUE (card_id, price_field, predict_date, model_used)
    )
    """
)

_CREATE_INDEX_SQL = text(
    f"""
    CREATE INDEX IF NOT EXISTS ix_{PREDICTIONS_INDEX_BASE}_lookup
        ON {PREDICTIONS_TABLE_FQ} (card_id, price_field, predict_date)
    """
)


def ensure_predictions_table():
    """Create our predictions table and index inside the ml schema.

    The ml schema already exists in your DB, so we don't strictly need to
    create it. We attempt CREATE SCHEMA IF NOT EXISTS for portability, but if
    the role lacks schema-creation privilege we ignore that specific failure
    and proceed — the table creation below is what actually matters.
    """
    engine = get_engine()
    # Best-effort schema creation in its own transaction so a permission error
    # here can't poison the table-creation transaction.
    try:
        with engine.begin() as conn:
            conn.execute(_CREATE_SCHEMA_SQL)
    except Exception:
        pass  # schema already exists / no privilege to create it — fine.

    with engine.begin() as conn:
        conn.execute(_CREATE_TABLE_SQL)
        conn.execute(_CREATE_INDEX_SQL)


# ---------------------------------------------------------------------------
# Reading history  (from the public schema)
# ---------------------------------------------------------------------------

def load_card_ids() -> list[int]:
    """All card_ids that have any history."""
    sql = text(
        f"SELECT DISTINCT {HISTORY_CARD_ID_COL} AS cid "
        f"FROM {HISTORY_TABLE_FQ} ORDER BY cid"
    )
    with get_engine().connect() as conn:
        return [row[0] for row in conn.execute(sql)]


def load_eligible_card_ids(min_obs: int = MIN_OBSERVATIONS) -> list[int]:
    """
    Only card_ids with enough history to actually forecast.

    The unfiltered load_card_ids() returns every card with any row at all, so
    the pipeline loops over ~600k cards just to skip 599k of them inside
    prepare_series — that's the source of the hour-long run (one DB round-trip
    per card to discover it has too little data). This pre-filters at the DB so
    we iterate only over the cards that can produce a forecast.

    A card counts as eligible if ANY field has >= min_obs distinct dates. It
    may still be skipped per-field by prepare_series (e.g. one grade is sparse),
    which is fine — the goal is just to stop iterating over the empty long tail.
    As history accumulates this query naturally returns more cards.
    """
    sql = text(
        f"""
        SELECT {HISTORY_CARD_ID_COL} AS cid
        FROM {HISTORY_TABLE_FQ}
        GROUP BY {HISTORY_CARD_ID_COL}
        HAVING COUNT(DISTINCT {HISTORY_DATE_COL}) >= :min_obs
        ORDER BY cid
        """
    )
    with get_engine().connect() as conn:
        return [row[0] for row in conn.execute(sql, {"min_obs": min_obs})]


def load_history(card_id: int) -> pd.DataFrame:
    """
    Return the raw history rows for one card, one row per date, with all price
    fields as columns. The date column is aliased to 'date' so the rest of the
    pipeline stays schema-agnostic.
    """
    cols = ", ".join(PRICE_FIELDS)
    sql = text(
        f"""
        SELECT {HISTORY_DATE_COL} AS date, {cols}
        FROM {HISTORY_TABLE_FQ}
        WHERE {HISTORY_CARD_ID_COL} = :cid
        ORDER BY {HISTORY_DATE_COL} ASC
        """
    )
    with get_engine().connect() as conn:
        df = pd.read_sql(sql, conn, params={"cid": card_id})
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# Writing predictions  (into the ml schema)
# ---------------------------------------------------------------------------

UPSERT_SQL = text(
    f"""
    INSERT INTO {PREDICTIONS_TABLE_FQ}
        (card_id, price_field, predict_date, predicted_value,
         lower_bound, upper_bound, model_used, generated_at)
    VALUES
        (:card_id, :price_field, :predict_date, :predicted_value,
         :lower_bound, :upper_bound, :model_used, now())
    ON CONFLICT (card_id, price_field, predict_date, model_used)
    DO UPDATE SET
        predicted_value = EXCLUDED.predicted_value,
        lower_bound     = EXCLUDED.lower_bound,
        upper_bound     = EXCLUDED.upper_bound,
        generated_at    = now()
    """
)


def save_predictions(rows: list[dict]):
    """Upsert a batch of prediction dicts."""
    if not rows:
        return
    with get_engine().begin() as conn:
        conn.execute(UPSERT_SQL, rows)


def load_model_predictions(card_id: int, exclude_ensemble: bool = True) -> pd.DataFrame:
    """Read back per-model predictions for one card so the ensemble can combine them.

    Only FUTURE-dated rows (predict_date >= CURRENT_DATE) are returned. As the
    7-day window slides forward each run, old per-model rows for past dates are
    orphaned (the upsert only overwrites the same date). Without this filter the
    ensemble would re-blend those stale rows and re-stamp them with today's
    generated_at, filling the table with fresh-looking forecasts for dates that
    have already passed.
    """
    clause = "AND model_used <> 'ensemble'" if exclude_ensemble else ""
    sql = text(
        f"""
        SELECT card_id, price_field, predict_date,
               predicted_value, lower_bound, upper_bound, model_used
        FROM {PREDICTIONS_TABLE_FQ}
        WHERE card_id = :cid
          AND predict_date >= CURRENT_DATE
          {clause}
        """
    )
    with get_engine().connect() as conn:
        df = pd.read_sql(sql, conn, params={"cid": card_id})
    if not df.empty:
        df["predict_date"] = pd.to_datetime(df["predict_date"])
    return df
