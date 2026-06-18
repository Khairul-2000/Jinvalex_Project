"""
Turn raw history into clean, regular, per-field time series.

Prophet expects a DataFrame with exactly two columns:
    ds  -> datetime
    y   -> numeric value

Your raw data is irregular (gaps between scrape dates) and has NULLs for
grades that don't exist on a given card. This module handles both.
"""
from __future__ import annotations

import pandas as pd

from .config import MAX_FFILL_DAYS, MIN_OBSERVATIONS


def prepare_series(history: pd.DataFrame, field: str) -> pd.DataFrame | None:
    """
    Extract one price field and return a Prophet-ready (ds, y) frame on a
    regular daily grid, or None if there isn't enough usable data.

    Steps:
      1. Keep date + the one field, drop NULL/zero rows (no real price).
      2. Collapse duplicate dates (take the last scrape that day).
      3. Reindex onto a continuous daily calendar.
      4. Forward-fill across short gaps only; leave long gaps as NaN, then
         drop them so Prophet trains on real-ish points.
    """
    # If this price field isn't a column in the history table at all, skip it.
    # (Your schema may not carry every grade.)
    if field not in history.columns:
        return None

    s = history[["date", field]].copy()
    s = s.rename(columns={"date": "ds", field: "y"})

    # Coerce to numeric in case the column came back as text/Decimal/None.
    s["y"] = pd.to_numeric(s["y"], errors="coerce")

    # Treat 0 and NULL as "no price recorded" for this grade.
    s = s.dropna(subset=["y"])
    s = s[s["y"] > 0]

    if len(s) < MIN_OBSERVATIONS:
        return None

    # One value per day (last scrape wins).
    s = s.sort_values("ds").groupby("ds", as_index=False).last()

    # Regular daily calendar from first to last observation.
    full_idx = pd.date_range(s["ds"].min(), s["ds"].max(), freq="D")
    s = s.set_index("ds").reindex(full_idx)
    s.index.name = "ds"

    # Forward-fill only across short gaps (a price holds until the next scrape),
    # but don't bridge long silences.
    s["y"] = s["y"].ffill(limit=MAX_FFILL_DAYS)
    s = s.dropna(subset=["y"]).reset_index()

    if len(s) < MIN_OBSERVATIONS:
        return None

    return s[["ds", "y"]]