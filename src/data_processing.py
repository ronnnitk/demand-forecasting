"""Data loading and filtering — separated from modeling on purpose.

Consolidates the loading/cleaning logic that was split across check.py and
process_data.py into one importable module.

Loads route through the partitioned panel store (:mod:`src.io_store`) when it
has been built, and fall back to a chunked read of the raw CSV otherwise, so
these functions work on a fresh checkout as well as a prepared one.
"""

import pandas as pd

from . import io_store
from .config import (
    DATE_COL,
    DEFAULT_FAMILY,
    DEFAULT_STORE_NBR,
    PROMO_COL,
    RAW_TRAIN_CSV,
    TARGET_COL,
)

USE_COLS = ["date", "store_nbr", "family", "sales"]


def load_store(store_nbr: int = DEFAULT_STORE_NBR, csv_path=RAW_TRAIN_CSV) -> pd.DataFrame:
    """Load a single store's rows, cheaply."""
    if io_store.is_built():
        return io_store.read_store(store_nbr)

    chunks = pd.read_csv(
        csv_path, usecols=USE_COLS, parse_dates=["date"], chunksize=200_000
    )
    filtered = [chunk[chunk["store_nbr"] == store_nbr] for chunk in chunks]
    return (
        pd.concat(filtered)
        .sort_values(["family", "date"])
        .reset_index(drop=True)
    )


def get_series(
    df_store: pd.DataFrame, family: str = DEFAULT_FAMILY
) -> pd.DataFrame:
    """Return a clean, sorted ``date``/``sales`` series for one product family."""
    cols = [DATE_COL, TARGET_COL]
    if PROMO_COL in df_store.columns:
        cols.append(PROMO_COL)
    return (
        df_store[df_store["family"] == family][cols]
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )


def regularize(series: pd.DataFrame, fill_value: float = 0.0) -> pd.DataFrame:
    """Reindex a series onto a complete daily calendar.

    Gaps in a demand series are not missing measurements — they are days the
    store was closed or the item was not stocked. Leaving them out silently
    corrupts every lag and rolling feature downstream, because ``shift(7)``
    then means "7 rows back", not "7 days back".
    """
    if series.empty:
        return series
    full = pd.DataFrame(
        {DATE_COL: pd.date_range(series[DATE_COL].min(), series[DATE_COL].max(), freq="D")}
    )
    out = full.merge(series, on=DATE_COL, how="left")
    out[TARGET_COL] = out[TARGET_COL].fillna(fill_value)
    if PROMO_COL in out.columns:
        out[PROMO_COL] = out[PROMO_COL].fillna(0)

    # Identity columns describe the series, not the day — carry them across the
    # inserted rows so the frame stays a valid panel slice.
    for col in ("store_nbr", "family"):
        if col in out.columns:
            out[col] = out[col].ffill().bfill()
    return out
