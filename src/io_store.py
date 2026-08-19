"""Partitioned on-disk panel store.

The original pipeline re-read and re-filtered the full 3M-row CSV for every
piece of work (and the API held the whole thing in RAM). That does not scale:
cost grows with the *dataset*, not with the slice you actually need.

This module converts the raw CSV **once**, in bounded-memory chunks, into one
partition per store. After that every consumer reads only the partition it
needs, so the working set is ~1/54th of the data and load time is
proportional to the slice rather than the corpus.

Format: Parquet when a Parquet engine is installed (columnar, compressed,
column-pruning), otherwise pickle. Both are transparent to callers.

    python -m src.io_store            # build the store
    python -m src.io_store --force    # rebuild from scratch
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from .config import (
    CHUNK_SIZE,
    DATE_COL,
    KEY_COLS,
    PANEL_DIR,
    PROMO_COL,
    RAW_TRAIN_CSV,
    TARGET_COL,
    USE_COLS,
)

INDEX_FILE = PANEL_DIR / "_series_index.csv"


# --- storage backend -----------------------------------------------------------

def _has_parquet() -> bool:
    for engine in ("pyarrow", "fastparquet"):
        try:
            __import__(engine)
            return True
        except ImportError:
            continue
    return False


PARQUET = _has_parquet()
EXT = ".parquet" if PARQUET else ".pkl"


def _write(df: pd.DataFrame, path: Path) -> None:
    if PARQUET:
        df.to_parquet(path, index=False)
    else:
        df.to_pickle(path, compression="infer")


def _read(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if PARQUET:
        return pd.read_parquet(path, columns=columns)
    df = pd.read_pickle(path)
    return df[columns] if columns else df


def partition_path(store_nbr: int) -> Path:
    return PANEL_DIR / f"store_{int(store_nbr):02d}{EXT}"


def is_built() -> bool:
    return INDEX_FILE.exists() and any(PANEL_DIR.glob(f"store_*{EXT}"))


# --- build ---------------------------------------------------------------------

def build(csv_path: Path = RAW_TRAIN_CSV, force: bool = False) -> pd.DataFrame:
    """Convert the raw CSV into per-store partitions. Returns the series index.

    Memory is bounded by ``CHUNK_SIZE`` rows plus the largest store's slice —
    the full CSV is never materialised.
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Raw dataset not found at {csv_path}. See data/README.md.")

    if is_built() and not force:
        return pd.read_csv(INDEX_FILE)

    if PANEL_DIR.exists() and force:
        shutil.rmtree(PANEL_DIR)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)

    buffers: dict[int, list[pd.DataFrame]] = {}
    n_rows = 0
    reader = pd.read_csv(
        csv_path, usecols=USE_COLS, parse_dates=[DATE_COL], chunksize=CHUNK_SIZE
    )
    for chunk in reader:
        n_rows += len(chunk)
        for store_nbr, part in chunk.groupby("store_nbr", sort=False):
            buffers.setdefault(int(store_nbr), []).append(part)

    index_rows = []
    for store_nbr, parts in sorted(buffers.items()):
        df = (
            pd.concat(parts, ignore_index=True)
            .sort_values([*KEY_COLS, DATE_COL])
            .reset_index(drop=True)
        )
        df[TARGET_COL] = df[TARGET_COL].astype("float32")
        df[PROMO_COL] = df[PROMO_COL].astype("int32")
        df["store_nbr"] = df["store_nbr"].astype("int16")
        df["family"] = df["family"].astype("category")
        _write(df, partition_path(store_nbr))

        for family, series in df.groupby("family", observed=True):
            index_rows.append(
                {
                    "store_nbr": store_nbr,
                    "family": family,
                    "n_obs": len(series),
                    "start": series[DATE_COL].min().date(),
                    "end": series[DATE_COL].max().date(),
                }
            )

    index = pd.DataFrame(index_rows)
    index.to_csv(INDEX_FILE, index=False)
    print(
        f"[panel] {n_rows:,} rows -> {len(buffers)} store partitions "
        f"({len(index)} series) in {PANEL_DIR}  format={EXT}"
    )
    return index


# --- read ----------------------------------------------------------------------

def series_index() -> pd.DataFrame:
    """Inventory of every (store, family) series in the store."""
    if not INDEX_FILE.exists():
        build()
    return pd.read_csv(INDEX_FILE, parse_dates=["start", "end"])


def read_store(store_nbr: int, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a single store partition (all families)."""
    path = partition_path(store_nbr)
    if not path.exists():
        build()
    if not path.exists():
        raise FileNotFoundError(f"No partition for store {store_nbr} at {path}")
    return _read(path, columns)


def read_series(store_nbr: int, family: str) -> pd.DataFrame:
    """Read one (store, family) series as a date/sales/onpromotion frame."""
    df = read_store(store_nbr)
    out = df[df["family"] == family]
    if out.empty:
        raise KeyError(f"No series for store={store_nbr} family={family!r}")
    return (
        out[[DATE_COL, TARGET_COL, PROMO_COL]]
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )


def iter_stores(stores: list[int] | None = None):
    """Yield ``(store_nbr, dataframe)`` one partition at a time.

    This is the scalability primitive: analysis and training stream over
    partitions, so peak memory is one store rather than the whole panel.
    """
    if not is_built():
        build()
    paths = sorted(PANEL_DIR.glob(f"store_*{EXT}"))
    for path in paths:
        store_nbr = int(path.stem.split("_")[1])
        if stores is not None and store_nbr not in stores:
            continue
        yield store_nbr, _read(path)


def read_panel(stores: list[int] | None = None) -> pd.DataFrame:
    """Concatenate selected store partitions. Use ``iter_stores`` when possible."""
    return pd.concat([df for _, df in iter_stores(stores)], ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the partitioned panel store.")
    parser.add_argument("--force", action="store_true", help="rebuild from scratch")
    args = parser.parse_args()
    build(force=args.force)
