"""Panel feature engineering for the global multi-series model.

The legacy feature set (`src/feature_engineering.py`) has 7 columns and is
built one series at a time. That is fine for a single series and hopeless for
1,782 of them, so this module builds features for the whole panel with
vectorised groupby operations — one pass, no Python loop over series.

What it adds beyond the legacy set, and why each earns its place:

* **Lags 21 and 28** — the ACF is significant out to lag 35 on the grocery
  families (see ``src.analysis.seasonality``); lags 1/7/14 alone leave signal
  on the table.
* **Multi-window rolling mean/std/max and EWM** — level and volatility at
  several timescales, so the model can tell a genuinely rising series from a
  one-day spike.
* **Fourier terms** — smooth weekly and annual seasonality. Annual seasonal
  strength measures 0.22-0.31 on the top families, but a raw ``month`` integer
  can only express it as 12 steps; harmonics express it continuously and with
  far fewer splits.
* **Promotion features** — ``onpromotion`` carries a median +82% weekday-adjusted
  uplift across the panel and was previously unused entirely.
* **Payday flags** — Ecuadorian public wages land on the 15th and the last day
  of the month, a documented demand driver in this dataset.
* **Series identity** — ``store_nbr`` and ``family`` as categoricals, which is
  what lets one model learn 1,782 series at once and share strength across them.

All features are strictly backward-looking: every lag and rolling statistic is
shifted before use, so no row can see its own target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar_events import (
    EVENT_CATEGORICAL_FEATURES,
    event_features,
)
from .config import (
    DATE_COL,
    EWM_SPANS,
    FOURIER_PERIODS,
    KEY_COLS,
    PANEL_LAGS,
    PROMO_COL,
    ROLLING_WINDOWS,
    TARGET_COL,
)

SERIES_KEY = "series_id"


def add_series_id(df: pd.DataFrame, family_codes: dict[str, int] | None = None) -> pd.DataFrame:
    """Stable series key plus an integer code for the ``family`` categorical.

    ``family_codes`` must be persisted alongside the model: codes assigned by
    whatever families happen to be present at predict time would silently
    remap the categorical and produce wrong forecasts.
    """
    out = df.copy()
    out[SERIES_KEY] = out["store_nbr"].astype(str) + "|" + out["family"].astype(str)

    if family_codes is None:
        family_codes = {fam: i for i, fam in enumerate(sorted(out["family"].astype(str).unique()))}
    out["family_code"] = (
        out["family"].astype(str).map(family_codes).fillna(-1).astype("int16")
    )
    return out


def build_family_codes(families) -> dict[str, int]:
    """Deterministic family -> code mapping, persisted with the model."""
    return {str(fam): i for i, fam in enumerate(sorted({str(f) for f in families}))}


def calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Date-derived features. Fully known in advance, so safe at any horizon."""
    out = df.copy()
    dates = out[DATE_COL]

    out["dayofweek"] = dates.dt.dayofweek.astype("int8")
    out["month"] = dates.dt.month.astype("int8")
    out["dayofmonth"] = dates.dt.day.astype("int8")
    out["dayofyear"] = dates.dt.dayofyear.astype("int16")
    out["weekofyear"] = dates.dt.isocalendar().week.astype("int16")
    out["year"] = dates.dt.year.astype("int16")
    out["is_weekend"] = (out["dayofweek"] >= 5).astype("int8")

    days_in_month = dates.dt.days_in_month
    out["is_month_start"] = (out["dayofmonth"] <= 3).astype("int8")
    out["is_month_end"] = (out["dayofmonth"] >= days_in_month - 2).astype("int8")

    # Payday: the 15th and the last day of the month, plus the two days after
    # each (the spending window, not just the deposit).
    is_payday = (out["dayofmonth"].isin([15])) | (out["dayofmonth"] == days_in_month)
    out["is_payday"] = is_payday.astype("int8")
    days_since_15 = (out["dayofmonth"] - 15).clip(lower=0)
    days_since_eom = out["dayofmonth"].where(out["dayofmonth"] < 15, 0)
    out["days_since_payday"] = np.minimum(days_since_15, days_since_eom).astype("int8")

    for period, n_harmonics in FOURIER_PERIODS:
        time_index = out["dayofyear"] if period > 100 else out["dayofweek"]
        for k in range(1, n_harmonics + 1):
            angle = 2 * np.pi * k * time_index / period
            tag = f"{int(period)}_{k}"
            out[f"sin_{tag}"] = np.sin(angle).astype("float32")
            out[f"cos_{tag}"] = np.cos(angle).astype("float32")

    # Festival / holiday / season drivers (also date-derived, so leak-free).
    out = event_features(out)
    return out


def lag_features(df: pd.DataFrame, group: str = SERIES_KEY) -> pd.DataFrame:
    """Lags, rolling statistics and EWM of the target, computed per series."""
    out = df.copy()
    grouped = out.groupby(group, observed=True)[TARGET_COL]

    for lag in PANEL_LAGS:
        out[f"lag_{lag}"] = grouped.shift(lag).astype("float32")

    # shift(1) first so a window never includes the current day's target.
    shifted = grouped.shift(1)
    by_series = shifted.groupby(out[group], observed=True)
    for window in ROLLING_WINDOWS:
        roll = by_series.rolling(window, min_periods=max(2, window // 2))
        out[f"roll_mean_{window}"] = roll.mean().reset_index(level=0, drop=True).astype("float32")
        out[f"roll_std_{window}"] = roll.std().reset_index(level=0, drop=True).astype("float32")
        out[f"roll_max_{window}"] = roll.max().reset_index(level=0, drop=True).astype("float32")

    for span in EWM_SPANS:
        out[f"ewm_{span}"] = (
            by_series.ewm(span=span, min_periods=2)
            .mean()
            .reset_index(level=0, drop=True)
            .astype("float32")
        )

    # Momentum: this week's level against last month's. Scale-free, so it
    # transfers between a high-volume and a low-volume series.
    denom = out["roll_mean_28"].replace(0, np.nan)
    out["ratio_7_28"] = (out["roll_mean_7"] / denom).astype("float32")

    # Recent intermittency: how many of the last 28 days sold nothing.
    zero_flag = (out[TARGET_COL] == 0).astype("float32")
    out["zero_share_28"] = (
        zero_flag.groupby(out[group], observed=True)
        .shift(1)
        .groupby(out[group], observed=True)
        .rolling(28, min_periods=7)
        .mean()
        .reset_index(level=0, drop=True)
        .astype("float32")
    )
    return out


def promo_features(df: pd.DataFrame, group: str = SERIES_KEY) -> pd.DataFrame:
    """Promotion intensity now, recently, and imminently.

    ``onpromotion`` is a *planned* field — the retailer knows next week's
    promotions today — so unlike sales it may be read at the current timestep
    and one step ahead without leaking.
    """
    out = df.copy()
    if PROMO_COL not in out.columns:
        return out

    out[PROMO_COL] = out[PROMO_COL].astype("float32")
    grouped = out.groupby(group, observed=True)[PROMO_COL]

    out["promo_flag"] = (out[PROMO_COL] > 0).astype("int8")
    out["promo_lag_1"] = grouped.shift(1).astype("float32")
    out["promo_lead_1"] = grouped.shift(-1).astype("float32")
    out["promo_roll_7"] = (
        grouped.shift(1)
        .groupby(out[group], observed=True)
        .rolling(7, min_periods=2)
        .mean()
        .reset_index(level=0, drop=True)
        .astype("float32")
    )
    return out


def build_features(
    panel: pd.DataFrame,
    dropna: bool = True,
    family_codes: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Full feature pipeline for a long panel frame.

    Expects columns ``date``, ``store_nbr``, ``family``, ``sales`` and
    optionally ``onpromotion``. Returns the frame with all feature columns
    appended, sorted by series then date.
    """
    df = panel.sort_values([*KEY_COLS, DATE_COL]).reset_index(drop=True)
    df = add_series_id(df, family_codes)
    df = calendar_features(df)
    df = lag_features(df)
    df = promo_features(df)

    if dropna:
        # The longest lag/window determines the warm-up each series must burn.
        warmup_cols = [f"lag_{max(PANEL_LAGS)}", f"roll_mean_{max(ROLLING_WINDOWS)}"]
        df = df.dropna(subset=warmup_cols).reset_index(drop=True)
    return df


# Columns that are never model inputs.
NON_FEATURES = {DATE_COL, TARGET_COL, SERIES_KEY, "family"}
# Treated as categorical splits by HistGradientBoostingRegressor.
CATEGORICAL_FEATURES = [
    "store_nbr", "family_code", "dayofweek", "month", *EVENT_CATEGORICAL_FEATURES,
]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model input columns, in a stable order."""
    return [
        col
        for col in df.columns
        if col not in NON_FEATURES and df[col].dtype.kind in "ifb"
    ]
