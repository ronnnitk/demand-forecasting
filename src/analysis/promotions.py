"""Promotion-effect analysis.

``onpromotion`` sits in the raw file, is loaded by every version of this
pipeline, and is used by none of them. 20.4% of rows carry a promotion, so if
promotions move demand at all, ignoring the column caps achievable accuracy —
and worse, the model attributes promo-driven spikes to the calendar.

This module measures the effect before it is fed to a model, because
"promotions lift sales" is an assumption until it is a number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DATE_COL, PROMO_COL, TARGET_COL
from ..io_store import iter_stores


def promo_uplift(series: pd.DataFrame) -> dict:
    """Compare demand on promo vs non-promo days for one series.

    The naive difference of means is confounded by day-of-week (promotions
    cluster on weekends) and by trend (promotions grew over time). We control
    for weekday by comparing each promo day against the mean of the *same
    weekday* on non-promo days, then average the ratios.
    """
    if PROMO_COL not in series.columns:
        return {"promo_days": 0, "uplift_raw": np.nan, "uplift_adjusted": np.nan}

    df = series[[DATE_COL, TARGET_COL, PROMO_COL]].copy()
    df["is_promo"] = df[PROMO_COL] > 0
    df["dow"] = df[DATE_COL].dt.dayofweek

    promo, base = df[df["is_promo"]], df[~df["is_promo"]]
    if promo.empty or base.empty:
        return {
            "promo_days": int(len(promo)),
            "promo_share": float(len(promo) / len(df)) if len(df) else 0.0,
            "mean_promo": float(promo[TARGET_COL].mean()) if len(promo) else np.nan,
            "mean_base": float(base[TARGET_COL].mean()) if len(base) else np.nan,
            "uplift_raw": np.nan,
            "uplift_adjusted": np.nan,
        }

    mean_promo, mean_base = float(promo[TARGET_COL].mean()), float(base[TARGET_COL].mean())
    uplift_raw = mean_promo / mean_base - 1 if mean_base > 0 else np.nan

    # Weekday-controlled uplift.
    base_by_dow = base.groupby("dow")[TARGET_COL].mean()
    expected = promo["dow"].map(base_by_dow)
    valid = expected > 0
    uplift_adj = (
        float((promo.loc[valid, TARGET_COL] / expected[valid]).mean() - 1)
        if valid.any()
        else np.nan
    )

    return {
        "promo_days": int(len(promo)),
        "promo_share": float(len(promo) / len(df)),
        "mean_promo": mean_promo,
        "mean_base": mean_base,
        "uplift_raw": float(uplift_raw) if np.isfinite(uplift_raw) else np.nan,
        "uplift_adjusted": uplift_adj,
        "mean_promo_intensity": float(promo[PROMO_COL].mean()),
    }


def promo_panel(stores: list[int] | None = None) -> pd.DataFrame:
    """Promotion uplift for every series, streamed by store partition."""
    rows = []
    for store_nbr, df in iter_stores(stores):
        for family, series in df.groupby("family", observed=True):
            stats = promo_uplift(series.sort_values(DATE_COL))
            rows.append({"store_nbr": store_nbr, "family": str(family), **stats})
    return pd.DataFrame(rows)


def uplift_by_family(promo_stats: pd.DataFrame) -> pd.DataFrame:
    """Roll promotion uplift up to product-family level across all stores.

    Series-level uplift is noisy; the family view is what a merchandising team
    can actually act on.
    """
    out = (
        promo_stats.groupby("family")
        .agg(
            n_series=("store_nbr", "size"),
            promo_days=("promo_days", "sum"),
            median_uplift_adjusted=("uplift_adjusted", "median"),
            mean_uplift_adjusted=("uplift_adjusted", "mean"),
            median_uplift_raw=("uplift_raw", "median"),
            mean_promo_intensity=("mean_promo_intensity", "mean"),
        )
        .reset_index()
    )
    return out.sort_values("median_uplift_adjusted", ascending=False).reset_index(drop=True)
