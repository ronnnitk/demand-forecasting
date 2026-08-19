"""Series profiling and demand-pattern classification.

31% of rows in this dataset are zero sales. That single fact governs model
choice: a series that sells every day and a series that sells eight times a
year need different treatment, and averaging them into one accuracy number
hides both. The Syntetos-Boylan scheme below splits series on two axes:

* **ADI** — average demand interval, mean days between non-zero sales.
  High ADI means sporadic demand.
* **CV²** — squared coefficient of variation of the *non-zero* demand sizes.
  High CV² means the size of a sale is unpredictable when it happens.

    ADI < 1.32, CV² < 0.49  -> smooth        (regular, easy)
    ADI < 1.32, CV² >= 0.49 -> erratic       (regular timing, volatile size)
    ADI >= 1.32, CV² < 0.49 -> intermittent  (sporadic, stable size)
    ADI >= 1.32, CV² >= 0.49-> lumpy         (sporadic and volatile: hardest)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import (
    ADI_CUTOFF,
    CV2_CUTOFF,
    DATE_COL,
    PROMO_COL,
    TARGET_COL,
)
from ..io_store import iter_stores


def classify_demand(adi: float, cv2: float) -> str:
    """Syntetos-Boylan demand-pattern label."""
    if not np.isfinite(adi) or not np.isfinite(cv2):
        return "no_demand"
    if adi < ADI_CUTOFF:
        return "smooth" if cv2 < CV2_CUTOFF else "erratic"
    return "intermittent" if cv2 < CV2_CUTOFF else "lumpy"


def profile_series(series: pd.DataFrame) -> dict:
    """Descriptive + forecastability statistics for one demand series."""
    sales = pd.to_numeric(series[TARGET_COL], errors="coerce").fillna(0.0).to_numpy(float)
    n = len(sales)
    nonzero = sales[sales > 0]
    n_nonzero = len(nonzero)

    adi = n / n_nonzero if n_nonzero else np.inf
    if n_nonzero:
        mean_nz = nonzero.mean()
        cv2 = float((nonzero.std(ddof=0) / mean_nz) ** 2) if mean_nz > 0 else np.inf
    else:
        cv2 = np.inf

    mean_all = float(sales.mean()) if n else 0.0
    cv_all = float(sales.std(ddof=0) / mean_all) if mean_all > 0 else np.inf

    # Share of the series' total volume in its final 90 days vs the prior 90 —
    # a cheap, robust trend read that does not assume linearity.
    recent = sales[-90:].sum()
    previous = sales[-180:-90].sum() if n >= 180 else np.nan
    growth = float(recent / previous - 1) if previous and previous > 0 else np.nan

    return {
        "n_obs": n,
        "n_nonzero": n_nonzero,
        "zero_share": float(1 - n_nonzero / n) if n else 1.0,
        "total_sales": float(sales.sum()),
        "mean_sales": mean_all,
        "median_sales": float(np.median(sales)) if n else 0.0,
        "std_sales": float(sales.std(ddof=0)) if n else 0.0,
        "cv": cv_all,
        "max_sales": float(sales.max()) if n else 0.0,
        "adi": float(adi),
        "cv2_nonzero": float(cv2),
        "pattern": classify_demand(adi, cv2),
        "growth_90d": growth,
        "promo_days": int((series[PROMO_COL] > 0).sum()) if PROMO_COL in series else 0,
    }


def profile_panel(stores: list[int] | None = None) -> pd.DataFrame:
    """Profile every series in the panel, streaming one store partition at a time."""
    rows = []
    for store_nbr, df in iter_stores(stores):
        for family, series in df.groupby("family", observed=True):
            series = series.sort_values(DATE_COL)
            rows.append(
                {
                    "store_nbr": store_nbr,
                    "family": str(family),
                    **profile_series(series),
                }
            )
    return pd.DataFrame(rows).sort_values("total_sales", ascending=False).reset_index(drop=True)


def pattern_summary(profile: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a panel profile by demand pattern.

    Answers the question that matters commercially: what fraction of *revenue*
    sits in each forecastability class, not just what fraction of series.
    """
    total = profile["total_sales"].sum()
    out = (
        profile.groupby("pattern")
        .agg(
            n_series=("family", "size"),
            total_sales=("total_sales", "sum"),
            mean_zero_share=("zero_share", "mean"),
            median_adi=("adi", "median"),
            median_cv2=("cv2_nonzero", "median"),
        )
        .reset_index()
    )
    out["series_share"] = out["n_series"] / len(profile)
    out["sales_share"] = out["total_sales"] / total if total else 0.0
    return out.sort_values("total_sales", ascending=False).reset_index(drop=True)
