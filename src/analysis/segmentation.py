"""ABC / XYZ segmentation — the 9-box that turns 1,782 series into a policy.

You cannot hand-tune 1,782 forecasts, and you should not spend equal effort on
all of them. Segmentation makes the trade-off explicit:

* **ABC** ranks series by contribution to total volume (Pareto). A-items are
  the ~80% of volume worth modelling carefully.
* **XYZ** ranks series by demand variability (coefficient of variation).
  X is stable and predictable, Z is volatile and may not be worth a complex model.

The AX corner deserves the best model; CZ is a candidate for a cheap baseline
plus safety stock rather than a forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ABC_CUTOFFS, XYZ_CUTOFFS


def _abc_label(cum_share_before: float) -> str:
    """Classify on the cumulative share *before* this item.

    Using the inclusive cumulative share breaks on concentrated panels: a
    single series worth 90% of volume would have cum_share 0.90 > 0.80 and be
    labelled B, which is exactly backwards. Testing the share accumulated by
    everything ranked above it means the largest series is always an A.
    """
    a_cut, b_cut = ABC_CUTOFFS
    if cum_share_before < a_cut:
        return "A"
    return "B" if cum_share_before < b_cut else "C"


def _xyz_label(cv: float) -> str:
    x_cut, y_cut = XYZ_CUTOFFS
    if not np.isfinite(cv):
        return "Z"
    if cv <= x_cut:
        return "X"
    return "Y" if cv <= y_cut else "Z"


def abc_xyz(profile: pd.DataFrame) -> pd.DataFrame:
    """Attach ABC, XYZ and combined segment labels to a panel profile.

    Expects the output of :func:`src.analysis.profiling.profile_panel`
    (needs ``total_sales`` and ``cv`` columns).
    """
    out = profile.sort_values("total_sales", ascending=False).reset_index(drop=True).copy()
    total = out["total_sales"].sum()

    out["sales_share"] = out["total_sales"] / total if total else 0.0
    out["cum_sales_share"] = out["sales_share"].cumsum()
    out["abc"] = (out["cum_sales_share"] - out["sales_share"]).apply(_abc_label)
    out["xyz"] = out["cv"].apply(_xyz_label)
    out["segment"] = out["abc"] + out["xyz"]
    return out


def segment_matrix(segmented: pd.DataFrame, value: str = "n_series") -> pd.DataFrame:
    """9-box pivot of series counts or revenue share."""
    if value == "n_series":
        grid = segmented.pivot_table(
            index="abc", columns="xyz", values="family", aggfunc="size", fill_value=0
        )
    else:
        grid = segmented.pivot_table(
            index="abc", columns="xyz", values="sales_share", aggfunc="sum", fill_value=0.0
        )
    return grid.reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0)


# Model policy implied by the segmentation: where to spend compute.
SEGMENT_POLICY = {
    "AX": "global GBM, full feature set, daily refresh",
    "AY": "global GBM, full feature set, daily refresh",
    "AZ": "global GBM + wider prediction interval; review manually",
    "BX": "global GBM, weekly refresh",
    "BY": "global GBM, weekly refresh",
    "BZ": "seasonal-naive baseline unless GBM beats it on backtest",
    "CX": "seasonal-naive baseline",
    "CY": "seasonal-naive baseline",
    "CZ": "moving-average baseline + safety stock; not worth a model",
}


def policy_table(segmented: pd.DataFrame) -> pd.DataFrame:
    """Series count, revenue share and recommended treatment per segment."""
    out = (
        segmented.groupby("segment")
        .agg(n_series=("family", "size"), sales_share=("sales_share", "sum"))
        .reset_index()
    )
    out["recommended_treatment"] = out["segment"].map(SEGMENT_POLICY)
    return out.sort_values("sales_share", ascending=False).reset_index(drop=True)
