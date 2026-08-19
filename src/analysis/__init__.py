"""Demand analytics: profiling, segmentation, seasonality, promotion effects.

These modules answer the questions that decide *how* to forecast — which
series are even forecastable, which ones carry the revenue, how strong the
weekly and annual seasonality is, and whether promotions move demand — rather
than jumping straight to fitting a model.
"""

from .profiling import classify_demand, profile_panel, profile_series
from .promotions import promo_uplift
from .seasonality import autocorrelation, seasonal_indices, trend_summary
from .segmentation import abc_xyz

__all__ = [
    "abc_xyz",
    "autocorrelation",
    "classify_demand",
    "profile_panel",
    "profile_series",
    "promo_uplift",
    "seasonal_indices",
    "trend_summary",
]
