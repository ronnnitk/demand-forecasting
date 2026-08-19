"""Seasonality, trend and autocorrelation analysis.

The original feature set used ``dayofweek``/``month``/``dayofmonth`` on faith,
without ever measuring whether those cycles exist or how strong they are.
This module quantifies them, which is what justifies (or kills) a feature.

Implemented directly on numpy/pandas so the package keeps its light
dependency footprint — no statsmodels required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DATE_COL, TARGET_COL


def seasonal_indices(series: pd.DataFrame, by: str = "dayofweek") -> pd.DataFrame:
    """Multiplicative seasonal index per calendar bucket.

    An index of 1.25 for Saturday means Saturday sells 25% above the series
    average. Computed against a centred 7-day moving average so that trend
    does not leak into the seasonal estimate.
    """
    df = series[[DATE_COL, TARGET_COL]].copy()
    df["trend"] = df[TARGET_COL].rolling(7, center=True, min_periods=4).mean()
    df["detrended"] = np.where(df["trend"] > 0, df[TARGET_COL] / df["trend"], np.nan)

    if by == "dayofweek":
        df["bucket"] = df[DATE_COL].dt.dayofweek
        labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    elif by == "month":
        df["bucket"] = df[DATE_COL].dt.month
        labels = None
    elif by == "dayofmonth":
        df["bucket"] = df[DATE_COL].dt.day
        labels = None
    else:
        raise ValueError(f"unsupported bucket: {by!r}")

    out = (
        df.groupby("bucket")["detrended"]
        .agg(index="mean", n="count")
        .reset_index()
        .sort_values("bucket")
    )
    # Normalise so the indices average exactly 1.0.
    mean_index = out["index"].mean()
    if mean_index and np.isfinite(mean_index):
        out["index"] = out["index"] / mean_index
    if labels:
        out["label"] = [labels[int(b)] for b in out["bucket"]]
    return out.reset_index(drop=True)


def seasonal_strength(series: pd.DataFrame, period: int = 7) -> float:
    """Strength of seasonality in [0, 1].

    ``1 - Var(remainder) / Var(detrended)``: the share of non-trend variance
    explained by the repeating seasonal profile. Above ~0.3 the cycle is worth
    modelling; near 0 it is noise.
    """
    values = series[TARGET_COL].to_numpy(float)
    if len(values) < 3 * period:
        return float("nan")

    trend = pd.Series(values).rolling(period, center=True, min_periods=period // 2).mean()
    detrended = pd.Series(values) - trend
    phase = np.arange(len(values)) % period
    seasonal = detrended.groupby(phase).transform("mean")
    remainder = detrended - seasonal

    var_detrended = np.nanvar(detrended)
    if not var_detrended:
        return 0.0
    return float(max(0.0, 1 - np.nanvar(remainder) / var_detrended))


def autocorrelation(series: pd.DataFrame, max_lag: int = 35) -> pd.DataFrame:
    """Autocorrelation function — evidence for which lags belong in the model."""
    values = series[TARGET_COL].to_numpy(float)
    values = values - values.mean()
    denom = float((values**2).sum())
    if denom == 0:
        return pd.DataFrame({"lag": range(1, max_lag + 1), "acf": np.nan})

    acf = [float((values[lag:] * values[:-lag]).sum() / denom) for lag in range(1, max_lag + 1)]
    out = pd.DataFrame({"lag": range(1, max_lag + 1), "acf": acf})
    # 95% white-noise band; |acf| above this is a real signal.
    out["significant"] = out["acf"].abs() > 1.96 / np.sqrt(len(values))
    return out


def trend_summary(series: pd.DataFrame) -> dict:
    """Linear trend (units/day) plus year-over-year growth."""
    df = series[[DATE_COL, TARGET_COL]].dropna()
    if len(df) < 30:
        return {"slope_per_day": np.nan, "slope_pct_per_year": np.nan, "yoy_growth": np.nan}

    x = (df[DATE_COL] - df[DATE_COL].min()).dt.days.to_numpy(float)
    y = df[TARGET_COL].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)

    mean_y = y.mean()
    slope_pct = float(slope * 365.25 / mean_y) if mean_y > 0 else np.nan

    last_year = df[df[DATE_COL] > df[DATE_COL].max() - pd.Timedelta(days=365)][TARGET_COL].sum()
    prior_year = df[
        (df[DATE_COL] <= df[DATE_COL].max() - pd.Timedelta(days=365))
        & (df[DATE_COL] > df[DATE_COL].max() - pd.Timedelta(days=730))
    ][TARGET_COL].sum()
    yoy = float(last_year / prior_year - 1) if prior_year > 0 else np.nan

    return {
        "slope_per_day": float(slope),
        "intercept": float(intercept),
        "slope_pct_per_year": slope_pct,
        "yoy_growth": yoy,
    }
