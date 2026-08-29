"""Forecast accuracy metrics.

The original project reported MAE plus an "accuracy" of
``1 - MAE / mean(actual)``. Two problems with that as the headline number:

1. **It is not comparable across series.** MAE is in units of the series, so a
   high-volume grocery family and a low-volume one cannot be averaged or ranked.
2. **It has no reference point.** 85% "accuracy" sounds good, but if repeating
   last week's sales scores 87%, the model is actively harmful.

The metrics here fix both. Scaled metrics (MASE, RMSSE) divide the error by the
in-sample error of a seasonal-naive forecast, so **1.0 means "as good as
seasonal naive"**, below 1.0 means better, and the numbers are unitless and
averageable across all 1,782 series. WAPE is the volume-weighted percentage
error that inventory teams actually use, and stays finite when demand is zero —
unlike MAPE, which is undefined on the 31% of days with no sales.
"""

from __future__ import annotations

import numpy as np

from .config import SEASONAL_PERIOD


def _arrays(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return y_true, y_pred


def mae(y_true, y_pred) -> float:
    """Mean absolute error, in units of the series."""
    y_true, y_pred = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error: sum|error| / sum|actual|.

    The percentage error inventory planners use. Finite whenever the series
    sells anything at all over the window, unlike MAPE.
    """
    y_true, y_pred = _arrays(y_true, y_pred)
    denom = float(np.sum(np.abs(y_true)))
    return float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else np.nan


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE in [0, 2]; zero-demand days contribute 0 rather than inf."""
    y_true, y_pred = _arrays(y_true, y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    ratio = np.divide(
        np.abs(y_true - y_pred), denom, out=np.zeros_like(denom), where=denom > 0
    )
    return float(np.mean(ratio))


def bias(y_true, y_pred) -> float:
    """Mean signed error as a share of mean actual: >0 over-forecasting.

    Bias is the metric that costs money — a model can have excellent MAE while
    systematically over-ordering and filling a warehouse.
    """
    y_true, y_pred = _arrays(y_true, y_pred)
    denom = float(np.mean(np.abs(y_true)))
    return float(np.mean(y_pred - y_true) / denom) if denom > 0 else np.nan


def _naive_scale(y_train, seasonal_period: int) -> float:
    """In-sample MAE of a seasonal-naive forecast — the MASE denominator."""
    y_train = np.asarray(y_train, dtype=float).ravel()
    if len(y_train) <= seasonal_period:
        return np.nan
    diffs = np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period])
    scale = float(np.mean(diffs))
    return scale if scale > 0 else np.nan


def mase(y_true, y_pred, y_train, seasonal_period: int = SEASONAL_PERIOD) -> float:
    """Mean absolute scaled error. 1.0 == seasonal naive; < 1.0 beats it."""
    scale = _naive_scale(y_train, seasonal_period)
    if not np.isfinite(scale):
        return np.nan
    return mae(y_true, y_pred) / scale


def rmsse(y_true, y_pred, y_train, seasonal_period: int = SEASONAL_PERIOD) -> float:
    """Root mean squared scaled error (the M5 competition metric)."""
    y_train = np.asarray(y_train, dtype=float).ravel()
    if len(y_train) <= seasonal_period:
        return np.nan
    scale = float(np.mean((y_train[seasonal_period:] - y_train[:-seasonal_period]) ** 2))
    if scale <= 0:
        return np.nan
    y_true, y_pred = _arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2) / scale))


def accuracy_pct(y_true, y_pred) -> float:
    """Legacy metric: MAE as accuracy relative to mean actual (0-100).

    Kept for backward compatibility with the original reports and the API
    response shape. Prefer :func:`wape` and :func:`mase` for new work — see the
    module docstring for why.
    """
    y_true, y_pred = _arrays(y_true, y_pred)
    avg = float(np.mean(y_true))
    denom = avg if avg > 0 else 1.0
    return max(0.0, float((1 - (mae(y_true, y_pred) / denom)) * 100))


def pinball_loss(y_true, y_pred, quantile: float) -> float:
    """Quantile (pinball) loss — the metric a probabilistic forecast is judged on.

    Penalises under-prediction by ``quantile`` and over-prediction by
    ``1 - quantile``, so the minimiser of the P90 loss really is the 90th
    percentile. Averaged over the window.
    """
    y_true, y_pred = _arrays(y_true, y_pred)
    error = y_true - y_pred
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def interval_coverage(y_true, lower, upper) -> float:
    """Share of actuals that fall inside [lower, upper].

    A calibrated 80% interval should cover ~0.80 of actuals. Much less means the
    band is too tight (stock-outs will surprise you); much more means it is too
    wide (you carry needless safety stock).
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    lower = np.asarray(lower, dtype=float).ravel()
    upper = np.asarray(upper, dtype=float).ravel()
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def evaluate_all(y_true, y_pred, y_train=None, seasonal_period: int = SEASONAL_PERIOD) -> dict:
    """Full metric set for one forecast window."""
    out = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "wape": wape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "accuracy_pct": accuracy_pct(y_true, y_pred),
    }
    if y_train is not None:
        out["mase"] = mase(y_true, y_pred, y_train, seasonal_period)
        out["rmsse"] = rmsse(y_true, y_pred, y_train, seasonal_period)
    return out
