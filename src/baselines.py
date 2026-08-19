"""Forecasting baselines.

A model without a baseline is an unfalsifiable claim. These are the references
every forecast must beat before it earns its complexity — they are free to
compute, need no training, and on retail data they are surprisingly hard to
beat. Each takes the training history and returns ``h`` future values.
"""

from __future__ import annotations

import numpy as np

from .config import SEASONAL_PERIOD


def naive(y_train, h: int) -> np.ndarray:
    """Repeat the last observed value."""
    y = np.asarray(y_train, dtype=float).ravel()
    return np.repeat(y[-1] if len(y) else 0.0, h)


def seasonal_naive(y_train, h: int, seasonal_period: int = SEASONAL_PERIOD) -> np.ndarray:
    """Repeat the value from one season ago — "same weekday last week".

    The reference forecast for daily retail demand, and the denominator of MASE.
    """
    y = np.asarray(y_train, dtype=float).ravel()
    if len(y) < seasonal_period:
        return naive(y, h)
    season = y[-seasonal_period:]
    return np.array([season[i % seasonal_period] for i in range(h)], dtype=float)


def moving_average(y_train, h: int, window: int = 28) -> np.ndarray:
    """Flat forecast at the mean of the last ``window`` days."""
    y = np.asarray(y_train, dtype=float).ravel()
    if not len(y):
        return np.zeros(h)
    return np.repeat(float(np.mean(y[-min(window, len(y)):])), h)


def seasonal_moving_average(
    y_train, h: int, seasonal_period: int = SEASONAL_PERIOD, n_seasons: int = 4
) -> np.ndarray:
    """Mean of the last ``n_seasons`` occurrences of each weekday.

    Less noisy than seasonal-naive because it averages several weeks, so a
    single odd Saturday does not propagate into the forecast.
    """
    y = np.asarray(y_train, dtype=float).ravel()
    if len(y) < seasonal_period * 2:
        return seasonal_naive(y, h, seasonal_period)

    profile = np.empty(seasonal_period, dtype=float)
    for phase in range(seasonal_period):
        # Positions matching this phase, counted back from the end of history.
        idx = np.arange(len(y) - seasonal_period + phase, -1, -seasonal_period)[:n_seasons]
        profile[phase] = float(np.mean(y[idx])) if len(idx) else float(np.mean(y))
    return np.array([profile[i % seasonal_period] for i in range(h)], dtype=float)


def croston(y_train, h: int, alpha: float = 0.1) -> np.ndarray:
    """Croston's method for intermittent demand.

    Smooths the non-zero demand *sizes* and the *intervals* between them
    separately, then forecasts size/interval as a flat rate. Designed exactly
    for the 864 sporadic series in this panel, where seasonal-naive mostly
    predicts zero.
    """
    y = np.asarray(y_train, dtype=float).ravel()
    nonzero_idx = np.flatnonzero(y > 0)
    if len(nonzero_idx) < 2:
        return np.repeat(float(np.mean(y)) if len(y) else 0.0, h)

    size = float(y[nonzero_idx[0]])
    interval = float(nonzero_idx[0] + 1)
    since_last = 0
    for value in y[nonzero_idx[0] + 1:]:
        since_last += 1
        if value > 0:
            size += alpha * (value - size)
            interval += alpha * (since_last - interval)
            since_last = 0
    rate = size / interval if interval > 0 else 0.0
    return np.repeat(max(0.0, rate), h)


BASELINES = {
    "naive": naive,
    "seasonal_naive": seasonal_naive,
    "moving_average_28": moving_average,
    "seasonal_moving_average": seasonal_moving_average,
    "croston": croston,
}


def run_baselines(y_train, h: int, names: list[str] | None = None) -> dict[str, np.ndarray]:
    """Run every baseline (or a named subset) and return their forecasts."""
    selected = names or list(BASELINES)
    return {name: BASELINES[name](y_train, h) for name in selected}
