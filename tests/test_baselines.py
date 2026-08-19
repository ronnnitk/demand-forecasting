"""Tests for the reference forecasts.

Baselines are the yardstick every model is measured against, so a bug here
would quietly move the bar rather than fail loudly.
"""

import numpy as np
import pytest

from src.baselines import (
    croston,
    moving_average,
    naive,
    run_baselines,
    seasonal_moving_average,
    seasonal_naive,
)


def test_naive_repeats_last_value():
    assert np.all(naive([1.0, 2.0, 9.0], 4) == 9.0)


def test_seasonal_naive_repeats_the_weekly_profile():
    y = np.arange(14, dtype=float)  # last week is 7..13
    out = seasonal_naive(y, 10, seasonal_period=7)
    assert list(out[:7]) == [7, 8, 9, 10, 11, 12, 13]
    assert out[7] == 7  # wraps around


def test_seasonal_naive_falls_back_when_history_is_short():
    out = seasonal_naive([5.0, 6.0], 3, seasonal_period=7)
    assert np.all(out == 6.0)


def test_moving_average_uses_only_the_window():
    y = np.concatenate([np.zeros(100), np.full(28, 10.0)])
    assert moving_average(y, 5, window=28)[0] == pytest.approx(10.0)


def test_seasonal_moving_average_preserves_weekday_shape():
    """Weekends must stay higher than weekdays after averaging several weeks."""
    week = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 20.0, 20.0])
    y = np.tile(week, 10)
    out = seasonal_moving_average(y, 7, seasonal_period=7, n_seasons=4)
    assert out[5] == pytest.approx(20.0)
    assert out[0] == pytest.approx(5.0)


def test_croston_estimates_the_demand_rate_for_sparse_series():
    """Demand of 10 every 5th day should forecast a rate near 2/day."""
    y = np.zeros(200)
    y[::5] = 10.0
    rate = croston(y, 3)[0]
    assert 1.0 < rate < 3.0


def test_croston_handles_a_series_that_never_sells():
    assert np.all(croston(np.zeros(50), 5) == 0.0)


def test_all_baselines_return_the_requested_horizon():
    y = np.tile([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 20)
    for name, forecast in run_baselines(y, 15).items():
        assert len(forecast) == 15, name
        assert np.isfinite(forecast).all(), name


def test_baselines_never_look_into_the_future():
    """A baseline given only the first half must be unaffected by the second."""
    y = np.tile([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 20)
    first_half = y[:70]
    tampered = np.concatenate([first_half, np.full(70, 999.0)])
    for name in run_baselines(first_half, 7):
        a = run_baselines(first_half, 7)[name]
        b = run_baselines(tampered[:70], 7)[name]
        assert np.allclose(a, b), name
