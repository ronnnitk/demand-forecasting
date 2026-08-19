"""Tests for the forecast metrics.

These matter more than usual: a metric bug does not crash anything, it just
silently reports the wrong quality and every downstream decision inherits it.
"""

import numpy as np
import pytest

from src.metrics import accuracy_pct, bias, evaluate_all, mae, mase, rmsse, smape, wape


def test_perfect_forecast_scores_zero_error():
    y = np.array([10.0, 20.0, 30.0])
    assert mae(y, y) == 0.0
    assert wape(y, y) == 0.0
    assert smape(y, y) == 0.0
    assert bias(y, y) == 0.0


def test_wape_is_scale_free_ratio():
    y_true = np.array([100.0, 100.0])
    y_pred = np.array([110.0, 90.0])
    # total absolute error 20 over total actual 200
    assert wape(y_true, y_pred) == pytest.approx(0.1)


def test_wape_handles_all_zero_actuals():
    """MAPE would divide by zero here; WAPE must return NaN, not explode."""
    assert np.isnan(wape(np.zeros(5), np.ones(5)))


def test_smape_ignores_days_where_both_are_zero():
    y_true = np.array([0.0, 10.0])
    y_pred = np.array([0.0, 10.0])
    assert smape(y_true, y_pred) == 0.0


def test_bias_sign_distinguishes_over_and_under_forecasting():
    y_true = np.array([10.0, 10.0])
    assert bias(y_true, np.array([12.0, 12.0])) > 0   # over-forecast
    assert bias(y_true, np.array([8.0, 8.0])) < 0     # under-forecast


def test_mase_of_seasonal_naive_is_about_one():
    """The defining property: MASE ~= 1 when the forecast IS seasonal naive."""
    rng = np.random.default_rng(0)
    season = np.array([5.0, 6.0, 7.0, 6.0, 8.0, 12.0, 13.0])
    y_train = np.tile(season, 30) + rng.normal(0, 0.5, 210)
    y_test = np.tile(season, 2)[:14]
    y_pred = np.tile(y_train[-7:], 2)[:14]

    score = mase(y_test, y_pred, y_train, seasonal_period=7)
    assert 0.3 < score < 3.0


def test_mase_rewards_better_forecasts():
    # Noise is required: a perfectly periodic history has zero seasonal-naive
    # error, which makes the MASE scale undefined (covered separately below).
    rng = np.random.default_rng(1)
    week = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
    y_train = np.tile(week, 20) + rng.normal(0, 2, 140)
    y_test = week

    good = mase(y_test, y_test, y_train, 7)
    bad = mase(y_test, np.full(7, 40.0), y_train, 7)
    assert good < bad


def test_scaled_metrics_return_nan_on_constant_history():
    """A flat series has zero seasonal-naive error, so the scale is undefined."""
    y_train = np.full(50, 7.0)
    assert np.isnan(mase(np.ones(5), np.zeros(5), y_train, 7))
    assert np.isnan(rmsse(np.ones(5), np.zeros(5), y_train, 7))


def test_legacy_accuracy_pct_is_preserved():
    """Backward compatibility with the original reports and the API contract."""
    y_true = np.array([100.0, 100.0])
    y_pred = np.array([90.0, 90.0])
    assert accuracy_pct(y_true, y_pred) == pytest.approx(90.0)


def test_accuracy_pct_never_goes_negative():
    assert accuracy_pct(np.array([1.0, 1.0]), np.array([100.0, 100.0])) == 0.0


def test_evaluate_all_includes_scaled_metrics_only_with_history():
    y = np.array([1.0, 2.0, 3.0])
    assert "mase" not in evaluate_all(y, y)
    assert "mase" in evaluate_all(y, y, y_train=np.tile([1.0, 2.0], 20))


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        mae(np.ones(3), np.ones(4))
