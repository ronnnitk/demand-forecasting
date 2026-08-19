"""Tests for the rolling-origin backtest harness.

The harness is the project's evidence base, so its own correctness — in
particular that no fold can see its test window — is worth asserting directly.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest import backtest_series, rolling_origins, summarize
from src.config import MIN_TRAIN_DAYS
from src.data_processing import regularize


WEEK = [10.0, 12.0, 11.0, 9.0, 13.0, 25.0, 27.0]


def _series(n=600, noise=1.5, seed=0):
    """A weekly-seasonal series with noise.

    The noise is deliberate: a perfectly periodic history has zero
    seasonal-naive error, which makes the MASE scale undefined and every
    scaled metric NaN. Real series are never noiseless.
    """
    rng = np.random.default_rng(seed)
    sales = np.tile(WEEK, n // 7 + 1)[:n]
    if noise:
        sales = np.clip(sales + rng.normal(0, noise, n), 0, None)
    return pd.DataFrame(
        {
            "date": pd.date_range("2016-01-04", periods=n, freq="D"),
            "sales": sales,
            "onpromotion": np.zeros(n),
        }
    )


def test_origins_are_spaced_one_horizon_apart():
    origins = rolling_origins(1000, folds=3, horizon=15)
    assert origins == [955, 970, 985]
    assert all(np.diff(origins) == 15)


def test_origins_are_dropped_when_training_history_is_too_short():
    assert rolling_origins(MIN_TRAIN_DAYS + 5, folds=3, horizon=15) == []


def test_backtest_scores_every_baseline_on_every_fold():
    results = backtest_series(_series(), folds=3, horizon=15)
    assert set(results["fold"]) == {1, 2, 3}
    assert "seasonal_naive" in set(results["model"])
    assert results["mase"].notna().any()


def test_seasonal_naive_is_exact_on_a_noiseless_seasonal_series():
    """Sanity check on the harness: a clean weekly cycle is predicted exactly."""
    results = backtest_series(_series(noise=0), folds=2, horizon=14)
    sn = results[results["model"] == "seasonal_naive"]
    assert sn["mae"].max() == pytest.approx(0.0, abs=1e-9)


def test_a_predictor_cannot_see_beyond_its_origin():
    """The history handed to a model must end at the origin, never later."""
    seen = {}

    def spy(history, horizon):
        seen[len(history)] = history["date"].max()
        return np.zeros(horizon)

    series = _series()
    results = backtest_series(series, predictors={"spy": spy}, folds=3, horizon=15)
    regularized = regularize(series)

    assert seen, "predictor was never called"
    for origin, last_date in seen.items():
        expected = regularized["date"].iloc[origin - 1]
        assert last_date == expected
        # The test window starts strictly after what the model saw.
        assert last_date < regularized["date"].iloc[origin]
    assert "spy" in set(results["model"])


def test_a_failing_predictor_does_not_abort_the_sweep():
    def broken(history, horizon):
        raise RuntimeError("model exploded")

    results = backtest_series(_series(), predictors={"broken": broken}, folds=2, horizon=15)
    # Baselines still scored despite the broken model.
    assert results[results["model"] == "seasonal_naive"]["mase"].notna().any()
    assert "model exploded" in set(results.get("error", pd.Series(dtype=str)).dropna())


def test_summarize_ranks_by_mase_and_reports_lift_over_the_reference():
    results = backtest_series(_series(), folds=3, horizon=15)
    leaderboard = summarize(results)
    assert leaderboard["mase"].is_monotonic_increasing
    assert "vs_seasonal_naive" in leaderboard.columns
    reference = leaderboard[leaderboard["model"] == "seasonal_naive"]
    assert float(reference["vs_seasonal_naive"].iloc[0]) == pytest.approx(0.0)
