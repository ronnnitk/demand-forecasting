"""Tests for the analysis layer.

Each test builds a series whose answer is known by construction, so the test
checks that the statistic measures what it claims rather than merely running.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.profiling import classify_demand, profile_series
from src.analysis.promotions import promo_uplift
from src.analysis.seasonality import (
    autocorrelation,
    seasonal_indices,
    seasonal_strength,
    trend_summary,
)
from src.analysis.segmentation import abc_xyz, segment_matrix


def _series(sales, promo=None, start="2016-01-01"):
    n = len(sales)
    return pd.DataFrame(
        {
            "date": pd.date_range(start, periods=n, freq="D"),
            "sales": np.asarray(sales, dtype=float),
            "onpromotion": np.zeros(n) if promo is None else np.asarray(promo),
        }
    )


# --- demand classification -----------------------------------------------------

def test_classify_demand_covers_the_four_quadrants():
    assert classify_demand(1.0, 0.1) == "smooth"
    assert classify_demand(1.0, 2.0) == "erratic"
    assert classify_demand(5.0, 0.1) == "intermittent"
    assert classify_demand(5.0, 2.0) == "lumpy"


def test_a_series_that_never_sells_is_flagged_not_scored():
    assert classify_demand(np.inf, np.inf) == "no_demand"
    profile = profile_series(_series(np.zeros(100)))
    assert profile["pattern"] == "no_demand"
    assert profile["zero_share"] == 1.0


def test_daily_seller_is_smooth_and_sparse_seller_is_not():
    steady = profile_series(_series(np.full(200, 10.0) + np.tile([0, 1, -1, 0.5], 50)))
    assert steady["pattern"] == "smooth"
    assert steady["adi"] == pytest.approx(1.0)

    sparse = np.zeros(200)
    sparse[::10] = 5.0
    assert profile_series(_series(sparse))["adi"] == pytest.approx(10.0)


def test_adi_counts_days_per_sale():
    sales = np.zeros(100)
    sales[::4] = 3.0  # sells every 4th day
    assert profile_series(_series(sales))["adi"] == pytest.approx(4.0)


# --- segmentation --------------------------------------------------------------

def test_abc_assigns_the_pareto_head_to_a():
    profile = pd.DataFrame(
        {
            "store_nbr": [1, 1, 1, 1],
            "family": list("PQRS"),
            "total_sales": [1000.0, 100.0, 10.0, 1.0],
            "cv": [0.1, 0.6, 1.5, 0.2],
        }
    )
    out = abc_xyz(profile)
    assert out.iloc[0]["abc"] == "A"
    assert out.iloc[-1]["abc"] == "C"


def test_the_dominant_series_is_a_even_when_it_alone_exceeds_the_cutoff():
    """A series worth 90% of volume must be an A, not a B.

    Classifying on the inclusive cumulative share gets this backwards, because
    the first row already sits past the 80% line.
    """
    profile = pd.DataFrame(
        {
            "store_nbr": [1, 1],
            "family": ["DOMINANT", "TINY"],
            "total_sales": [900.0, 100.0],
            "cv": [0.1, 0.1],
        }
    )
    assert abc_xyz(profile).iloc[0]["abc"] == "A"


def test_xyz_tracks_variability():
    profile = pd.DataFrame(
        {
            "store_nbr": [1, 1, 1],
            "family": list("PQR"),
            "total_sales": [100.0, 100.0, 100.0],
            "cv": [0.1, 0.75, 5.0],
        }
    )
    assert abc_xyz(profile)["xyz"].tolist() == ["X", "Y", "Z"]


def test_segment_matrix_is_always_the_full_nine_box():
    profile = pd.DataFrame(
        {"store_nbr": [1], "family": ["P"], "total_sales": [10.0], "cv": [0.1]}
    )
    grid = segment_matrix(abc_xyz(profile))
    assert grid.shape == (3, 3)
    assert grid.loc["A", "X"] == 1


# --- seasonality ---------------------------------------------------------------

def test_seasonal_indices_recover_a_known_weekly_shape():
    week = [10.0, 10.0, 10.0, 10.0, 10.0, 20.0, 20.0]  # weekend double
    out = seasonal_indices(_series(np.tile(week, 30), start="2016-01-04"), "dayofweek")
    weekend = out[out["bucket"] >= 5]["index"].mean()
    weekday = out[out["bucket"] < 5]["index"].mean()
    assert weekend > weekday
    assert out["index"].mean() == pytest.approx(1.0)


def test_seasonal_strength_is_high_for_seasonal_and_low_for_noise():
    week = np.tile([1.0, 2.0, 3.0, 4.0, 5.0, 20.0, 22.0], 40)
    assert seasonal_strength(_series(week), 7) > 0.5

    rng = np.random.default_rng(0)
    assert seasonal_strength(_series(rng.normal(50, 10, 280)), 7) < 0.35


def test_autocorrelation_peaks_at_the_seasonal_lag():
    y = np.tile([1.0, 2.0, 3.0, 4.0, 5.0, 20.0, 22.0], 40)
    acf = autocorrelation(_series(y), 21)
    assert acf.loc[acf["lag"] == 7, "acf"].iloc[0] > acf.loc[acf["lag"] == 3, "acf"].iloc[0]


def test_trend_summary_detects_direction():
    assert trend_summary(_series(np.arange(400, dtype=float)))["slope_per_day"] > 0
    assert trend_summary(_series(np.arange(400, 0, -1, dtype=float)))["slope_per_day"] < 0


# --- promotions ----------------------------------------------------------------

def test_promo_uplift_recovers_a_known_effect():
    """Promo days built to sell exactly double must report ~+100%."""
    n = 280
    promo = np.zeros(n)
    promo[::14] = 1  # every other Monday, so non-promo Mondays remain as a baseline
    sales = np.full(n, 10.0)
    sales[promo > 0] = 20.0

    stats = promo_uplift(_series(sales, promo, start="2016-01-04"))
    assert stats["uplift_adjusted"] == pytest.approx(1.0, abs=0.05)


def test_promo_uplift_is_nan_without_any_promotions():
    stats = promo_uplift(_series(np.full(50, 5.0)))
    assert np.isnan(stats["uplift_adjusted"])


def test_promo_uplift_controls_for_weekday_confounding():
    """Promos concentrated on naturally-strong weekends must not look causal.

    Raw uplift sees a 10x effect that is purely the weekend. The adjusted
    figure compares each promo Saturday against non-promo Saturdays, so it
    must come back at roughly zero.
    """
    n = 560
    dates = pd.date_range("2016-01-02", periods=n, freq="D")  # starts Saturday
    dow = dates.dayofweek.to_numpy()
    sales = np.where(dow >= 5, 100.0, 10.0)
    # Promo on every other weekend, leaving non-promo weekends as the control.
    promo = ((dow >= 5) & ((np.arange(n) // 7) % 2 == 0)).astype(float)

    stats = promo_uplift(
        pd.DataFrame({"date": dates, "sales": sales, "onpromotion": promo})
    )
    # Raw uplift reports a large spurious effect (the non-promo weekends left
    # in the baseline dilute it, but it is still wildly positive);
    # the weekday-adjusted figure correctly reports ~zero.
    assert stats["uplift_raw"] > 2
    assert abs(stats["uplift_adjusted"]) < 0.1


def test_promo_uplift_is_nan_when_promotion_is_perfectly_confounded():
    """If a weekday is *always* on promotion there is no control, so no answer.

    Returning NaN here is the correct behaviour: the effect is not identifiable
    from this data, and inventing a number would be worse than admitting that.
    """
    n = 280
    dates = pd.date_range("2016-01-02", periods=n, freq="D")
    dow = dates.dayofweek.to_numpy()
    sales = np.where(dow >= 5, 100.0, 10.0)
    promo = (dow >= 5).astype(float)  # every weekend, without exception

    stats = promo_uplift(
        pd.DataFrame({"date": dates, "sales": sales, "onpromotion": promo})
    )
    assert np.isnan(stats["uplift_adjusted"])
