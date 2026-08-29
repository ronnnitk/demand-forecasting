"""Tests for probabilistic metrics and the inventory policy.

The economics must be right: a higher service level must never lower the reorder
point, and a wider forecast interval must raise safety stock.
"""

import numpy as np
import pandas as pd
import pytest

from src.inventory import (
    inventory_plan,
    reorder_point_for_series,
    z_score,
)
from src.metrics import interval_coverage, pinball_loss


# --- metrics -------------------------------------------------------------------

def test_pinball_loss_is_zero_for_perfect_forecast():
    y = np.array([10.0, 20.0, 30.0])
    assert pinball_loss(y, y, 0.5) == pytest.approx(0.0)


def test_pinball_penalizes_under_and_over_asymmetrically():
    y = np.array([10.0])
    # For the P90, under-forecasting should hurt much more than over-forecasting.
    under = pinball_loss(y, np.array([8.0]), 0.9)
    over = pinball_loss(y, np.array([12.0]), 0.9)
    assert under > over


def test_interval_coverage_counts_inside_band():
    y = np.array([1.0, 5.0, 9.0, 12.0])
    lower = np.array([0.0, 4.0, 10.0, 0.0])
    upper = np.array([2.0, 6.0, 11.0, 20.0])
    # inside: idx 0,1,3 ; outside: idx 2 -> 3/4
    assert interval_coverage(y, lower, upper) == pytest.approx(0.75)


# --- inventory -----------------------------------------------------------------

def test_z_score_matches_known_service_levels():
    assert z_score(0.95) == pytest.approx(1.645, abs=1e-3)
    assert z_score(0.975) == pytest.approx(1.960, abs=1e-3)
    assert z_score(0.50) == pytest.approx(0.0, abs=1e-6)


def test_higher_service_level_raises_reorder_point():
    point = np.full(7, 10.0)
    lower = np.full(7, 6.0)
    upper = np.full(7, 14.0)
    low = reorder_point_for_series(point, lower, upper, lead_time=7, service_level=0.90)
    high = reorder_point_for_series(point, lower, upper, lead_time=7, service_level=0.99)
    assert high["reorder_point"] > low["reorder_point"]
    assert high["mean_lead_demand"] == pytest.approx(70.0)  # 7 days x 10


def test_wider_interval_means_more_safety_stock():
    point = np.full(7, 10.0)
    narrow = reorder_point_for_series(point, np.full(7, 9.0), np.full(7, 11.0), service_level=0.95)
    wide = reorder_point_for_series(point, np.full(7, 4.0), np.full(7, 16.0), service_level=0.95)
    assert wide["safety_stock"] > narrow["safety_stock"]


def test_zero_uncertainty_gives_zero_safety_stock():
    point = np.full(7, 10.0)
    plan = reorder_point_for_series(point, point.copy(), point.copy(), service_level=0.95)
    assert plan["safety_stock"] == pytest.approx(0.0)
    assert plan["reorder_point"] == pytest.approx(70.0)


def test_inventory_plan_ranks_by_reorder_point():
    dates = pd.date_range("2017-06-01", periods=7, freq="D")

    def band(store, fam, level):
        return pd.DataFrame({
            "store_nbr": store, "family": fam, "date": dates,
            "forecast": float(level),
            "forecast_lower": float(level) * 0.8,
            "forecast_upper": float(level) * 1.2,
        })

    df = pd.concat([band(1, "SMALL", 5), band(2, "BIG", 100)], ignore_index=True)
    plan = inventory_plan(df, lead_time=7, service_level=0.95)
    assert plan.iloc[0]["family"] == "BIG"          # highest reorder point first
    assert set(plan["family"]) == {"BIG", "SMALL"}


def test_inventory_plan_requires_interval_columns():
    df = pd.DataFrame({"store_nbr": [1], "family": ["A"], "date": ["2017-01-01"], "forecast": [1.0]})
    with pytest.raises(ValueError):
        inventory_plan(df)
