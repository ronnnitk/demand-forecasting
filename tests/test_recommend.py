"""Tests for the prescriptive recommender.

The economics must be right (a discount that dilutes margin should never be
recommended) and the action policy must respond correctly to the demand signals.
"""

import numpy as np
import pandas as pd

from src.recommend import (
    constant_elasticity_multiplier,
    elasticity_from_uplift,
    optimize_discount,
    recommend_panel,
    recommend_series,
)


def test_discount_raises_demand_for_negative_elasticity():
    mult = constant_elasticity_multiplier(0.20, elasticity=-1.2)
    assert mult > 1.0                              # cheaper -> more units
    assert constant_elasticity_multiplier(0.0, -1.2) == 1.0


def test_elasticity_inferred_from_uplift_is_more_elastic_than_default():
    # A strongly promo-responsive item should read as more elastic (more negative).
    responsive = elasticity_from_uplift(1.0)       # +100% on promo
    assert responsive < -1.2
    # Missing / non-positive uplift falls back to the default.
    assert elasticity_from_uplift(np.nan) == -1.2
    assert elasticity_from_uplift(-0.1) == -1.2


def test_optimizer_never_recommends_below_cost():
    # Thin 5% margin: no discount in the grid (min 5%) can be profitable.
    plan = optimize_discount(gross_margin=0.05, elasticity=-1.2)
    assert plan["discount"] < 0.05
    assert plan["profit_change"] >= 0.0


def test_optimizer_finds_a_profitable_discount_for_elastic_high_margin_item():
    # A discount only raises profit when |elasticity| > 1/margin (here 1/0.40 = 2.5),
    # so an item must be genuinely elastic before the optimizer marks it down.
    plan = optimize_discount(gross_margin=0.40, elasticity=-3.5)
    assert plan["discount"] > 0.0
    assert plan["profit_change"] > 0.0


def test_optimizer_declines_discount_when_elasticity_below_breakeven():
    # |elasticity| = 2.0 < 1/0.40 = 2.5 -> discounting always loses profit.
    plan = optimize_discount(gross_margin=0.40, elasticity=-2.0)
    assert plan["discount"] == 0.0
    assert plan["profit_change"] == 0.0


def test_softening_elastic_item_gets_a_discount():
    rec = recommend_series(
        trailing_daily=100, forecast_daily=70,       # -30% slowdown
        uplift_adjusted=1.0, gross_margin=0.40,
    )
    assert rec["action"] == "DISCOUNT"
    assert rec["suggested_discount_pct"] > 0


def test_rising_demand_triggers_stock_up_not_discount():
    rec = recommend_series(
        trailing_daily=100, forecast_daily=140, uplift_adjusted=1.0, gross_margin=0.40
    )
    assert rec["action"] == "STOCK_UP"
    assert rec["suggested_discount_pct"] == 0.0


def test_festival_in_horizon_forces_stock_up():
    rec = recommend_series(
        trailing_daily=100, forecast_daily=100, uplift_adjusted=1.0, gross_margin=0.40,
        upcoming_festival=(True, "christmas_season"),
    )
    assert rec["action"] == "STOCK_UP"
    assert "christmas" in rec["rationale"]


def test_volatile_series_is_sent_to_review():
    rec = recommend_series(
        trailing_daily=100, forecast_daily=100, uplift_adjusted=1.0,
        gross_margin=0.40, cv=1.5,
    )
    assert rec["action"] == "REVIEW"


def test_stable_demand_holds():
    rec = recommend_series(
        trailing_daily=100, forecast_daily=101, uplift_adjusted=0.5, gross_margin=0.40
    )
    assert rec["action"] == "HOLD"


def test_recommend_panel_ranks_discounts_first():
    dates = pd.date_range("2017-06-01", periods=15, freq="D")   # no festival window
    hist_dates = pd.date_range("2017-04-15", "2017-05-31", freq="D")

    def series(store, fam, hist_level, fc_level):
        h = pd.DataFrame({"date": hist_dates, "store_nbr": store, "family": fam,
                          "sales": float(hist_level)})
        f = pd.DataFrame({"date": dates, "store_nbr": store, "family": fam,
                          "forecast": float(fc_level)})
        return h, f

    h1, f1 = series(1, "A", 100, 60)     # softening -> DISCOUNT
    h2, f2 = series(2, "B", 100, 105)    # stable -> HOLD
    history = pd.concat([h1, h2], ignore_index=True)
    forecasts = pd.concat([f1, f2], ignore_index=True)
    promo = pd.DataFrame({"store_nbr": [1, 2], "family": ["A", "B"],
                          "uplift_adjusted": [1.2, 0.3]})

    out = recommend_panel(forecasts, history, promo, margins={"A": 0.40, "B": 0.40})
    assert out.iloc[0]["action"] == "DISCOUNT"    # ranked first
    assert set(out["action"]) == {"DISCOUNT", "HOLD"}
