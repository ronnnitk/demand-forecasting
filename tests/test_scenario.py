"""Tests for the scenario planner (what-if simulation over the forecast).

Two layers, tested separately: the pure economics (price elasticity, profit
accounting) need no model at all, while the promotion lever re-runs the
recursive forecast — exercised here against a tiny deterministic stand-in
model rather than a real fitted `HistGradientBoostingRegressor`, so the tests
stay fast and pin the *wiring* (promo plan reaches the model; the model's
response reaches the scenario's numbers) without depending on a training run.
"""

import numpy as np
import pandas as pd
import pytest

from src.scenario import (
    BUSINESS_AS_USUAL,
    Scenario,
    compare_scenarios,
    default_scenarios,
    price_multiplier,
    promo_intensity,
    scenario_economics,
    simulate_series,
)


# --- price elasticity --------------------------------------------------------

def test_price_cut_raises_demand_for_negative_elasticity():
    assert price_multiplier(-0.10, elasticity=-1.2) > 1.0


def test_price_rise_lowers_demand_for_negative_elasticity():
    assert price_multiplier(0.10, elasticity=-1.2) < 1.0


def test_no_price_change_is_neutral():
    assert price_multiplier(0.0, elasticity=-1.2) == pytest.approx(1.0)


def test_price_multiplier_brackets_one_around_zero():
    lo = price_multiplier(-1e-4, -1.2)
    hi = price_multiplier(1e-4, -1.2)
    assert lo > 1.0 > hi


def test_extreme_price_moves_stay_finite():
    # A ">100% cut" and a huge markup are clamped, not blown up to inf/nan.
    assert np.isfinite(price_multiplier(-5.0, -1.2))
    assert np.isfinite(price_multiplier(50.0, -1.2))


# --- promo intensity resolution ----------------------------------------------

def _promo_history():
    return pd.DataFrame(
        {
            "date": pd.date_range("2017-01-01", periods=10, freq="D"),
            "onpromotion": [0, 0, 4, 4, 0, 0, 6, 0, 0, 0],
        }
    )


def test_promo_intensity_none_passes_through():
    assert promo_intensity(_promo_history(), None) is None


def test_promo_intensity_false_forces_zero():
    assert promo_intensity(_promo_history(), False) == 0.0


def test_promo_intensity_explicit_number_is_used_directly():
    assert promo_intensity(_promo_history(), 3.5) == 3.5


def test_promo_intensity_true_averages_the_series_own_on_days():
    assert promo_intensity(_promo_history(), True) == pytest.approx(np.mean([4, 4, 6]))


def test_promo_intensity_true_falls_back_when_series_never_promoted():
    never = pd.DataFrame({"date": pd.date_range("2017-01-01", periods=5, freq="D"), "onpromotion": 0})
    assert promo_intensity(never, True) == 1.0


# --- scenario economics --------------------------------------------------------

def test_scenario_economics_bau_has_zero_deltas():
    daily = np.full(15, 12.0)
    econ = scenario_economics(daily, daily, gross_margin=0.30, price_change_pct=0.0)
    assert econ["demand_delta_pct"] == 0.0
    assert econ["revenue_delta_pct"] == 0.0
    assert econ["profit_delta_pct"] == 0.0


def test_scenario_economics_matches_recommend_breakeven_logic():
    # Same breakeven fact `test_recommend.py` pins for optimize_discount:
    # |elasticity| below 1/margin means a cut always dilutes profit even
    # though it sells more units.
    baseline = np.full(15, 100.0)
    mult = price_multiplier(-0.10, elasticity=-2.0)  # breakeven at 1/0.40 = 2.5
    scenario = baseline * mult
    econ = scenario_economics(baseline, scenario, gross_margin=0.40, price_change_pct=-0.10)
    assert econ["demand_delta_pct"] > 0
    assert econ["profit_delta_pct"] < 0


def test_scenario_economics_profitable_when_elastic_enough():
    baseline = np.full(15, 100.0)
    mult = price_multiplier(-0.10, elasticity=-3.5)
    scenario = baseline * mult
    econ = scenario_economics(baseline, scenario, gross_margin=0.40, price_change_pct=-0.10)
    assert econ["profit_delta_pct"] > 0


def test_scenario_economics_price_rise_can_beat_bau_when_inelastic():
    baseline = np.full(15, 100.0)
    mult = price_multiplier(0.05, elasticity=-1.2)  # well under 1/0.25=4.0 breakeven
    scenario = baseline * mult
    econ = scenario_economics(baseline, scenario, gross_margin=0.25, price_change_pct=0.05)
    assert econ["demand_delta_pct"] < 0     # fewer units
    assert econ["profit_delta_pct"] > 0      # but more profitable per unit wins out


# --- simulate_series / compare_scenarios, against a fake model ---------------

class _TogglePromoModel:
    """Deterministic stand-in for a fitted model: a flat level plus a bump
    whenever `promo_flag` is on. Lets the promo lever's wiring be tested
    without a real HistGradientBoostingRegressor fit."""

    def __init__(self, base: float = 20.0, promo_bump: float = 8.0):
        self.base = base
        self.promo_bump = promo_bump

    def predict(self, X):
        out = np.full(len(X), self.base, dtype=float)
        if "promo_flag" in X.columns:
            out = out + self.promo_bump * X["promo_flag"].to_numpy(float)
        return out


def _fake_history_and_meta(n: int = 60):
    from src.features import build_family_codes, build_features, feature_columns

    dates = pd.date_range("2017-01-01", periods=n, freq="D")
    history = pd.DataFrame(
        {"date": dates, "store_nbr": 1, "family": "A", "sales": 20.0, "onpromotion": 0.0}
    )
    family_codes = build_family_codes(["A"])
    feats = build_features(history, family_codes=family_codes)
    meta = {"feature_columns": feature_columns(feats), "family_codes": family_codes}
    return history, meta


def test_business_as_usual_matches_the_plain_forecast():
    history, meta = _fake_history_and_meta()
    result = simulate_series(history, _TogglePromoModel(), meta, BUSINESS_AS_USUAL, days=7)
    assert result["profit_delta_pct"] == 0.0
    assert np.allclose(result["daily"]["baseline"], result["daily"]["scenario"])


def test_promotion_scenario_lifts_demand_over_baseline():
    history, meta = _fake_history_and_meta()
    result = simulate_series(
        history, _TogglePromoModel(promo_bump=8.0), meta,
        Scenario(name="promo", promotion=5.0), days=7,
    )
    assert result["demand_delta_pct"] > 0
    assert (result["daily"]["scenario"] > result["daily"]["baseline"]).all()


def test_forcing_promotion_off_never_raises_demand():
    history, meta = _fake_history_and_meta()
    result = simulate_series(
        history, _TogglePromoModel(promo_bump=8.0), meta,
        Scenario(name="off", promotion=False), days=7,
    )
    assert result["demand_delta_pct"] <= 0.0


def test_demand_shock_scales_the_whole_path():
    history, meta = _fake_history_and_meta()
    result = simulate_series(
        history, _TogglePromoModel(), meta,
        Scenario(name="shock", demand_multiplier=1.5), days=7,
    )
    assert result["daily"]["scenario"].to_numpy() == pytest.approx(
        result["daily"]["baseline"].to_numpy() * 1.5, rel=1e-6
    )


def test_default_scenarios_have_no_duplicate_of_business_as_usual():
    names = [s.name for s in default_scenarios()]
    assert names[0] == "business_as_usual"
    assert len(names) == len(set(names))
    # No other scenario is a no-op duplicate of BAU (same promo, same price).
    for sc in default_scenarios()[1:]:
        assert sc.promotion is not None or sc.price_change_pct != 0.0


def test_compare_scenarios_ranks_by_profit_and_includes_bau():
    history, meta = _fake_history_and_meta()
    table = compare_scenarios(history, _TogglePromoModel(), meta, days=7)
    assert "business_as_usual" in set(table["scenario"])
    assert table["profit_delta_pct"].is_monotonic_decreasing
    bau_row = table[table["scenario"] == "business_as_usual"].iloc[0]
    assert bau_row["profit_delta_pct"] == 0.0
