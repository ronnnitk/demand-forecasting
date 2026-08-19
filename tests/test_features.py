"""Tests for panel feature engineering.

The critical property is **no leakage**: every lag and rolling statistic must
be strictly backward-looking. A leak here would produce excellent backtest
numbers and a model that fails the moment it is deployed, which is the most
expensive kind of bug in a forecasting system.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    SERIES_KEY,
    build_family_codes,
    build_features,
    calendar_features,
    feature_columns,
    lag_features,
)


def _panel(n_days=120, n_series=3):
    frames = []
    dates = pd.date_range("2016-01-01", periods=n_days, freq="D")
    for i in range(n_series):
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "store_nbr": 1 + i,
                    "family": f"FAM_{i}",
                    "sales": np.arange(n_days, dtype=float) + i * 100,
                    "onpromotion": (np.arange(n_days) % 5 == 0).astype(int),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_build_features_returns_rows_for_every_series():
    out = build_features(_panel())
    assert out[SERIES_KEY].nunique() == 3


def test_lags_do_not_cross_series_boundaries():
    """Series B's first lag must be NaN, not the tail of series A."""
    out = build_features(_panel(n_days=60), dropna=False)
    first_rows = out.groupby(SERIES_KEY).head(1)
    assert first_rows["lag_1"].isna().all()


def test_lag_values_are_correct_within_a_series():
    out = build_features(_panel(n_days=60), dropna=False)
    one = out[out["store_nbr"] == 1].reset_index(drop=True)
    assert one["lag_7"].iloc[10] == pytest.approx(one["sales"].iloc[3])
    assert one["lag_28"].iloc[40] == pytest.approx(one["sales"].iloc[12])


def test_rolling_features_exclude_the_current_day():
    """roll_mean_7 at row i must average rows i-7..i-1, never row i itself."""
    out = build_features(_panel(n_days=60), dropna=False)
    one = out[out["store_nbr"] == 1].reset_index(drop=True)
    expected = one["sales"].iloc[13:20].mean()
    assert one["roll_mean_7"].iloc[20] == pytest.approx(expected)


def test_no_feature_correlates_perfectly_with_the_target():
    """A blunt leakage tripwire: any column equal to sales is a leak."""
    out = build_features(_panel(n_days=200))
    for col in feature_columns(out):
        assert not np.allclose(out[col].fillna(0), out["sales"]), f"{col} leaks the target"


def test_promo_lead_is_allowed_but_promo_never_uses_sales():
    out = build_features(_panel(n_days=60), dropna=False)
    one = out[out["store_nbr"] == 1].reset_index(drop=True)
    # onpromotion is a planned field, so reading one step ahead is legitimate.
    assert one["promo_lead_1"].iloc[5] == pytest.approx(one["onpromotion"].iloc[6])


def test_calendar_features_are_deterministic_from_the_date():
    # 2017-01-14 Sat, 2017-01-15 Sun, 2017-01-31 Tue (and the month end).
    df = pd.DataFrame({"date": pd.to_datetime(["2017-01-14", "2017-01-15", "2017-01-31"])})
    out = calendar_features(df)
    assert out["dayofweek"].tolist() == [5, 6, 1]
    assert out["is_weekend"].tolist() == [1, 1, 0]
    assert out["is_payday"].tolist() == [0, 1, 1]       # the 15th and month end
    assert out["is_month_end"].tolist() == [0, 0, 1]


def test_family_codes_are_stable_across_calls():
    codes_a = build_family_codes(["B", "A", "C"])
    codes_b = build_family_codes(["C", "B", "A"])
    assert codes_a == codes_b


def test_unknown_family_maps_to_sentinel_not_a_silent_remap():
    codes = build_family_codes(["FAM_0", "FAM_1"])
    out = build_features(_panel(n_days=60), dropna=False, family_codes=codes)
    assert (out.loc[out["family"] == "FAM_2", "family_code"] == -1).all()


def test_feature_columns_excludes_identifiers_and_target():
    out = build_features(_panel())
    cols = feature_columns(out)
    for excluded in ("date", "sales", "family", SERIES_KEY):
        assert excluded not in cols
    assert "family_code" in cols
    assert "store_nbr" in cols
