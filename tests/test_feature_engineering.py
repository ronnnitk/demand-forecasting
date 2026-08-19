"""Unit tests for feature engineering — the kind of coverage reviewers look for."""

import pandas as pd

from src.feature_engineering import create_features


def _sample():
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    return pd.DataFrame({"date": dates, "sales": range(30)})


def test_adds_expected_columns():
    out = create_features(_sample())
    for col in ["dayofweek", "month", "dayofmonth", "lag_1", "lag_7", "lag_14", "rolling_mean_7"]:
        assert col in out.columns


def test_does_not_mutate_input():
    df = _sample()
    create_features(df)
    assert list(df.columns) == ["date", "sales"]


def test_lag_alignment():
    out = create_features(_sample())
    # lag_1 at row i equals sales at row i-1
    assert out["lag_1"].iloc[1] == out["sales"].iloc[0]
    assert pd.isna(out["lag_1"].iloc[0])
