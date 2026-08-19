"""Tests for data loading and calendar regularization.

``regularize`` is quietly one of the most important functions in the project:
this dataset omits store-closure days (Christmas) entirely, so without it a
``shift(7)`` silently means "7 rows back" rather than "7 days back" and every
seasonal feature is misaligned for the rest of the series.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_processing import get_series, regularize


def test_regularize_fills_calendar_gaps_with_zero():
    sparse = pd.DataFrame(
        {
            "date": pd.to_datetime(["2016-01-01", "2016-01-02", "2016-01-05"]),
            "sales": [1.0, 2.0, 3.0],
        }
    )
    out = regularize(sparse)
    assert len(out) == 5
    assert out["sales"].tolist() == [1.0, 2.0, 0.0, 0.0, 3.0]


def test_regularize_makes_lag_7_mean_seven_days():
    dates = pd.date_range("2016-01-01", periods=15, freq="D")
    dropped = dates.delete(3)  # simulate a closure day
    df = pd.DataFrame({"date": dropped, "sales": np.arange(len(dropped), dtype=float)})

    out = regularize(df)
    assert (out["date"].diff().dropna() == pd.Timedelta(days=1)).all()
    assert out["date"].iloc[7] - out["date"].iloc[0] == pd.Timedelta(days=7)


def test_regularize_carries_series_identity_across_inserted_rows():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2016-01-01", "2016-01-04"]),
            "sales": [1.0, 2.0],
            "store_nbr": [7, 7],
            "family": ["EGGS", "EGGS"],
        }
    )
    out = regularize(df)
    assert out["store_nbr"].tolist() == [7, 7, 7, 7]
    assert out["family"].tolist() == ["EGGS"] * 4


def test_regularize_zero_fills_promotions_not_forward_fills_them():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2016-01-01", "2016-01-04"]),
            "sales": [1.0, 2.0],
            "onpromotion": [5, 5],
        }
    )
    out = regularize(df)
    assert out["onpromotion"].tolist() == [5, 0, 0, 5]


def test_regularize_is_a_no_op_on_a_complete_calendar():
    df = pd.DataFrame(
        {"date": pd.date_range("2016-01-01", periods=10, freq="D"), "sales": np.arange(10.0)}
    )
    assert len(regularize(df)) == 10


def test_regularize_handles_an_empty_frame():
    empty = pd.DataFrame({"date": pd.to_datetime([]), "sales": []})
    assert regularize(empty).empty


def test_get_series_selects_one_family_and_sorts_by_date():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2016-01-03", "2016-01-01", "2016-01-02"]),
            "family": ["EGGS", "EGGS", "MEATS"],
            "sales": [3.0, 1.0, 99.0],
        }
    )
    out = get_series(df, "EGGS")
    assert out["sales"].tolist() == [1.0, 3.0]
    assert out["date"].is_monotonic_increasing
