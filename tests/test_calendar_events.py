"""Tests for festival/holiday features.

Two properties matter: the movable-feast maths must be right (a wrong Easter
shifts Carnival and Good Friday for every year), and the features must stay
strictly date-derived so they never leak the target.
"""

import numpy as np
import pandas as pd
import pytest

from src.calendar_events import (
    EVENT_FEATURE_COLUMNS,
    build_holiday_frame,
    easter_sunday,
    event_features,
    holidays_for_year,
)


def test_easter_matches_known_dates():
    # Reference Gregorian Easter Sundays.
    assert easter_sunday(2013).isoformat() == "2013-03-31"
    assert easter_sunday(2016).isoformat() == "2016-03-27"
    assert easter_sunday(2017).isoformat() == "2017-04-16"


def test_fixed_and_movable_holidays_present():
    hol = holidays_for_year(2016)
    assert hol[pd.Timestamp("2016-12-25").date()] == "christmas"
    assert hol[pd.Timestamp("2016-01-01").date()] == "new_year"
    # Carnival Tuesday 2016 = Easter (Mar 27) - 47 = Feb 9.
    assert hol[pd.Timestamp("2016-02-09").date()] == "carnival"
    # Good Friday 2016 = Mar 25.
    assert hol[pd.Timestamp("2016-03-25").date()] == "good_friday"


def test_event_features_flag_christmas_eve_proximity():
    df = pd.DataFrame({"date": pd.date_range("2016-12-20", "2016-12-26", freq="D")})
    out = event_features(df)
    dec24 = out[out["date"] == "2016-12-24"].iloc[0]
    dec25 = out[out["date"] == "2016-12-25"].iloc[0]
    assert dec25["is_holiday"] == 1
    assert dec24["is_holiday_eve"] == 1          # day before Christmas
    assert dec24["days_to_holiday"] == 1
    assert dec25["is_christmas_season"] == 0      # season window ends Dec 24
    assert dec24["is_christmas_season"] == 1


def test_event_features_added_for_every_row_without_dropping():
    df = pd.DataFrame({"date": pd.date_range("2015-01-01", periods=400, freq="D")})
    out = event_features(df)
    assert len(out) == len(df)
    for col in EVENT_FEATURE_COLUMNS:
        assert col in out.columns
        assert out[col].notna().all()


def test_earthquake_response_window_is_flagged():
    df = pd.DataFrame({"date": pd.to_datetime(
        ["2016-04-15", "2016-04-16", "2016-05-31", "2016-06-01", "2017-04-16"]
    )})
    out = event_features(df)
    # Window is 2016-04-16 .. 2016-05-31 inclusive; nothing before/after or in 2017.
    assert out["is_earthquake_response"].tolist() == [0, 1, 1, 0, 0]


def test_events_do_not_depend_on_sales():
    """Same dates, different sales -> identical event features (no leakage)."""
    dates = pd.date_range("2016-01-01", periods=120, freq="D")
    a = event_features(pd.DataFrame({"date": dates, "sales": np.arange(120.0)}))
    b = event_features(pd.DataFrame({"date": dates, "sales": np.zeros(120)}))
    for col in EVENT_FEATURE_COLUMNS:
        assert a[col].equals(b[col])
