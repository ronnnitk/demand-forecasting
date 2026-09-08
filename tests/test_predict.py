"""Tests for `forecast_panel`'s recursive multi-step loop.

Focused on the `promo_plan` path in particular: it lets a caller supply known
future promotion values (a *planned* field, unlike sales), and until
`src.scenario` started using it, nothing exercised it — so it's worth pinning
directly rather than only indirectly through the scenario tests.
"""

import numpy as np
import pandas as pd

from src.config import DATE_COL, KEY_COLS
from src.features import build_family_codes, build_features, feature_columns
from src.predict import forecast_panel


class _PromoResponsiveModel:
    """Predicts a flat level plus a bump whenever `promo_flag` is on."""

    def __init__(self, base: float = 20.0, promo_bump: float = 8.0):
        self.base = base
        self.promo_bump = promo_bump

    def predict(self, X):
        out = np.full(len(X), self.base, dtype=float)
        if "promo_flag" in X.columns:
            out = out + self.promo_bump * X["promo_flag"].to_numpy(float)
        return out


def _history_and_meta(n: int = 40):
    dates = pd.date_range("2017-01-01", periods=n, freq="D")
    history = pd.DataFrame(
        {"date": dates, "store_nbr": 1, "family": "A", "sales": 20.0, "onpromotion": 0.0}
    )
    family_codes = build_family_codes(["A"])
    feats = build_features(history, family_codes=family_codes)
    meta = {"feature_columns": feature_columns(feats), "family_codes": family_codes}
    return history, meta


def test_forecast_panel_runs_without_a_promo_plan():
    history, meta = _history_and_meta()
    out = forecast_panel(history, _PromoResponsiveModel(), meta, days=5)
    assert len(out) == 5
    assert (out["forecast"] == 20.0).all()  # history never promoted -> fallback stays 0


def test_forecast_panel_applies_an_explicit_promo_plan():
    history, meta = _history_and_meta()
    dates = pd.date_range(history[DATE_COL].max() + pd.Timedelta(days=1), periods=5, freq="D")
    plan = pd.DataFrame({"store_nbr": 1, "family": "A", "date": dates, "onpromotion": 6.0})

    out = forecast_panel(history, _PromoResponsiveModel(promo_bump=8.0), meta, days=5, promo_plan=plan)
    assert (out["forecast"] == 28.0).all()  # base 20 + bump 8, every planned day


def test_forecast_panel_falls_back_when_a_plan_is_missing_a_date():
    history, meta = _history_and_meta()
    dates = pd.date_range(history[DATE_COL].max() + pd.Timedelta(days=1), periods=5, freq="D")
    # Only the first 2 of 5 horizon days are planned; the rest must fall back
    # to the recent-average promo plan (0, since history never promoted)
    # rather than crash or silently leave NaN.
    plan = pd.DataFrame({"store_nbr": 1, "family": "A", "date": dates[:2], "onpromotion": 6.0})

    out = forecast_panel(history, _PromoResponsiveModel(promo_bump=8.0), meta, days=5, promo_plan=plan).sort_values(DATE_COL)
    forecasts = out["forecast"].to_numpy()
    assert list(forecasts[:2]) == [28.0, 28.0]     # planned days: promo applied
    assert list(forecasts[2:]) == [20.0, 20.0, 20.0]  # unplanned days: fallback (no promo)


def test_forecast_panel_covers_every_series_in_a_multi_series_plan():
    dates = pd.date_range("2017-01-01", periods=40, freq="D")
    history = pd.concat(
        [
            pd.DataFrame({"date": dates, "store_nbr": 1, "family": "A", "sales": 20.0, "onpromotion": 0.0}),
            pd.DataFrame({"date": dates, "store_nbr": 2, "family": "A", "sales": 20.0, "onpromotion": 0.0}),
        ],
        ignore_index=True,
    )
    family_codes = build_family_codes(["A"])
    feats = build_features(history, family_codes=family_codes)
    meta = {"feature_columns": feature_columns(feats), "family_codes": family_codes}

    future_dates = pd.date_range(dates.max() + pd.Timedelta(days=1), periods=3, freq="D")
    # Plan covers only store 1; store 2 must still get a (fallback) value, not NaN.
    plan = pd.DataFrame({"store_nbr": 1, "family": "A", "date": future_dates, "onpromotion": 6.0})

    out = forecast_panel(history, _PromoResponsiveModel(), meta, days=3, promo_plan=plan)
    assert set(out[KEY_COLS].drop_duplicates().itertuples(index=False)) == {(1, "A"), (2, "A")}
    assert out.loc[out["store_nbr"] == 1, "forecast"].eq(28.0).all()
    assert out.loc[out["store_nbr"] == 2, "forecast"].eq(20.0).all()
