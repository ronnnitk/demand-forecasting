"""Feature engineering for the demand-forecasting model.

This is the *single* home for `create_features`, which was previously
copy-pasted into check.py, check (1).py and backend/main.py.
"""

import pandas as pd

from .config import LAGS, ROLLING_WINDOW


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, lag, and rolling-mean features to a date/sales frame.

    Expects columns: ``date`` (datetime) and ``sales`` (numeric).
    """
    data = data.copy()
    data["dayofweek"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month
    data["dayofmonth"] = data["date"].dt.day
    for lag in LAGS:
        data[f"lag_{lag}"] = data["sales"].shift(lag)
    data["rolling_mean_7"] = (
        data["sales"].shift(1).rolling(window=ROLLING_WINDOW).mean()
    )
    return data
