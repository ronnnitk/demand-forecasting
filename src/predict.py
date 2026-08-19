"""Multi-step forecasting.

Two paths, both recursive (each predicted day is fed back as the next day's lag):

* :func:`predict_future` — the original single-series path, kept unchanged so
  the legacy per-family models and the existing notebooks keep working.
* :func:`forecast_panel` — the global-model path, which rolls **every series
  forward in the same loop**. That matters for scale: one vectorised
  ``model.predict`` call per horizon step covers all 1,782 series, instead of
  1,782 separate recursive loops.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from .config import (
    DATE_COL,
    FORECAST_HORIZON,
    KEY_COLS,
    PROMO_COL,
    TARGET_COL,
)
from .feature_engineering import create_features
from .features import build_features

# Days of history to carry into the recursive loop: enough to satisfy the
# longest lag (28) and rolling window (28) with room to spare.
TAIL_DAYS = 70


def predict_future(current_df: pd.DataFrame, model, days: int = FORECAST_HORIZON) -> pd.DataFrame:
    """Roll a single-series model forward ``days`` steps, feeding predictions back as lags."""
    temp_df = current_df.copy()
    future_preds = []
    last_date = temp_df[DATE_COL].max()

    for _ in range(days):
        next_date = last_date + timedelta(days=1)
        new_row = pd.DataFrame({DATE_COL: [next_date], TARGET_COL: [0.0]})
        temp_df = pd.concat([temp_df, new_row], ignore_index=True)

        features_df = create_features(temp_df)
        last_features = features_df.tail(1).drop([DATE_COL, TARGET_COL], axis=1)

        pred = max(0.0, float(model.predict(last_features)[0]))  # no negative sales
        future_preds.append({DATE_COL: next_date, "prediction": pred})

        temp_df.loc[temp_df.index[-1], TARGET_COL] = pred
        last_date = next_date

    return pd.DataFrame(future_preds)


def _default_promo_plan(history: pd.DataFrame) -> pd.Series:
    """Assume future promotion intensity continues the last 28 days' average.

    ``onpromotion`` is a planned field, so in production the real promotion
    calendar should be passed in instead — this is the neutral fallback for
    when it is not available, and it is stated rather than hidden.
    """
    recent = history[history[DATE_COL] > history[DATE_COL].max() - pd.Timedelta(days=28)]
    return recent.groupby(KEY_COLS, observed=True)[PROMO_COL].mean()


def forecast_panel(
    history: pd.DataFrame,
    model,
    meta: dict,
    days: int = FORECAST_HORIZON,
    promo_plan: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Recursive multi-step forecast for every series in ``history``, together.

    ``history`` is a long panel (``date``, ``store_nbr``, ``family``, ``sales``,
    ``onpromotion``). ``promo_plan`` may supply known future promotion values
    with the same keys plus ``date``; otherwise the recent average is carried
    forward.

    Returns a long frame of ``date``, ``store_nbr``, ``family``, ``forecast``.
    """
    cols = meta["feature_columns"]
    family_codes = meta["family_codes"]

    last_date = history[DATE_COL].max()
    work = history[history[DATE_COL] > last_date - pd.Timedelta(days=TAIL_DAYS)].copy()
    work = work[[DATE_COL, *KEY_COLS, TARGET_COL, PROMO_COL]]

    keys = work[KEY_COLS].drop_duplicates().reset_index(drop=True)
    fallback_promo = _default_promo_plan(work)

    outputs = []
    for step in range(1, days + 1):
        next_date = last_date + timedelta(days=step)

        future = keys.copy()
        future[DATE_COL] = next_date
        future[TARGET_COL] = 0.0
        if promo_plan is not None:
            planned = promo_plan[promo_plan[DATE_COL] == next_date]
            future = future.merge(planned[[*KEY_COLS, PROMO_COL]], on=KEY_COLS, how="left")
            future[PROMO_COL] = future[PROMO_COL].fillna(
                future.set_index(KEY_COLS).index.map(fallback_promo)
            )
        else:
            future[PROMO_COL] = future.set_index(KEY_COLS).index.map(fallback_promo)
        future[PROMO_COL] = future[PROMO_COL].astype(float).fillna(0.0)

        work = pd.concat([work, future[work.columns]], ignore_index=True)

        feats = build_features(work, dropna=False, family_codes=family_codes)
        mask = feats[DATE_COL] == next_date
        step_feats = feats.loc[mask, cols].fillna(0.0)

        preds = np.clip(model.predict(step_feats), 0, None)

        step_out = feats.loc[mask, [DATE_COL, *KEY_COLS]].copy()
        step_out["forecast"] = preds
        outputs.append(step_out)

        # Feed the prediction back so the next step's lags see it.
        write_idx = work.index[work[DATE_COL] == next_date]
        aligned = step_out.set_index(KEY_COLS)["forecast"]
        work.loc[write_idx, TARGET_COL] = (
            work.loc[write_idx].set_index(KEY_COLS).index.map(aligned).astype(float)
        )

    return pd.concat(outputs, ignore_index=True).sort_values([*KEY_COLS, DATE_COL]).reset_index(drop=True)


def global_predictor(model, meta: dict):
    """Adapt the global model to the :mod:`src.backtest` predictor interface.

    Lets the global model be scored on exactly the same rolling-origin folds as
    the baselines, which is the only way the comparison means anything.
    """

    def predict(history: pd.DataFrame, horizon: int) -> np.ndarray:
        panel = history.copy()
        if PROMO_COL not in panel.columns:
            panel[PROMO_COL] = 0.0
        out = forecast_panel(panel, model, meta, days=horizon)
        return out.sort_values(DATE_COL)["forecast"].to_numpy(float)

    return predict
