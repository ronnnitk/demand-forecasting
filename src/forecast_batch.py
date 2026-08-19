"""Batch-forecast every series once, so serving is a lookup.

Computing a forecast on the request path costs ~1.5s per series even with a
pre-trained model, because the recursive loop has to rebuild features for each
of the 15 steps. Doing that per request is the same mistake as training per
request, just smaller.

Demand forecasts are refreshed daily at best, and every user asking for
store 2 / GROCERY II wants the same answer. So compute all 1,728 series in one
vectorised pass — one ``model.predict`` per horizon step covering every series
at once — and persist the result. Serving then costs a dictionary lookup, and
the batch job is the thing that gets scheduled, monitored and alerted on.

    python -m src.forecast_batch                  # all series, 15 days
    python -m src.forecast_batch --days 30
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from .baselines import seasonal_moving_average
from .config import (
    DATE_COL,
    FORECAST_HORIZON,
    KEY_COLS,
    PROCESSED_DIR,
    TARGET_COL,
)
from .metrics import evaluate_all
from .predict import forecast_panel
from .train_global import load, load_training_panel

FORECAST_FILE = PROCESSED_DIR / "batch_forecasts.csv"
METRICS_FILE = PROCESSED_DIR / "batch_metrics.csv"


def run(
    stores: list[int] | None = None,
    days: int = FORECAST_HORIZON,
    min_date: str | None = "2016-01-01",
    with_metrics: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forecast every series and score the model on a held-out final window.

    ``min_date`` only bounds the *history* fed to the recursive loop; it does
    not affect the model, which was trained separately.
    """
    model, meta = load()
    panel = load_training_panel(stores, min_date)
    n_series = panel.groupby(KEY_COLS, observed=True).ngroups
    print(f"[batch] {len(panel):,} rows / {n_series} series, horizon {days}d")

    t0 = time.time()
    forecasts = forecast_panel(panel, model, meta, days=days)
    print(f"[batch] forecast for {n_series} series in {time.time() - t0:.1f}s "
          f"({1000 * (time.time() - t0) / max(n_series, 1):.0f} ms/series)")

    metrics = pd.DataFrame()
    if with_metrics:
        metrics = backtest_last_window(panel, model, meta, days)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(FORECAST_FILE, index=False)
    if not metrics.empty:
        metrics.to_csv(METRICS_FILE, index=False)
    print(f"[batch] wrote {FORECAST_FILE.name}" + (f" and {METRICS_FILE.name}" if not metrics.empty else ""))
    return forecasts, metrics


def backtest_last_window(panel: pd.DataFrame, model, meta, days: int) -> pd.DataFrame:
    """Score the model per series on the final ``days`` it has never seen used.

    Withholds the last ``days`` from the recursive history, forecasts them, and
    compares against the seasonal moving-average baseline on the same window.
    Gives the API a per-series accuracy figure without any request-time work.
    """
    cutoff = panel[DATE_COL].max() - pd.Timedelta(days=days)
    history = panel[panel[DATE_COL] <= cutoff]
    actual = panel[panel[DATE_COL] > cutoff]
    if history.empty or actual.empty:
        return pd.DataFrame()

    t0 = time.time()
    predicted = forecast_panel(history, model, meta, days=days)
    print(f"[batch] holdout forecast in {time.time() - t0:.1f}s")

    actual_idx = actual.set_index([*KEY_COLS, DATE_COL])[TARGET_COL]
    pred_idx = predicted.set_index([*KEY_COLS, DATE_COL])["forecast"]

    rows = []
    for key, series_hist in history.groupby(KEY_COLS, observed=True):
        try:
            y_true = actual_idx.loc[key].sort_index().to_numpy(float)
            y_pred = pred_idx.loc[key].sort_index().to_numpy(float)
        except KeyError:
            continue
        if not len(y_true) or len(y_true) != len(y_pred):
            continue

        y_train = series_hist.sort_values(DATE_COL)[TARGET_COL].to_numpy(float)
        baseline = seasonal_moving_average(y_train, len(y_true))

        row = {"store_nbr": key[0], "family": key[1]}
        row.update({f"model_{k}": v for k, v in evaluate_all(y_true, y_pred, y_train).items()})
        row.update({f"base_{k}": v for k, v in evaluate_all(y_true, baseline, y_train).items()})
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    scored = out.dropna(subset=["model_mase", "base_mase"])
    beat = float((scored["model_mase"] < scored["base_mase"]).mean()) if len(scored) else float("nan")
    print(
        f"[batch] holdout: model MASE {scored['model_mase'].mean():.3f} vs "
        f"baseline {scored['base_mase'].mean():.3f} | "
        f"model wins on {beat:.1%} of {len(scored)} series"
    )
    return out


def load_forecasts() -> pd.DataFrame | None:
    if not FORECAST_FILE.exists():
        return None
    return pd.read_csv(FORECAST_FILE, parse_dates=[DATE_COL])


def load_metrics() -> pd.DataFrame | None:
    if not METRICS_FILE.exists():
        return None
    return pd.read_csv(METRICS_FILE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-forecast every series.")
    parser.add_argument("--stores", type=int, nargs="*", default=None)
    parser.add_argument("--days", type=int, default=FORECAST_HORIZON)
    parser.add_argument("--min-date", type=str, default="2016-01-01")
    parser.add_argument("--no-metrics", action="store_true")
    args = parser.parse_args()
    run(args.stores, args.days, args.min_date, not args.no_metrics)
