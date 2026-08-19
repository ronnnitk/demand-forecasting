"""Rolling-origin backtesting.

The original evaluation was a single 15-day holdout at the end of one series.
That gives one number from one window, so it cannot distinguish a good model
from a lucky fortnight, and it says nothing about how accuracy decays as the
horizon grows.

Rolling-origin evaluation (a.k.a. time-series cross-validation) instead walks
the cut-off forward, refitting on everything before each origin and scoring the
next ``horizon`` days:

    |--------- train ---------|-- test --|                fold 1
    |------------ train ----------|-- test --|            fold 2
    |--------------- train -----------|-- test --|        fold 3

Nothing after an origin is ever visible to the model fitted at that origin, so
the score is an honest estimate of live performance. Every model is scored
against the same folds as the baselines in :mod:`src.baselines`, which is what
turns "the model is accurate" into "the model beats seasonal-naive by X%".

    python -m src.backtest --stores 2 44 47 --folds 3
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .baselines import run_baselines
from .config import (
    ANALYSIS_DIR,
    BACKTEST_FOLDS,
    BACKTEST_HORIZON,
    DATE_COL,
    MIN_TRAIN_DAYS,
    SEASONAL_PERIOD,
    TARGET_COL,
)
from .data_processing import regularize
from .io_store import iter_stores
from .metrics import evaluate_all


def rolling_origins(n_obs: int, folds: int, horizon: int) -> list[int]:
    """Cut-off indices for each fold, earliest first.

    Origins are spaced one horizon apart so the test windows tile the end of
    the series without overlapping.
    """
    origins = [n_obs - horizon * (folds - i) for i in range(folds)]
    return [o for o in origins if o >= MIN_TRAIN_DAYS]


def backtest_series(
    series: pd.DataFrame,
    predictors: dict | None = None,
    folds: int = BACKTEST_FOLDS,
    horizon: int = BACKTEST_HORIZON,
    seasonal_period: int = SEASONAL_PERIOD,
) -> pd.DataFrame:
    """Score baselines (and optional models) on one series across folds.

    ``predictors`` maps a name to ``fn(history_df, horizon) -> array`` so any
    model that can forecast from a history frame plugs into the same harness.
    """
    series = regularize(series.sort_values(DATE_COL).reset_index(drop=True))
    values = series[TARGET_COL].to_numpy(float)
    n_obs = len(values)

    rows = []
    for fold, origin in enumerate(rolling_origins(n_obs, folds, horizon), start=1):
        y_train = values[:origin]
        y_test = values[origin: origin + horizon]
        if len(y_test) < horizon:
            continue

        forecasts = run_baselines(y_train, horizon)
        if predictors:
            history = series.iloc[:origin]
            for name, fn in predictors.items():
                try:
                    forecasts[name] = np.asarray(fn(history, horizon), dtype=float)
                except Exception as exc:  # a broken model must not kill the sweep
                    forecasts[name] = np.full(horizon, np.nan)
                    rows.append({"fold": fold, "model": name, "error": str(exc)})

        for name, y_pred in forecasts.items():
            if not np.isfinite(y_pred).all():
                continue
            rows.append(
                {
                    "fold": fold,
                    "origin_date": series[DATE_COL].iloc[origin - 1],
                    "model": name,
                    "n_train": origin,
                    **evaluate_all(y_test, y_pred, y_train, seasonal_period),
                }
            )
    return pd.DataFrame(rows)


def backtest_panel(
    stores: list[int] | None = None,
    families: list[str] | None = None,
    predictors: dict | None = None,
    folds: int = BACKTEST_FOLDS,
    horizon: int = BACKTEST_HORIZON,
    min_total_sales: float = 0.0,
) -> pd.DataFrame:
    """Backtest every selected series, streaming one store partition at a time."""
    results = []
    for store_nbr, df in iter_stores(stores):
        for family, series in df.groupby("family", observed=True):
            family = str(family)
            if families and family not in families:
                continue
            if series[TARGET_COL].sum() <= min_total_sales:
                continue  # a series that never sold anything has nothing to score

            fold_results = backtest_series(series, predictors, folds, horizon)
            if fold_results.empty:
                continue
            fold_results["store_nbr"] = store_nbr
            fold_results["family"] = family
            results.append(fold_results)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def summarize(results: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Aggregate fold-level results into a model leaderboard.

    MASE and RMSSE are averaged (they are already scale-free, so the mean is
    meaningful). MAE is *volume-weighted* rather than averaged, because a plain
    mean would let one high-volume series dominate the ranking.
    """
    scored = results[results["model"].notna() & results["mase"].notna()]
    group = ["model", *(by or [])]

    out = (
        scored.groupby(group)
        .agg(
            n_folds=("mae", "size"),
            mase=("mase", "mean"),
            rmsse=("rmsse", "mean"),
            wape=("wape", "median"),
            smape=("smape", "mean"),
            bias=("bias", "mean"),
            mae=("mae", "mean"),
        )
        .reset_index()
        .sort_values("mase")
    )
    # Improvement over the reference forecast, the number that decides adoption.
    reference = out[out["model"] == "seasonal_naive"]["mase"]
    if len(reference):
        out["vs_seasonal_naive"] = 1 - out["mase"] / float(reference.iloc[0])
    return out.reset_index(drop=True)


def backtest_global(
    stores: list[int] | None = None,
    folds: int = BACKTEST_FOLDS,
    horizon: int = BACKTEST_HORIZON,
    min_date: str | None = "2015-01-01",
    seasonal_period: int = SEASONAL_PERIOD,
) -> pd.DataFrame:
    """Head-to-head: the global model against every baseline, same folds.

    The model is **refit at each origin** on data up to that origin only. This
    is the expensive but honest version — reusing one model fitted on the full
    history would let it see the test windows and would flatter it badly.

    Returns one row per (series, fold, model).
    """
    from .predict import forecast_panel
    from .train_global import fit_global, load_training_panel

    panel = load_training_panel(stores, min_date)
    max_date = panel[DATE_COL].max()

    rows = []
    for fold in range(folds, 0, -1):
        origin_date = max_date - pd.Timedelta(days=horizon * fold)
        train = panel[panel[DATE_COL] <= origin_date]
        test = panel[
            (panel[DATE_COL] > origin_date)
            & (panel[DATE_COL] <= origin_date + pd.Timedelta(days=horizon))
        ]
        if test.empty:
            continue

        print(f"[backtest] fold {folds - fold + 1}/{folds}: origin {origin_date.date()}, "
              f"train={len(train):,} rows, test={len(test):,} rows")
        model, meta = fit_global(train, validation_days=0)
        forecast = forecast_panel(train, model, meta, days=horizon)

        actual = test.set_index(["store_nbr", "family", DATE_COL])[TARGET_COL]
        predicted = forecast.set_index(["store_nbr", "family", DATE_COL])["forecast"]

        for (store_nbr, family), series_train in train.groupby(["store_nbr", "family"], observed=True):
            key = (store_nbr, family)
            try:
                y_true = actual.loc[key].sort_index().to_numpy(float)
                y_pred = predicted.loc[key].sort_index().to_numpy(float)
            except KeyError:
                continue
            if len(y_true) != len(y_pred) or not len(y_true):
                continue

            y_train = series_train.sort_values(DATE_COL)[TARGET_COL].to_numpy(float)
            candidates = {"global_gbm": y_pred, **run_baselines(y_train, len(y_true))}
            for name, pred in candidates.items():
                rows.append(
                    {
                        "fold": folds - fold + 1,
                        "origin_date": origin_date,
                        "store_nbr": store_nbr,
                        "family": family,
                        "model": name,
                        **evaluate_all(y_true, pred, y_train, seasonal_period),
                    }
                )
    return pd.DataFrame(rows)


def main(stores=None, folds=BACKTEST_FOLDS, horizon=BACKTEST_HORIZON, out_name="backtest_baselines.csv"):
    results = backtest_panel(stores=stores, folds=folds, horizon=horizon)
    if results.empty:
        print("[backtest] no series produced results")
        return results

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(ANALYSIS_DIR / f"raw_{out_name}", index=False)
    leaderboard = summarize(results)
    leaderboard.to_csv(ANALYSIS_DIR / out_name, index=False)
    print(leaderboard.to_string(index=False))
    return leaderboard


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rolling-origin backtest of the baselines.")
    parser.add_argument("--stores", type=int, nargs="*", default=None)
    parser.add_argument("--folds", type=int, default=BACKTEST_FOLDS)
    parser.add_argument("--horizon", type=int, default=BACKTEST_HORIZON)
    args = parser.parse_args()
    main(stores=args.stores, folds=args.folds, horizon=args.horizon)
