"""Probabilistic forecasting: prediction intervals via quantile models.

A point forecast answers "how much will sell?" but an inventory decision needs
"how much *might* sell?". Ordering to the mean stocks out roughly half the time;
ordering to the P90 sets a 90% service level on purpose.

This module trains one gradient-boosting model per quantile (using the pinball
loss, whose minimiser *is* that quantile) on exactly the same features and
family codes as the point model, so the bands are aligned with the central
forecast. The point model still drives the recursive lags — the quantile models
only widen or narrow the band around that path (see :func:`forecast_panel`).

    python -m src.quantiles                    # train P10/P90 for the saved model
    python -m src.quantiles --quantiles 0.05 0.5 0.95
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import DATE_COL, GLOBAL_MODEL_PARAMS, MODELS_DIR, TARGET_COL, VALIDATION_DAYS
from .features import build_features
from .metrics import interval_coverage, pinball_loss
from .predict import forecast_panel
from .train_global import load as load_point_model, load_training_panel

QUANTILE_MODEL_PATH = MODELS_DIR / "quantile_models.joblib"
DEFAULT_QUANTILES = (0.1, 0.9)


def _quantile_params(quantile: float) -> dict:
    """Base global-model params retuned for quantile (pinball) loss."""
    params = {k: v for k, v in GLOBAL_MODEL_PARAMS.items() if k != "loss"}
    params["loss"] = "quantile"
    params["quantile"] = quantile
    return params


def train_quantile_models(
    panel: pd.DataFrame,
    meta: dict,
    quantiles=DEFAULT_QUANTILES,
) -> dict[float, HistGradientBoostingRegressor]:
    """Fit one quantile model per requested quantile.

    Reuses the point model's ``feature_columns``, ``family_codes`` and
    categorical mask (via ``meta``) so every model consumes an identical feature
    matrix and the intervals line up with the point forecast.
    """
    cols = meta["feature_columns"]
    cat_features = meta["categorical_features"]
    feats = build_features(panel, family_codes=meta["family_codes"])
    cat_mask = [col in cat_features for col in cols]

    models = {}
    for q in quantiles:
        model = HistGradientBoostingRegressor(
            categorical_features=cat_mask, **_quantile_params(q)
        )
        model.fit(feats[cols], feats[TARGET_COL])
        models[float(q)] = model
    return models


def save(models: dict, quantiles, path: Path = QUANTILE_MODEL_PATH, scale: float = 1.0) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "quantiles": list(quantiles), "scale": float(scale)}, path)
    return path


def load(path: Path = QUANTILE_MODEL_PATH) -> dict[float, HistGradientBoostingRegressor]:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No quantile models at {path}. Run:  python -m src.quantiles"
        )
    return joblib.load(path)["models"]


def load_bundle(path: Path = QUANTILE_MODEL_PATH) -> dict:
    """Full quantile bundle: models, quantiles, and the calibration scale."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No quantile models at {path}. Run:  python -m src.quantiles"
        )
    bundle = joblib.load(path)
    bundle.setdefault("scale", 1.0)
    return bundle


def _calibrate_scale(median, lower, upper, actual, target: float) -> tuple[float, float]:
    """Find the width multiplier that makes empirical coverage match ``target``.

    Per-step quantile models understate the uncertainty that accumulates over a
    recursive multi-step forecast, so the raw band is typically too tight. This
    is a conformal-style fix: widen (or tighten) the band symmetrically about the
    median by a single factor chosen on the holdout so that the realised coverage
    tracks the nominal interval. Returns ``(scale, achieved_coverage)``.
    """
    median = np.asarray(median, float)
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    actual = np.asarray(actual, float)

    best_scale, best_gap, best_cov = 1.0, float("inf"), 0.0
    for scale in np.arange(0.5, 5.01, 0.1):
        lo = np.clip(median - scale * (median - lower), 0, None)
        hi = median + scale * (upper - median)
        cov = interval_coverage(actual, lo, hi)
        gap = abs(cov - target)
        if gap < best_gap:
            best_scale, best_gap, best_cov = float(scale), gap, cov
    return best_scale, best_cov


def forecast_with_intervals(
    history: pd.DataFrame,
    model,
    meta: dict,
    quantile_models: dict[float, object],
    days: int,
    lower_q: float = 0.1,
    upper_q: float = 0.9,
    scale: float = 1.0,
) -> pd.DataFrame:
    """Point forecast plus a ``forecast_lower`` / ``forecast_upper`` band.

    ``scale`` widens the band about the point forecast by the calibration factor
    learned in :func:`main` (1.0 = raw quantile models).
    """
    out = forecast_panel(history, model, meta, days=days, quantile_models=quantile_models)
    lo, hi = f"q{int(round(lower_q * 100))}", f"q{int(round(upper_q * 100))}"
    rename = {}
    if lo in out.columns:
        rename[lo] = "forecast_lower"
    if hi in out.columns:
        rename[hi] = "forecast_upper"
    out = out.rename(columns=rename)

    if {"forecast_lower", "forecast_upper"}.issubset(out.columns):
        # A quantile crossing (lower > upper) is possible when models disagree on
        # a sparse series; clamp so the band is always well-formed.
        lower = out[["forecast_lower", "forecast_upper"]].min(axis=1)
        upper = out[["forecast_lower", "forecast_upper"]].max(axis=1)
        # Apply the conformal calibration scale about the point forecast.
        out["forecast_lower"] = np.clip(out["forecast"] - scale * (out["forecast"] - lower), 0, None)
        out["forecast_upper"] = out["forecast"] + scale * (upper - out["forecast"])
    return out


def main(stores=None, min_date="2015-01-01", quantiles=DEFAULT_QUANTILES, holdout_days=VALIDATION_DAYS):
    model, meta = load_point_model()
    panel = load_training_panel(stores, min_date)

    cutoff = panel[DATE_COL].max() - pd.Timedelta(days=holdout_days)
    train = panel[panel[DATE_COL] <= cutoff]

    print(f"[quantiles] training {list(quantiles)} on {len(train):,} rows ...")
    qmodels = train_quantile_models(train, meta, quantiles)

    # Calibrate on the held-out window: forecast the raw band, learn the width
    # scale that makes empirical coverage match the nominal interval, and persist
    # it with the models so serving applies the same correction.
    lo_q, hi_q = min(quantiles), max(quantiles)
    nominal = hi_q - lo_q
    band = forecast_with_intervals(train, model, meta, qmodels, holdout_days, lo_q, hi_q, scale=1.0)
    actual = panel[panel[DATE_COL] > cutoff]
    keys = ["store_nbr", "family", DATE_COL]
    merged = band.merge(actual[[*keys, TARGET_COL]], on=keys, how="inner")

    scale, cov_after = 1.0, float("nan")
    if not merged.empty and {"forecast_lower", "forecast_upper"}.issubset(merged.columns):
        cov_before = interval_coverage(merged[TARGET_COL], merged["forecast_lower"], merged["forecast_upper"])
        scale, cov_after = _calibrate_scale(
            merged["forecast"], merged["forecast_lower"], merged["forecast_upper"],
            merged[TARGET_COL], nominal,
        )
        print(f"[quantiles] nominal {nominal:.0%} interval | coverage {cov_before:.1%} raw "
              f"-> {cov_after:.1%} after x{scale:.1f} calibration")

    path = save(qmodels, quantiles, scale=scale)
    print(f"[quantiles] saved {path} (calibration scale {scale:.2f})")
    return qmodels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train quantile models for prediction intervals.")
    parser.add_argument("--stores", type=int, nargs="*", default=None)
    parser.add_argument("--min-date", type=str, default="2015-01-01")
    parser.add_argument("--quantiles", type=float, nargs="*", default=list(DEFAULT_QUANTILES))
    args = parser.parse_args()
    main(args.stores, args.min_date, tuple(args.quantiles))
