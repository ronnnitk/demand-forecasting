"""Train ONE model across all series — the change that makes this scale.

The per-series approach (`src.train`) fits an independent RandomForest for each
(store, family) pair. On this panel that is 1,782 models to fit, persist,
version, monitor and refresh, and each one sees only its own ~1,700 rows. A new
store means 33 more models trained from scratch with no history.

A **global** model inverts that. One gradient-boosting model is fitted on all
series at once, with ``store_nbr`` and ``family`` as categorical features. The
consequences are the point:

* **One artifact** to train, ship and monitor instead of 1,782.
* **Shared strength.** Weekend and payday effects are learned from 3M rows
  rather than re-learned 1,782 times from 1,700 rows each. Series that are too
  sparse to fit alone — the 864 intermittent/lumpy ones — borrow the pattern.
* **Cold start works.** A new store or family gets a forecast on day one from
  its categorical neighbours.
* **Cost grows with data, not with series count.** Adding stores adds rows;
  it does not multiply the number of things to operate.

Poisson loss is used because demand is non-negative and count-like with 31%
zeros; squared error on such a target pulls predictions toward the mean of a
skewed distribution and can go negative.

    python -m src.train_global                       # all stores, since 2015
    python -m src.train_global --stores 1 2 3        # a subset, for a fast loop
    python -m src.train_global --min-date 2013-01-01 # the full history
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import (
    ANALYSIS_DIR,
    DATE_COL,
    GLOBAL_MODEL_PARAMS,
    KEY_COLS,
    MODELS_DIR,
    TARGET_COL,
    VALIDATION_DAYS,
)
from .data_processing import regularize
from .features import (
    CATEGORICAL_FEATURES,
    build_family_codes,
    build_features,
    feature_columns,
)
from .io_store import iter_stores, series_index
from .metrics import evaluate_all

GLOBAL_MODEL_PATH = MODELS_DIR / "global_model.joblib"
DEFAULT_MIN_DATE = "2015-01-01"


def load_training_panel(
    stores: list[int] | None = None, min_date: str | None = DEFAULT_MIN_DATE
) -> pd.DataFrame:
    """Assemble a regularized long panel, streaming one store at a time.

    Each series is put on a complete daily calendar before concatenation, so
    that lag 7 always means "seven days ago" rather than "seven rows ago".
    """
    frames = []
    for store_nbr, df in iter_stores(stores):
        if min_date:
            df = df[df[DATE_COL] >= pd.Timestamp(min_date)]
        for family, series in df.groupby("family", observed=True):
            if series[TARGET_COL].sum() <= 0:
                continue  # never sold anything: nothing to learn, nothing to serve
            reg = regularize(series.sort_values(DATE_COL).reset_index(drop=True))
            reg["store_nbr"] = store_nbr
            reg["family"] = str(family)
            frames.append(reg)
    if not frames:
        raise ValueError("no series matched the selection")
    return pd.concat(frames, ignore_index=True)


def fit_global(
    panel: pd.DataFrame,
    validation_days: int = VALIDATION_DAYS,
    params: dict | None = None,
) -> tuple[HistGradientBoostingRegressor, dict]:
    """Fit the global model with a time-based holdout. Returns model + metadata.

    The split is by **date**, not by row: every series contributes its last
    ``validation_days`` to validation. A random split would leak, because a
    neighbouring day of the same series carries almost the same information.
    """
    families = sorted(panel["family"].astype(str).unique())
    family_codes = build_family_codes(families)

    t0 = time.time()
    feats = build_features(panel, family_codes=family_codes)
    build_secs = time.time() - t0

    cols = feature_columns(feats)
    cat_mask = [col in CATEGORICAL_FEATURES for col in cols]

    cutoff = feats[DATE_COL].max() - pd.Timedelta(days=validation_days)
    train = feats[feats[DATE_COL] <= cutoff]
    val = feats[feats[DATE_COL] > cutoff]

    model = HistGradientBoostingRegressor(
        categorical_features=cat_mask, **(params or GLOBAL_MODEL_PARAMS)
    )
    t0 = time.time()
    model.fit(train[cols], train[TARGET_COL])
    fit_secs = time.time() - t0

    metrics = {}
    if len(val):
        y_pred = np.clip(model.predict(val[cols]), 0, None)
        metrics = evaluate_all(val[TARGET_COL], y_pred)

    meta = {
        "feature_columns": cols,
        "categorical_features": CATEGORICAL_FEATURES,
        "family_codes": family_codes,
        "stores": sorted(int(s) for s in panel["store_nbr"].unique()),
        "families": families,
        "n_series": int(panel.groupby(KEY_COLS, observed=True).ngroups),
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "train_start": str(feats[DATE_COL].min().date()),
        "train_end": str(feats[DATE_COL].max().date()),
        "validation_days": validation_days,
        "params": params or GLOBAL_MODEL_PARAMS,
        "holdout_metrics": metrics,
        "feature_build_seconds": round(build_secs, 1),
        "fit_seconds": round(fit_secs, 1),
        "trained_at": pd.Timestamp.utcnow().isoformat(),
    }
    return model, meta


def save(model, meta: dict, path: Path = GLOBAL_MODEL_PATH) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "meta": meta}, path)
    return path


def load(path: Path = GLOBAL_MODEL_PATH) -> tuple[HistGradientBoostingRegressor, dict]:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No global model at {path}. Run:  python -m src.train_global"
        )
    bundle = joblib.load(path)
    return bundle["model"], bundle["meta"]


def permutation_importance_report(
    model, feats: pd.DataFrame, meta: dict, n_repeats: int = 3, sample: int = 60_000
) -> pd.DataFrame:
    """Which features actually carry the signal, measured by permutation.

    Split-count importance is biased toward high-cardinality columns.
    Permuting a column and measuring the MAE increase asks the question that
    matters: how much worse is the model without this feature?
    """
    from sklearn.inspection import permutation_importance

    cols = meta["feature_columns"]
    subset = feats.sample(min(sample, len(feats)), random_state=0)
    result = permutation_importance(
        model,
        subset[cols],
        subset[TARGET_COL],
        n_repeats=n_repeats,
        random_state=0,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "feature": cols,
                "importance": result.importances_mean,
                "std": result.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def main(stores=None, min_date=DEFAULT_MIN_DATE, validation_days=VALIDATION_DAYS, importance=False):
    print(f"[global] loading panel (stores={stores or 'all'}, since {min_date}) ...")
    panel = load_training_panel(stores, min_date)
    print(f"[global] {len(panel):,} rows across {panel.groupby(KEY_COLS, observed=True).ngroups} series")

    model, meta = fit_global(panel, validation_days=validation_days)
    path = save(model, meta)

    m = meta["holdout_metrics"]
    print(
        f"[global] saved {path}\n"
        f"         features={len(meta['feature_columns'])} "
        f"train_rows={meta['train_rows']:,} fit={meta['fit_seconds']}s\n"
        f"         holdout MAE={m.get('mae', float('nan')):.2f} "
        f"WAPE={m.get('wape', float('nan')):.3f} bias={m.get('bias', float('nan')):+.3f}"
    )

    if importance:
        feats = build_features(panel, family_codes=meta["family_codes"])
        imp = permutation_importance_report(model, feats, meta)
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        imp.to_csv(ANALYSIS_DIR / "feature_importance.csv", index=False)
        print("\n[global] top features by permutation importance:")
        print(imp.head(15).to_string(index=False))
    return model, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the global multi-series model.")
    parser.add_argument("--stores", type=int, nargs="*", default=None)
    parser.add_argument("--min-date", type=str, default=DEFAULT_MIN_DATE)
    parser.add_argument("--validation-days", type=int, default=VALIDATION_DAYS)
    parser.add_argument("--importance", action="store_true", help="compute permutation importance")
    args = parser.parse_args()
    main(args.stores, args.min_date, args.validation_days, args.importance)
