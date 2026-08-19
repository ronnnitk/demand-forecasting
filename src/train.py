"""Train a per-family RandomForest and PERSIST it to models/.

This closes the biggest production gap in the original project: the backend
retrained a model on every request. Here we train once and save a reusable
artifact that predict.py / the API can load.

Run:
    python -m src.train
"""

import argparse

import joblib
from sklearn.ensemble import RandomForestRegressor

from .config import (
    DEFAULT_FAMILY,
    DEFAULT_STORE_NBR,
    MODELS_DIR,
    N_ESTIMATORS,
    RANDOM_STATE,
    VALIDATION_DAYS,
)
from .data_processing import get_series, load_store
from .evaluate import accuracy_pct, mae
from .feature_engineering import create_features


def train_family(df_series, n_estimators=N_ESTIMATORS):
    """Fit a RandomForest on the engineered features; return model + metrics."""
    df_feat = create_features(df_series).dropna().reset_index(drop=True)
    train = df_feat.iloc[:-VALIDATION_DAYS]
    val = df_feat.iloc[-VALIDATION_DAYS:]

    x_train, y_train = train.drop(["date", "sales"], axis=1), train["sales"]
    x_val, y_val = val.drop(["date", "sales"], axis=1), val["sales"]

    model = RandomForestRegressor(
        n_estimators=n_estimators, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    return model, {"mae": mae(y_val, y_pred), "accuracy": accuracy_pct(y_val, y_pred)}


def main(store_nbr=DEFAULT_STORE_NBR, family=DEFAULT_FAMILY):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df_store = load_store(store_nbr)
    series = get_series(df_store, family)
    model, metrics = train_family(series)

    safe_family = family.replace(" ", "_").lower()
    out = MODELS_DIR / f"store{store_nbr}_{safe_family}.joblib"
    joblib.dump(model, out)
    print(f"Saved {out}  |  MAE={metrics['mae']:.2f}  Accuracy={metrics['accuracy']:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and persist a demand model.")
    parser.add_argument("--store", type=int, default=DEFAULT_STORE_NBR)
    parser.add_argument("--family", type=str, default=DEFAULT_FAMILY)
    args = parser.parse_args()
    main(args.store, args.family)
