"""Small shared helpers."""

from pathlib import Path

import joblib

from .config import MODELS_DIR


def model_path(store_nbr: int, family: str) -> Path:
    safe_family = family.replace(" ", "_").lower()
    return MODELS_DIR / f"store{store_nbr}_{safe_family}.joblib"


def load_model(store_nbr: int, family: str):
    """Load a persisted model, or raise if it hasn't been trained yet."""
    path = model_path(store_nbr, family)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run:  python -m src.train "
            f"--store {store_nbr} --family '{family}'"
        )
    return joblib.load(path)
