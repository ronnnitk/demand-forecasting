"""Single source of truth for paths, dataset selectors, and model hyperparameters.

Previously these values were hard-coded and duplicated across check.py,
process_data.py and backend/main.py. Centralizing them here is what makes the
pipeline configurable instead of copy-pasted.
"""

from pathlib import Path

# --- Paths (repo-root relative) ------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PANEL_DIR = PROCESSED_DIR / "panel"          # partitioned series store
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
ANALYSIS_DIR = REPORTS_DIR / "analysis"      # generated analysis tables

RAW_TRAIN_CSV = RAW_DIR / "train.csv"

# --- Dataset schema ------------------------------------------------------------
KEY_COLS = ["store_nbr", "family"]           # identifies one time series
DATE_COL = "date"
TARGET_COL = "sales"
PROMO_COL = "onpromotion"
USE_COLS = [DATE_COL, *KEY_COLS, TARGET_COL, PROMO_COL]

CHUNK_SIZE = 250_000                         # rows per CSV chunk on ingest

# --- Default dataset selectors -------------------------------------------------
DEFAULT_STORE_NBR = 2
DEFAULT_FAMILY = "GROCERY II"

# --- Feature engineering -------------------------------------------------------
LAGS = [1, 7, 14]                            # legacy per-series feature set
ROLLING_WINDOW = 7

# Richer global-model feature set.
PANEL_LAGS = [1, 2, 3, 7, 14, 21, 28]
ROLLING_WINDOWS = [7, 14, 28]
EWM_SPANS = [7, 28]
FOURIER_PERIODS = [(7, 2), (365.25, 3)]      # (period, n harmonics)
# Ecuador public-sector wages are paid on the 15th and the last day of the month,
# a documented driver of demand spikes in this dataset.
PAYDAY_DAYS = (15,)

# --- Model hyperparameters -----------------------------------------------------
N_ESTIMATORS = 50
RANDOM_STATE = 42
VALIDATION_DAYS = 15
FORECAST_HORIZON = 15

# Global gradient-boosting model (sklearn HistGradientBoostingRegressor).
GLOBAL_MODEL_PARAMS = {
    "loss": "poisson",                       # non-negative, count-like demand
    "learning_rate": 0.06,
    "max_iter": 400,
    "max_leaf_nodes": 63,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": RANDOM_STATE,
}

# --- Self-adaptive retraining --------------------------------------------------
RETRAIN_MAX_AGE_DAYS = 7          # refresh at least weekly even absent drift
DRIFT_TOLERANCE = 0.15            # retrain if recent WAPE is >15% worse than at train time
PROMOTE_MARGIN = 0.01            # challenger must beat champion WAPE by >1% to be promoted
ADAPTIVE_HOLDOUT_DAYS = 15        # window both models are scored on, head-to-head

# --- Prescriptive recommendations (offers / discounts) -------------------------
# Baseline price-elasticity of demand used when a series has too little promo
# history to estimate its own. Negative: a discount raises units sold.
DEFAULT_PRICE_ELASTICITY = -1.2
# Gross margin assumed when a per-family margin is unknown (share of price).
DEFAULT_GROSS_MARGIN = 0.25
# Candidate discount depths the recommender evaluates.
DISCOUNT_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
# A series whose forecast falls this far below its trailing demand is "slowing".
SLOWDOWN_THRESHOLD = 0.15

# --- Analysis / evaluation -----------------------------------------------------
SEASONAL_PERIOD = 7                          # weekly seasonality for MASE/RMSSE
BACKTEST_FOLDS = 3                           # rolling-origin folds
BACKTEST_HORIZON = 15                        # days forecast per fold
MIN_TRAIN_DAYS = 180                         # a series needs this much history

# ABC segmentation cut-offs (cumulative share of revenue).
ABC_CUTOFFS = (0.80, 0.95)
# XYZ segmentation cut-offs (coefficient of variation of demand).
XYZ_CUTOFFS = (0.50, 1.00)
# Intermittency (Syntetos-Boylan) cut-offs: average demand interval, CV squared.
ADI_CUTOFF = 1.32
CV2_CUTOFF = 0.49
