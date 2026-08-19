# Notebooks

The exploratory work currently lives in a single monolithic notebook,
`00_original_monolith.ipynb` (173 cells), preserved here as-is.

## Target split (next pass)
Industry practice is a numbered, single-responsibility pipeline. This notebook
should be broken into:

| Notebook | Responsibility |
|---|---|
| `01_eda.ipynb` | Data profiling, distributions, seasonality |
| `02_preprocessing.ipynb` | Cleaning, filtering, missing values |
| `03_feature_engineering.ipynb` | Lags, rolling means, calendar features (imports `src.feature_engineering`) |
| `04_modeling.ipynb` | Training experiments (imports `src.train`) |
| `05_evaluation.ipynb` | Metrics, error analysis (imports `src.evaluate`) |
| `06_business_insights.ipynb` | Demand narrative for stakeholders |

Notebooks should import from `src/` rather than redefining logic, and be
committed **with rendered outputs** so they read without re-execution.
