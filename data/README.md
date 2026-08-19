# Data

Raw data is **not** committed to version control (see `.gitignore`). This file
documents provenance so the pipeline is reproducible.

## Layout
| Folder | Contents | Committed? |
|---|---|---|
| `raw/` | Original, immutable source data (`train.csv`) | No |
| `processed/` | Derived tables produced by `src/` (e.g. `store_2_all_forecasts.csv`) | No |
| `external/` | Third-party reference data (holidays, oil prices, etc.) | No |

## Source dataset
**Store Sales — Time Series Forecasting** (Corporación Favorita / Kaggle).
- Grain: daily `sales` per `store_nbr` × product `family`.
- Columns used: `date`, `store_nbr`, `family`, `sales`.

## How to obtain
1. Download `train.csv` from the Kaggle competition page.
2. Place it at `data/raw/train.csv`.
3. Run `python -m src.train` to produce a model in `models/`.
