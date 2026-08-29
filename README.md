# Demand Forecasting

Daily demand forecasting for retail store sales — from raw data to a served API
and an interactive dashboard — with a **self-adapting global model** and a
**prescriptive layer** that turns forecasts into stock and pricing actions.

Predicts sales for every `store × product family` series (1,782 of them) using a
single global gradient-boosting model with lag, rolling, Fourier-seasonal,
promotion, and **festival/holiday** features.

> Dataset: *Store Sales — Time Series Forecasting* (Corporación Favorita, Ecuador — Kaggle).

---

## What makes this more than a notebook

| Capability | Where | Why it matters |
|---|---|---|
| **One global model** for all 1,782 series | `src/train_global.py` | One artifact to ship/monitor instead of 1,782; shares strength across series; cold-starts new stores. |
| **Festival & seasonal drivers** | `src/calendar_events.py` | Ecuador holidays (fixed + Easter-derived movable feasts), Christmas/Mother's-Day/Black-Friday windows, holiday-proximity and wet-season features — the events that actually move grocery demand. |
| **Self-adapting retraining** | `src/adaptive.py` | Monitors live accuracy, retrains on **drift or age**, and promotes a challenger only if it beats the champion. The engine keeps itself current and logs every decision. |
| **Prescriptive offers/discounts** | `src/recommend.py` | Detects softening demand, estimates each item's price-elasticity from its measured promo uplift, and recommends the **profit-optimal discount** — or *stock up* ahead of festivals. |
| **Prediction intervals** | `src/quantiles.py` | Quantile (pinball-loss) models give P10/P90 bands, **conformally calibrated** on a holdout (raw 54% → 80% coverage). |
| **Inventory policy** | `src/inventory.py` | Turns the interval into safety stock and **reorder points** at a target service level. |
| **Honest evaluation** | `src/backtest.py`, `src/baselines.py`, `src/metrics.py` | Rolling-origin backtest against seasonal-naive/Croston baselines; MASE/RMSSE/WAPE + pinball/coverage, not just MAE. |
| **Scales with data, not series** | `src/io_store.py` | Raw CSV converted once into per-store partitions; every job streams the slice it needs. |
| **Segmentation → policy** | `src/analysis/` | ABC/XYZ 9-box, promo-uplift, seasonality profiling drive where compute is spent. |

---

## Architecture

```
demand-forecasting/
├── data/           # raw / processed(panel store) / external   (gitignored)
├── notebooks/      # exploratory analysis
├── src/
│   ├── config.py              # single source of truth: paths, schema, hyperparams
│   ├── io_store.py            # partitioned on-disk panel store
│   ├── data_processing.py     # load / filter / regularize to a daily calendar
│   ├── feature_engineering.py # legacy per-series features
│   ├── features.py            # panel features for the global model
│   ├── calendar_events.py     # festival / holiday / season features   ← new
│   ├── train.py               # legacy per-series RandomForest
│   ├── train_global.py        # global HistGradientBoosting model
│   ├── predict.py             # recursive multi-step forecast (single + panel)
│   ├── forecast_batch.py      # batch-forecast every series → serving lookup
│   ├── adaptive.py            # self-adapting drift-triggered retraining  ← new
│   ├── recommend.py           # prescriptive offers / discounts / stock    ← new
│   ├── baselines.py  backtest.py  metrics.py  evaluate.py
│   ├── visualization.py  utils.py
│   └── analysis/     # profiling, seasonality, promotions, segmentation, report
├── models/         # global_model.joblib, archive/, retrain_log.csv   (gitignored)
├── reports/        # figures/ + analysis/ + written reports
├── app/            # backend/ (FastAPI) + frontend/ (React + TypeScript)
├── tests/          # 85 pytest unit tests
└── config/  requirements.txt  environment.yml  LICENSE  .gitignore
```

Everything imports from `src/` — notebooks, CLIs, and the API share the same
functions, so there is no copy-pasted logic anywhere in the pipeline.

---

## Quickstart

```bash
pip install -r requirements.txt        # or: conda env create -f environment.yml

# 1. Data — place the Kaggle file (see data/README.md)
#    data/raw/train.csv
python -m src.io_store                  # build the partitioned panel store

# 2. Train the global model (festival features included)
python -m src.train_global              # all stores, since 2015

# 3. Batch-forecast every series (serving becomes a lookup)
python -m src.forecast_batch

# 4. Prescriptive actions: offers / discounts / stock-ups
python -m src.recommend                 # → reports/analysis/recommendations.csv

# 5. Prediction intervals + inventory policy
python -m src.quantiles                  # train + calibrate P10/P90 models
python -m src.inventory                  # → reports/analysis/inventory_plan.csv

# 6. Keep the model current, automatically
python -m src.adaptive                  # monitor → (drift/age) → challenge → promote
```

### Honest accuracy check

```bash
python -m src.backtest --stores 2 44 47 --folds 3   # model vs baselines, same folds
```

On stores 2 & 44 (since 2016-06), the global model holds **WAPE ≈ 0.10** and
beats the seasonal moving-average baseline on **~77%** of series.

---

## The self-adapting loop (`src/adaptive.py`)

1. **Monitor** — score the live champion on the most recent holdout; compare its
   WAPE to training time.
2. **Trigger** — retrain on **drift** (`recent WAPE > train WAPE × (1+tol)`) *or*
   **age** (`> RETRAIN_MAX_AGE_DAYS`).
3. **Challenge** — fit a challenger on data up to the same cut-off; both forecast
   the identical unseen window.
4. **Promote conservatively** — adopt the challenger only if it beats the
   champion by `PROMOTE_MARGIN`; otherwise keep the champion. Every decision is
   appended to `models/retrain_log.csv`.

Schedule step 5 (`src.adaptive`) daily (cron / Task Scheduler / Airflow) and the
model maintains itself.

## The recommender (`src/recommend.py`)

For each series it compares the forecast to trailing demand and emits one action:

- **DISCOUNT** — demand softening *and* elastic → the profit-optimal markdown,
  chosen from a grid using `Q(d)=Q0·(1-d)^ε`; never dilutes margin below zero.
- **STOCK_UP** — demand rising, or a festival falls in the horizon → protect
  availability, hold price.
- **HOLD** / **REVIEW** — stable, or too volatile to automate.

Elasticity `ε` is inferred from each item's weekday-adjusted promo uplift; every
recommendation ships with expected demand, revenue and profit deltas.

## Testing

```bash
pytest            # 85 tests: leakage tripwires, calendar maths, adaptive & discount logic
```

## License

MIT — see [LICENSE](LICENSE).
