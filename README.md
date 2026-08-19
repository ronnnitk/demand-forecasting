# Demand Forecasting

Daily demand forecasting for retail store sales, from raw data to a served API
and interactive dashboard. Predicts sales per store × product family using
gradient-free tree ensembles with lag and calendar features.

> Dataset: *Store Sales — Time Series Forecasting* (Corporación Favorita, Kaggle).

---

## Architecture

```
demand-forecasting/
├── data/              # raw / processed / external  (gitignored, see data/README.md)
├── notebooks/         # numbered analysis pipeline (01_eda … 06_business_insights)
├── src/               # importable library: the single source of truth
│   ├── config.py            # paths, selectors, hyperparameters
│   ├── data_processing.py   # load + filter (separated from modeling)
│   ├── feature_engineering.py
│   ├── train.py             # trains AND persists a model to models/
│   ├── predict.py           # recursive multi-step forecast
│   ├── evaluate.py          # MAE / accuracy
│   ├── visualization.py
│   └── utils.py             # model load/save helpers
├── models/            # persisted .joblib artifacts
├── reports/           # figures/ + technical_report.docx + business_insights.docx
├── app/
│   ├── backend/       # FastAPI service (imports src/, no duplicated model code)
│   └── frontend/      # React + TypeScript dashboard
├── tests/             # pytest unit tests
├── config/            # config.yaml
├── requirements.txt   # environment.yml  •  LICENSE  •  .gitignore
```

**Separation of concerns:** data loading, feature engineering, training,
prediction, and evaluation are independent modules in `src/`. The notebooks, the
CLI (`python -m src.train`), and the API all import the *same* functions — no
copy-pasted logic.

---

## Quickstart

```bash
# 1. Environment
pip install -r requirements.txt        # or: conda env create -f environment.yml

# 2. Data — place the Kaggle file (see data/README.md)
#    data/raw/train.csv

# 3. Train and persist a model
python -m src.train --store 2 --family "GROCERY II"

# 4. Run the API
uvicorn app.backend.main:app --reload

# 5. Run the dashboard
cd app/frontend && npm install && npm start
```

## Testing

```bash
pytest
```

## Results

Validation metric: **MAE** per product family, plus an accuracy score relative
to mean actual sales. See `reports/` for the written analysis and
`reports/figures/` for forecast plots.

## License

MIT — see [LICENSE](LICENSE).
