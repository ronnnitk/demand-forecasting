"""ForeSight IQ API — serves forecasts from persisted artifacts.

What changed and why it matters:

The previous version trained **two RandomForests on every single request**
(a validation model and a full model), after loading all 3,000,888 rows into
process memory at startup. That put a multi-second model fit on the request
path, made latency scale with the size of the series, burned CPU re-deriving an
identical answer for identical inputs, and meant two users could get different
numbers for the same query depending on timing.

Now: the model is trained offline (`python -m src.train_global`), persisted, and
loaded once. Requests do feature construction and inference only. Data is read
from the partitioned store one store at a time, with an LRU cache, so memory is
bounded by the partitions actually requested rather than by the whole dataset.

    uvicorn app.backend.main:app --reload
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Make the shared src/ library importable (repo root is three levels up).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src import io_store  # noqa: E402
from src.analysis.promotions import promo_uplift  # noqa: E402
from src.baselines import seasonal_moving_average  # noqa: E402
from src.config import DATE_COL, PROMO_COL, SCENARIO_HORIZON, TARGET_COL  # noqa: E402
from src.data_processing import regularize  # noqa: E402
from src.forecast_batch import load_forecasts, load_metrics  # noqa: E402
from src.metrics import evaluate_all  # noqa: E402
from src.predict import forecast_panel  # noqa: E402
from src.recommend import elasticity_from_uplift  # noqa: E402
from src.scenario import Scenario, compare_scenarios, simulate_series  # noqa: E402
from src.train_global import load as load_global_model  # noqa: E402

app = FastAPI(
    title="ForeSight IQ API",
    description="Demand forecasting service — serves a persisted global model.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # restrict to the frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE: dict = {
    "model": None, "meta": None, "index": None, "error": None,
    "quantile_models": None, "quantile_scale": 1.0,
}


@app.on_event("startup")
def load_artifacts() -> None:
    """Load the model and the series index once. No data is loaded eagerly."""
    try:
        STATE["index"] = io_store.series_index()
    except FileNotFoundError as exc:
        STATE["error"] = f"panel store not built: {exc}"
        print(f"[startup] {STATE['error']}", flush=True)
        return

    try:
        model, meta = load_global_model()
        STATE["model"], STATE["meta"] = model, meta
        print(
            f"[startup] global model loaded: {meta['n_series']} series, "
            f"{len(meta['feature_columns'])} features, trained {meta['trained_at'][:19]}",
            flush=True,
        )
    except FileNotFoundError as exc:
        # The API still serves history and baseline forecasts without a model.
        STATE["error"] = str(exc)
        print(f"[startup] no global model: {exc}", flush=True)

    # Optional: prediction-interval models. Absent them, forecasts stay point-only.
    try:
        from src.quantiles import load_bundle  # noqa: E402

        bundle = load_bundle()
        STATE["quantile_models"] = bundle["models"]
        STATE["quantile_scale"] = bundle.get("scale", 1.0)
        print(f"[startup] quantile models loaded (scale {STATE['quantile_scale']:.2f})", flush=True)
    except FileNotFoundError:
        print("[startup] no quantile models — forecasts will be point-only", flush=True)


@lru_cache(maxsize=8)
def _store_partition(store_nbr: int) -> pd.DataFrame:
    """Cached per-store read. Bounds memory to the stores actually requested."""
    return io_store.read_store(store_nbr)


def _series(store_nbr: int, family: str) -> pd.DataFrame:
    df = _store_partition(store_nbr)
    out = df[df["family"] == family]
    if out.empty:
        raise HTTPException(status_code=404, detail="Store or product family not found.")
    slice_ = out[[DATE_COL, TARGET_COL, PROMO_COL]].sort_values(DATE_COL).reset_index(drop=True)
    regularized = regularize(slice_)
    regularized["store_nbr"] = store_nbr
    regularized["family"] = family
    return regularized


@app.get("/api/health")
def health() -> dict:
    """Readiness plus provenance of the artifact being served."""
    meta = STATE["meta"]
    return {
        "status": "ok" if meta else "degraded",
        "model_loaded": meta is not None,
        "panel_built": STATE["index"] is not None,
        "detail": STATE["error"],
        "model": None
        if not meta
        else {
            "n_series": meta["n_series"],
            "n_features": len(meta["feature_columns"]),
            "train_start": meta["train_start"],
            "train_end": meta["train_end"],
            "trained_at": meta["trained_at"],
            "holdout_metrics": meta["holdout_metrics"],
        },
    }


@app.get("/api/stores")
def get_stores() -> list[int]:
    if STATE["index"] is None:
        raise HTTPException(status_code=503, detail="Panel store not built.")
    return sorted(STATE["index"]["store_nbr"].unique().tolist())


@app.get("/api/families")
def get_families() -> list[str]:
    if STATE["index"] is None:
        raise HTTPException(status_code=503, detail="Panel store not built.")
    return sorted(STATE["index"]["family"].unique().tolist())


@app.get("/api/series")
def series_catalog(
    store_nbr: int | None = Query(None, description="Filter to one store"),
) -> list[dict]:
    """Inventory of available series with their coverage — served from the index."""
    if STATE["index"] is None:
        raise HTTPException(status_code=503, detail="Panel store not built.")
    index = STATE["index"]
    if store_nbr is not None:
        index = index[index["store_nbr"] == store_nbr]
    return index.assign(
        start=lambda d: d["start"].astype(str), end=lambda d: d["end"].astype(str)
    ).to_dict("records")


@app.get("/api/forecast")
def get_forecast(
    store_nbr: int = Query(2, description="Store number"),
    family: str = Query("GROCERY II", description="Product family"),
    days: int = Query(15, ge=1, le=30, description="Forecast horizon in days"),
    history_limit: str = Query("6m", description="History window: '3m', '6m' or '1y'"),
    model: str = Query("global", description="'global' (persisted GBM) or 'baseline'"),
) -> dict:
    """Historical demand plus an out-of-sample forecast.

    Inference only — no training happens here.
    """
    series = _series(store_nbr, family)
    if len(series) < 60:
        raise HTTPException(status_code=400, detail="Insufficient history for this series.")

    if model == "global":
        if STATE["model"] is None:
            raise HTTPException(
                status_code=503,
                detail=f"Global model unavailable ({STATE['error']}). "
                "Train it with: python -m src.train_global",
            )
        if STATE["quantile_models"]:
            from src.quantiles import forecast_with_intervals  # noqa: E402

            forecast_df = forecast_with_intervals(
                series, STATE["model"], STATE["meta"], STATE["quantile_models"],
                days=days, scale=STATE["quantile_scale"],
            )
            future = [
                {"date": row[DATE_COL].strftime("%Y-%m-%d"), "sales": None,
                 "forecast": round(float(row["forecast"]), 2),
                 "forecast_lower": round(float(row["forecast_lower"]), 2),
                 "forecast_upper": round(float(row["forecast_upper"]), 2)}
                for _, row in forecast_df.iterrows()
            ]
            model_name = "global_gbm+intervals"
        else:
            forecast_df = forecast_panel(series, STATE["model"], STATE["meta"], days=days)
            future = [
                {"date": row[DATE_COL].strftime("%Y-%m-%d"), "sales": None,
                 "forecast": round(float(row["forecast"]), 2)}
                for _, row in forecast_df.iterrows()
            ]
            model_name = "global_gbm"
    else:
        values = series[TARGET_COL].to_numpy(float)
        preds = seasonal_moving_average(values, days)
        last_date = series[DATE_COL].max()
        future = [
            {"date": (last_date + pd.Timedelta(days=i + 1)).strftime("%Y-%m-%d"),
             "sales": None, "forecast": round(float(p), 2)}
            for i, p in enumerate(preds)
        ]
        model_name = "seasonal_moving_average"

    # Backtest metrics on the last `days` of actual history, so the numbers
    # returned describe out-of-sample behaviour rather than a re-fit on itself.
    holdout = series.iloc[-days:]
    history_for_holdout = series.iloc[:-days]
    y_true = holdout[TARGET_COL].to_numpy(float)
    y_train = history_for_holdout[TARGET_COL].to_numpy(float)
    y_pred = seasonal_moving_average(y_train, days)
    if model == "global" and STATE["model"] is not None:
        backtest_df = forecast_panel(history_for_holdout, STATE["model"], STATE["meta"], days=days)
        y_pred = backtest_df.sort_values(DATE_COL)["forecast"].to_numpy(float)
    metrics = evaluate_all(y_true, y_pred, y_train)

    limit_days = {"3m": 90, "6m": 180, "1y": 365}.get(history_limit, 180)
    recent = series.tail(limit_days)
    records = [
        {"date": row[DATE_COL].strftime("%Y-%m-%d"),
         "sales": float(row[TARGET_COL]), "forecast": None}
        for _, row in recent.iterrows()
    ]

    return {
        "store_nbr": store_nbr,
        "family": family,
        "model": model_name,
        "forecast_days": days,
        "history_limit": history_limit,
        "metrics": {k: (None if v is None or not np.isfinite(v) else round(float(v), 4))
                    for k, v in metrics.items()},
        "data": records + future,
    }


@app.get("/api/scenario")
def scenario(
    store_nbr: int = Query(2, description="Store number"),
    family: str = Query("GROCERY II", description="Product family"),
    days: int = Query(SCENARIO_HORIZON, ge=1, le=30, description="Days to simulate"),
    promo: bool | None = Query(
        None, description="Force promotion on/off for the horizon; omit to leave it unchanged"
    ),
    price_change: float = Query(
        0.0, ge=-0.9, le=1.0, description="Fractional price move, e.g. -0.10 for a 10% cut"
    ),
    demand_multiplier: float = Query(
        1.0, gt=0.0, le=5.0, description="Blanket shock factor, 1.15 = +15%"
    ),
    compare: bool = Query(
        False, description="Return the standard comparison set instead of one custom scenario"
    ),
) -> dict:
    """What-if simulation: price a merchandising plan against the forecast.

    Runs the recursive forecast for one series (once for business-as-usual, plus
    once more when the scenario changes promotion) and returns demand / revenue /
    profit deltas vs baseline. Single series only — this re-runs the model, so it
    is inference, not a lookup like ``/api/recommendations``.
    """
    if STATE["model"] is None:
        raise HTTPException(
            status_code=503,
            detail=f"Global model unavailable ({STATE['error']}). "
            "Train it with: python -m src.train_global",
        )
    series = _series(store_nbr, family)
    if len(series) < 60:
        raise HTTPException(status_code=400, detail="Insufficient history for this series.")

    uplift = promo_uplift(series).get("uplift_adjusted")
    elasticity = elasticity_from_uplift(uplift)

    if compare:
        table = compare_scenarios(
            series, STATE["model"], STATE["meta"], days=days, uplift_adjusted=uplift
        )
        return {
            "store_nbr": store_nbr, "family": family, "days": days,
            "elasticity": round(float(elasticity), 2),
            "scenarios": table.replace({np.nan: None}).to_dict("records"),
        }

    plan = Scenario(
        name="custom", promotion=promo, price_change_pct=price_change,
        demand_multiplier=demand_multiplier,
    )
    result = simulate_series(
        series, STATE["model"], STATE["meta"], plan, days=days, uplift_adjusted=uplift
    )
    daily = result.pop("daily")
    result["daily"] = [
        {"date": row[DATE_COL].strftime("%Y-%m-%d"),
         "baseline": float(row["baseline"]), "scenario": float(row["scenario"])}
        for _, row in daily.iterrows()
    ]
    return {"store_nbr": store_nbr, "family": family, **result}


@app.get("/api/analysis/segments")
def segments(limit: int = Query(50, ge=1, le=500)) -> list[dict]:
    """ABC/XYZ segmentation, read from the generated analysis pack."""
    from src.config import ANALYSIS_DIR

    path = ANALYSIS_DIR / "abc_xyz_segments.csv"
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail="Analysis pack not generated. Run: python -m src.analysis.report",
        )
    return pd.read_csv(path).head(limit).replace({np.nan: None}).to_dict("records")


@app.get("/api/recommendations")
def recommendations(
    store_nbr: int | None = Query(None, description="Filter to one store"),
    action: str | None = Query(None, description="Filter by action, e.g. DISCOUNT"),
    limit: int = Query(100, ge=1, le=1000),
) -> dict:
    """Prescriptive offer / discount / stock actions from the recommender.

    Served from the pre-computed ``reports/analysis/recommendations.csv`` so the
    request path stays a lookup — the recommendations are refreshed by the batch
    job (`python -m src.recommend`), not on request.
    """
    from src.config import ANALYSIS_DIR

    path = ANALYSIS_DIR / "recommendations.csv"
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail="Recommendations not generated. Run: python -m src.recommend",
        )
    df = pd.read_csv(path)
    if store_nbr is not None:
        df = df[df["store_nbr"] == store_nbr]
    if action is not None:
        df = df[df["action"].str.upper() == action.upper()]

    counts = df["action"].value_counts().to_dict()
    rows = df.head(limit).replace({np.nan: None}).to_dict("records")
    return {"count": len(df), "action_counts": counts, "recommendations": rows}


@app.get("/api/inventory")
def inventory(
    store_nbr: int | None = Query(None, description="Filter to one store"),
    limit: int = Query(100, ge=1, le=1000),
) -> dict:
    """Safety-stock and reorder points from the probabilistic forecast.

    Served from ``reports/analysis/inventory_plan.csv``, refreshed by the batch
    job (`python -m src.inventory`).
    """
    from src.config import ANALYSIS_DIR

    path = ANALYSIS_DIR / "inventory_plan.csv"
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail="Inventory plan not generated. Run: python -m src.inventory",
        )
    df = pd.read_csv(path)
    if store_nbr is not None:
        df = df[df["store_nbr"] == store_nbr]
    return {"count": len(df), "items": df.head(limit).replace({np.nan: None}).to_dict("records")}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
