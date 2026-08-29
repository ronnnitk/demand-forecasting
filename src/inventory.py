"""Inventory policy: turn a probabilistic forecast into safety stock and
reorder points.

This is where uncertainty pays for itself. Given the point forecast and a
prediction interval, we can set a **reorder point** that hits a chosen service
level instead of guessing:

    reorder_point = expected demand over the lead time
                    + z(service_level) · sigma over the lead time

The daily demand standard deviation is backed out of the interval width under a
normal approximation — an 80% interval spans ``(z_hi - z_lo)`` standard
deviations, so ``sigma_day ≈ (upper - lower) / (z_hi - z_lo)``. Over an
``L``-day lead time, independent daily variances add, so
``sigma_lead = sqrt(Σ sigma_day²)``.

The output tells a planner, per series: how much to expect, how much buffer the
target service level needs, and the level at which to reorder.

    python -m src.inventory                         # lead 7d, 95% service level
    python -m src.inventory --lead-time 14 --service-level 0.98
"""

from __future__ import annotations

import argparse
from statistics import NormalDist

import numpy as np
import pandas as pd

from .config import (
    ANALYSIS_DIR,
    DATE_COL,
    DEFAULT_LEAD_TIME_DAYS,
    DEFAULT_SERVICE_LEVEL,
    KEY_COLS,
    QUANTILES,
)

_NORM = NormalDist()


def z_score(service_level: float) -> float:
    """Standard-normal quantile for a service level (0.95 -> 1.645)."""
    service_level = float(np.clip(service_level, 0.5, 0.9999))
    return float(_NORM.inv_cdf(service_level))


def _sigma_per_day(lower: np.ndarray, upper: np.ndarray, lower_q: float, upper_q: float) -> np.ndarray:
    """Back out daily sigma from an interval width under a normal approximation."""
    span = z_score(upper_q) - z_score(lower_q)  # e.g. P10..P90 -> ~2.563
    if span <= 0:
        return np.zeros_like(upper, dtype=float)
    return np.clip(upper - lower, 0, None) / span


def reorder_point_for_series(
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    lead_time: int = DEFAULT_LEAD_TIME_DAYS,
    service_level: float = DEFAULT_SERVICE_LEVEL,
    lower_q: float = QUANTILES[0],
    upper_q: float = QUANTILES[1],
) -> dict:
    """Reorder point / safety stock for one series over its lead time.

    Uses the first ``lead_time`` days of the forecast. Daily variances add, so
    the lead-time sigma is the root-sum-square of the per-day sigmas.
    """
    horizon = min(lead_time, len(point))
    point = np.asarray(point, dtype=float)[:horizon]
    sigma_day = _sigma_per_day(np.asarray(lower, float)[:horizon], np.asarray(upper, float)[:horizon], lower_q, upper_q)

    mean_lead = float(np.sum(point))
    sigma_lead = float(np.sqrt(np.sum(sigma_day ** 2)))
    safety = z_score(service_level) * sigma_lead
    return {
        "lead_time_days": horizon,
        "mean_lead_demand": round(mean_lead, 2),
        "sigma_lead": round(sigma_lead, 2),
        "safety_stock": round(safety, 2),
        "reorder_point": round(mean_lead + safety, 2),
        "service_level": service_level,
    }


def inventory_plan(
    interval_forecasts: pd.DataFrame,
    lead_time: int = DEFAULT_LEAD_TIME_DAYS,
    service_level: float = DEFAULT_SERVICE_LEVEL,
    lower_q: float = QUANTILES[0],
    upper_q: float = QUANTILES[1],
) -> pd.DataFrame:
    """Reorder points for every series in an interval-forecast frame.

    ``interval_forecasts`` needs ``store_nbr``, ``family``, ``date``,
    ``forecast``, ``forecast_lower``, ``forecast_upper`` (from
    :func:`src.quantiles.forecast_with_intervals`).
    """
    required = {"forecast", "forecast_lower", "forecast_upper"}
    if not required.issubset(interval_forecasts.columns):
        raise ValueError(f"interval_forecasts missing columns: {required - set(interval_forecasts.columns)}")

    rows = []
    for key, g in interval_forecasts.sort_values(DATE_COL).groupby(KEY_COLS, observed=True):
        plan = reorder_point_for_series(
            g["forecast"].to_numpy(float),
            g["forecast_lower"].to_numpy(float),
            g["forecast_upper"].to_numpy(float),
            lead_time, service_level, lower_q, upper_q,
        )
        rows.append({"store_nbr": int(key[0]), "family": str(key[1]), **plan})

    out = pd.DataFrame(rows)
    return out.sort_values("reorder_point", ascending=False).reset_index(drop=True)


def run(stores=None, lead_time=DEFAULT_LEAD_TIME_DAYS, service_level=DEFAULT_SERVICE_LEVEL, min_date="2016-01-01"):
    """Load models, forecast intervals, and write an inventory plan."""
    from .quantiles import forecast_with_intervals, load_bundle
    from .train_global import load as load_point_model, load_training_panel

    model, meta = load_point_model()
    bundle = load_bundle()
    qmodels, scale = bundle["models"], bundle.get("scale", 1.0)
    panel = load_training_panel(stores, min_date)

    lo_q, hi_q = min(QUANTILES), max(QUANTILES)
    band = forecast_with_intervals(
        panel, model, meta, qmodels, days=lead_time, lower_q=lo_q, upper_q=hi_q, scale=scale
    )
    plan = inventory_plan(band, lead_time, service_level, lo_q, hi_q)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANALYSIS_DIR / "inventory_plan.csv"
    plan.to_csv(out_path, index=False)
    print(f"[inventory] {len(plan)} series | lead {lead_time}d | service {service_level:.0%}")
    print(f"[inventory] wrote {out_path}")
    print(plan.head(12).to_string(index=False))
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute safety stock and reorder points.")
    parser.add_argument("--stores", type=int, nargs="*", default=None)
    parser.add_argument("--lead-time", type=int, default=DEFAULT_LEAD_TIME_DAYS)
    parser.add_argument("--service-level", type=float, default=DEFAULT_SERVICE_LEVEL)
    parser.add_argument("--min-date", type=str, default="2016-01-01")
    args = parser.parse_args()
    run(args.stores, args.lead_time, args.service_level, args.min_date)
