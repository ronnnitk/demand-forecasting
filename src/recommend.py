"""Prescriptive layer: turn forecasts into offers, discounts and stock actions.

A forecast says *what will happen*. A merchandising team needs *what to do about
it*. This module closes that gap: for every (store, family) it reads the demand
forecast, compares it to trailing demand, and recommends one action -

* **DISCOUNT / PROMOTE** when demand is softening and the item is price-responsive.
  The discount depth is *optimised*, not guessed: using each item's own
  price-elasticity (estimated from its measured promotion uplift) it evaluates a
  grid of discount depths and picks the one that maximises expected profit,
  refusing to dilute margin below zero.
* **STOCK UP** when demand is rising or a festival falls inside the horizon - the
  costly mistake there is a stock-out, and discounting into rising demand just
  gives away margin.
* **HOLD** when the series is stable and healthy.
* **REVIEW** when the series is too volatile for an automated call.

Elasticity, not vibes: the demand response to a discount ``d`` is modelled as the
constant-elasticity curve ``Q(d) = Q0 · (1-d)^ε`` with ``ε < 0``. Where an item
has enough promotion history, ``ε`` is inferred from its weekday-adjusted promo
uplift (:mod:`src.analysis.promotions`); otherwise a documented default is used.
Every recommendation ships with the numbers behind it - expected uplift, revenue
and profit change - so a human can overrule it on sight.

    python -m src.recommend                 # write reports/analysis/recommendations.csv
    python -m src.recommend --stores 2 44
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .analysis.promotions import promo_uplift
from .calendar_events import event_features
from .config import (
    ANALYSIS_DIR,
    DATE_COL,
    DEFAULT_GROSS_MARGIN,
    DEFAULT_PRICE_ELASTICITY,
    DISCOUNT_GRID,
    KEY_COLS,
    SLOWDOWN_THRESHOLD,
    TARGET_COL,
)

TRAILING_DAYS = 28
# The effective price cut a "promotion" represents, used to back out an
# elasticity from a measured on/off promo uplift.
ASSUMED_PROMO_DISCOUNT = 0.20
ELASTICITY_FLOOR, ELASTICITY_CAP = -4.0, -0.2  # keep inferred elasticity sane


def constant_elasticity_multiplier(discount: float, elasticity: float) -> float:
    """Demand multiplier for a price cut of fraction ``discount``.

    ``Q(d)/Q0 = (1-d)^ε``. With ε negative a lower price lifts demand, and the
    curve is convex, so successive discount points buy less extra volume.
    """
    discount = float(np.clip(discount, 0.0, 0.95))
    return float((1.0 - discount) ** elasticity)


def elasticity_from_uplift(uplift_adjusted: float) -> float:
    """Back out a constant elasticity from a measured on/off promotion uplift.

    If turning promotion on multiplies demand by ``(1+u)`` and a promotion is
    worth roughly ``ASSUMED_PROMO_DISCOUNT`` off, then
    ``(1+u) = (1-d_promo)^ε`` ⇒ ``ε = ln(1+u) / ln(1-d_promo)``.
    Falls back to the configured default when the uplift is missing or non-positive.
    """
    if uplift_adjusted is None or not np.isfinite(uplift_adjusted) or uplift_adjusted <= 0:
        return DEFAULT_PRICE_ELASTICITY
    eps = np.log(1.0 + uplift_adjusted) / np.log(1.0 - ASSUMED_PROMO_DISCOUNT)
    return float(np.clip(eps, ELASTICITY_FLOOR, ELASTICITY_CAP))


def optimize_discount(
    gross_margin: float,
    elasticity: float,
    grid=DISCOUNT_GRID,
) -> dict:
    """Pick the discount depth that maximises expected profit per unit of Q0.

    Profit per baseline-unit at discount ``d`` is ``(1-d)^ε · (m - d)`` where
    ``m`` is the full-price gross margin. A discount deeper than the margin sells
    at a loss, so those depths are excluded. Returns the best depth and the
    demand/revenue/profit deltas it implies.
    """
    best = {"discount": 0.0, "demand_uplift": 0.0, "revenue_change": 0.0, "profit_change": 0.0}
    base_profit = gross_margin  # per baseline unit at full price
    if base_profit <= 0:
        return best

    best_profit_change = 0.0
    for d in grid:
        if d >= gross_margin:  # would sell below cost
            continue
        mult = constant_elasticity_multiplier(d, elasticity)
        profit = mult * (gross_margin - d)
        profit_change = profit / base_profit - 1.0
        if profit_change > best_profit_change:
            best_profit_change = profit_change
            best = {
                "discount": float(d),
                "demand_uplift": float(mult - 1.0),
                "revenue_change": float(mult * (1.0 - d) - 1.0),
                "profit_change": float(profit_change),
            }
    return best


def _horizon_festival(forecast_dates: pd.Series) -> tuple[bool, str]:
    """Does the forecast horizon contain a festival/holiday? Name the first one."""
    ev = event_features(pd.DataFrame({DATE_COL: pd.to_datetime(forecast_dates)}))
    hit = ev[(ev["is_festival"] == 1) | (ev["is_holiday"] == 1)]
    if hit.empty:
        return False, ""
    for flag, name in [
        ("is_christmas_season", "christmas_season"),
        ("is_mothers_day", "mothers_day"),
        ("is_black_friday", "black_friday"),
    ]:
        if hit[flag].any():
            return True, name
    return True, "holiday"


def recommend_series(
    trailing_daily: float,
    forecast_daily: float,
    uplift_adjusted: float,
    gross_margin: float = DEFAULT_GROSS_MARGIN,
    cv: float | None = None,
    upcoming_festival: tuple[bool, str] = (False, ""),
    slowdown_threshold: float = SLOWDOWN_THRESHOLD,
) -> dict:
    """Decide the action for one series from its demand signals.

    Pure function of the inputs, so the policy is unit-testable in isolation.
    """
    has_festival, festival_name = upcoming_festival
    trend_ratio = forecast_daily / trailing_daily - 1.0 if trailing_daily > 0 else 0.0

    base = {
        "trailing_daily": round(trailing_daily, 2),
        "forecast_daily": round(forecast_daily, 2),
        "trend_pct": round(trend_ratio, 3),
        "suggested_discount_pct": 0.0,
        "expected_demand_uplift_pct": 0.0,
        "expected_profit_change_pct": 0.0,
    }

    # Too volatile to automate: hand it to a human.
    if cv is not None and cv > 1.0:
        return {**base, "action": "REVIEW",
                "rationale": f"demand too volatile to automate (CV={cv:.2f})"}

    # Rising demand or an upcoming festival: protect availability, don't discount.
    if has_festival and trend_ratio > -slowdown_threshold:
        return {**base, "action": "STOCK_UP",
                "rationale": f"{festival_name.replace('_', ' ')} in the forecast window; "
                             f"demand trend {trend_ratio:+.0%} - secure inventory, hold price"}
    if trend_ratio > slowdown_threshold:
        return {**base, "action": "STOCK_UP",
                "rationale": f"forecast up {trend_ratio:+.0%} vs trailing demand - avoid stock-out"}

    # Softening demand: consider a margin-aware discount.
    if trend_ratio < -slowdown_threshold:
        elasticity = elasticity_from_uplift(uplift_adjusted)
        plan = optimize_discount(gross_margin, elasticity)
        if plan["discount"] > 0 and plan["profit_change"] > 0:
            return {**base,
                    "action": "DISCOUNT",
                    "suggested_discount_pct": round(plan["discount"] * 100, 1),
                    "expected_demand_uplift_pct": round(plan["demand_uplift"] * 100, 1),
                    "expected_profit_change_pct": round(plan["profit_change"] * 100, 1),
                    "rationale": f"demand softening {trend_ratio:+.0%}; ε={elasticity:.2f} - a "
                                 f"{plan['discount']:.0%} offer lifts units {plan['demand_uplift']:+.0%} "
                                 f"for {plan['profit_change']:+.0%} profit"}
        # Slowing but no profitable discount exists (inelastic / thin margin).
        return {**base, "action": "HOLD",
                "rationale": f"demand softening {trend_ratio:+.0%} but no discount clears the "
                             f"margin bar (ε={elasticity:.2f}); monitor rather than dilute"}

    return {**base, "action": "HOLD", "rationale": "demand stable - no action needed"}


def recommend_panel(
    forecasts: pd.DataFrame,
    history: pd.DataFrame,
    promo_stats: pd.DataFrame | None = None,
    margins: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Recommend an action for every series present in ``forecasts``.

    ``forecasts``: long frame (store_nbr, family, date, forecast).
    ``history``:   long panel with recent actuals (store_nbr, family, date, sales).
    ``promo_stats``: optional output of :func:`src.analysis.promotions.promo_panel`
    for per-series uplift; missing rows fall back to the default elasticity.
    """
    uplift_lookup = {}
    if promo_stats is not None and not promo_stats.empty:
        uplift_lookup = {
            (int(r.store_nbr), str(r.family)): r.uplift_adjusted
            for r in promo_stats.itertuples()
        }

    cutoff = history[DATE_COL].max() - pd.Timedelta(days=TRAILING_DAYS)
    recent = history[history[DATE_COL] > cutoff]
    trailing = recent.groupby(KEY_COLS, observed=True)[TARGET_COL].mean()
    variability = recent.groupby(KEY_COLS, observed=True)[TARGET_COL].agg(
        lambda s: s.std() / s.mean() if s.mean() > 0 else np.nan
    )

    rows = []
    for key, fc in forecasts.groupby(KEY_COLS, observed=True):
        store_nbr, family = int(key[0]), str(key[1])
        trailing_daily = float(trailing.get(key, 0.0))
        forecast_daily = float(fc["forecast"].mean())
        margin = (margins or {}).get(family, DEFAULT_GROSS_MARGIN)

        rec = recommend_series(
            trailing_daily=trailing_daily,
            forecast_daily=forecast_daily,
            uplift_adjusted=uplift_lookup.get((store_nbr, family)),
            gross_margin=margin,
            cv=float(variability.get(key)) if pd.notna(variability.get(key, np.nan)) else None,
            upcoming_festival=_horizon_festival(fc[DATE_COL]),
        )
        rows.append({"store_nbr": store_nbr, "family": family, **rec})

    out = pd.DataFrame(rows)
    # Rank so the biggest, most actionable moves are at the top of the list.
    action_priority = {"DISCOUNT": 0, "STOCK_UP": 1, "REVIEW": 2, "HOLD": 3}
    out["_p"] = out["action"].map(action_priority).fillna(9)
    out = out.sort_values(
        ["_p", "trailing_daily"], ascending=[True, False]
    ).drop(columns="_p").reset_index(drop=True)
    return out


def run(stores: list[int] | None = None) -> pd.DataFrame:
    """Load batch forecasts + history + promo uplift, write recommendations."""
    from .analysis.promotions import promo_panel
    from .forecast_batch import load_forecasts
    from .train_global import load_training_panel

    forecasts = load_forecasts()
    if forecasts is None:
        raise FileNotFoundError(
            "No batch forecasts found. Run:  python -m src.forecast_batch"
        )
    if stores is not None:
        forecasts = forecasts[forecasts["store_nbr"].isin(stores)]

    history = load_training_panel(stores, min_date="2016-01-01")
    promo_stats = promo_panel(stores)
    recs = recommend_panel(forecasts, history, promo_stats)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANALYSIS_DIR / "recommendations.csv"
    recs.to_csv(out_path, index=False)

    counts = recs["action"].value_counts().to_dict()
    print(f"[recommend] {len(recs)} series -> {counts}")
    print(f"[recommend] wrote {out_path}")
    print(recs.head(12).to_string(index=False))
    return recs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recommend offers / discounts / stock actions.")
    parser.add_argument("--stores", type=int, nargs="*", default=None)
    args = parser.parse_args()
    run(args.stores)
