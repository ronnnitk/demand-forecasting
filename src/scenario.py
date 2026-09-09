"""Scenario planning: test a merchandising plan against the forecast before
committing to it.

A forecast says what demand will be under business-as-usual. A merchandiser is
never in business-as-usual — they are deciding whether to run a promotion, cut
a price, or brace for a demand shock. This module answers the counterfactual,
"what does demand look like *if I do X*?", and prices the difference in units,
revenue and gross profit so a plan can be judged on money rather than vibes.

Three levers, each modelled the way the data actually supports:

* **Promotion** (``Scenario.promotion``) flows through the model natively.
  ``onpromotion`` is a trained feature carrying a measured median +82%
  weekday-adjusted uplift (:mod:`src.analysis.promotions`), so switching it on
  for the horizon and re-running the recursive forecast (:func:`src.predict.
  forecast_panel`, which already accepts a ``promo_plan``) gives the model's
  own learned promo response — not a bolt-on multiplier.
* **Price change** (``Scenario.price_change_pct``) has no direct feature — the
  Favorita dataset ships no price column for the model to learn from — so it is
  applied through the same constant-elasticity curve the recommender uses for
  discounts (:func:`src.recommend.optimize_discount`), extended to cover a
  price *rise* as well as a cut. Elasticity is inferred from the series' own
  measured promotion uplift, exactly as the recommender infers it.
* **Demand shock** (``Scenario.demand_multiplier``) is a manual blanket factor
  for events the model cannot know about — a heat wave, a competitor closing,
  a marketing push — stated explicitly rather than hidden in the forecast.

    python -m src.scenario --store 44 --family "GROCERY II" --promo --days 15
    python -m src.scenario --store 44 --family "GROCERY II" --price-change -0.10
    python -m src.scenario --store 44 --family "GROCERY II" --compare
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    DATE_COL,
    DEFAULT_GROSS_MARGIN,
    KEY_COLS,
    PROMO_COL,
    SCENARIO_HORIZON,
    SCENARIO_PRICE_GRID,
)
from .predict import forecast_panel
from .recommend import elasticity_from_uplift

# How far back to look when a scenario asks for promotion "on" without saying
# at what intensity — the series' own recent typical "on" level.
PROMO_LOOKBACK_DAYS = 90
# Guard rails so a price scenario can't be pushed into a non-finite multiplier
# (a >100% "cut" or an absurd mark-up).
MIN_PRICE_RATIO, MAX_PRICE_RATIO = 0.05, 3.0


@dataclass(frozen=True)
class Scenario:
    """One merchandising plan to test against the forecast.

    ``promotion``: ``None`` leaves the promo plan unchanged (the recent
    average is carried forward, same as a plain forecast); ``True`` turns
    promotion on at the series' own typical "on" intensity (see
    :func:`promo_intensity`); ``False`` (or ``0``) forces it off; any other
    number sets an explicit ``onpromotion`` intensity for the whole horizon.
    ``price_change_pct``: e.g. ``-0.10`` for a 10% cut, ``0.05`` for a 5% rise.
    ``demand_multiplier``: a blanket factor for shocks the model cannot know
    about — 1.15 = +15%, 0.85 = -15%.
    ``elasticity``: override the inferred elasticity; ``None`` infers it from
    the series' own measured promotion uplift (falling back to the configured
    default), exactly as :mod:`src.recommend` does.
    """

    name: str = "scenario"
    promotion: bool | float | None = None
    price_change_pct: float = 0.0
    demand_multiplier: float = 1.0
    elasticity: float | None = None


BUSINESS_AS_USUAL = Scenario(name="business_as_usual")


def promo_intensity(history: pd.DataFrame, promotion: bool | float | None) -> float | None:
    """Resolve a ``Scenario.promotion`` value to a concrete ``onpromotion`` level.

    ``None`` passes through unchanged (the caller should leave the promo plan
    as-is). ``True`` reads the series' own recent history for the average
    intensity on days it *was* promoted, so "turn promotion on" means
    something calibrated to this series rather than a guessed constant.
    """
    if promotion is None:
        return None
    if isinstance(promotion, bool):
        if not promotion:
            return 0.0
        cutoff = history[DATE_COL].max() - pd.Timedelta(days=PROMO_LOOKBACK_DAYS)
        recent = history[history[DATE_COL] > cutoff]
        on_days = recent.loc[recent[PROMO_COL] > 0, PROMO_COL]
        return float(on_days.mean()) if len(on_days) else 1.0
    return float(promotion)


def _promo_plan(history: pd.DataFrame, dates: pd.Series, intensity: float) -> pd.DataFrame:
    """A flat promo plan at ``intensity`` for every series, every horizon date."""
    keys = history[KEY_COLS].drop_duplicates()
    calendar = pd.DataFrame({DATE_COL: pd.to_datetime(pd.Series(dates).unique())})
    plan = keys.merge(calendar, how="cross")
    plan[PROMO_COL] = float(intensity)
    return plan


def price_multiplier(price_change_pct: float, elasticity: float) -> float:
    """Demand multiplier under constant price elasticity: ``Q1/Q0 = (P1/P0)^ε``.

    Same functional form as a discount (``P1/P0 = 1 - d``,
    :func:`src.recommend.constant_elasticity_multiplier`); this version isn't
    clamped to non-negative discounts, so it covers a price *rise* too — a
    scenario is as often "what if we mark it up" as "what if we mark it down".
    """
    price_ratio = float(np.clip(1.0 + price_change_pct, MIN_PRICE_RATIO, MAX_PRICE_RATIO))
    return float(price_ratio ** elasticity)


def scenario_economics(
    baseline_daily: np.ndarray,
    scenario_daily: np.ndarray,
    gross_margin: float,
    price_change_pct: float,
) -> dict:
    """Expected demand / revenue / profit deltas of a scenario vs baseline.

    Accounted the same way :func:`src.recommend.optimize_discount` prices a
    discount, so a scenario and a recommendation read on the same basis:
    profit per baseline unit moves from ``margin`` to ``margin +
    price_change_pct`` (a cut is negative, a rise positive) and units move by
    whatever the scenario's demand path implies.
    """
    base_units = float(np.sum(baseline_daily))
    scen_units = float(np.sum(scenario_daily))
    demand_delta_pct = scen_units / base_units - 1.0 if base_units > 0 else 0.0

    base_revenue = base_units  # full-price baseline: 1 unit of price per unit
    scen_revenue = scen_units * (1.0 + price_change_pct)
    revenue_delta_pct = scen_revenue / base_revenue - 1.0 if base_revenue > 0 else 0.0

    base_profit = base_units * gross_margin
    scen_profit = scen_units * (gross_margin + price_change_pct)
    profit_delta_pct = scen_profit / base_profit - 1.0 if base_profit > 0 else 0.0

    return {
        "baseline_units": round(base_units, 1),
        "scenario_units": round(scen_units, 1),
        "demand_delta_pct": round(demand_delta_pct * 100, 1),
        "revenue_delta_pct": round(revenue_delta_pct * 100, 1),
        "profit_delta_pct": round(profit_delta_pct * 100, 1),
    }


def simulate_series(
    history: pd.DataFrame,
    model,
    meta: dict,
    scenario: Scenario = BUSINESS_AS_USUAL,
    days: int = SCENARIO_HORIZON,
    uplift_adjusted: float | None = None,
    gross_margin: float = DEFAULT_GROSS_MARGIN,
) -> dict:
    """Simulate one scenario for one series. Returns day-by-day paths + priced deltas.

    ``history`` is a single-series long frame (``date``, ``store_nbr``,
    ``family``, ``sales``, ``onpromotion``) — the same shape
    :func:`src.predict.forecast_panel` expects. The promotion lever only costs
    a second recursive forecast when the scenario actually changes it;
    business-as-usual reuses the one baseline pass.
    """
    history = history.sort_values(DATE_COL)

    baseline_fc = forecast_panel(history, model, meta, days=days).sort_values(DATE_COL)
    dates = baseline_fc[DATE_COL].reset_index(drop=True)
    baseline_daily = baseline_fc["forecast"].to_numpy(float)

    elasticity = (
        scenario.elasticity
        if scenario.elasticity is not None
        else elasticity_from_uplift(uplift_adjusted)
    )

    intensity = promo_intensity(history, scenario.promotion)
    if intensity is None:
        scenario_daily = baseline_daily.copy()
    else:
        plan = _promo_plan(history, dates, intensity)
        scen_fc = forecast_panel(history, model, meta, days=days, promo_plan=plan).sort_values(DATE_COL)
        scenario_daily = scen_fc["forecast"].to_numpy(float)

    mult = price_multiplier(scenario.price_change_pct, elasticity) * scenario.demand_multiplier
    scenario_daily = np.clip(scenario_daily * mult, 0, None)

    econ = scenario_economics(baseline_daily, scenario_daily, gross_margin, scenario.price_change_pct)
    daily = pd.DataFrame(
        {
            DATE_COL: dates,
            "baseline": np.round(baseline_daily, 2),
            "scenario": np.round(scenario_daily, 2),
        }
    )

    return {
        "scenario": scenario.name,
        "elasticity": round(elasticity, 2),
        "promotion_intensity": None if intensity is None else round(intensity, 2),
        "price_change_pct": round(scenario.price_change_pct * 100, 1),
        "demand_multiplier": scenario.demand_multiplier,
        **econ,
        "daily": daily,
    }


def default_scenarios() -> list[Scenario]:
    """The standard "should we run a deal or move the price" comparison set."""
    return [
        BUSINESS_AS_USUAL,
        Scenario(name="promotion", promotion=True),
        *[
            Scenario(name=f"price_{p:+.0%}", price_change_pct=p)
            for p in SCENARIO_PRICE_GRID
            if p != 0.0  # already covered by business_as_usual
        ],
    ]


def compare_scenarios(
    history: pd.DataFrame,
    model,
    meta: dict,
    scenarios: list[Scenario] | None = None,
    days: int = SCENARIO_HORIZON,
    uplift_adjusted: float | None = None,
    gross_margin: float = DEFAULT_GROSS_MARGIN,
) -> pd.DataFrame:
    """Run a set of scenarios for one series and rank them by profit delta."""
    rows = []
    for sc in scenarios or default_scenarios():
        result = simulate_series(history, model, meta, sc, days, uplift_adjusted, gross_margin)
        rows.append({k: v for k, v in result.items() if k != "daily"})

    out = pd.DataFrame(rows)
    return out.sort_values("profit_delta_pct", ascending=False).reset_index(drop=True)


def run(
    store_nbr: int,
    family: str,
    days: int = SCENARIO_HORIZON,
    promotion: bool | float | None = None,
    price_change_pct: float = 0.0,
    demand_multiplier: float = 1.0,
    compare: bool = False,
    min_date: str = "2016-01-01",
):
    """CLI entry point: simulate one series against the persisted global model."""
    from .analysis.promotions import promo_uplift
    from .data_processing import regularize
    from .io_store import read_series
    from .train_global import load as load_point_model

    model, meta = load_point_model()
    series = read_series(store_nbr, family)
    history = regularize(series)
    history["store_nbr"] = store_nbr
    history["family"] = family
    if min_date:
        history = history[history[DATE_COL] >= pd.Timestamp(min_date)]

    uplift = promo_uplift(series).get("uplift_adjusted")
    elasticity = elasticity_from_uplift(uplift)

    if compare:
        table = compare_scenarios(history, model, meta, days=days, uplift_adjusted=uplift)
        print(f"[scenario] {store_nbr} / {family} — {days}d comparison (eps={elasticity:.2f}):")
        print(table.drop(columns=["elasticity", "demand_multiplier"]).to_string(index=False))
        return table

    scenario = Scenario(
        name="custom", promotion=promotion, price_change_pct=price_change_pct,
        demand_multiplier=demand_multiplier,
    )
    result = simulate_series(history, model, meta, scenario, days, uplift)
    print(
        f"[scenario] {store_nbr} / {family} — {scenario.name}: "
        f"demand {result['demand_delta_pct']:+.1f}% revenue {result['revenue_delta_pct']:+.1f}% "
        f"profit {result['profit_delta_pct']:+.1f}% (eps={elasticity:.2f})"
    )
    print(result["daily"].to_string(index=False))
    return result["daily"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate a merchandising scenario against the forecast.")
    parser.add_argument("--store", type=int, required=True)
    parser.add_argument("--family", type=str, required=True)
    parser.add_argument("--days", type=int, default=SCENARIO_HORIZON)
    parser.add_argument("--promo", action="store_true", help="turn promotion on at typical intensity")
    parser.add_argument("--no-promo", action="store_true", help="force promotion off")
    parser.add_argument("--price-change", type=float, default=0.0, help="e.g. -0.10 for a 10%% cut")
    parser.add_argument("--demand-shock", type=float, default=1.0, help="blanket multiplier, e.g. 1.15")
    parser.add_argument("--compare", action="store_true", help="compare the standard scenario set")
    args = parser.parse_args()

    promo_arg = True if args.promo else (False if args.no_promo else None)
    run(
        args.store, args.family, args.days,
        promotion=promo_arg, price_change_pct=args.price_change,
        demand_multiplier=args.demand_shock, compare=args.compare,
    )
