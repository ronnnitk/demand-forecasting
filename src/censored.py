"""Stockout-aware demand: recover the demand a stock-out hid.

When an item is out of stock the register records zero sales — but the demand
was still there, it just walked out unmet. Train a forecaster on those zeros and
it learns that demand is lower than it is; it then forecasts low, the buyer
orders low, the shelf runs out sooner, and the next training round sees even
more zeros. Censored demand is a feedback loop, and the way out is to spot the
likely stock-outs and put back an estimate of what would have sold.

The Favorita data ships no inventory column, so a stock-out is *inferred*, not
observed, and only where the inference is safe:

* the series is a **regular seller** (sells on most days) — a zero on a sporadic
  item is genuine no-demand, not a stock-out, and is left alone;
* the day's **expected demand is clearly positive** — estimated from the same
  weekday in a window *either side* of the gap, so a normally-dead Sunday is not
  "corrected", and a long assortment gap collapses the expectation to ~0 and is
  not touched;
* corrections are **capped** per series, so a mis-specified expectation cannot
  rewrite the history wholesale.

Every corrected day is flagged (``suspected_stockout``) and the original value
is kept (``sales_raw``), so nothing is silently overwritten and the call stays
auditable. :func:`correct_panel` is the hook :mod:`src.train_global` can opt into
with ``--correct-stockouts``.

    python -m src.censored --stores 2 44                    # scan, write a report
    python -m src.censored --store 2 --family "GROCERY II"   # one series, verbose
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    ANALYSIS_DIR,
    DATE_COL,
    KEY_COLS,
    STOCKOUT_MAX_CORRECTION_SHARE,
    STOCKOUT_MIN_EXPECTED_UNITS,
    STOCKOUT_MIN_SELLING_SHARE,
    STOCKOUT_WINDOW_DAYS,
    TARGET_COL,
)
from .io_store import iter_stores


@dataclass(frozen=True)
class StockoutConfig:
    """Knobs for the stock-out inference. Defaults come from :mod:`src.config`.

    ``min_selling_share``: skip series that sell on fewer than this fraction of
    days — their zeros are the demand pattern, not a fault.
    ``window_days``: half-width (each side) of the same-weekday window used to
    estimate what a zero day would have sold.
    ``min_expected_units``: a zero day is only suspicious if at least this much
    demand looks to have been lost.
    ``max_correction_share``: if more than this fraction of a series' days would
    be corrected, the expectation is presumed wrong and the series is left as-is.
    """

    min_selling_share: float = STOCKOUT_MIN_SELLING_SHARE
    window_days: int = STOCKOUT_WINDOW_DAYS
    min_expected_units: float = STOCKOUT_MIN_EXPECTED_UNITS
    max_correction_share: float = STOCKOUT_MAX_CORRECTION_SHARE


DEFAULT_CONFIG = StockoutConfig()


def expected_demand(
    sales: np.ndarray, dow: np.ndarray, window_days: int = STOCKOUT_WINDOW_DAYS
) -> np.ndarray:
    """What each day *would* have sold, from the same weekday either side of it.

    Per weekday phase, a centered rolling median over the neighbouring
    occurrences of that weekday. Centered on purpose: this is not a forecast,
    it is reconstruction of censored history, and the actuals surrounding a gap
    are the best evidence of what the gap hid. The median shrugs off the odd
    other stock-out in the window, and a p95 clip stops a thin window from
    imputing an implausible spike.
    """
    sales = np.asarray(sales, dtype=float)
    dow = np.asarray(dow)
    out = np.full(len(sales), np.nan)
    span = 2 * max(1, window_days // 7) + 1  # e.g. 28d -> +-4 same-weekday obs -> 9

    nonzero = sales[sales > 0]
    ceiling = float(np.quantile(nonzero, 0.95)) if len(nonzero) else np.inf

    for phase in range(7):
        idx = np.flatnonzero(dow == phase)
        if len(idx) == 0:
            continue
        med = (
            pd.Series(sales[idx])
            .rolling(span, center=True, min_periods=2)
            .median()
            .to_numpy()
        )
        out[idx] = med

    return np.clip(out, 0.0, ceiling)


def flag_series(series: pd.DataFrame, config: StockoutConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Add ``expected`` / ``suspected_stockout`` / ``lost_sales`` columns.

    ``series`` is a single-series daily frame (``date``, ``sales``); it should
    already be on a complete calendar (:func:`src.data_processing.regularize`).
    Flagging is computed for every series; whether it is *acted on* is decided
    by :func:`correct_series` (the selling-share and cap gates).
    """
    out = series.sort_values(DATE_COL).reset_index(drop=True).copy()
    sales = out[TARGET_COL].to_numpy(float)
    dow = out[DATE_COL].dt.dayofweek.to_numpy()

    exp = expected_demand(sales, dow, config.window_days)
    is_zero = sales <= 0
    suspected = is_zero & np.isfinite(exp) & (exp >= config.min_expected_units)

    selling_share = float(np.mean(sales > 0)) if len(sales) else 0.0
    if selling_share < config.min_selling_share:
        suspected[:] = False
    elif suspected.mean() > config.max_correction_share:
        # Too many "stock-outs" to be credible — distrust the expectation here.
        suspected[:] = False

    out["expected"] = np.round(exp, 2)
    out["suspected_stockout"] = suspected
    out["lost_sales"] = np.where(suspected, np.round(exp - sales, 2), 0.0)
    return out


def correct_series(series: pd.DataFrame, config: StockoutConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Return ``series`` with censored zeros replaced by their expected demand.

    Adds ``sales_raw`` (the untouched original) and ``suspected_stockout``.
    ``sales`` is raised to ``expected`` only on flagged days; every other value
    is left exactly as it was.
    """
    flagged = flag_series(series, config)
    flagged["sales_raw"] = series.sort_values(DATE_COL).reset_index(drop=True)[TARGET_COL].to_numpy(float)
    flagged[TARGET_COL] = np.where(
        flagged["suspected_stockout"], flagged["expected"], flagged["sales_raw"]
    )
    return flagged.drop(columns=["expected", "lost_sales"])


def stockout_summary(flagged: pd.DataFrame) -> dict:
    """One-row summary of a flagged series: how much demand the zeros hid."""
    sales_raw = (
        flagged["sales_raw"] if "sales_raw" in flagged else flagged[TARGET_COL]
    ).to_numpy(float)
    n = len(flagged)
    n_suspected = int(flagged["suspected_stockout"].sum())
    lost = float(flagged["lost_sales"].sum()) if "lost_sales" in flagged else float(
        (flagged["expected"] - sales_raw).where(flagged["suspected_stockout"], 0).sum()
    )
    observed = float(sales_raw.sum())
    return {
        "n_days": n,
        "selling_share": round(float(np.mean(sales_raw > 0)), 3) if n else 0.0,
        "suspected_stockout_days": n_suspected,
        "stockout_day_share": round(n_suspected / n, 4) if n else 0.0,
        "observed_units": round(observed, 1),
        "lost_units": round(lost, 1),
        "lost_demand_share": round(lost / (observed + lost), 4) if observed + lost > 0 else 0.0,
    }


def scan_panel(stores: list[int] | None = None, config: StockoutConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Per-series stock-out summary for the whole panel, streaming one store at a time."""
    from .data_processing import regularize

    rows = []
    for store_nbr, df in iter_stores(stores):
        for family, series in df.groupby("family", observed=True):
            reg = regularize(series.sort_values(DATE_COL).reset_index(drop=True))
            flagged = flag_series(reg, config)
            rows.append(
                {"store_nbr": store_nbr, "family": str(family), **stockout_summary(flagged)}
            )
    out = pd.DataFrame(rows)
    return out.sort_values("lost_units", ascending=False).reset_index(drop=True)


def correct_panel(panel: pd.DataFrame, config: StockoutConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Apply :func:`correct_series` to every series in a long panel.

    Same columns in, same columns out (plus ``sales_raw`` /
    ``suspected_stockout``), so a caller can swap this in ahead of feature
    building without touching anything downstream.
    """
    parts = []
    for _, series in panel.groupby(KEY_COLS, observed=True, sort=False):
        corrected = correct_series(series, config)
        for col in KEY_COLS:
            corrected[col] = series[col].iloc[0]
        parts.append(corrected)
    out = pd.concat(parts, ignore_index=True)
    return out[[*[c for c in panel.columns], "sales_raw", "suspected_stockout"]]


def run(
    stores: list[int] | None = None,
    family: str | None = None,
    store: int | None = None,
) -> pd.DataFrame:
    """CLI entry: scan the panel (or one series) and write a stock-out report."""
    if store is not None and family is not None:
        from .data_processing import regularize
        from .io_store import read_series

        series = regularize(read_series(store, family))
        flagged = flag_series(series)
        summary = stockout_summary(flagged)
        print(f"[censored] {store} / {family}: {summary}")
        hits = flagged[flagged["suspected_stockout"]]
        if len(hits):
            print(hits[[DATE_COL, TARGET_COL, "expected", "lost_sales"]].to_string(index=False))
        return flagged

    report = scan_panel(stores)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANALYSIS_DIR / "stockouts.csv"
    report.to_csv(out_path, index=False)

    total_lost = report["lost_units"].sum()
    total_obs = report["observed_units"].sum()
    affected = int((report["suspected_stockout_days"] > 0).sum())
    print(
        f"[censored] {len(report)} series scanned — {affected} show suspected stock-outs; "
        f"recovered {total_lost:,.0f} lost units "
        f"({total_lost / (total_obs + total_lost):.1%} of demand)"
    )
    print(f"[censored] wrote {out_path}")
    print(report.head(12).to_string(index=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and correct stock-out-censored demand.")
    parser.add_argument("--stores", type=int, nargs="*", default=None, help="limit the panel scan")
    parser.add_argument("--store", type=int, default=None, help="inspect a single series (with --family)")
    parser.add_argument("--family", type=str, default=None, help="inspect a single series (with --store)")
    args = parser.parse_args()
    run(stores=args.stores, family=args.family, store=args.store)
