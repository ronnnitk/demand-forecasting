"""Self-adapting retraining: the loop that lets the engine keep itself current.

A forecasting model decays. Demand drifts, new products and stores appear, and a
model trained last quarter quietly gets worse until someone notices. The manual
answer is "retrain on a cron"; the problem with a blind cron is that it retrains
when nothing has changed (waste) and *doesn't* retrain when something breaks
between runs (risk).

This module makes retraining a **decision** instead of a schedule:

1. **Monitor.** Score the live champion on the most recent holdout window and
   compare its WAPE to the WAPE it had at training time. A meaningful increase
   is drift.
2. **Trigger.** Retrain when the model has drifted *or* has simply aged past
   ``RETRAIN_MAX_AGE_DAYS`` — whichever comes first.
3. **Challenge.** Fit a challenger on data up to the same cut-off and forecast
   the same holdout. Champion and challenger are judged on identical, unseen
   days, so the comparison is fair.
4. **Promote conservatively.** Adopt the challenger only if it beats the champion
   by more than ``PROMOTE_MARGIN``. Otherwise keep the champion — a new model
   that is merely *different* is not an improvement, and churn has its own cost.

Every cycle appends a row to ``models/retrain_log.csv``, so the model's history
is auditable: what was decided, why, and whether accuracy moved.

    python -m src.adaptive                    # one monitored cycle, all stores
    python -m src.adaptive --stores 2 44 47   # a fast subset for a quick loop
    python -m src.adaptive --force            # retrain regardless of the trigger
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ADAPTIVE_HOLDOUT_DAYS,
    DATE_COL,
    DRIFT_TOLERANCE,
    KEY_COLS,
    MODELS_DIR,
    PROMOTE_MARGIN,
    RETRAIN_MAX_AGE_DAYS,
    TARGET_COL,
)
from .metrics import evaluate_all
from .predict import forecast_panel
from .train_global import (
    GLOBAL_MODEL_PATH,
    fit_global,
    load,
    load_training_panel,
    save,
)

RETRAIN_LOG = MODELS_DIR / "retrain_log.csv"
ARCHIVE_DIR = MODELS_DIR / "archive"


def evaluate_on_holdout(model, meta: dict, panel: pd.DataFrame, holdout_days: int) -> dict:
    """Score a model by forecasting the final ``holdout_days`` from the cut-off.

    Both champion and challenger go through this same path, so neither is
    advantaged: each forecasts unseen days from history up to the identical
    cut-off, and is measured against the same actuals.
    """
    cutoff = panel[DATE_COL].max() - pd.Timedelta(days=holdout_days)
    history = panel[panel[DATE_COL] <= cutoff]
    actual = panel[panel[DATE_COL] > cutoff]
    if history.empty or actual.empty:
        return {}

    forecast = forecast_panel(history, model, meta, days=holdout_days)
    a = actual.set_index([*KEY_COLS, DATE_COL])[TARGET_COL]
    p = forecast.set_index([*KEY_COLS, DATE_COL])["forecast"]
    joined = pd.concat([a.rename("y"), p.rename("yhat")], axis=1).dropna()
    if joined.empty:
        return {}
    return evaluate_all(joined["y"].to_numpy(float), joined["yhat"].to_numpy(float))


def model_age_days(meta: dict) -> float:
    """Days since the champion was trained, from its recorded timestamp."""
    trained_at = meta.get("trained_at")
    if not trained_at:
        return float("inf")
    age = pd.Timestamp.utcnow() - pd.Timestamp(trained_at)
    return age.total_seconds() / 86_400


def decide_retrain(
    meta: dict,
    recent_wape: float,
    max_age_days: float = RETRAIN_MAX_AGE_DAYS,
    drift_tolerance: float = DRIFT_TOLERANCE,
) -> tuple[bool, str]:
    """Should we retrain, and why? Pure function of the monitoring signals."""
    baseline_wape = (meta.get("holdout_metrics") or {}).get("wape")
    age = model_age_days(meta)

    if baseline_wape and np.isfinite(recent_wape):
        drift = recent_wape / baseline_wape - 1
        if drift > drift_tolerance:
            return True, (
                f"drift: recent WAPE {recent_wape:.3f} is {drift:+.0%} vs "
                f"training WAPE {baseline_wape:.3f} (tol {drift_tolerance:.0%})"
            )
    if age >= max_age_days:
        return True, f"stale: model is {age:.1f}d old (max {max_age_days}d)"
    return False, (
        f"healthy: recent WAPE {recent_wape:.3f}, age {age:.1f}d — no retrain"
    )


def _archive_champion(path: Path) -> Path | None:
    """Move the current champion aside before a promotion, timestamped."""
    if not path.exists():
        return None
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%S")
    dest = ARCHIVE_DIR / f"{path.stem}_{stamp}{path.suffix}"
    shutil.copy2(path, dest)
    return dest


def _log(row: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(RETRAIN_LOG, mode="a", header=not RETRAIN_LOG.exists(), index=False)


def run_cycle(
    stores: list[int] | None = None,
    min_date: str | None = "2015-01-01",
    holdout_days: int = ADAPTIVE_HOLDOUT_DAYS,
    force: bool = False,
    promote_margin: float = PROMOTE_MARGIN,
) -> dict:
    """One monitor -> decide -> challenge -> promote cycle. Returns the decision."""
    panel = load_training_panel(stores, min_date)
    decision = {
        "run_at": pd.Timestamp.utcnow().isoformat(),
        "n_series": int(panel.groupby(KEY_COLS, observed=True).ngroups),
        "holdout_days": holdout_days,
    }

    # --- 1. Monitor the incumbent champion (if any) ---------------------------
    champion = champion_meta = None
    recent_wape = float("nan")
    try:
        champion, champion_meta = load()
        champ_metrics = evaluate_on_holdout(champion, champion_meta, panel, holdout_days)
        recent_wape = champ_metrics.get("wape", float("nan"))
        decision["champion_recent_wape"] = round(recent_wape, 4) if np.isfinite(recent_wape) else None
        decision["champion_train_wape"] = (champion_meta.get("holdout_metrics") or {}).get("wape")
    except FileNotFoundError:
        decision["champion_recent_wape"] = None
        decision["champion_train_wape"] = None

    # --- 2. Decide -------------------------------------------------------------
    if champion is None:
        retrain, reason = True, "cold start: no champion model exists"
    elif force:
        retrain, reason = True, "forced by operator"
    else:
        retrain, reason = decide_retrain(champion_meta, recent_wape)
    decision["retrain"] = retrain
    decision["reason"] = reason

    if not retrain:
        decision["action"] = "kept_champion"
        print(f"[adaptive] {reason}")
        _log(decision)
        return decision

    # --- 3. Challenge: fit on history up to the holdout cut-off ---------------
    cutoff = panel[DATE_COL].max() - pd.Timedelta(days=holdout_days)
    challenger, challenger_meta = fit_global(panel[panel[DATE_COL] <= cutoff], validation_days=0)
    chal_metrics = evaluate_on_holdout(challenger, challenger_meta, panel, holdout_days)
    challenger_wape = chal_metrics.get("wape", float("nan"))
    decision["challenger_wape"] = round(challenger_wape, 4) if np.isfinite(challenger_wape) else None

    # --- 4. Promote conservatively --------------------------------------------
    promote = champion is None or (
        np.isfinite(challenger_wape)
        and np.isfinite(recent_wape)
        and challenger_wape < recent_wape * (1 - promote_margin)
    )

    if promote:
        archived = _archive_champion(GLOBAL_MODEL_PATH)
        # Refit on ALL data (including the holdout) before shipping: the holdout
        # was only withheld to judge the challenger; the model that goes live
        # should use every day available.
        final_model, final_meta = fit_global(panel, validation_days=holdout_days)
        save(final_model, final_meta)
        decision["action"] = "promoted_challenger"
        decision["archived_to"] = str(archived) if archived else None
        print(f"[adaptive] promoted challenger (WAPE {challenger_wape:.3f} < "
              f"champion {recent_wape:.3f}). {reason}")
    else:
        decision["action"] = "kept_champion_challenger_worse"
        print(f"[adaptive] retrained but kept champion: challenger WAPE "
              f"{challenger_wape:.3f} did not beat {recent_wape:.3f} by "
              f"{promote_margin:.0%}. {reason}")

    _log(decision)
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one self-adaptive retraining cycle.")
    parser.add_argument("--stores", type=int, nargs="*", default=None)
    parser.add_argument("--min-date", type=str, default="2015-01-01")
    parser.add_argument("--holdout-days", type=int, default=ADAPTIVE_HOLDOUT_DAYS)
    parser.add_argument("--force", action="store_true", help="retrain regardless of trigger")
    args = parser.parse_args()
    run_cycle(args.stores, args.min_date, args.holdout_days, args.force)
