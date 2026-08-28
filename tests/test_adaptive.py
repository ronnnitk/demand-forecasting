"""Tests for the self-adaptive retraining decision logic.

The full cycle needs the panel store and a real fit; these tests pin the pure
decision rules that decide *whether* to retrain and *whether* to promote, which
is where a wrong threshold would either churn models needlessly or let a decayed
model run.
"""

import pandas as pd

from src.adaptive import decide_retrain, model_age_days


def _meta(train_wape=0.30, age_days=1.0):
    trained_at = (pd.Timestamp.utcnow() - pd.Timedelta(days=age_days)).isoformat()
    return {"holdout_metrics": {"wape": train_wape}, "trained_at": trained_at}


def test_healthy_model_is_not_retrained():
    retrain, reason = decide_retrain(_meta(0.30, age_days=1.0), recent_wape=0.31)
    assert retrain is False
    assert "healthy" in reason


def test_drift_triggers_retrain():
    # Recent WAPE 0.40 vs training 0.30 = +33%, over the 15% tolerance.
    retrain, reason = decide_retrain(_meta(0.30, age_days=1.0), recent_wape=0.40)
    assert retrain is True
    assert "drift" in reason


def test_staleness_triggers_retrain_even_when_accurate():
    retrain, reason = decide_retrain(
        _meta(0.30, age_days=30.0), recent_wape=0.30, max_age_days=7
    )
    assert retrain is True
    assert "stale" in reason


def test_small_degradation_within_tolerance_is_ignored():
    # +10% is under the 15% default tolerance -> no drift retrain.
    retrain, reason = decide_retrain(_meta(0.30, age_days=1.0), recent_wape=0.33)
    assert retrain is False


def test_missing_timestamp_reads_as_infinitely_old():
    assert model_age_days({}) == float("inf")
    # ...which forces a retrain on the staleness branch.
    retrain, _ = decide_retrain({"holdout_metrics": {}}, recent_wape=float("nan"))
    assert retrain is True
