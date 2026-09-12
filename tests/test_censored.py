"""Tests for stockout-aware demand correction."""

import numpy as np
import pandas as pd
import pytest

from src.censored import (
    StockoutConfig,
    correct_panel,
    correct_series,
    expected_demand,
    flag_series,
)
from src.config import DATE_COL, KEY_COLS, TARGET_COL


@pytest.fixture
def sample_series():
    """Create a sample time series with synthetic stock-outs."""
    dates = pd.date_range("2023-01-01", periods=56, freq="D")  # 8 weeks
    # Regular pattern with occasional "stock-outs" (zeros that look suspicious)
    sales = np.array([5, 6, 0, 7, 5, 6, 5, 6, 0, 7, 5, 6, 5, 6, 0, 7, 5, 6, 5, 6,
                      0, 7, 5, 6, 5, 6, 0, 7, 5, 6, 5, 6, 0, 7, 5, 6, 5, 6, 0, 7,
                      5, 6, 5, 6, 0, 7, 5, 6, 5, 6, 0, 7, 5, 6, 5, 6], dtype=float)

    return pd.DataFrame({
        DATE_COL: dates,
        TARGET_COL: sales,
        "store_nbr": 2,
        "family": "GROCERY II",
    })


def test_expected_demand_basic():
    """Test that expected demand estimates from same weekday neighbors."""
    sales = np.array([5.0, 6.0, 0.0, 7.0, 5.0, 6.0, 5.0, 6.0, 0.0, 7.0])
    dow = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2])  # day of week

    expected = expected_demand(sales, dow, window_days=7)

    # Should produce values for each position
    assert len(expected) == len(sales)
    # Most positions should have positive expectations
    assert np.sum(expected > 0) > 0


def test_flag_series_detects_stockouts(sample_series):
    """Test that flag_series marks suspicious zeros."""
    flagged = flag_series(sample_series)

    # Should add expected, suspected_stockout, lost_sales columns
    assert "expected" in flagged.columns
    assert "suspected_stockout" in flagged.columns
    assert "lost_sales" in flagged.columns

    # Some zeros should be flagged as suspected stock-outs
    assert flagged["suspected_stockout"].sum() > 0


def test_flag_series_respects_min_selling_share():
    """Test that series with low selling share are not corrected."""
    # Series that rarely sells (only 10% of days)
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    sales = np.array([0.0] * 27 + [5.0, 6.0, 7.0], dtype=float)

    series = pd.DataFrame({
        DATE_COL: dates,
        TARGET_COL: sales,
    })

    config = StockoutConfig(min_selling_share=0.70)
    flagged = flag_series(series, config)

    # All zeros should be left untouched (not flagged) because selling share < 0.70
    assert flagged["suspected_stockout"].sum() == 0


def test_correct_series_preserves_original(sample_series):
    """Test that correct_series keeps sales_raw for audit trail."""
    corrected = correct_series(sample_series)

    # Should have sales_raw and suspected_stockout columns
    assert "sales_raw" in corrected.columns
    assert "suspected_stockout" in corrected.columns

    # Original values should be preserved in sales_raw
    assert np.allclose(
        corrected["sales_raw"].to_numpy(),
        sample_series[TARGET_COL].to_numpy()
    )


def test_correct_series_only_updates_flagged_days(sample_series):
    """Test that only flagged zeros are corrected."""
    corrected = correct_series(sample_series)
    original = sample_series[TARGET_COL].to_numpy()
    corrected_sales = corrected[TARGET_COL].to_numpy()

    flagged_mask = corrected["suspected_stockout"].to_numpy()

    # Non-flagged days should match original
    assert np.allclose(
        corrected_sales[~flagged_mask],
        original[~flagged_mask]
    )


def test_correct_panel_maintains_grouping():
    """Test that correct_panel preserves store_nbr and family grouping."""
    dates = pd.date_range("2023-01-01", periods=56, freq="D")

    # Create a small panel with 2 series
    panel = pd.concat([
        pd.DataFrame({
            DATE_COL: dates,
            TARGET_COL: np.array([5, 6, 0, 7] * 14, dtype=float),
            "store_nbr": 2,
            "family": "GROCERY II",
        }),
        pd.DataFrame({
            DATE_COL: dates,
            TARGET_COL: np.array([10, 12, 0, 15] * 14, dtype=float),
            "store_nbr": 3,
            "family": "DAIRY",
        }),
    ], ignore_index=True)

    corrected = correct_panel(panel)

    # Should maintain the grouping columns
    assert "store_nbr" in corrected.columns
    assert "family" in corrected.columns

    # Should have correct counts per group
    assert len(corrected[corrected["store_nbr"] == 2]) == 56
    assert len(corrected[corrected["store_nbr"] == 3]) == 56


def test_stockout_config_defaults():
    """Test that StockoutConfig uses sensible defaults."""
    config = StockoutConfig()

    assert config.min_selling_share == 0.70
    assert config.window_days == 28
    assert config.min_expected_units == 1.0
    assert config.max_correction_share == 0.10


def test_stockout_config_custom_values():
    """Test that StockoutConfig accepts custom values."""
    config = StockoutConfig(
        min_selling_share=0.80,
        window_days=14,
        min_expected_units=2.0,
        max_correction_share=0.05
    )

    assert config.min_selling_share == 0.80
    assert config.window_days == 14
    assert config.min_expected_units == 2.0
    assert config.max_correction_share == 0.05
