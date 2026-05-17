import pandas as pd
import numpy as np
import pytest
from models.load import LoadModel


def make_index(days=7):
    """Generate a timezone-aware hourly DatetimeIndex for testing."""
    return pd.date_range(
        start="2023-06-01",
        periods=days * 24,
        freq="h",
        tz="America/Los_Angeles"
    )


def test_output_is_correct_length():
    """Output should have one value per hour in the index."""
    model = LoadModel(annual_kwh=6000.0, random_seed=42)
    index = make_index(days=7)
    result = model.generate(index)
    assert len(result) == len(index)


def test_output_never_negative():
    """Load should never go below zero."""
    model = LoadModel(annual_kwh=6000.0, random_seed=42)
    index = make_index(days=30)
    result = model.generate(index)
    assert (result >= 0).all(), "Load should never be negative"


def test_evening_peak_higher_than_overnight():
    """Evening hours should average higher load than overnight hours."""
    model = LoadModel(annual_kwh=6000.0, random_seed=42)

    # Use a full year for a stable average
    index = pd.date_range(
        start="2023-01-01",
        periods=365 * 24,
        freq="h",
        tz="America/Los_Angeles"
    )
    result = model.generate(index)

    # Evening peak: 6pm - 9pm
    evening = result[result.index.hour.isin([18, 19, 20])].mean()

    # Overnight: 1am - 4am
    overnight = result[result.index.hour.isin([1, 2, 3])].mean()

    assert evening > overnight, (
        f"Expected evening ({evening:.3f} kW) > overnight ({overnight:.3f} kW)"
    )


def test_summer_higher_than_winter():
    """August load should average higher than January due to AC."""
    model = LoadModel(annual_kwh=6000.0, random_seed=42)

    index = pd.date_range(
        start="2023-01-01",
        periods=365 * 24,
        freq="h",
        tz="America/Los_Angeles"
    )
    result = model.generate(index)

    august = result[result.index.month == 8].mean()
    january = result[result.index.month == 1].mean()

    assert august > january, (
        f"Expected August ({august:.3f} kW) > January ({january:.3f} kW)"
    )


def test_annual_consumption_close_to_target():
    """Total annual consumption should be within 5% of the target."""
    target_kwh = 6000.0
    model = LoadModel(annual_kwh=target_kwh, random_seed=42)

    index = pd.date_range(
        start="2023-01-01",
        periods=365 * 24,
        freq="h",
        tz="America/Los_Angeles"
    )
    result = model.generate(index)

    # Each value is kW over 1 hour = kWh
    total_kwh = result.sum()
    error_pct = abs(total_kwh - target_kwh) / target_kwh

    assert error_pct < 0.05, (
        f"Annual consumption {total_kwh:.1f} kWh is more than 5% "
        f"from target {target_kwh} kWh (error: {error_pct:.1%})"
    )