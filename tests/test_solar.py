import pandas as pd
import numpy as np
import pytest
from models.solar import SolarModel

# Standard San Diego rooftop system we'll reuse across tests
def make_model():
    return SolarModel(
        panel_area_m2=34.0,      # 20 panels × 1.7 m² each
        efficiency=0.20,
        temp_coefficient=0.004,
        tilt_deg=20,
        azimuth_deg=180,         # south-facing
        latitude=32.7,
        longitude=-117.2,
        altitude_m=50,
    )


def make_climate_row(month, day, hour, ghi, dni, dhi, temp_air):
    """Helper that builds a single-row climate DataFrame for a given hour."""
    timestamp = pd.Timestamp(
        year=2023, month=month, day=day, hour=hour,
        tz="America/Los_Angeles"
    )
    return pd.DataFrame(
        {"GHI": [ghi], "DNI": [dni], "DHI": [dhi], "temp_air": [temp_air]},
        index=[timestamp]
    )


def test_zero_output_at_night():
    """Panel should produce no power when there is no irradiance."""
    model = make_model()
    climate = make_climate_row(
        month=6, day=15, hour=2,  # 2am
        ghi=0, dni=0, dhi=0, temp_air=18.0
    )
    result = model.generate(climate)
    assert result.iloc[0] == 0.0, "Expected zero output at night"


def test_positive_output_at_noon():
    """Panel should produce meaningful power at solar noon on a clear day."""
    model = make_model()
    climate = make_climate_row(
        month=6, day=15, hour=12,  # solar noon
        ghi=950, dni=850, dhi=100, temp_air=25.0
    )
    result = model.generate(climate)
    assert result.iloc[0] > 1.0, "Expected at least 1 kW output at noon"


def test_output_decreases_with_higher_temperature():
    """Higher cell temperature should reduce output — same irradiance, different temp."""
    model = make_model()

    cool_climate = make_climate_row(
        month=6, day=15, hour=12,
        ghi=950, dni=850, dhi=100, temp_air=15.0  # cool day
    )
    hot_climate = make_climate_row(
        month=6, day=15, hour=12,
        ghi=950, dni=850, dhi=100, temp_air=40.0  # hot day
    )

    cool_output = model.generate(cool_climate).iloc[0]
    hot_output = model.generate(hot_climate).iloc[0]

    assert cool_output > hot_output, (
        f"Expected cool output ({cool_output:.3f} kW) > "
        f"hot output ({hot_output:.3f} kW)"
    )


def test_output_never_negative():
    """Output should never go below zero regardless of conditions."""
    model = make_model()
    climate = make_climate_row(
        month=1, day=15, hour=12,
        ghi=200, dni=150, dhi=50, temp_air=-5.0  # cold winter day
    )
    result = model.generate(climate)
    assert result.iloc[0] >= 0.0, "Output should never be negative"