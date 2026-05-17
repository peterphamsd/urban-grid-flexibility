import pandas as pd
import numpy as np
import pytest
from models.battery import BatteryModel
from models.optimizer import optimize_dispatch


def make_battery(initial_soc=0.5):
    return BatteryModel(
        capacity_kwh=10.0,
        soc_min=0.2,
        soc_max=0.9,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        max_charge_rate_kw=5.0,
        max_discharge_rate_kw=5.0,
        initial_soc=initial_soc,
    )


def make_inputs(hours=24, solar_kw=0.0, load_kw=1.0, price=0.30):
    """Build flat solar, load, and price series for a given number of hours."""
    index = pd.RangeIndex(hours)
    solar = pd.Series([solar_kw] * hours, index=index)
    load = pd.Series([load_kw] * hours, index=index)
    prices = pd.Series([price] * hours, index=index)
    return solar, load, prices


def test_energy_balance_holds():
    """At every hour: solar + discharge + grid must equal load + charge."""
    solar, load, prices = make_inputs(solar_kw=3.0, load_kw=2.0)
    battery = make_battery()

    result = optimize_dispatch(solar, load, battery, prices)

    for _, row in result.iterrows():
        lhs = row.solar_kw + row.p_discharge_kw + row.p_grid_kw
        rhs = row.load_kw + row.p_charge_kw + row.p_curtail_kw
        assert abs(lhs - rhs) < 1e-4, (
            f"Energy balance violated at hour {row.hour}: "
            f"supply {lhs:.4f} != demand {rhs:.4f}"
        )


def test_no_grid_when_solar_covers_load():
    """When solar exceeds load all day, grid import should be zero."""
    # Solar 5 kW, load 1 kW — surplus all day
    solar, load, prices = make_inputs(solar_kw=5.0, load_kw=1.0)
    battery = make_battery(initial_soc=0.2)  # start low so battery absorbs surplus

    result = optimize_dispatch(solar, load, battery, prices)

    total_grid = result.p_grid_kw.sum()
    assert total_grid < 1e-4, (
        f"Expected zero grid import when solar covers load, got {total_grid:.4f} kW"
    )


def test_soc_stays_within_bounds():
    """SOC should never exceed soc_max or fall below soc_min."""
    solar, load, prices = make_inputs(solar_kw=3.0, load_kw=2.5)
    battery = make_battery()

    result = optimize_dispatch(solar, load, battery, prices)

    assert result.soc.max() <= 0.9 + 1e-4, (
        f"SOC exceeded ceiling: {result.soc.max():.4f}"
    )
    assert result.soc.min() >= 0.2 - 1e-4, (
        f"SOC fell below floor: {result.soc.min():.4f}"
    )


def test_grid_cost_minimized_by_using_battery():
    """System with battery should have lower grid cost than without storage."""
    # Peak prices in evening, solar only in midday
    solar = pd.Series(
        [0]*6 + [2, 4, 5, 5, 4, 2] + [0]*6 + [0]*4 + [0]*2,
        index=pd.RangeIndex(24)
    )
    load = pd.Series(
        [0.8]*6 + [1.0]*6 + [1.2]*6 + [2.5]*4 + [1.0]*2,
        index=pd.RangeIndex(24)
    )
    prices = pd.Series(
        [0.10]*6 + [0.20]*6 + [0.20]*6 + [0.35]*4 + [0.15]*2,
        index=pd.RangeIndex(24)
    )

    # With battery
    battery = make_battery(initial_soc=0.3)
    result_with = optimize_dispatch(solar, load, battery, prices)
    cost_with = result_with.net_cost.sum()

    # Without battery — grid must cover all load not met by solar
    no_battery_cost = sum(
        max(load.iloc[h] - solar.iloc[h], 0) * prices.iloc[h]
        for h in range(24)
    )

    assert cost_with < no_battery_cost, (
        f"Battery should reduce cost: with={cost_with:.3f}, "
        f"without={no_battery_cost:.3f}"
    )


def test_output_has_correct_columns():
    """Result DataFrame should contain all expected columns."""
    solar, load, prices = make_inputs(solar_kw=2.0, load_kw=1.5)
    battery = make_battery()

    result = optimize_dispatch(solar, load, battery, prices)

    expected_cols = {
        "hour", "solar_kw", "load_kw",
        "p_charge_kw", "p_discharge_kw", "p_grid_kw",
        "p_curtail_kw", "soc", "price", "net_cost"
    }
    assert expected_cols.issubset(set(result.columns))