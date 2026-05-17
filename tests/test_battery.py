import pytest
from models.battery import BatteryModel


def make_battery():
    """Standard 10 kWh home battery reused across tests."""
    return BatteryModel(
        capacity_kwh=10.0,
        soc_min=0.2,
        soc_max=0.9,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        max_charge_rate_kw=5.0,
        max_discharge_rate_kw=5.0,
        initial_soc=0.5,
    )


def test_charging_increases_soc():
    """Charging should increase SOC by the correct amount."""
    battery = make_battery()

    # Charge at 3 kW for 1 hour
    # Expected delta: (3 * 0.95) / 10 = 0.285
    # Expected new SOC: 0.5 + 0.285 = 0.785
    new_soc = battery.step(p_charge_kw=3.0, p_discharge_kw=0.0)

    assert abs(new_soc - 0.785) < 1e-6, (
        f"Expected SOC 0.785, got {new_soc:.6f}"
    )


def test_discharging_decreases_soc():
    """Discharging should decrease SOC by the correct amount."""
    battery = make_battery()

    # Discharge at 2 kW for 1 hour
    # Expected delta: (2 / 0.95) / 10 = 0.2105
    # Expected new SOC: 0.5 - 0.2105 = 0.2895
    new_soc = battery.step(p_charge_kw=0.0, p_discharge_kw=2.0)

    assert abs(new_soc - 0.2895) < 1e-4, (
        f"Expected SOC ~0.2895, got {new_soc:.6f}"
    )


def test_soc_ceiling_enforced():
    """Charging beyond soc_max should raise a ValueError."""
    battery = make_battery()

    # Battery at 0.5, capacity 10 kWh, soc_max 0.9
    # Charging 5 kW for 1 hour adds (5 * 0.95) / 10 = 0.475
    # New SOC would be 0.975 — above the 0.9 ceiling
    with pytest.raises(ValueError):
        battery.step(p_charge_kw=5.0, p_discharge_kw=0.0)


def test_soc_floor_enforced():
    """Discharging below soc_min should raise a ValueError."""
    battery = make_battery()

    # Battery at 0.5, discharging 4 kW for 1 hour
    # Delta: (4 / 0.95) / 10 = 0.421
    # New SOC would be 0.079 — below the 0.2 floor
    with pytest.raises(ValueError):
        battery.step(p_charge_kw=0.0, p_discharge_kw=4.0)


def test_simultaneous_charge_discharge_rejected():
    """Charging and discharging at the same time should raise a ValueError."""
    battery = make_battery()

    with pytest.raises(ValueError):
        battery.step(p_charge_kw=2.0, p_discharge_kw=2.0)


def test_exceeding_max_charge_rate_rejected():
    """Charging above max rate should raise a ValueError."""
    battery = make_battery()

    with pytest.raises(ValueError):
        battery.step(p_charge_kw=6.0, p_discharge_kw=0.0)


def test_history_tracks_correctly():
    """SOC history should record every timestep including initial state."""
    battery = make_battery()

    battery.step(p_charge_kw=2.0, p_discharge_kw=0.0)
    battery.step(p_charge_kw=0.0, p_discharge_kw=1.0)

    # History should have 3 entries: initial + 2 steps
    assert len(battery.soc_history) == 3


def test_reset_restores_initial_state():
    """Reset should return battery to starting SOC and clear history."""
    battery = make_battery()

    battery.step(p_charge_kw=2.0, p_discharge_kw=0.0)
    battery.reset(initial_soc=0.5)

    assert battery.soc == 0.5
    assert len(battery.soc_history) == 1