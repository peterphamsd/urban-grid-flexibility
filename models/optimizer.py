import pandas as pd
import numpy as np
from pyomo.environ import (
    ConcreteModel,
    Var,
    Objective,
    Constraint,
    NonNegativeReals,
    minimize,
    value,
    SolverFactory,
)
from models.battery import BatteryModel


def optimize_dispatch(
    solar_kw: pd.Series,
    load_kw: pd.Series,
    battery: BatteryModel,
    price_per_kwh: pd.Series,
) -> pd.DataFrame:
    """
    Optimize hourly battery dispatch to minimize grid electricity cost.

    Solves a linear program across the full time horizon simultaneously,
    finding the optimal charge, discharge, and grid import at each hour
    subject to physical constraints.

    Parameters -
    solar_kw : pd.Series (Hourly solar generation in kW)
    load_kw : pd.Series (Hourly load demand in kW)
    battery : BatteryModel (Battery with physical parameters defined)
    price_per_kwh : pd.Series (Hourly electricity price in $/kWh)

    Returns -
    pd.DataFrame
        Hourly dispatch decisions with columns:
        p_charge_kw, p_discharge_kw, p_grid_kw, soc, net_cost
    """

    hours = list(range(len(solar_kw)))
    n = len(hours)

    # -------------------------------------------------------------------------
    # Build the linear program using pyomo
    # -------------------------------------------------------------------------

    model = ConcreteModel()

    # --- Decision Variables ---
    # One value per hour for each variable
    # NonNegativeReals enforces the >= 0 constraint automatically
    model.p_curtail = Var(hours, domain=NonNegativeReals)
    model.p_charge = Var(hours, domain=NonNegativeReals)
    model.p_discharge = Var(hours, domain=NonNegativeReals)
    model.p_grid = Var(hours, domain=NonNegativeReals)

    # SOC is also a variable — the optimizer tracks it across hours
    # It's bounded between soc_min and soc_max
    model.soc = Var(
        hours,
        domain=NonNegativeReals,
        bounds=(battery.soc_min, battery.soc_max)
    )

    # --- Objective Function ---
    # Minimize total grid electricity cost across all hours
    # Cost = price(h) * p_grid(h) * 1 hour for each hour h

    def objective_rule(model):
        return sum(
            price_per_kwh.iloc[h] * model.p_grid[h]
            for h in hours
        )

    model.objective = Objective(rule=objective_rule, sense=minimize)

    # --- Constraint 1: Energy Balance ---
    # At every hour: solar + discharge + grid = load + charge
    # This is the fundamental physical law the system cannot violate

    def energy_balance_rule(model, h):
        return (
            solar_kw.iloc[h] + model.p_discharge[h] + model.p_grid[h] == load_kw.iloc[h] + model.p_charge[h] + model.p_curtail[h]
        )

    model.energy_balance = Constraint(hours, rule=energy_balance_rule)

    # --- Constraint 2: SOC Dynamics ---
    # SOC at next hour = SOC now + effect of charge/discharge
    # This is the difference equation we derived in the math foundation:
    #   SOC(t+1) = SOC(t) + (P_charge * η_charge - P_discharge / η_discharge) / E_max

    def soc_dynamics_rule(model, h):
        if h == 0:
            # First hour: SOC starts at battery's initial state
            return model.soc[0] == battery.soc

        return (
            model.soc[h] == model.soc[h - 1]
            + (
                model.p_charge[h - 1] * battery.charge_efficiency
                - model.p_discharge[h - 1] / battery.discharge_efficiency
            ) / battery.capacity_kwh
        )

    model.soc_dynamics = Constraint(hours, rule=soc_dynamics_rule)

    # --- Constraint 3: Charge Rate Limit ---
    def max_charge_rule(model, h):
        return model.p_charge[h] <= battery.max_charge_rate_kw

    model.max_charge = Constraint(hours, rule=max_charge_rule)

    # --- Constraint 4: Discharge Rate Limit ---
    def max_discharge_rule(model, h):
        return model.p_discharge[h] <= battery.max_discharge_rate_kw

    model.max_discharge = Constraint(hours, rule=max_discharge_rule)

    # --- Solve the linear program ---
    # GLPK is a free open source LP solver
    solver = SolverFactory("highs")
    result = solver.solve(model)

    # Check the solver actually found a valid solution
    if result.solver.termination_condition.name != "optimal":
        raise RuntimeError(
            f"Optimizer did not find an optimal solution. "
            f"Termination condition: {result.solver.termination_condition}"
        )

    # --- Extract results into a DataFrame ---
    # Pull the optimal value of each decision variable at each hour
    records = []
    for h in hours:
        p_charge = value(model.p_charge[h])
        p_discharge = value(model.p_discharge[h])
        p_grid = value(model.p_grid[h])
        soc = value(model.soc[h])
        net_cost = price_per_kwh.iloc[h] * p_grid

        records.append({
            "hour": h,
            "solar_kw": solar_kw.iloc[h],
            "load_kw": load_kw.iloc[h],
            "p_charge_kw": p_charge,
            "p_discharge_kw": p_discharge,
            "p_grid_kw": p_grid,
            "p_curtail_kw": value(model.p_curtail[h]),
            "soc": soc,
            "price": price_per_kwh.iloc[h],
            "net_cost": net_cost,
        })

    return pd.DataFrame(records)
