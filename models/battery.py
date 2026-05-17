import numpy as np


class BatteryModel:
    """
    Tracks battery State of Charge (SOC) across hourly timesteps.

    Models a lithium-ion battery with realistic round-trip efficiency
    losses and hard physical constraints.

    Parameters
    ----------
    capacity_kwh : float
        Total energy capacity in kilowatt-hours
    soc_min : float
        Minimum allowable SOC as a fraction (e.g. 0.2)
    soc_max : float
        Maximum allowable SOC as a fraction (e.g. 0.9)
    charge_efficiency : float
        Fraction of input power that reaches storage (e.g. 0.95)
    discharge_efficiency : float
        Fraction of stored power that reaches output (e.g. 0.95)
    max_charge_rate_kw : float
        Maximum charging power in kilowatts
    max_discharge_rate_kw : float
        Maximum discharging power in kilowatts
    initial_soc : float
        Starting SOC as a fraction (default 0.5)
    """

    def __init__(
        self,
        capacity_kwh: float,
        soc_min: float,
        soc_max: float,
        charge_efficiency: float,
        discharge_efficiency: float,
        max_charge_rate_kw: float,
        max_discharge_rate_kw: float,
        initial_soc: float = 0.5,
    ):
        self.capacity_kwh = capacity_kwh
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.max_charge_rate_kw = max_charge_rate_kw
        self.max_discharge_rate_kw = max_discharge_rate_kw

        # Current state — this updates as we step through time
        self.soc = initial_soc

        # History log — records SOC after every timestep for analysis
        self.soc_history = [initial_soc]

    def step(self, p_charge_kw: float, p_discharge_kw: float) -> float:
        """
        Advance the battery one timestep (one hour).

        Applies the SOC update equation:
            Charging:    SOC(t+1) = SOC(t) + (P_charge * η_charge) / E_max
            Discharging: SOC(t+1) = SOC(t) - (P_discharge / η_discharge) / E_max

        Parameters
        ----------
        p_charge_kw : float
            Power flowing into the battery this hour (kW)
        p_discharge_kw : float
            Power flowing out of the battery this hour (kW)

        Returns
        -------
        float
            Updated SOC after this timestep

        Raises
        ------
        ValueError
            If any physical constraint is violated
        """

        # Constraint 1: no negative power flows
        if p_charge_kw < 0:
            raise ValueError(f"p_charge_kw must be >= 0, got {p_charge_kw}")
        if p_discharge_kw < 0:
            raise ValueError(f"p_discharge_kw must be >= 0, got {p_discharge_kw}")

        # Constraint 2: power rates within physical limits
        if p_charge_kw > self.max_charge_rate_kw:
            raise ValueError(
                f"p_charge_kw {p_charge_kw} exceeds max {self.max_charge_rate_kw}"
            )
        if p_discharge_kw > self.max_discharge_rate_kw:
            raise ValueError(
                f"p_discharge_kw {p_discharge_kw} exceeds max {self.max_discharge_rate_kw}"
            )

        # Constraint 3: no simultaneous charge and discharge
        if p_charge_kw > 0 and p_discharge_kw > 0:
            raise ValueError(
                "Cannot charge and discharge simultaneously"
            )

        # SOC update — Δt = 1 hour so it drops out of the equation
        # Charging: energy stored = power * efficiency
        # Discharging: energy drawn from storage = power / efficiency
        delta_soc = (
            (p_charge_kw * self.charge_efficiency)
            - (p_discharge_kw / self.discharge_efficiency)
        ) / self.capacity_kwh

        new_soc = self.soc + delta_soc

        # Constraint 4: SOC stays within bounds
        if new_soc > self.soc_max + 1e-6:
            raise ValueError(
                f"SOC {new_soc:.4f} would exceed soc_max {self.soc_max}"
            )
        if new_soc < self.soc_min - 1e-6:
            raise ValueError(
                f"SOC {new_soc:.4f} would fall below soc_min {self.soc_min}"
            )

        # Clip tiny floating point drift at the boundaries
        self.soc = float(np.clip(new_soc, self.soc_min, self.soc_max))
        self.soc_history.append(self.soc)

        return self.soc

    def reset(self, initial_soc: float = 0.5):
        """Reset battery to initial state — useful between simulation runs."""
        self.soc = initial_soc
        self.soc_history = [initial_soc]